#!/usr/bin/env python3
"""
Test script to open multiple cameras via v4l/by-path and display them in real-time.
Usage:
    python test_camera_stream.py
    
    Or with specific devices:
    python test_camera_stream.py --devices /dev/v4l/by-path/device1 /dev/v4l/by-path/device2
    
Press 'q' to quit, 's' to save current frames.
"""

import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from threading import Thread
import queue


def get_available_by_path_devices():
    """Get all available v4l/by-path devices."""
    by_path_dir = "/dev/v4l/by-path"
    if not os.path.exists(by_path_dir):
        print(f"Directory {by_path_dir} not found!")
        return []
    
    devices = []
    for item in sorted(os.listdir(by_path_dir)):
        item_path = os.path.join(by_path_dir, item)
        if os.path.islink(item_path):
            real_path = os.path.realpath(item_path)
            devices.append({
                'by_path': item_path,
                'real_path': real_path,
                'name': item
            })
    
    return devices


def resolve_device_path(device_or_symlink):
    """Resolve device path (by-path or /dev/videoX) to real path."""
    if device_or_symlink.startswith('/dev/v4l/by-path/'):
        real_path = os.path.realpath(device_or_symlink)
        return real_path
    return device_or_symlink


def open_camera(device_path, width=640, height=480):
    """Open a camera device with OpenCV."""
    try:
        # Resolve by-path to real device
        real_path = resolve_device_path(device_path)
        print(f"Opening {device_path}")
        print(f"  -> Real path: {real_path}")
        
        cap = cv2.VideoCapture(real_path, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            print(f"  ERROR: Failed to open!")
            return None
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Get actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"  SUCCESS: {actual_width}x{actual_height} @ {fps}fps")
        
        return cap
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def capture_frames(cap, device_name, frame_queue):
    """Continuously capture frames from a camera."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add device name to frame
        cv2.putText(frame, device_name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Drop frame if queue is full


def display_single_camera(device_path, width=640, height=480):
    """Display a single camera in real-time."""
    cap = open_camera(device_path, width, height)
    if cap is None:
        return
    
    device_name = os.path.basename(device_path)
    window_name = f"Camera: {device_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    print(f"\nDisplaying {device_name}... Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from {device_name}")
            break
        
        # Add info to frame
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        cv2.imshow(window_name, frame)
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"camera_frame_{device_name}_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()


def display_multiple_cameras(device_paths, width=640, height=480):
    """Display multiple cameras in a grid layout."""
    caps = {}
    threads = {}
    frame_queues = {}
    
    # Open all cameras
    print("\nOpening cameras...\n")
    for device_path in device_paths:
        device_name = os.path.basename(device_path)
        cap = open_camera(device_path, width, height)
        if cap is not None:
            caps[device_name] = cap
            frame_queues[device_name] = queue.Queue(maxsize=2)
            
            # Start capture thread
            thread = Thread(target=capture_frames, args=(cap, device_name, frame_queues[device_name]), daemon=True)
            thread.start()
            threads[device_name] = thread
    
    if not caps:
        print("Failed to open any cameras!")
        return
    
    print(f"\nOpened {len(caps)} camera(s). Press 'q' to quit, 's' to save frames\n")
    
    # Calculate grid layout
    num_cameras = len(caps)
    cols = int(np.ceil(np.sqrt(num_cameras)))
    rows = int(np.ceil(num_cameras / cols))
    
    # Get frame dimensions from first camera
    ret, test_frame = list(caps.values())[0].read()
    frame_h, frame_w = test_frame.shape[:2]
    
    display_w = min(1920 // cols, 640)
    display_h = int(display_w * frame_h / frame_w)
    
    window_name = "Multi-Camera View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, cols * display_w, rows * display_h)
    
    device_names = list(caps.keys())
    frame_count = 0
    # Cache last received frame per camera to avoid black flicker when queue is momentarily empty
    cached_frames = {name: np.zeros((display_h, display_w, 3), dtype=np.uint8) for name in device_names}

    while True:
        # Create grid canvas
        canvas = np.zeros((rows * display_h, cols * display_w, 3), dtype=np.uint8)

        # Collect frames from all cameras
        for idx, device_name in enumerate(device_names):
            try:
                frame = frame_queues[device_name].get_nowait()
                cached_frames[device_name] = cv2.resize(frame, (display_w, display_h))
            except queue.Empty:
                pass  # Reuse cached frame to avoid flickering

            # Place in grid
            row = idx // cols
            col = idx % cols
            y_start = row * display_h
            y_end = y_start + display_h
            x_start = col * display_w
            x_end = x_start + display_w

            canvas[y_start:y_end, x_start:x_end] = cached_frames[device_name]
        
        # Add frame counter
        cv2.putText(canvas, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(window_name, canvas)
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"multicam_screen_{frame_count}.jpg"
            cv2.imwrite(filename, canvas)
            print(f"Saved: {filename}")
    
    # Cleanup
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Test v4l2 cameras and display them in real-time"
    )
    parser.add_argument(
        "--devices", 
        nargs="+",
        help="Specific device paths to open (v4l/by-path or /dev/videoX)"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Display only the first device in single window mode"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera frame width (default: 640)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera frame height (default: 480)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available v4l/by-path devices and exit"
    )
    
    args = parser.parse_args()
    
    # List available devices
    if args.list:
        devices = get_available_by_path_devices()
        if not devices:
            print("No v4l/by-path devices found!")
        else:
            print("\n" + "=" * 80)
            print("AVAILABLE V4L/BY-PATH DEVICES")
            print("=" * 80 + "\n")
            for i, dev in enumerate(devices, 1):
                print(f"{i}. {dev['by_path']}")
                print(f"   -> {dev['real_path']}\n")
        return
    
    # Determine which devices to open
    if args.devices:
        device_paths = args.devices
    else:
        devices = get_available_by_path_devices()
        if not devices:
            print("No v4l/by-path devices found!")
            print("Run with --list to see available devices")
            return
        device_paths = [dev['by_path'] for dev in devices]
    
    print(f"Opening {len(device_paths)} device(s)...")
    
    # Display cameras
    if args.single or len(device_paths) == 1:
        display_single_camera(device_paths[0], args.width, args.height)
    else:
        display_multiple_cameras(device_paths, args.width, args.height)


if __name__ == "__main__":
    main()

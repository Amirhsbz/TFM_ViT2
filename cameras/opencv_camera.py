from typing import Optional, Tuple
import json
import os

import cv2
import numpy as np


def _resolve_v4l_device_path(device_path: str) -> str:
    """Resolve v4l/by-path symlink to actual /dev/videoX device.
    
    Args:
        device_path: v4l2 device path, either:
            - Symlink: '/dev/v4l/by-path/pci-...'
            - Direct: '/dev/videoX'
    
    Returns:
        Resolved device path (e.g., '/dev/video0')
    """
    if not isinstance(device_path, str) or not device_path.startswith('/dev/'):
        return device_path
    
    # If it's already a /dev/videoX path, return as is
    if device_path.startswith('/dev/video'):
        if os.path.exists(device_path):
            return device_path
        else:
            raise FileNotFoundError(f"Device {device_path} not found")
    
    # If it's a v4l/by-path symlink, resolve it
    if device_path.startswith('/dev/v4l/by-path/'):
        # Check if symlink exists first
        if not os.path.exists(device_path) and not os.path.islink(device_path):
            # List available by-path devices for debugging
            by_path_dir = '/dev/v4l/by-path/'
            if os.path.exists(by_path_dir):
                available = os.listdir(by_path_dir)
                print(f"Warning: Device {device_path} not found!")
                print(f"Available v4l/by-path devices: {available}")
            raise FileNotFoundError(f"v4l/by-path device symlink not found: {device_path}")
        
        try:
            # Resolve the symlink to get the real path
            real_path = os.path.realpath(device_path)
            if os.path.exists(real_path):
                print(f"Resolved {device_path} -> {real_path}")
                return real_path
            else:
                raise FileNotFoundError(f"Symlink target {real_path} not found")
        except Exception as e:
            raise RuntimeError(f"Failed to resolve v4l/by-path device {device_path}: {e}")
    
    return device_path


class OpenCVCamera:
    """Simple OpenCV camera driver for webcams and v4l2 devices."""

    def __init__(
        self,
        camera_id,
        width: int = 640,
        height: int = 480,
        perspective_config_path: Optional[str] = None,
        perspective_key: Optional[str] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        """Initialize the OpenCV camera.

        Args:
            camera_id: The camera device path (str) or device ID (int).
                      - For v4l2 by-path: str like '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:1:1.0'
                      - For v4l2 video: str like '/dev/video0' or '/dev/video1'
                      - For int ID: int like 0, 1, 2, etc.
            width: The width of the camera frame.
            height: The height of the camera frame.
            perspective_config_path: Optional path to a JSON config with four corner
                points for perspective rectification.
            perspective_key: Optional key inside the JSON config, e.g. "tactile_left".
            output_size: Optional default returned image size as (width, height).
        """
        self.camera_id = camera_id
        self._perspective_transform = None
        self._perspective_size = None
        self._perspective_config_path = perspective_config_path
        self._perspective_key = perspective_key
        self._output_size = output_size
        self._last_raw_frame_rgb = None
        
        # Support v4l2 by-path, regular v4l2 paths, and int device IDs
        try:
            if isinstance(camera_id, str):
                # Resolve v4l/by-path symlink if necessary
                resolved_path = _resolve_v4l_device_path(camera_id)
                print(f"Opening v4l2 device: {resolved_path}")
                self.cap = cv2.VideoCapture(resolved_path, cv2.CAP_V4L2)
            else:
                # int device ID
                print(f"Opening device ID: {camera_id}")
                self.cap = cv2.VideoCapture(camera_id)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error: {e}")
            raise
        
        if not self.cap.isOpened():
            # Try to provide more helpful error messages
            error_msg = f"Failed to open camera {camera_id}."
            if isinstance(camera_id, str):
                if camera_id.startswith('/dev/v4l/by-path/'):
                    error_msg += " The v4l/by-path device symlink exists but cannot be opened by OpenCV."
                    error_msg += " Try checking if the device is already in use or has permission issues."
                else:
                    error_msg += " The device path doesn't exist or is not accessible."
            else:
                error_msg += f" Device ID {camera_id} is not available or already in use."
            raise RuntimeError(error_msg)
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if perspective_config_path:
            self._load_perspective_transform(perspective_config_path, perspective_key)
        else:
            print(f"Perspective warp disabled for {camera_id}")
        
        print(f"Camera {camera_id} initialized: {self.width}x{self.height}")

    def _load_perspective_transform(
        self, config_path: str, config_key: Optional[str]
    ) -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Perspective config not found: {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if config_key:
            entry = config.get(config_key)
            if entry is None and config_key.startswith("tactile_"):
                entry = config.get(config_key.replace("tactile_", ""))
            if entry is None and "points" in config:
                entry = config
            if entry is None:
                raise KeyError(
                    f"Key '{config_key}' not found in perspective config: {config_path}"
                )
        else:
            entry = config

        if "points" not in entry:
            raise KeyError(
                f"Perspective config entry must contain 'points': {config_path}"
            )

        src_pts = np.array(entry["points"], dtype=np.float32)
        if src_pts.shape != (4, 2):
            raise ValueError(
                f"Perspective config must contain 4 points, got shape {src_pts.shape}"
            )

        out_width, out_height = self._resolve_output_size(entry, src_pts)
        dst_pts = np.array(
            [
                [0, 0],
                [out_width, 0],
                [out_width, out_height],
                [0, out_height],
            ],
            dtype=np.float32,
        )

        self._perspective_transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self._perspective_size = (int(out_width), int(out_height))
        print(
            f"Loaded perspective warp for {self.camera_id} from {config_path}"
            f" (key={config_key or 'root'} -> {self._perspective_size[0]}x{self._perspective_size[1]})"
        )

    @staticmethod
    def _resolve_output_size(entry: dict, src_pts: np.ndarray) -> Tuple[int, int]:
        if "output_size" in entry:
            out_width, out_height = entry["output_size"]
        elif "size" in entry:
            out_width, out_height = entry["size"]
        elif "out_width" in entry and "out_height" in entry:
            out_width = entry["out_width"]
            out_height = entry["out_height"]
        else:
            tl, tr, br, bl = src_pts
            out_width = int(np.mean([tr[0], br[0]]) - np.mean([tl[0], bl[0]]))
            out_height = int(np.mean([br[1], bl[1]]) - np.mean([tl[1], tr[1]]))

        out_width = int(round(out_width))
        out_height = int(round(out_height))
        if out_width <= 0 or out_height <= 0:
            raise ValueError(
                f"Invalid perspective output size: {out_width}x{out_height}"
            )
        return out_width, out_height

    def read(
        self, img_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.

        Returns:
            np.ndarray: The color image in RGB format (H, W, 3).
            np.ndarray: Empty depth image (H, W) - webcams don't have depth.
        """
        ret, frame = self.cap.read()
        
        if not ret:
            raise RuntimeError(f"Failed to read from camera {self.camera_id}")

        self._last_raw_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._perspective_transform is not None and self._perspective_size is not None:
            frame = cv2.warpPerspective(
                frame, self._perspective_transform, self._perspective_size
            )
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        output_size = img_size or self._output_size
        if output_size is not None:
            frame_rgb = cv2.resize(frame_rgb, output_size, interpolation=cv2.INTER_AREA)
        
        # Create empty depth image (webcams don't have depth)
        depth = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint16)
        
        return frame_rgb, depth

    def get_last_raw_frame_rgb(self) -> Optional[np.ndarray]:
        if self._last_raw_frame_rgb is None:
            return None
        return self._last_raw_frame_rgb.copy()

    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            print(f"Camera {self.camera_id} released")

    def __del__(self):
        """Destructor to ensure camera is released."""
        self.release()

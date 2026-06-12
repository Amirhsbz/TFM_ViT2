from dataclasses import dataclass
from multiprocessing import Process
from typing import List, Optional, Tuple

import numpy as np
import tyro

from camera_node import ZMQServerCamera, ZMQServerCameraFaster
from robot_node import ZMQServerRobot
from robots.robot import BimanualRobot


@dataclass
class Args:
    robot: str = "ur"
    hand_type: str = ""
    hostname: str = "127.0.0.1"
    robot_ip: str = "10.40.101.10"
    faster: bool = True
    cam_names: Tuple[str, ...] = ("first_view", "third_view")
    ability_gripper_grip_range: int = 110
    realsense_width: int = 640
    realsense_height: int = 480
    realsense_fps: int = 30
    img_size: Optional[Tuple[int, int]] = None  # (320, 240)
    logitech_cam_id: str = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Logi_C310_HD_WebCam_Logi_C310_HD_WebCam-video-index0"  # full /dev/v4l/by-path or by-id path for first_view Logitech


class MixedCamera:
    """Wraps an OpenCV camera (Logitech, index 0) + a single RealSense (index 1).

    Returns stacked arrays matching the shape RealSenseCamera produces for two cameras:
      image: (2, H, W, 3) uint8
      depth: (2, H, W)    uint16  — depth[0] is zeros (webcam has no depth)
    """

    def __init__(self, opencv_camera, realsense_camera):
        self._opencv = opencv_camera
        self._realsense = realsense_camera

    def read(self, img_size=None):
        rgb_logi, _ = self._opencv.read(img_size)         # (H, W, 3)
        rgb_rs, depth_rs = self._realsense.read(img_size)  # (1, H, W, 3), (1, H, W)
        H, W = rgb_logi.shape[:2]
        image = np.concatenate([rgb_logi[np.newaxis], rgb_rs], axis=0)
        depth = np.concatenate([np.zeros((1, H, W), dtype=np.uint16), depth_rs], axis=0)
        return image, depth

    def release(self):
        self._opencv.release()
        self._realsense.release()


def launch_server_cameras(port: int, camera_ports: List[str], args: Args):
    from cameras.opencv_camera import OpenCVCamera
    from cameras.realsense_camera import RealSenseCamera

    realsense_ports = [p for p in camera_ports if RealSenseCamera.supports_identifier(p)]
    opencv_ports = [p for p in camera_ports if not RealSenseCamera.supports_identifier(p)]

    if not opencv_ports:
        # All RealSense
        camera = RealSenseCamera(
            realsense_ports,
            width=args.realsense_width,
            height=args.realsense_height,
            fps=args.realsense_fps,
            img_size=args.img_size,
        )
    elif len(opencv_ports) == 1 and not realsense_ports:
        # Single OpenCV camera
        print(opencv_ports)
        camera = OpenCVCamera(opencv_ports[0], width=args.realsense_width, height=args.realsense_height)
    elif len(opencv_ports) == 1 and len(realsense_ports) == 1:
        # Mixed: Logitech at index 0, RealSense at index 1
        opencv_cam = OpenCVCamera(
            opencv_ports[0],
            width=args.realsense_width,
            height=args.realsense_height,
        )
        rs_cam = RealSenseCamera(
            realsense_ports,
            width=args.realsense_width,
            height=args.realsense_height,
            fps=args.realsense_fps,
            img_size=args.img_size,
        )
        camera = MixedCamera(opencv_cam, rs_cam)
    else:
        raise ValueError(
            f"Unsupported camera mix: {len(opencv_ports)} OpenCV + {len(realsense_ports)} RealSense. "
            "Supported: all-RealSense, single OpenCV, or exactly one of each."
        )

    if args.faster:
        server = ZMQServerCameraFaster(camera, port=port, host=args.hostname)
    else:
        server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting camera server on port {port}")
    server.serve()


def launch_robot_server(port: int, args: Args):
    if args.robot == "ur":
        from robots.ur import URRobot

        robot = URRobot(robot_ip=args.robot_ip)
    elif args.robot == "bimanual_ur":
        from robots.ur import URRobot

        if args.hand_type == "ability":
            # 6 DoF Ability Hand
            # robot_l - right hand; robot_r - left hand
            _robot_l = URRobot(
                robot_ip="111.111.1.3",
                no_gripper=False,
                gripper_type="ability",
                grip_range=args.ability_gripper_grip_range,
                port_idx=1,
            )
            _robot_r = URRobot(
                robot_ip="111.111.2.3",
                no_gripper=False,
                gripper_type="ability",
                grip_range=args.ability_gripper_grip_range,
                port_idx=2,
            )
        else:
            # Robotiq gripper
            _robot_l = URRobot(robot_ip="111.111.1.3", no_gripper=False)
            _robot_r = URRobot(robot_ip="111.111.2.3", no_gripper=False)
        robot = BimanualRobot(_robot_l, _robot_r)
    else:
        raise NotImplementedError(f"Robot {args.robot} not implemented")
    server = ZMQServerRobot(robot, port=port, host=args.hostname)
    print(f"Starting robot server on port {port}")
    server.serve()


def create_camera_server(args: Args) -> List[Process]:
    # first_view: Logitech webcam (device index or /dev/video* path)
    # third_view: RealSense D435i serial number
    cam_ports = {
        "first_view": args.logitech_cam_id,
        "third_view": "213622078586",
    }
    ports = [cam_ports.get(name, name) for name in args.cam_names]
    camera_port = 5000
    # start a single python process for all cameras
    print(f"Launching cameras {ports} on port {camera_port}")
    server = Process(target=launch_server_cameras, args=(camera_port, ports, args))
    return server

def main(args):
    camera_server = create_camera_server(args)
    print("Starting camera server process")
    camera_server.start()
    launch_robot_server(6000, args)

if __name__ == "__main__":
    main(tyro.cli(Args))

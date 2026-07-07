from dataclasses import dataclass
from multiprocessing import Process
from typing import List, Optional, Tuple

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


def launch_server_cameras(port: int, camera_ports: List[str], args: Args):
    from cameras.opencv_camera import OpenCVCamera
    from cameras.realsense_camera import RealSenseCamera

    if all(RealSenseCamera.supports_identifier(camera_id) for camera_id in camera_ports):
        camera = RealSenseCamera(
            camera_ports,
            width=args.realsense_width,
            height=args.realsense_height,
            fps=args.realsense_fps,
            img_size=args.img_size,
        )
    elif len(camera_ports) == 1:
        print(camera_ports)
        camera = OpenCVCamera(camera_ports[0], width=640, height=480)
    else:
        raise ValueError(
            "Multiple camera nodes currently require RealSense serials or aliases."
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


# Camera aliases. RealSense aliases are resolved to serial numbers at runtime.
CAM_PORTS = {
    "first_view": "338622073293",
    "third_view": "213622078586",
}

# CAM_PORTS = {
#     "435": "000000000000"
# }

def create_camera_server(args: Args) -> List[Process]:
    ports = [CAM_PORTS.get(name, name) for name in args.cam_names]
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

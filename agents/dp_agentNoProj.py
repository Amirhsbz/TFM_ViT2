import collections
import json
import os
from typing import Any, Dict

import numpy as np
import quaternion
import torch

from learning.dp.pipelineNoProj import Agent as DPAgent

IMAGE_KEYS = {"img", "tactile_img"}
DEFAULT_RESET_JOINTS = np.deg2rad([-87.0, -88.0, -112.0, -67.0, 90.0, 0.0])


def parse_txt_to_json(input_file_path, output_file_path):
    data = {}
    with open(input_file_path, "r") as file:
        for line in file:
            kv = line.strip().split(": ", 1)
            if len(kv) != 2:
                continue
            key, value = kv
            if key == "camera_indices":
                data[key] = list(map(int, value))
            elif key == "representation_type":
                data[key] = value.split("-")
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value == "True":
                            value = True
                        elif value == "False":
                            value = False
                        elif value == "None":
                            value = None
                data[key] = value

    with open(output_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    return data


def get_reset_joints(ur_eef=False):
    if ur_eef:
        return np.array(
            [-0.10244499, -0.74927846, 0.14209882, -0.36223588, -1.434728, 0.86917898, 0.0],
            dtype=np.float32,
        )
    return np.concatenate([DEFAULT_RESET_JOINTS, np.array([0.0], dtype=np.float32)])


def get_eef_pose(eef_pose, eef_delta):
    pos_delta = eef_delta[:3]
    rot_delta = eef_delta[3:6]
    pos = eef_pose[:3] + pos_delta
    rot = quaternion.as_rotation_vector(
        quaternion.from_rotation_vector(rot_delta)
        * quaternion.from_rotation_vector(eef_pose[3:6])
    )
    return np.concatenate((pos, rot, eef_delta[6:7]), axis=-1)


class BimanualDPAgent:
    """Single-arm diffusion-policy deployment agent.

    The class name is kept for compatibility with existing deployment scripts.
    """

    def __init__(
        self,
        ckpt_path,
        dp_args=None,
        gripper_min=0.0,
        gripper_max=1.0,
    ):
        if dp_args is None:
            dp_args = self.get_default_dp_args()

        args_txt = os.path.join(os.path.dirname(ckpt_path), "args_log.txt")
        args_json = os.path.join(os.path.dirname(ckpt_path), "args_log.json")
        args = parse_txt_to_json(args_txt, args_json)
        for k in dp_args.keys():
            if k == "output_sizes":
                dp_args[k]["img"] = None # unprojected 512-dim GAP feature, paper-aligned
                dp_args[k]["tactile_img"] = args["image_output_size"]
            elif k in args:
                dp_args[k] = args[k]

        ckpt_dir = os.path.dirname(ckpt_path)
        args_path = os.path.join(ckpt_dir, "dp_args.json")
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                dp_args = json.load(f)
        else:
            with open(args_path, "w") as f:
                json.dump(dp_args, f)

        torch.cuda.set_device(0)
        self.dp = DPAgent(
            output_sizes=dp_args["output_sizes"],
            representation_type=dp_args["representation_type"],
            identity_encoder=dp_args["identity_encoder"],
            camera_indices=dp_args["camera_indices"],
            pred_horizon=dp_args["pred_horizon"],
            obs_horizon=dp_args["obs_horizon"],
            action_horizon=dp_args["action_horizon"],
            without_sampling=dp_args["without_sampling"],
            predict_eef_delta=dp_args["predict_eef_delta"],
            predict_pos_delta=dp_args["predict_pos_delta"],
            use_ddim=dp_args["use_ddim"],
            joint_state_dim=dp_args["joint_state_dim"],
            action_dim=dp_args["action_dim"],
            eef_state_dim=dp_args["eef_dim"],
            touch_dim=dp_args["touch_dim"],
            hand_pos_dim=dp_args["hand_pos_dim"],
        )
        self.dp_args = dp_args
        self.obsque = collections.deque(maxlen=dp_args["obs_horizon"])
        self.dp.load(ckpt_path)
        self.action_queue = collections.deque(maxlen=dp_args["action_horizon"])
        self.predict_eef_delta = dp_args["predict_eef_delta"]
        self.predict_pos_delta = dp_args["predict_pos_delta"]
        assert not (self.predict_eef_delta and self.predict_pos_delta)
        self.control = get_reset_joints(ur_eef=self.predict_eef_delta)
        self.num_diffusion_iters = dp_args["num_diffusion_iters"]
        self.trigger_state = True
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max

    def reset_temporal_state(self) -> None:
        self.obsque.clear()
        self.action_queue.clear()
        self.control = get_reset_joints(ur_eef=self.predict_eef_delta)

    @staticmethod
    def get_default_dp_args():
        return {
            "output_sizes": {
                "eef": 64,
                "hand_pos": 64,
                "img": None,  # unprojected 512-dim GAP feature, paper-aligned
                "tactile_img": 128,
                "pos": 128,
                "touch": 64,
            },
            "representation_type": ["img", "tactile_img", "pos"],
            "identity_encoder": False,
            "camera_indices": [0, 1],
            "obs_horizon": 1,
            "pred_horizon": 16,
            "action_horizon": 8,
            "num_diffusion_iters": 15,
            "without_sampling": False,
            "clip_far": False,
            "predict_eef_delta": False,
            "predict_pos_delta": False,
            "use_ddim": False,
            "joint_state_dim": 7,
            "action_dim": 7,
            "eef_dim": 6,
            "touch_dim": 30,
            "hand_pos_dim": 12,
        }

    def compile_inference(self, example_obs, precision="high", num_inference_iters=5):
        torch.set_float32_matmul_precision(precision)
        self.dp.policy.forward = torch.compile(torch.no_grad(self.dp.policy.forward))
        self.num_diffusion_iters = num_inference_iters
        for _ in range(25):
            self.act(example_obs)
        self.reset_temporal_state()

    def _preprocess_obs(self, obs: Dict[str, Any]):
        obs = self.dp.get_observation([obs], load_img=True)
        for image_key in IMAGE_KEYS:
            if image_key in obs:
                obs[image_key] = self.dp.eval_transform(obs[image_key].squeeze(0))
        return obs

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        curr_joint_pos = np.asarray(obs["joint_positions"], dtype=np.float32)
        curr_eef_pose = np.asarray(obs["ee_pos_quat"], dtype=np.float32)
        obs = self._preprocess_obs(obs)

        if len(self.obsque) == 0:
            self.obsque.extend([obs] * self.dp_args["obs_horizon"])
        else:
            self.obsque.append(obs)

        if len(self.action_queue) > 0:
            act = self.action_queue.popleft()
        else:
            pred = self.dp.predict(
                self.obsque, num_diffusion_iters=self.num_diffusion_iters
            )
            for i in range(self.dp_args["action_horizon"]):
                self.action_queue.append(pred[i])
            act = self.action_queue.popleft()

        act = np.asarray(act, dtype=np.float32)

        if self.predict_pos_delta:
            self.control[: len(curr_joint_pos)] = curr_joint_pos
            self.control = self.control + act
            act = self.control
        elif self.predict_eef_delta:
            act = get_eef_pose(curr_eef_pose, act)

        if not self.predict_eef_delta:
            act[-1] = np.clip(act[-1], self.gripper_min, self.gripper_max)

        return act

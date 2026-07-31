import argparse
import collections
import concurrent
import os
import pickle
import sys
import time
import hashlib

mypath = os.path.dirname(os.path.realpath(__file__))

print("adding", mypath, "to the sys path")

sys.path.append(mypath)

import data_processing
import numpy as np
import torch
from dataset import Dataset
from learnerNoProj import DiffusionPolicy
from modelsNoProj import GaussianNoise, ImageEncoder, StateEncoder
from torch import nn
from torch.nn import ModuleList
from torchvision import transforms
from utils import WandBLogger, generate_random_string, get_eef_delta, save_args

LEFT_UR_IDX = list(range(0, 6))
RIGHT_UR_IDX = list(range(12, 18))
LEFT_HAND_IDX = list(range(6, 12))
RIGHT_HAND_IDX = list(range(18, 24))
HAND_IDX = LEFT_HAND_IDX + RIGHT_HAND_IDX
IMAGE_KEYS = {"img", "tactile_img"}
TEST_INPUT = {
    "joint_positions": torch.zeros(7),
    "ee_pos_quat": torch.zeros(6),
    "base_rgb": torch.zeros(3, 480, 640, 3),
    "base_depth": torch.zeros(3, 480, 640),
    "control": torch.zeros(7),
    "touch": torch.zeros(30),
    "hand_pos": torch.zeros(12),
}


class Agent:
    def __init__(
        self,
        output_sizes={
            "eef": 64,
            "hand_pos": 64,
            "img": None,
            "tactile_img": 128,
            "pos": 128,
            "touch": 64,
        },
        dropout={
            "eef": 0.0,
            "hand_pos": 0.0,
            "img": 0.0,
            "tactile_img": 0.0,
            "pos": 0.0,
            "touch": 0.0,
        },
        action_dim=24,
        camera_indices=[0, 1, 2],
        representation_type=["eef", "hand_pos", "img", "touch", "depth"],
        pred_horizon=4,
        obs_horizon=1,
        action_horizon=2,
        identity_encoder=False,
        without_sampling=False,
        predict_eef_delta=False,
        predict_pos_delta=False,
        clip_far=False,
        color_jitter=False,
        num_diffusion_iters=100,
        load_img=False,
        weight_decay=1e-6,
        num_workers=64,
        use_ddim=False,
        binarize_touch=False,
        policy_dropout_rate=0.0,
        state_noise=0.0,
        img_gaussian_noise=0.0,
        img_masking_prob=0.0,
        img_patch_size=16,
        compile_train=False,
        joint_state_dim=24,
        eef_state_dim=12,
        touch_dim=60,
        hand_pos_dim=12,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_workers = num_workers
        self.binarize_touch = binarize_touch

        # Merge depth and rgb image
        if "depth" in representation_type:
            self.image_channel = 4
            representation_type.remove("depth")
        else:
            self.image_channel = 3
        self.representation_type = representation_type

        self.camera_indices = camera_indices
        self.predict_pos_delta = predict_pos_delta
        self.image_num = len(camera_indices)
        self.tactile_image_num = 2
        self.tactile_image_channel = 3
        self.cpu = torch.device("cpu")
        image_encoder, touch_encoder, pos_encoder = None, None, None
        eef_dim, hand_pos_dim, image_dim, touch_dim, pos_dim = 0, 0, 0, 0, 0
        self.rt_dim = {
            "eef": eef_state_dim,
            "hand_pos": hand_pos_dim,
            "pos": joint_state_dim,
            "touch": touch_dim,
            "action": action_dim,
        }
        self.clip_far = clip_far
        self.load_img = load_img
        self.color_jitter = color_jitter
        self.epi_dir = []
        self.compile_train = compile_train

        # Use color jitter to augment the image
        if self.color_jitter:
            if self.image_channel == 3:
                # no depth
                self.downsample = nn.Sequential(
                    transforms.Resize(
                        (240, 320),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                    transforms.ColorJitter(brightness=0.1),
                )
            else:
                # with depth, only jitter the rgb part
                self.downsample = lambda x: transforms.Resize(
                    (240, 320),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                )(
                    torch.concat(
                        [transforms.ColorJitter(brightness=0.1)(x[:, :3]), x[:, 3:]],
                        axis=1,
                    )
                )

        # Not using color jitter, only downsample the image
        else:
            self.downsample = nn.Sequential(
                transforms.Resize(
                    (240, 320),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
            )

        mean_vec = [128.0] * 3 + [35767.0] * (self.image_channel - 3)
        std_vec = [128.0] * 3 + [35767.0] * (self.image_channel - 3)

        # Crop randomization, normalization
        self.transform = nn.Sequential(
            transforms.RandomCrop((216, 288)),
            transforms.Normalize(
                mean=mean_vec,
                std=std_vec,
            ),
        )

        # Add gaussian noise to the image
        if img_gaussian_noise > 0.0:
            self.transform = nn.Sequential(
                self.transform,
                GaussianNoise(img_gaussian_noise),
            )

        def mask_img(x):
            # Divide the image into patches and randomly mask some of them
            img_patch = x.unfold(2, img_patch_size, img_patch_size).unfold(
                3, img_patch_size, img_patch_size
            )
            mask = (
                torch.rand(
                    (
                        x.shape[0],
                        x.shape[-2] // img_patch_size,
                        x.shape[-1] // img_patch_size,
                    )
                )
                < img_masking_prob
            )
            mask = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(img_patch)
            x = x.clone()
            x.unfold(2, img_patch_size, img_patch_size).unfold(
                3, img_patch_size, img_patch_size
            )[mask] = 0
            return x

        if img_masking_prob > 0.0:
            self.transform = lambda x: mask_img(
                nn.Sequential(
                    transforms.RandomCrop((216, 288)),
                    transforms.Normalize(mean=mean_vec, std=std_vec),
                )(x)
            )
        # For evaluation, only center crop and normalize
        self.eval_transform = nn.Sequential(
            transforms.CenterCrop((216, 288)),
            transforms.Normalize(
                mean=mean_vec,
                std=std_vec,
            ),
        )

        self.stats = None
        obs_dim = 0
        encoders = {}
        if "eef" in self.representation_type:
            eef_dim = self.rt_dim["eef"]
            if identity_encoder:
                eef_encoder = nn.Identity(eef_dim)
            else:
                eef_encoder = StateEncoder(
                    input_size=eef_dim,
                    output_size=output_sizes["eef"],
                    hidden_size=128,
                    dropout=dropout["eef"],
                )
                eef_dim = output_sizes["eef"]
            encoders["eef"] = eef_encoder
            obs_dim += eef_dim
        if "hand_pos" in self.representation_type:
            hand_pos_dim = self.rt_dim["hand_pos"]
            if identity_encoder:
                hand_pos_encoder = nn.Identity(hand_pos_dim)
            else:
                hand_pos_encoder = StateEncoder(
                    input_size=hand_pos_dim,
                    output_size=output_sizes["hand_pos"],
                    hidden_size=128,
                    dropout=dropout["hand_pos"],
                )
                hand_pos_dim = output_sizes["hand_pos"]
            encoders["hand_pos"] = hand_pos_encoder
            obs_dim += hand_pos_dim
        if "img" in self.representation_type:
            img_output_size = (output_sizes["img"] if output_sizes["img"] is not None 
                               else ImageEncoder.RAW_OUTPUT_SIZE)
            image_encoder = ModuleList(
                [
                    # Use different image encoders for each camera
                    ImageEncoder(
                        output_sizes["img"], self.image_channel, dropout["img"]
                    )
                    for i in range(self.image_num)
                ]
            )
            image_dim = img_output_size * self.image_num
            encoders["img"] = image_encoder
            obs_dim += image_dim
        if "tactile_img" in self.representation_type:
            tactile_encoder = ModuleList(
                [
                    ImageEncoder(
                        output_sizes["tactile_img"],
                        self.tactile_image_channel,
                        dropout["tactile_img"],
                    )
                    for _ in range(self.tactile_image_num)
                ]
            )
            tactile_image_dim = output_sizes["tactile_img"] * self.tactile_image_num
            encoders["tactile_img"] = tactile_encoder
            obs_dim += tactile_image_dim
        if "pos" in self.representation_type:
            pos_dim = self.rt_dim["pos"]
            if identity_encoder:
                pos_encoder = nn.Identity(pos_dim)
            else:
                pos_encoder = StateEncoder(
                    pos_dim, output_sizes["pos"], dropout=dropout["pos"]
                )
                pos_dim = output_sizes["pos"]
            encoders["pos"] = pos_encoder
            obs_dim += pos_dim
        if "touch" in self.representation_type:
            touch_dim = self.rt_dim["touch"]
            if identity_encoder:
                touch_encoder = nn.Identity(touch_dim)
            else:
                touch_encoder = StateEncoder(
                    touch_dim,
                    output_sizes["touch"],
                    dropout=dropout["touch"],
                    binarize_touch=binarize_touch,
                )
                touch_dim = output_sizes["touch"]
            encoders["touch"] = touch_encoder
            obs_dim += touch_dim

        self.policy = DiffusionPolicy(
            obs_horizon=obs_horizon,
            obs_dim=obs_dim,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            action_dim=action_dim,
            representation_type=representation_type,
            encoders=encoders,
            num_diffusion_iters=num_diffusion_iters,
            without_sampling=without_sampling,
            weight_decay=weight_decay,
            use_ddim=use_ddim,
            binarize_touch=self.binarize_touch,
            policy_dropout_rate=policy_dropout_rate,
        )

        # Compile the forward function to accelerate deployment inference
        if self.compile_train:
            self.policy.nets["noise_pred_net"].forward = torch.compile(
                self.policy.nets["noise_pred_net"].forward
            )

        self.policy.to(self.device)
        self.iter = 0
        self.obs_deque = None
        self.threshold = 8000
        self.state_noise = state_noise

        self.predict_eef_delta = predict_eef_delta

    def _get_image_observation(self, data, image_key="img"):
        # allocate memory for the image
        if image_key == "img":
            image_num = len(self.camera_indices)
            image_channel = self.image_channel
        elif image_key == "tactile_img":
            image_num = self.tactile_image_num
            image_channel = self.tactile_image_channel
        else:
            raise ValueError(f"Unsupported image key: {image_key}")

        img = torch.zeros(
            (len(data), image_num, image_channel, 240, 320), dtype=torch.float32
        )

        if image_key == "img":
            base_rgb_key = "base_rgb" if "base_rgb" in data[0] else "base_camera_rgb"
            base_depth_key = (
                "base_depth" if "base_depth" in data[0] else "base_camera_depth"
            )
            image_size = data[0][base_rgb_key].shape
            H, W = image_size[1], image_size[2]

            if self.image_channel == 4:
            # Use depth
                def process_rgbd(d):
                    rgbd = np.concatenate(
                        [d[base_rgb_key], d[base_depth_key][..., None]], axis=-1
                    ).reshape(-1, H, W, self.image_channel)
                    if self.clip_far:
                        clip_back_view = d[base_depth_key][0] > (self.threshold / 10)
                        clip_wrist = d[base_depth_key][1:] > (self.threshold)
                        clip = np.concatenate([clip_back_view, clip_wrist], axis=0)
                        clip = np.concatenate(
                            [clip[..., None]] * self.image_channel, axis=-1
                        )
                        rgbd = rgbd * clip
                    rgbd = rgbd[self.camera_indices].astype(np.float32)
                    rgbd = np.moveaxis(rgbd, -1, 1)
                    if H == 480 and W == 640 and self.color_jitter is False:
                        rgbd = self.downsample(torch.tensor(rgbd))
                    else:
                        rgbd = torch.tensor(rgbd)
                    return rgbd

                fn = process_rgbd

            else:
                def process_rgb(d):
                    rgb = d[base_rgb_key].reshape(-1, H, W, self.image_channel)
                    rgb = rgb[self.camera_indices].astype(np.float32)
                    rgb = np.moveaxis(rgb, -1, 1)
                    if H == 480 and W == 640 and self.color_jitter is False:
                        rgb = self.downsample(torch.tensor(rgb))
                    else:
                        rgb = torch.tensor(rgb)
                    return rgb

                fn = process_rgb
        else:
            tactile_keys = ["tactile_left_rgb", "tactile_right_rgb"]
            image_size = data[0][tactile_keys[0]].shape
            H, W = image_size[0], image_size[1]

            def process_tactile(d):
                rgb = np.stack([d[k] for k in tactile_keys], axis=0).astype(np.float32)
                rgb = np.moveaxis(rgb, -1, 1)
                if H == 480 and W == 640 and self.color_jitter is False:
                    rgb = self.downsample(torch.tensor(rgb))
                else:
                    rgb = torch.tensor(rgb)
                return rgb

            fn = process_tactile

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            future_to_data = {
                executor.submit(fn, d): (d, i) for i, d in enumerate(data)
            }
            failures = []
            for future in concurrent.futures.as_completed(future_to_data):
                d, i = future_to_data[future]
                try:
                    img[i] = future.result()
                except Exception as exc:
                    failures.append((i, d.get("file_path", "<live_obs>"), exc))

        if failures:
            examples = ", ".join(
                f"index={i}, source={source}, error={exc}"
                for i, source, exc in failures[:3]
            )
            raise RuntimeError(
                f"Failed to load {len(failures)} {image_key} observation(s): {examples}"
            )

        return img

    def get_observation(self, data, load_img=False):
        input_data = {}
        for rt in self.representation_type:
            if rt in IMAGE_KEYS:
                if load_img:
                    input_data[rt] = self._get_image_observation(data, image_key=rt)
                else:
                    input_data[rt] = np.stack([d["file_path"] for d in data])
            elif rt == "eef":
                input_data[rt] = np.stack([d["ee_pos_quat"] for d in data])
            elif rt == "hand_pos":
                if self.rt_dim["hand_pos"] != len(HAND_IDX):
                    raise NotImplementedError(
                        "hand_pos representation is only implemented for the legacy 12-dim dual-hand layout."
                    )
                input_data[rt] = np.stack(
                    [d["joint_positions"][HAND_IDX] for d in data]
                )
            elif rt == "pos":
                input_data[rt] = np.stack([d["joint_positions"] for d in data])
            else:
                input_data[rt] = np.stack([d[rt] for d in data])
        return input_data

    def predict(self, obs_deque: collections.deque, num_diffusion_iters=15):
        """
        data: dict
            data['image']: torch.tensor (1,5,224,224)
            data['touch']: torch.tensor (1,6)
            data['pos']: torch.tensor (1,24)
        """
        pred = self.policy.forward(
            self.stats, obs_deque, num_diffusion_iters=num_diffusion_iters
        )
        return pred

    def _get_init_train_data(
        self,
        total_data_points,
        memmap_loader_path="",
        store_img_in_memory=False,
    ):
        init_data = {}
        for rt in self.representation_type + ["action"]:
            if rt in IMAGE_KEYS:
                use_memmap_for_this_key = rt in IMAGE_KEYS and memmap_loader_path != ""
                if not use_memmap_for_this_key:
                    if store_img_in_memory:
                        image_num = self.image_num if rt == "img" else self.tactile_image_num
                        image_channel = (
                            self.image_channel
                            if rt == "img"
                            else self.tactile_image_channel
                        )
                        init_data[rt] = np.empty(
                            (
                                total_data_points,
                                image_num,
                                image_channel,
                                240,
                                320,
                            ),
                            dtype=np.uint16,
                        )
                    else:
                        init_data[rt] = np.empty((total_data_points,), dtype=object)
            else:
                init_data[rt] = np.zeros(
                    (total_data_points, self.rt_dim[rt]), dtype=np.float32
                )
        return init_data

    def _get_image_memmap_path(self, memmap_loader_path, image_key):
        if image_key == "img" or memmap_loader_path == "":
            return memmap_loader_path
        root, ext = os.path.splitext(memmap_loader_path)
        if ext == "":
            return f"{memmap_loader_path}-{image_key}"
        return f"{root}-{image_key}{ext}"

    def _open_episode_image_memmaps(self, total_data_points, memmap_loader_path=""):
        image_memmaps = {}
        cache_memmap = False
        required_image_keys = [k for k in IMAGE_KEYS if k in self.representation_type]
        if memmap_loader_path == "":
            return image_memmaps, cache_memmap

        for image_key in required_image_keys:
            image_num = self.image_num if image_key == "img" else self.tactile_image_num
            image_channel = (
                self.image_channel
                if image_key == "img"
                else self.tactile_image_channel
            )
            image_shape = (total_data_points, image_num, image_channel, 240, 320)
            image_memmap_path = self._get_image_memmap_path(
                memmap_loader_path, image_key
            )
            if os.path.exists(image_memmap_path):
                image_memmaps[image_key] = np.memmap(
                    image_memmap_path,
                    dtype=np.uint16,
                    mode="r",
                    shape=image_shape,
                )
            else:
                cache_memmap = True
                image_memmap_dir = os.path.dirname(image_memmap_path)
                if image_memmap_dir != "":
                    os.makedirs(image_memmap_dir, exist_ok=True)
                image_memmaps[image_key] = np.memmap(
                    image_memmap_path,
                    dtype=np.uint16,
                    mode="w+",
                    shape=image_shape,
                )
        return image_memmaps, cache_memmap

    def get_train_loader(self, batch_size, memmap_loader_path="", eval=False):
        current_epi_dir = self.epi_dir

        total_data_points = sum(
            [data_processing.get_episode_length(epi) for epi in current_epi_dir]
        )
        has_h5_episode = any(
            os.path.exists(os.path.join(epi, "trajectory.h5")) for epi in current_epi_dir
        )
        store_img_in_memory = self.load_img
        train_data = {"data": {}, "meta": {}}
        image_memmaps, cache_memmap = self._open_episode_image_memmaps(
            total_data_points, memmap_loader_path
        )
        for image_key, image_memmap in image_memmaps.items():
            train_data["data"][image_key] = image_memmap

        # H5 trajectories only need image decoding when memmap cache is missing.
        if has_h5_episode and (memmap_loader_path == "" or cache_memmap):
            store_img_in_memory = True

        train_data["data"] = self._get_init_train_data(
            total_data_points,
            memmap_loader_path=memmap_loader_path,
            store_img_in_memory=store_img_in_memory,
        )
        train_data["meta"] = {"episode_ends": []}
        if memmap_loader_path != "":
            for image_key, image_memmap in image_memmaps.items():
                if image_key not in train_data["data"]:
                    train_data["data"][image_key] = image_memmap

        data_index = 0

        print("Loading training data    ")
        for i, epi in enumerate(current_epi_dir):
            print("loading {}-th data from {}\r".format(i, epi), end="")
            data = data_processing.iterate(epi, load_img=store_img_in_memory)
            if len(data) == 0:
                continue

            data_length = len(data)

            # images - (N, num_cams, self.image_channel, 240, 320)
            obs = self.get_observation(data, store_img_in_memory or cache_memmap)

            # obs space
            for rt in self.representation_type:
                if rt in IMAGE_KEYS:
                    if rt in image_memmaps and cache_memmap:
                        image_memmaps[rt][data_index : data_index + data_length] = obs[rt]
                        image_memmaps[rt].flush()
                    elif memmap_loader_path == "" or rt not in image_memmaps:
                        train_data["data"][rt][data_index : data_index + data_length] = obs[
                            rt
                        ]
                else:
                    train_data["data"][rt][data_index : data_index + data_length] = obs[
                        rt
                    ]

            # action space
            train_data["data"]["action"][data_index : data_index + data_length] = (
                self.get_train_action(data)
            )

            if len(train_data["meta"]["episode_ends"]) == 0:
                train_data["meta"]["episode_ends"].append(data_length)
            else:
                train_data["meta"]["episode_ends"].append(
                    data_length + train_data["meta"]["episode_ends"][-1]
                )
            data_index += data_length

        if cache_memmap:
            for image_key in image_memmaps:
                image_memmaps[image_key] = np.memmap(
                    self._get_image_memmap_path(memmap_loader_path, image_key),
                    dtype=np.uint16,
                    mode="r",
                    shape=image_memmaps[image_key].shape,
                )
                train_data["data"][image_key] = image_memmaps[image_key]

        print("Train data loaded")
        for k, v in train_data["data"].items():
            print(k, v.shape)

        train_dataset = Dataset(
            data=train_data,
            representation_type=self.representation_type,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            stats=self.stats,
            load_img=store_img_in_memory or memmap_loader_path != "",
            transform=self.transform if not eval else self.eval_transform,
            get_img=self._get_image_observation,
            binarize_touch=self.binarize_touch,
            state_noise=self.state_noise if not eval else 0.0,
        )
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=not eval,
            pin_memory=True,
            persistent_workers=True,
        )
        self.policy.data_stat = train_dataset.stats
        return dataloader

    def train(
        self,
        path_list,
        batch_size=4,
        epochs=300,
        traj_type="all",
        prefix="",
        save_path=None,
        save_freq=10,
        eval_freq=10,
        wandb_logger=None,
        memmap_loader_path="",
        eval_memmap_loader_path="",
        train_path=None,
        test_path=None,
    ):
        torch.cuda.empty_cache()

        if train_path is not None and test_path is not None:
            print(train_path)
            print(test_path)
            self.epi_dir = data_processing.get_epi_dir(
                train_path, traj_type=traj_type, prefix=prefix
            )
            print(self.epi_dir)
            eval_trajs = data_processing.get_epi_dir(
                test_path, traj_type=traj_type, prefix=prefix
            )
            if len(eval_trajs) == 0:
                raise ValueError(f"No eval trajectories found in {test_path}")
            print(f"eval trajectories: {len(eval_trajs)}")
        else:
            if type(path_list) != list:
                path_list = [path_list]
            for path in path_list:
                self.epi_dir += data_processing.get_epi_dir(
                    path, traj_type=traj_type, prefix=prefix
                )
            eval_traj = self.epi_dir[-1]
            self.epi_dir.remove(eval_traj)
            print("eval traj:", eval_traj)
            eval_trajs = [eval_traj]

        eval_round = 0

        def get_rotating_eval_data(epoch_idx):
            nonlocal eval_round
            eval_traj = eval_trajs[eval_round % len(eval_trajs)]
            eval_round += 1
            # A single shared memmap path cannot safely cache multiple eval
            # trajectories with different lengths, so rotating eval loads H5
            # images directly when more than one test trajectory is present.
            current_eval_memmap_path = (
                eval_memmap_loader_path if len(eval_trajs) == 1 else ""
            )
            eval_data = self.get_eval_data(eval_traj, current_eval_memmap_path)
            return os.path.basename(eval_traj), eval_data

        train_loader = self.get_train_loader(batch_size, memmap_loader_path)
        self.policy.set_lr_scheduler(len(train_loader) * epochs)
        if self.stats is None:
            self.stats = train_loader.dataset.stats
            self.save_stats(save_path)
        if self.compile_train:
            train_loader.dataset.__getitem__ = torch.compile(
                train_loader.dataset.__getitem__
            )

        self.policy.train(
            epochs,
            train_loader,
            save_path=save_path,
            eval_data=get_rotating_eval_data,
            eval_freq=eval_freq,
            save_freq=save_freq,
            wandb_logger=wandb_logger,
        )

        self.policy.to_ema()
        self.eval_all(
            eval_trajs,
            save_path=save_path,
            memmap_loader_path=eval_memmap_loader_path if len(eval_trajs) == 1 else "",
        )

    def get_train_action(self, data):
        act = self.get_eval_action(data)
        return np.stack(act)

    def get_eval_action(self, data):
        if self.predict_eef_delta:
            if self.rt_dim["action"] != 24 or self.rt_dim["eef"] != 12:
                raise NotImplementedError(
                    "predict_eef_delta currently only supports the legacy 24-dim dual-arm configuration."
                )
            # TODO: make sure this is only used when "control" is eef pose
            act = []
            for d in data:
                left_arm_act = get_eef_delta(
                    d["ee_pos_quat"][:6], d["control"][LEFT_UR_IDX]
                )
                left_hand_act = d["control"][LEFT_HAND_IDX]
                right_arm_act = get_eef_delta(
                    d["ee_pos_quat"][6:], d["control"][RIGHT_UR_IDX]
                )
                right_hand_act = d["control"][RIGHT_HAND_IDX]
                act.append(
                    np.concatenate(
                        [left_arm_act, left_hand_act, right_arm_act, right_hand_act],
                        axis=-1,
                    )
                )
            return act
        elif self.predict_pos_delta:
            # TODO: make sure this is only used when "control" is joint pos
            act = [d["control"] for d in data]
            act = np.diff(act, axis=0, append=act[-1:])
            return act
        else:
            return [d["control"] for d in data]

    def get_eval_data(self, data_path, memmap_loader_path=""):
        use_disk_eval_cache = not any(
            image_key in self.representation_type for image_key in IMAGE_KEYS
        )
        if use_disk_eval_cache:
            cache_key = {
                "representation_type": tuple(self.representation_type),
                "camera_indices": tuple(self.camera_indices),
                "image_channel": self.image_channel,
                "obs_horizon": self.obs_horizon,
                "pred_horizon": self.pred_horizon,
                "action_horizon": self.action_horizon,
            }
            cache_name = "dp_eval_cache_" + hashlib.md5(
                repr(cache_key).encode("utf-8")
            ).hexdigest()[:12] + ".pkl"
            cache_path = os.path.join(data_path, cache_name)
            if os.path.exists(cache_path):
                print(f"Loading eval cache from {cache_path}")
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
        else:
            cache_path = None

        print("GETTING EVAL DATA", end="\r")
        total_data_points = data_processing.get_episode_length(data_path)
        image_memmaps, cache_memmap = self._open_episode_image_memmaps(
            total_data_points, memmap_loader_path
        )
        has_h5_episode = os.path.exists(os.path.join(data_path, "trajectory.h5"))
        load_eval_images = self.load_img or (
            has_h5_episode and (memmap_loader_path == "" or cache_memmap)
        )
        data = data_processing.iterate(data_path, load_img=load_eval_images)

        action = self.get_eval_action(data)
        B = len(data)

        print("GETTING EVAL OBSERVATION", end="\r")
        obs = self.get_observation(data, load_img=load_eval_images or cache_memmap)
        for image_key, image_memmap in image_memmaps.items():
            if cache_memmap:
                image_memmap[:B] = obs[image_key]
                image_memmap.flush()
                image_memmaps[image_key] = np.memmap(
                    self._get_image_memmap_path(memmap_loader_path, image_key),
                    dtype=np.uint16,
                    mode="r",
                    shape=image_memmap.shape,
                )
            obs[image_key] = torch.tensor(
                np.asarray(image_memmaps[image_key][:B]).astype(np.float32),
                dtype=torch.float32,
            )
        obs_list = []

        for image_key in IMAGE_KEYS:
            if image_key in self.representation_type:
                obs[image_key] = obs[image_key].float()
                obs[image_key] = self.eval_transform(obs[image_key])
        for i in range(B):
            obs_list.append({rt: obs[rt][i] for rt in self.representation_type})
        eval_data = (obs_list, action)
        if cache_path is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(eval_data, f)
            print(f"Saved eval cache to {cache_path}")
        return eval_data

    def eval(self, data_path, save_path=None, memmap_loader_path=""):
        print("GETTING EVAL DATA")
        eval_data = self.get_eval_data(data_path, memmap_loader_path)
        obs, action = eval_data
        print("EVALUATING")
        action, mse, norm_mse = self.policy.eval(obs, action)
        print("ACTION_MSE: {}, NORM_MSE: {}".format(mse, norm_mse))
        if save_path is None:
            save_path = "./eval/{}".format(
                data_path.split("/")[-1]
                + "_"
                + time.strftime("%m%d_%H%M%S", time.localtime())
            )
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(action)):
            if save_path is not None:
                with open(os.path.join(save_path, str(i) + ".pkl"), "wb") as f:
                    pickle.dump(
                        {
                            "control": action[i],
                            "joint_positions": eval_data[0][i]["pos"],
                        },
                        f,
                    )

    def eval_all(self, data_paths, save_path=None, memmap_loader_path=""):
        if type(data_paths) != list:
            data_paths = [data_paths]
        results = []
        print(f"EVALUATING {len(data_paths)} TEST TRAJECTORIES")
        for data_path in data_paths:
            eval_data = self.get_eval_data(data_path, memmap_loader_path)
            obs, action = eval_data
            _, mse, norm_mse = self.policy.eval(obs, action)
            result = {
                "traj": data_path,
                "action_mse": float(mse),
                "normalized_mse": float(norm_mse),
            }
            results.append(result)
            print(
                f"{os.path.basename(data_path)}: "
                f"ACTION_MSE={result['action_mse']}, "
                f"NORM_MSE={result['normalized_mse']}"
            )

        if len(results) > 0:
            mean_action_mse = float(np.mean([r["action_mse"] for r in results]))
            mean_norm_mse = float(np.mean([r["normalized_mse"] for r in results]))
        else:
            mean_action_mse = float("nan")
            mean_norm_mse = float("nan")
        summary = {
            "results": results,
            "mean_action_mse": mean_action_mse,
            "mean_normalized_mse": mean_norm_mse,
        }
        print(
            f"FULL_TEST_ACTION_MSE: {mean_action_mse}, "
            f"FULL_TEST_NORMALIZED_MSE: {mean_norm_mse}"
        )
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "full_eval_summary.pkl"), "wb") as f:
                pickle.dump(summary, f)
        return summary

    def get_eval_loader(
        self, dir_path, traj_type="plain", prefix="0", batch_size=32, num_workers=16
    ):
        self.num_workers = num_workers
        print(f"GETTING EVAL DATA FROM {dir_path}")
        self.epi_dir = data_processing.get_epi_dir(dir_path, traj_type, prefix)
        eval_loader = self.get_train_loader(batch_size, "", eval=True)
        return eval_loader

    def eval_dir(self, eval_loader, num_diffusion_iters=15):
        self.policy.num_diffusion_iters = num_diffusion_iters
        with torch.no_grad():
            mse, action_mse = self.policy.eval_loader(eval_loader)
        print(f"MSE: {mse}", f"ACTION_MSE: {action_mse}")
        return mse, action_mse

    def load(self, path):
        model_path = os.path.join(path)
        dir_path = os.path.dirname(path)
        stat_path = os.path.join(dir_path, "stats.pkl")
        self.stats = pickle.load(open(stat_path, "rb"))
        self.policy.data_stat = self.stats
        self.policy.load(model_path)
        print("model loaded")

    def save_stats(self, path):
        os.makedirs(path, exist_ok=True)
        stat_path = os.path.join(path, "stats.pkl")
        if not os.path.exists(stat_path):
            with open(stat_path, "wb") as f:
                pickle.dump(self.stats, f)
        print("stats saved")

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "model.ckpt")
        stat_path = os.path.join(path, "stats.pkl")
        self.policy.save(model_path)

        # if stat not exist, create one
        if not os.path.exists(stat_path):
            with open(stat_path, "wb") as f:
                pickle.dump(self.stats, f)
        print("model saved")


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


if __name__ == "__main__":
    # TODO: better config management
    args = argparse.ArgumentParser()
    # train config
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--obs_horizon", type=int, default=1)
    args.add_argument("--action_horizon", type=int, default=8)
    args.add_argument("--pred_horizon", type=int, default=16)
    args.add_argument("--epochs", type=int, default=300)

    # input config
    args.add_argument("--traj_type", type=str, default="plain")
    args.add_argument("--prefix", type=str, default=None)
    args.add_argument("--save_path", type=str, default=None)
    args.add_argument("--load_path", type=str, default=None)

    args.add_argument("--eval", type=boolean_string, default=False)
    args.add_argument(
        "--representation_type", type=str, default="img-depth-eef-hand_pos-touch"
    )

    args.add_argument("--base_path", type=str, default="./shared")
    args.add_argument("--data_name", type=str, default="test_data")
    args.add_argument("--data_path", type=str, default=None)
    args.add_argument("--data_prefix", type=str, default=None)
    args.add_argument("--model_save_path", type=str, default=None)

    args.add_argument("--clip_far", type=boolean_string, default=False)
    args.add_argument("--color_jitter", type=boolean_string, default=False)
    args.add_argument("--predict_eef_delta", type=boolean_string, default=False)
    args.add_argument("--predict_pos_delta", type=boolean_string, default=False)
    args.add_argument("--use_ddim", type=boolean_string, default=False)

    args.add_argument("--policy_dropout_rate", type=float, default=0.0)  # For simple BC
    args.add_argument(
        "--dropout_rate", type=float, default=0.0
    )  # For the state encoder of Diffusion Policy
    args.add_argument(
        "--img_dropout_rate", type=float, default=0.0
    )  # For the image encoder of Diffusion Policy
    args.add_argument("--weight_decay", type=float, default=1e-5)
    args.add_argument("--image_output_size", type=int, default=32)
    args.add_argument("--joint_state_dim", type=int, default=24)
    args.add_argument("--action_dim", type=int, default=24)
    args.add_argument("--eef_dim", type=int, default=12)
    args.add_argument("--touch_dim", type=int, default=60)
    args.add_argument("--hand_pos_dim", type=int, default=12)
    args.add_argument("--state_noise", type=float, default=0.0)
    args.add_argument("--img_gaussian_noise", type=float, default=0.0)
    args.add_argument("--img_masking_prob", type=float, default=0.0)
    args.add_argument("--img_patch_size", type=int, default=16)
    args.add_argument("--num_workers", type=int, default=16)

    args.add_argument("--eval_path", type=str, default=None)
    args.add_argument("--identity_encoder", type=boolean_string, default=False)
    args.add_argument("--gpu", type=int, default=0)

    args.add_argument("--camera_indices", type=str, default="012")
    args.add_argument("--save_freq", type=int, default=10)
    args.add_argument("--eval_freq", type=int, default=10)

    args.add_argument("--add_model_save_path_suffix", type=boolean_string, default=True)
    args.add_argument("--use_wandb", type=boolean_string, default=False)
    args.add_argument("--without_sampling", type=boolean_string, default=False)
    args.add_argument("--binarize_touch", type=boolean_string, default=False)

    # model config
    args.add_argument("--num_diffusion_iters", type=int, default=100)
    args.add_argument("--wandb_exp_name", type=str, default=None)
    args.add_argument("--load_img", type=boolean_string, default=False)
    args.add_argument("--train_suffix", type=str, default="")

    args.add_argument("--use_train_test_split", type=boolean_string, default=True)
    args.add_argument("--use_memmap_cache", type=boolean_string, default=False)
    args.add_argument("--memmap_loader_path", type=str, default=None)
    args.add_argument("--prepare_cache_only", type=boolean_string, default=False)
    args.add_argument("--compile_train", type=boolean_string, default=False)

    # wandb config
    args.add_argument("--wandb_entity_name", type=str, default=None)
    args.add_argument("--wandb_project_name", type=str, default=None)
    args = args.parse_args()

    if args.gpu is not None:
        torch.cuda.set_device("cuda:{}".format(args.gpu))
        print("Using gpu: {}".format(args.gpu))

    # automatic naming
    curr_time = time.strftime("%m%d_%H%M%S", time.localtime())
    random_tag = generate_random_string()
    model_id = f"{curr_time}_{random_tag}"

    if args.use_wandb:
        wandb_config = WandBLogger.get_default_config()
        wandb_config.entity = args.wandb_entity_name
        wandb_config.project = args.wandb_project_name
        if args.wandb_exp_name is not None:
            wandb_config.exp_name = args.wandb_exp_name + "_" + model_id
        else:
            wandb_config.exp_name = model_id
        wandb_logger = WandBLogger(
            config=wandb_config, variant=vars(args), prefix="logging"
        )
    else:
        wandb_logger = None

    agent = Agent(
        dropout={
            "eef": args.dropout_rate,
            "hand_pos": args.dropout_rate,
            "img": args.img_dropout_rate,
            "tactile_img": args.img_dropout_rate,
            "pos": args.dropout_rate,
            "touch": args.dropout_rate,
        },
        output_sizes={
            "eef": 64,
            "hand_pos": 64,
            "img": None,  # unprojected 512-dim GAP feature, paper-aligned
            "tactile_img": args.image_output_size,
            "pos": 128,
            "touch": 64,
        },
        representation_type=args.representation_type.split("-"),
        identity_encoder=args.identity_encoder,
        camera_indices=list(map(int, args.camera_indices)),
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        without_sampling=args.without_sampling,
        predict_eef_delta=args.predict_eef_delta,
        predict_pos_delta=args.predict_pos_delta,
        clip_far=args.clip_far,
        color_jitter=args.color_jitter,
        num_diffusion_iters=args.num_diffusion_iters,
        load_img=args.load_img,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        use_ddim=args.use_ddim,
        binarize_touch=args.binarize_touch,
        policy_dropout_rate=args.policy_dropout_rate,
        state_noise=args.state_noise,
        img_gaussian_noise=args.img_gaussian_noise,
        img_masking_prob=args.img_masking_prob,
        img_patch_size=args.img_patch_size,
        compile_train=args.compile_train,
        joint_state_dim=args.joint_state_dim,
        action_dim=args.action_dim,
        eef_state_dim=args.eef_dim,
        touch_dim=args.touch_dim,
        hand_pos_dim=args.hand_pos_dim,
    )
    if args.load_path is not None:
        agent.load(args.load_path)

    use_depth = "depth" in args.representation_type.split("-")

    if not args.eval:
        if args.model_save_path is None:
            data_name = args.data_name.replace("/", "-")
            model_path = os.path.join(args.base_path, "ckpts", data_name)
        else:
            model_path = args.model_save_path

        model_path_suffix = f"{model_id}"

        if args.add_model_save_path_suffix:
            args_ = [
                ("camera", args.camera_indices),
                ("identity", args.identity_encoder),
                (
                    "repr",
                    "".join(
                        [(x[0]).upper() for x in args.representation_type.split("-")]
                    ),
                ),
                ("oh", args.obs_horizon),
                ("ah", args.action_horizon),
                ("ph", args.pred_horizon),
                ("prefix", args.prefix),
                ("do", args.dropout_rate),
                ("imgos", args.image_output_size),
                ("wd", args.weight_decay),
                ("use_ddim", args.use_ddim),
                ("binarize_touch", args.binarize_touch),
            ]
            args_str = "-".join([f"{k}={v}" for k, v in args_])
            if args.without_sampling:
                args_str += "-ws"
            if args.predict_pos_delta:
                args_str += "-posdelta"
            if args.predict_eef_delta:
                args_str += "-eefdelta"
            model_path_suffix += "-" + args_str
        model_path = os.path.join(model_path, model_path_suffix)

        print(f"Saving to model path {model_path}")

        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)

        save_args(args, model_path)

        if agent.stats is not None:
            agent.save_stats(model_path)

        if args.data_path is not None:
            data_path = args.data_path
        else:
            if args.data_prefix is not None:
                data_path = os.path.join(
                    args.base_path, args.data_prefix, args.data_name
                )
            else:
                data_path = os.path.join(args.base_path, args.data_name)

        if args.use_train_test_split:
            train_path = data_path + "_train" + args.train_suffix
            test_path = data_path + "_test"
        else:
            train_path = test_path = None
        print(f"using data path {data_path}")
        if args.use_memmap_cache:
            if args.memmap_loader_path is not None:
                memmap_loader_path = args.memmap_loader_path
                if test_path is not None:
                    root, ext = os.path.splitext(memmap_loader_path)
                    eval_memmap_loader_path = (
                        f"{root}-eval{ext}" if ext != "" else f"{memmap_loader_path}-eval"
                    )
                else:
                    eval_memmap_loader_path = memmap_loader_path
            else:
                memmap_base_path = train_path if train_path is not None else data_path
                memmap_loader_path = os.path.join(
                    memmap_base_path, f"{args.camera_indices}-{use_depth}-mem.dat"
                )
                eval_memmap_base_path = (
                    test_path if test_path is not None else memmap_base_path
                )
                eval_memmap_loader_path = os.path.join(
                    eval_memmap_base_path, f"{args.camera_indices}-{use_depth}-mem.dat"
                )
        else:
            memmap_loader_path = ""
            eval_memmap_loader_path = ""

        print("using memmap loader path:", memmap_loader_path)
        print("using eval memmap loader path:", eval_memmap_loader_path)
        if args.prepare_cache_only:
            cache_source_path = train_path if train_path is not None else data_path
            print(f"Preparing cache from {cache_source_path}")
            agent.epi_dir = data_processing.get_epi_dir(
                cache_source_path, traj_type=args.traj_type, prefix=args.prefix
            )
            agent.get_train_loader(
                batch_size=args.batch_size, memmap_loader_path=memmap_loader_path
            )
            if test_path is not None:
                eval_trajs = data_processing.get_epi_dir(
                    test_path, traj_type=args.traj_type, prefix=args.prefix
                )
                print(f"Found {len(eval_trajs)} eval trajectories.")
                if len(eval_trajs) == 0:
                    raise ValueError(f"No eval trajectories found in {test_path}")
                if len(eval_trajs) == 1:
                    print(f"Preparing eval cache from {eval_trajs[0]}")
                    agent.get_eval_data(
                        eval_trajs[0], memmap_loader_path=eval_memmap_loader_path
                    )
                else:
                    print(
                        "Skipping eval memmap cache: rotating eval has "
                        f"{len(eval_trajs)} test trajectories with different lengths, "
                        "so eval data will be loaded directly from H5."
                    )
            print("Cache preparation finished.")
            sys.exit(0)

        agent.train(
            data_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            traj_type=args.traj_type,
            prefix=args.prefix,
            save_path=model_path,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            wandb_logger=wandb_logger,
            train_path=train_path,
            test_path=test_path,
            memmap_loader_path=memmap_loader_path,
            eval_memmap_loader_path=eval_memmap_loader_path,
        )

    else:
        agent.eval(args.eval_path, save_path=args.save_path)

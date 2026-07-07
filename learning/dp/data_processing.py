import concurrent.futures
import os
import pickle
import re
import tempfile

import cv2
import natsort
import numpy as np


def from_pickle(path, load_img = True, num_cam = 3):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "base_rgb" not in data and load_img:
        rgb = []
        for i in range(num_cam):
            rgb_path = path.replace(".pkl", f"-{i}.png")
            if os.path.exists(rgb_path):
                rgb.append(cv2.imread(rgb_path))
        data["base_rgb"] = np.stack(rgb, axis=0)

    return data


def _decode_h5_video_frames(dataset, *, is_depth=False):
    video_bytes = np.asarray(dataset, dtype=np.uint8).tobytes()
    codec = str(dataset.attrs.get("codec", "mp4v"))
    suffix = ".avi" if codec == "MJPG" else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open embedded video stream '{dataset.name}'")

        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_depth:
                gray = frame[..., 0].astype(np.float32)
                depth_min = float(dataset.attrs.get("depth_min_mm", 200.0))
                depth_max = float(dataset.attrs.get("depth_max_mm", 1500.0))
                valid = gray > 0
                depth = np.zeros(gray.shape, dtype=np.float32)
                if depth_max > depth_min:
                    depth[valid] = (
                        gray[valid] / 255.0 * (depth_max - depth_min) + depth_min
                    )
                frames.append(depth)
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    finally:
        os.unlink(tmp_path)

    if not frames:
        raise RuntimeError(f"No frames decoded from embedded video stream '{dataset.name}'")
    return np.stack(frames, axis=0)


def from_h5(path, load_img=True):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Loading trajectory.h5 for diffusion-policy training requires 'h5py'."
        ) from exc

    with h5py.File(path, "r") as f:
        frames_group = f["frames"]
        frame_count = int(f.attrs.get("frame_count", 0))
        if frame_count <= 0:
            first_key = next(iter(frames_group.keys()), None)
            frame_count = len(frames_group[first_key]) if first_key is not None else 0

        numeric_data = {
            key: np.asarray(dataset[:]) for key, dataset in frames_group.items()
        }

        decoded_videos = {}
        if load_img and "videos" in f:
            for key, dataset in f["videos"].items():
                attrs = dict(dataset.attrs)
                is_depth = attrs.get("source_kind") == "depth" or key.endswith("_depth")
                decoded_videos[key] = _decode_h5_video_frames(
                    dataset, is_depth=is_depth
                )

        multi_camera_streams = {}
        tactile_streams = {}
        if decoded_videos:
            for key, frames in decoded_videos.items():
                match = re.match(r"^(.*)_(\d+)$", key)
                if match:
                    base_key = match.group(1)
                    camera_idx = int(match.group(2))
                    multi_camera_streams.setdefault(base_key, []).append(
                        (camera_idx, frames)
                    )
                else:
                    tactile_streams[key] = frames

            for key in multi_camera_streams:
                multi_camera_streams[key].sort(key=lambda item: item[0])

        data = []
        for i in range(frame_count):
            frame = {key: value[i] for key, value in numeric_data.items()}

            activated = frame.get("activated")
            if activated is not None:
                if isinstance(activated, np.ndarray):
                    activated = bool(np.asarray(activated).item())
                else:
                    activated = bool(activated)
                frame["activated"] = activated

            if load_img:
                for base_key, streams in multi_camera_streams.items():
                    frame[base_key] = np.stack(
                        [stream_frames[i] for _, stream_frames in streams], axis=0
                    )
                for key, stream_frames in tactile_streams.items():
                    frame[key] = stream_frames[i]

            frame["file_path"] = f"{path}::{i}"
            data.append(frame)

    return data


def get_episode_length(path):
    h5_path = os.path.join(path, "trajectory.h5")
    if os.path.exists(h5_path):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "Counting frames in trajectory.h5 requires 'h5py'."
            ) from exc

        with h5py.File(h5_path, "r") as f:
            frame_count = int(f.attrs.get("frame_count", 0))
            if frame_count > 0:
                return frame_count
            if "timestamps" in f:
                return len(f["timestamps"])
            if "frames" in f:
                first_key = next(iter(f["frames"].keys()), None)
                if first_key is not None:
                    return len(f["frames"][first_key])
            return 0

    return len([d for d in os.listdir(path) if d.endswith(".pkl")])

# Get the trajectory data from the given directory
def iterate(path, workers=32, load_img=True, num_cam=3):
    h5_path = os.path.join(path, "trajectory.h5")
    if os.path.exists(h5_path):
        return from_h5(h5_path, load_img=load_img)

    dir = os.listdir(path)
    dir = [d for d in dir if d.endswith(".pkl")]
    dir = natsort.natsorted(dir)
    dirname = os.path.basename(path)
    root_path = "./mask_cache"
    data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers = workers) as executor:
        futures = {executor.submit(from_pickle, os.path.join(path, file), load_img, num_cam): (i, file) for i, file in enumerate(dir)}
        for future in futures:
            try:
                i, file = futures[future]
                d = future.result()
                if not d["activated"]["l"] and not d["activated"]["r"]:
                    continue
                basedirfile = os.path.join(dirname, file)
                maskfile = os.path.join(root_path, basedirfile)
                if os.path.exists(maskfile):
                    d["mask"] = from_pickle(maskfile)
                d["mask_path"] = maskfile
                d["file_path"] = os.path.join(path, file)
                data.append(d)
            except:
                print(f"Failed to load {file}")
                pass
    return data


def get_latest(path):
    dir = os.listdir(path)
    dir = natsort.natsorted(dir)
    return from_pickle(os.path.join(path, dir[-1]))

# Get all trajectory directories from the given path
def get_epi_dir(path, traj_type, prefix=None):
    dir = natsort.natsorted(os.listdir(path))
    if prefix is not None:
        prefixs = prefix.split("-")

    new_dir = []
    for d in dir:
        if os.path.isdir(os.path.join(path, d)):
            matched = False
            if prefix is None:
                matched = True
            else:
                for prefix in prefixs:
                    if d.startswith(prefix):
                        matched = True
            if matched:
                new_dir.append(d)

    print("All Directories")
    print(new_dir)
    print("==========")
    dir = new_dir
    if traj_type == "plain":
        dir = [
            d
            for d in dir
            if not d.endswith("failed")
            and not d.endswith("ood")
            and not d.endswith("ikbad")
            and not d.endswith("heated")
            and not d.endswith("stop")
            and not d.endswith("hard")
        ]
    elif traj_type == "all":
        dir = dir
    else:
        raise NotImplementedError
    dir_list = [
        os.path.join(path, d) for d in dir if os.path.isdir(os.path.join(path, d))
    ]
    return dir_list

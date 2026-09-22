#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap


EVALUATOR = r"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

# Import data_loader before config/policy_config to avoid a local native
# extension segfault seen when OpenPI imports normalize first.
from openpi.training import data_loader as openpi_data_loader
from openpi.training import config as openpi_config
from openpi.policies import policy_config


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_int(value):
    return int(to_numpy(value).reshape(-1)[0])


def new_accumulator(action_dim):
    return {
        "sample_count": 0,
        "per_sample_mse": [],
        "per_dim_sse": np.zeros((action_dim,), dtype=np.float64),
        "per_dim_count": np.zeros((action_dim,), dtype=np.float64),
        "total_sse": 0.0,
        "total_count": 0,
    }


def update_accumulator(acc, sq):
    acc["sample_count"] += 1
    acc["per_sample_mse"].append(float(np.mean(sq)))
    acc["total_sse"] += float(np.sum(sq))
    acc["total_count"] += int(sq.size)
    acc["per_dim_sse"] += np.sum(sq, axis=0)
    acc["per_dim_count"] += sq.shape[0]


def summarize_accumulator(acc):
    if acc["total_count"] == 0:
        return {
            "sample_count": 0,
            "mse": None,
            "rmse": None,
            "per_dim_mse": None,
            "per_sample_mse_mean": None,
            "per_sample_mse_median": None,
            "per_sample_mse_p90": None,
        }
    per_sample_mse = np.asarray(acc["per_sample_mse"], dtype=np.float64)
    mse = acc["total_sse"] / acc["total_count"]
    return {
        "sample_count": acc["sample_count"],
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "per_dim_mse": (acc["per_dim_sse"] / np.maximum(acc["per_dim_count"], 1)).tolist(),
        "per_sample_mse_mean": float(np.mean(per_sample_mse)),
        "per_sample_mse_median": float(np.median(per_sample_mse)),
        "per_sample_mse_p90": float(np.percentile(per_sample_mse, 90)),
    }


def phase_name(progress):
    if progress < 0.3:
        return "0.0-0.3"
    if progress < 0.6:
        return "0.3-0.6"
    return "0.6-1.0"


def main():
    config_name = sys.argv[1]
    checkpoint_dir = Path(sys.argv[2])
    max_samples = None if sys.argv[3] == "None" else int(sys.argv[3])
    action_dim = int(sys.argv[4])
    default_prompt = None if sys.argv[5] == "None" else sys.argv[5]
    num_steps = int(sys.argv[6])

    cfg = openpi_config.get_config(config_name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = openpi_data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    sample_count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    if sample_count <= 0:
        raise SystemExit("No samples available for evaluation.")

    episode_lengths = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        episode_index = scalar_int(sample["episode_index"])
        frame_index = scalar_int(sample["frame_index"])
        episode_lengths[episode_index] = max(episode_lengths.get(episode_index, 0), frame_index + 1)

    policy = policy_config.create_trained_policy(
        cfg,
        checkpoint_dir,
        repack_transforms=data_config.repack_transforms,
        default_prompt=default_prompt,
        sample_kwargs={"num_steps": num_steps},
    )

    overall = new_accumulator(action_dim)
    phases = {
        "0.0-0.3": new_accumulator(action_dim),
        "0.3-0.6": new_accumulator(action_dim),
        "0.6-1.0": new_accumulator(action_dim),
    }

    # Spread the sampled frames evenly over the whole dataset rather than taking the first
    # sample_count of them. LeRobot stores frames in episode order, so a prefix only covers the
    # start of the first episode or two -- with --max-samples 200 against ~333-frame episodes,
    # every sample came from the first 60% of episode 0 and the late-episode phase scored nothing.
    eval_indices = [round(i * (len(dataset) - 1) / max(sample_count - 1, 1)) for i in range(sample_count)]

    for evaluated, i in enumerate(eval_indices, start=1):
        sample = dataset[i]
        episode_index = scalar_int(sample["episode_index"])
        frame_index = scalar_int(sample["frame_index"])
        episode_len = max(episode_lengths.get(episode_index, 1), 1)
        progress = frame_index / max(episode_len - 1, 1)
        phase = phase_name(progress)

        target = to_numpy(sample["action"]).astype(np.float32)
        if target.ndim == 1:
            target = target[None, :]
        target = target[:, :action_dim]

        result = policy.infer(sample)
        pred = to_numpy(result["actions"]).astype(np.float32)
        if pred.ndim == 1:
            pred = pred[None, :]
        pred = pred[:, :action_dim]

        horizon = min(len(pred), len(target))
        if horizon <= 0:
            continue
        diff = pred[:horizon] - target[:horizon]
        sq = np.square(diff, dtype=np.float64)
        update_accumulator(overall, sq)
        update_accumulator(phases[phase], sq)

        if evaluated % 25 == 0 or evaluated == sample_count:
            print(f"evaluated {evaluated}/{sample_count}", flush=True)

    if overall["total_count"] == 0:
        raise SystemExit("No valid action chunks were evaluated.")

    overall_metrics = summarize_accumulator(overall)
    metrics = {
        "config_name": config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "sample_count": sample_count,
        "action_dim": action_dim,
        "action_horizon": int(cfg.model.action_horizon),
        "num_steps": num_steps,
        **{k: v for k, v in overall_metrics.items() if k != "sample_count"},
        "phase_metrics": {name: summarize_accumulator(acc) for name, acc in phases.items()},
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
"""


def find_uv() -> str:
    uv = shutil.which("uv")
    if uv is not None:
        return uv
    local_uv = Path.home() / ".local" / "bin" / "uv"
    if local_uv.exists():
        return str(local_uv)
    raise SystemExit("Could not find uv. Install uv or set PATH so uv is available.")


def infer_asset_id(checkpoint_dir: Path) -> str | None:
    assets_dir = checkpoint_dir / "assets"
    norm_stats = sorted(assets_dir.glob("**/norm_stats.json"))
    if len(norm_stats) != 1:
        return None
    return str(norm_stats[0].parent.relative_to(assets_dir))


def read_state_dim(dataset_root: Path) -> int | None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return int(info["features"]["observation.state"]["shape"][0])


def read_action_format(dataset_root: Path) -> str | None:
    report_path = dataset_root / "conversion_report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report.get("action_format")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained pi0_ur5e policy on LeRobot data with action MSE.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--config-name", default="pi0_ur5e_cup")
    parser.add_argument("--lerobot-repo-id", default="local/pi0_ur5e_eval")
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--action-format", default=None, choices=["ee_delta_6d_gripper", "joint_position_gripper", "joint_delta_gripper"])
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=10, help="Diffusion sampling steps used by policy.sample_actions.")
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--use-tactile-input", default="true")
    args = parser.parse_args()

    # HaptileTactilePI0Pytorch configs read their own PI0_UR5E_TACTILE_* environment variables and
    # live in files install_openpi_config.py doesn't install, so evaluating one needs a second
    # installer and a separate env block (same split as serve_policy.py). Keyed off the config
    # name so the plain pi0_ur5e_cup path stays exactly as it was.
    is_tactile = args.config_name.endswith("_tactile")

    installers = ["install_openpi_config.py"]
    if is_tactile:
        installers.append("install_openpi_pytorch_patch.py")
    for installer in installers:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name(installer)),
                "--openpi-root",
                str(args.openpi_root),
            ],
            check=True,
        )

    dataset_root = args.dataset_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_home = Path("/tmp/pi0_ur5e_eval_lerobot_home").resolve()
    link_path = output_home / args.lerobot_repo_id
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(dataset_root, target_is_directory=True)

    env = os.environ.copy()
    env["HF_LEROBOT_HOME"] = str(output_home)
    env["PI0_UR5E_LEROBOT_REPO_ID"] = args.lerobot_repo_id
    env["PI0_UR5E_ASSET_ID"] = args.asset_id or infer_asset_id(checkpoint_dir) or args.lerobot_repo_id
    env["PI0_UR5E_ACTION_FORMAT"] = args.action_format or read_action_format(dataset_root) or "joint_position_gripper"
    if (state_dim := read_state_dim(dataset_root)) is not None:
        env["PI0_UR5E_STATE_DIM"] = str(state_dim)

    if is_tactile:
        for name in ("LEROBOT_REPO_ID", "ASSET_ID", "ACTION_FORMAT", "STATE_DIM"):
            if (value := env.get(f"PI0_UR5E_{name}")) is not None:
                env[f"PI0_UR5E_TACTILE_{name}"] = value
        env["PI0_UR5E_TACTILE_USE_TACTILE_INPUT"] = str(args.use_tactile_input).lower()
        # The checkpoint already carries the fine-tuned tactile encoder, so fetching the pretrained
        # T3 checkpoint would only download weights that are immediately overwritten.
        env["PI0_UR5E_TACTILE_LOAD_T3_CHECKPOINT"] = "false"

    env.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

    command = [
        find_uv(),
        "run",
        "python",
        "-c",
        textwrap.dedent(EVALUATOR),
        args.config_name,
        str(checkpoint_dir),
        str(args.max_samples),
        str(args.action_dim),
        "None" if args.default_prompt is None else args.default_prompt,
        str(args.num_steps),
    ]
    subprocess.run(command, cwd=args.openpi_root, env=env, check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi0_ur5e.io_utils import write_json
from pi0_ur5e.openpi_config import write_openpi_config_patch


def parse_args():
    parser = argparse.ArgumentParser(description="Launch pi0/pi0.5 fine-tuning from an external OpenPI checkout.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--openpi-config", default="pi0_ur5e_cup")
    parser.add_argument("--lerobot-repo-id", default="local/pi0_ur5e_cup")
    parser.add_argument("--checkpoint", default="pi05_base")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--max-token-len", type=int, default=None)
    parser.add_argument("--lora", default="true")
    parser.add_argument("--pi05", default="true")
    parser.add_argument("--model-family", default=None, choices=["pi0", "pi05", "pi0_fast"])
    parser.add_argument("--use-delta-actions", default="true")
    parser.add_argument("--freeze-mode", default="default", choices=["default", "vision_action_head", "action_head"])
    parser.add_argument("--include-tactile", default="false")
    parser.add_argument("--tactile-feature-mode", default="none", choices=["none", "low_dim", "image_embedding", "raw_image"])
    parser.add_argument("--tactile-embedding-dim", default=128, type=int)
    parser.add_argument("--action-format", default=None, choices=["ee_delta_6d_gripper", "joint_position_gripper", "joint_delta_gripper"])
    parser.add_argument("--camera-padding-strategy", default="zeros")
    parser.add_argument("--wandb", default="false")
    parser.add_argument("--exp-name", default="ur5e_cup_pi05_base")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", default="false")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    include_tactile = str(args.include_tactile).lower() == "true"
    patch_path = args.output_dir / "openpi_config_patch.json"
    conversion_report_path = args.dataset_root / "conversion_report.json"
    action_format = args.action_format
    if action_format is None and conversion_report_path.exists():
        conversion_report = json.loads(conversion_report_path.read_text(encoding="utf-8"))
        action_format = conversion_report.get("action_format")
    action_format = action_format or "joint_position_gripper"
    write_openpi_config_patch(
        patch_path,
        dataset_root=str(args.dataset_root),
        checkpoint=args.checkpoint,
        include_tactile=include_tactile,
        action_format=action_format,
        camera_padding_strategy=args.camera_padding_strategy,
    )
    snapshot = vars(args).copy()
    snapshot["dataset_root"] = str(args.dataset_root)
    snapshot["output_dir"] = str(args.output_dir)
    snapshot["openpi_root"] = str(args.openpi_root)
    write_json(args.output_dir / "train_args_snapshot.json", snapshot)
    for name in ("conversion_report.json", "norm_stats.json"):
        src = args.dataset_root / name
        if src.exists():
            shutil.copy2(src, args.output_dir / name)
    info_path = args.dataset_root / "meta" / "info.json"
    state_dim = None
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        state_dim = int(info["features"]["observation.state"]["shape"][0])
    env = os.environ.copy()
    lerobot_home = (args.output_dir / "lerobot_home").resolve()
    link_path = lerobot_home / args.lerobot_repo_id
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(args.dataset_root.resolve(), target_is_directory=True)
    env.update(
        {
            "HF_LEROBOT_HOME": str(lerobot_home),
            "PI0_UR5E_LEROBOT_REPO_ID": args.lerobot_repo_id,
            "PI0_UR5E_ASSET_ID": args.lerobot_repo_id,
            "PI0_UR5E_TRAIN_STEPS": str(args.steps),
            "PI0_UR5E_BATCH_SIZE": str(args.batch_size),
            "PI0_UR5E_ACTION_HORIZON": str(args.action_horizon),
            "PI0_UR5E_ACTION_FORMAT": action_format,
            "PI0_UR5E_LORA": str(args.lora).lower(),
            "PI0_UR5E_PI05": str(args.pi05).lower(),
            "PI0_UR5E_USE_DELTA_ACTIONS": str(args.use_delta_actions).lower(),
            "PI0_UR5E_FREEZE_MODE": args.freeze_mode,
            "PI0_UR5E_CAMERA_PADDING": args.camera_padding_strategy,
            "PI0_UR5E_ASSETS_BASE_DIR": str((args.output_dir / "assets").resolve()),
            "PI0_UR5E_CHECKPOINT_BASE_DIR": str((args.output_dir / "checkpoints").resolve()),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": env.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75"),
        }
    )
    if args.max_token_len is not None:
        env["PI0_UR5E_MAX_TOKEN_LEN"] = str(args.max_token_len)
    if args.model_family is not None:
        env["PI0_UR5E_MODEL_FAMILY"] = args.model_family
        if args.model_family == "pi0":
            env["PI0_UR5E_PI05"] = "false"
        elif args.model_family == "pi05":
            env["PI0_UR5E_PI05"] = "true"
    if state_dim is not None:
        env["PI0_UR5E_STATE_DIM"] = str(state_dim)
    if str(args.wandb).lower() != "true":
        env["WANDB_MODE"] = "disabled"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("install_openpi_config.py")),
            "--openpi-root",
            str(args.openpi_root),
        ],
        check=True,
    )
    compute_command = ["uv", "run", "scripts/compute_norm_stats.py", "--config-name", args.openpi_config]
    train_command = ["uv", "run", "scripts/train.py", args.openpi_config, "--exp-name", args.exp_name]
    if args.resume:
        train_command += ["--resume"]
    else:
        train_command += ["--overwrite"]
    if str(args.wandb).lower() != "true":
        train_command += ["--no-wandb-enabled"]
    print("OpenPI compute norm stats command:")
    print(" ".join(compute_command))
    print("OpenPI train command:")
    print(" ".join(train_command))
    if str(args.dry_run).lower() == "true":
        return
    try:
        subprocess.run(compute_command, cwd=args.openpi_root, check=True, env=env)
        subprocess.run(train_command, cwd=args.openpi_root, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print("Training failed. If this is GPU memory related, try: lower --batch-size, keep --lora true, enable gradient checkpointing in OpenPI, or freeze the VLM and train only the action expert/projection.")
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()

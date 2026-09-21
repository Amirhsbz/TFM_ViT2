#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def find_uv() -> str:
    uv = shutil.which("uv")
    if uv is not None:
        return uv

    local_uv = Path.home() / ".local" / "bin" / "uv"
    if local_uv.exists():
        return str(local_uv)

    raise SystemExit("Could not find uv. Install it or add it to PATH before serving the policy.")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch OpenPI's official policy server for pi0_ur5e_cup.")
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--config-name", default="pi0_ur5e_cup")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", default=None, type=Path)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-token-len", type=int, default=None)
    parser.add_argument("--model-family", default=None, choices=["pi0", "pi05", "pi0_fast"])
    parser.add_argument("--use-delta-actions", default="true")
    parser.add_argument("--camera-padding-strategy", default="zeros")
    parser.add_argument("--freeze-mode", default="default", choices=["default", "vision_action_head", "action_head"])
    parser.add_argument("--default-prompt", default="pick up the paper cup and place it on the target")
    parser.add_argument("--record", default="false")
    parser.add_argument("--dry-run", default="false")
    parser.add_argument("--use-tactile-input", default="true")
    args = parser.parse_args()

    # HaptileTactilePI0Pytorch configs read their own PI0_UR5E_TACTILE_* environment variables and
    # live in files that install_openpi_config.py doesn't install, so serving one needs a second
    # installer and a separate env block. Keyed off the config name so the plain pi0_ur5e_cup
    # serving path stays exactly as it was.
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
    env = os.environ.copy()
    if args.dataset_root is not None:
        env["PI0_UR5E_LEROBOT_REPO_ID"] = str(args.dataset_root.resolve())
        if (state_dim := read_state_dim(args.dataset_root)) is not None:
            env["PI0_UR5E_STATE_DIM"] = str(state_dim)
    if args.max_token_len is not None:
        env["PI0_UR5E_MAX_TOKEN_LEN"] = str(args.max_token_len)
    if args.model_family is not None:
        env["PI0_UR5E_MODEL_FAMILY"] = args.model_family
        if args.model_family == "pi0":
            env["PI0_UR5E_PI05"] = "false"
        elif args.model_family == "pi05":
            env["PI0_UR5E_PI05"] = "true"
    env["PI0_UR5E_USE_DELTA_ACTIONS"] = str(args.use_delta_actions).lower()
    env["PI0_UR5E_CAMERA_PADDING"] = args.camera_padding_strategy
    env["PI0_UR5E_FREEZE_MODE"] = args.freeze_mode

    checkpoint_dir = args.checkpoint_dir.resolve()
    asset_id = args.asset_id or infer_asset_id(checkpoint_dir)
    if asset_id is not None:
        env["PI0_UR5E_ASSET_ID"] = asset_id

    if is_tactile:
        for name in (
            "LEROBOT_REPO_ID",
            "STATE_DIM",
            "MAX_TOKEN_LEN",
            "PI05",
            "USE_DELTA_ACTIONS",
            "CAMERA_PADDING",
            "ASSET_ID",
        ):
            if (value := env.get(f"PI0_UR5E_{name}")) is not None:
                env[f"PI0_UR5E_TACTILE_{name}"] = value
        env["PI0_UR5E_TACTILE_USE_TACTILE_INPUT"] = str(args.use_tactile_input).lower()
        # The trained checkpoint already carries the fine-tuned tactile encoder, so fetching the
        # pretrained T3 checkpoint here would only download weights that are immediately
        # overwritten -- and would make serving depend on network access.
        env["PI0_UR5E_TACTILE_LOAD_T3_CHECKPOINT"] = "false"
        # vision_tower_mode / vision LoRA rank / alpha are deliberately not set here:
        # HaptileTactileConfig.load_pytorch reconstructs those from the checkpoint itself
        # (its LoRA keys, plus the training config recorded in metadata.pt).

    uv = find_uv()
    command = [
        uv,
        "run",
        "python",
        "-c",
        "import runpy; import openpi.training.data_loader; runpy.run_path('scripts/serve_policy.py', run_name='__main__')",
        "--port",
        str(args.port),
        "--default-prompt",
        args.default_prompt,
    ]
    if str(args.record).lower() == "true":
        command.append("--record")
    command += [
        "policy:checkpoint",
        "--policy.config",
        args.config_name,
        "--policy.dir",
        str(checkpoint_dir),
    ]
    print("OpenPI serve command:")
    print(shlex.join(command))
    if str(args.dry_run).lower() == "true":
        return
    subprocess.run(command, cwd=args.openpi_root, env=env, check=True)


if __name__ == "__main__":
    main()

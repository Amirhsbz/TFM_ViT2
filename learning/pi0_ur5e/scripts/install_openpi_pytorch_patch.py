#!/usr/bin/env python3
"""Installs the Haptile tactile-expert PyTorch model into an OpenPI checkout.

Sibling to install_openpi_config.py, same marker-splice mechanism, but for the PyTorch backend:

1. Copies three vendored FTP1-sourced files (openpi_patches_pytorch/_vendor/, see that
   directory's README.md for provenance) plus three new Haptile files
   (haptile_tactile_encoder.py, haptile_tactile_pytorch.py, haptile_tactile_config.py) into
   $OPENPI_ROOT/src/openpi/models_pytorch/.
2. Splices haptile_train_config_patch.py's TrainConfig block into
   $OPENPI_ROOT/src/openpi/training/config.py, at the same _CONFIGS anchor
   install_openpi_config.py uses.

Prerequisite: run install_openpi_config.py first (or after -- order doesn't matter for the file
copies, but haptile_train_config_patch.py's TrainConfig reuses TeleGsyUR5eInputs/
TeleGsyLeRobotUR5eDataConfig/TeleGsyShapeTolerantCheckpointWeightLoader, which only exist in
$OPENPI_ROOT once the TELE_GSY_PI0_UR5E_CUP block from install_openpi_config.py is present).

Also prerequisite (not automated by this script -- see docs/ftp1_tactile_expert_port.md):
- transformers_replace (adaRMS patch) copied over $OPENPI_ROOT's installed transformers package.
- timm installed in $OPENPI_ROOT's environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

BEGIN = "# BEGIN TELE_GSY_PI0_UR5E_CUP_TACTILE"
END = "# END TELE_GSY_PI0_UR5E_CUP_TACTILE"
ANCHOR = "if len({config.name for config in _CONFIGS}) != len(_CONFIGS):"

_PATCH_DIR = Path(__file__).resolve().parents[1] / "openpi_patches_pytorch"
_VENDOR_DIR = _PATCH_DIR / "_vendor"

_VENDORED_FILES = ("ftp1_attention_masks.py", "ftp1_gemma_pytorch.py", "t3_tactile_encoder.py")
_HAPTILE_FILES = ("haptile_tactile_encoder.py", "haptile_tactile_pytorch.py", "haptile_tactile_config.py")


def install_models_pytorch_files(openpi_root: Path) -> list[Path]:
    dest_dir = openpi_root / "src" / "openpi" / "models_pytorch"
    if not dest_dir.exists():
        raise FileNotFoundError(f"OpenPI models_pytorch dir not found: {dest_dir}")
    installed = []
    for filename in _VENDORED_FILES:
        src = _VENDOR_DIR / filename
        dst = dest_dir / filename
        shutil.copyfile(src, dst)
        installed.append(dst)
    for filename in _HAPTILE_FILES:
        src = _PATCH_DIR / filename
        dst = dest_dir / filename
        shutil.copyfile(src, dst)
        installed.append(dst)
    return installed


def install_train_config_patch(openpi_root: Path, patch_path: Path) -> Path:
    config_path = openpi_root / "src" / "openpi" / "training" / "config.py"
    if not config_path.exists():
        raise FileNotFoundError(f"OpenPI config.py not found: {config_path}")
    patch = patch_path.read_text(encoding="utf-8").strip() + "\n"
    text = config_path.read_text(encoding="utf-8")
    if BEGIN in text and END in text:
        start = text.index(BEGIN)
        end = text.index(END, start) + len(END)
        text = text[:start] + patch.strip() + text[end:]
    else:
        if ANCHOR not in text:
            raise RuntimeError(f"Could not find insertion anchor in {config_path}")
        # Use the LAST occurrence, not the first: the real executable anchor line is always
        # last (the _CONFIGS uniqueness check, at the very end of the config list), but earlier
        # installed patch blocks (e.g. TELE_GSY_PI0_UR5E_CUP) quote this same anchor text inside
        # their own header comment ("Paste this block ... immediately before: if len(...)"), so
        # a first-occurrence replace can corrupt an already-installed block instead of inserting
        # before the real anchor.
        anchor_index = text.rindex(ANCHOR)
        text = text[:anchor_index] + patch + "\n" + text[anchor_index:]
    config_path.write_text(text, encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the Haptile tactile-expert PyTorch model into an OpenPI checkout."
    )
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument(
        "--patch",
        default=_PATCH_DIR / "haptile_train_config_patch.py",
        type=Path,
    )
    args = parser.parse_args()

    installed_files = install_models_pytorch_files(args.openpi_root)
    for path in installed_files:
        print(f"Installed {path}")

    config_path = install_train_config_patch(args.openpi_root, args.patch)
    print(f"Spliced pi0_ur5e_cup_tactile TrainConfig into {config_path}")


if __name__ == "__main__":
    main()

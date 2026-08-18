#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


BEGIN = "# BEGIN TELE_GSY_PI0_UR5E_CUP"
END = "# END TELE_GSY_PI0_UR5E_CUP"
ANCHOR = "if len({config.name for config in _CONFIGS}) != len(_CONFIGS):"


def install_config(openpi_root: Path, patch_path: Path) -> Path:
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
        text = text.replace(ANCHOR, patch + "\n" + ANCHOR, 1)
    config_path.write_text(text, encoding="utf-8")
    _patch_compute_norm_stats_import_order(openpi_root)
    return config_path


def _patch_compute_norm_stats_import_order(openpi_root: Path) -> None:
    """Avoid a local segfault from importing normalize before data_loader.

    On the installed OpenPI checkout used here, `scripts/compute_norm_stats.py`
    exits with 139 before argument parsing. Importing LeRobot/data_loader before
    `openpi.shared.normalize` avoids the crash while keeping the official entry
    point and script behavior intact.
    """
    script_path = openpi_root / "scripts" / "compute_norm_stats.py"
    if not script_path.exists():
        return
    text = script_path.read_text(encoding="utf-8")
    old = (
        "import openpi.models.model as _model\n"
        "import openpi.shared.normalize as normalize\n"
        "import openpi.training.config as _config\n"
        "import openpi.training.data_loader as _data_loader\n"
        "import openpi.transforms as transforms\n"
    )
    new = (
        "import openpi.models.model as _model\n"
        "import openpi.training.config as _config\n"
        "import openpi.training.data_loader as _data_loader\n"
        "import openpi.shared.normalize as normalize\n"
        "import openpi.transforms as transforms\n"
    )
    if old in text:
        text = text.replace(old, new, 1)
    workaround = (
        "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as "
        "_TeleGsyLeRobotImportOrderWorkaround\n"
    )
    if workaround not in text:
        text = text.replace("import tyro\n\n", "import tyro\n" + workaround + "\n", 1)
    script_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject the pi0_ur5e_cup TrainConfig into an OpenPI checkout.")
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument(
        "--patch",
        default=Path(__file__).resolve().parents[1] / "openpi_patches" / "pi0_ur5e_cup_config.py",
        type=Path,
    )
    args = parser.parse_args()
    config_path = install_config(args.openpi_root, args.patch)
    print(f"Installed pi0_ur5e_cup config into {config_path}")


if __name__ == "__main__":
    main()

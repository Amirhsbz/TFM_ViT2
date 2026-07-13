#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi0_ur5e.dataset_reader import DatasetReader
from pi0_ur5e.lerobot_writer import write_lerobot_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Convert UR5e raw trajectories to a LeRobot/OpenPI-style dataset.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", default=Path("learning/pi0_ur5e/configs/dataset_schema.yaml"), type=Path)
    parser.add_argument("--task-name", default="cup_pick_place")
    parser.add_argument("--default-prompt", default="pick up the paper cup and place it on the target")
    parser.add_argument(
        "--episode-prompt-cutoff",
        default=None,
        help="Episode id cutoff for assigning two prompts lexicographically, e.g. 0519_181949.",
    )
    parser.add_argument(
        "--prompt-before-or-at-cutoff",
        default=None,
        help="Prompt used for episodes with episode_id <= --episode-prompt-cutoff.",
    )
    parser.add_argument(
        "--prompt-after-cutoff",
        default=None,
        help="Prompt used for episodes with episode_id > --episode-prompt-cutoff.",
    )
    parser.add_argument("--action-mode", default=None, choices=["ee_delta_6d_gripper", "ee_absolute_6d_gripper", "joint_position_gripper", "joint_delta_gripper"])
    parser.add_argument("--include-tactile", default="false")
    parser.add_argument("--tactile-feature-mode", default=None, choices=["none", "low_dim", "image_embedding"])
    parser.add_argument("--tactile-embedding-dim", default=None, type=int)
    parser.add_argument("--skip-invalid-episodes", default="false")
    parser.add_argument("--repo-id", default="local/pi0_ur5e_cup")
    parser.add_argument("--overwrite", default="false")
    return parser.parse_args()


def main():
    args = parse_args()
    include_tactile = str(args.include_tactile).lower() == "true"
    config = {"include_tactile": include_tactile, "default_prompt": args.default_prompt}
    if args.action_mode is not None:
        config["action_mode"] = args.action_mode
    if args.tactile_feature_mode is not None:
        config["tactile_feature_mode"] = args.tactile_feature_mode
    if args.tactile_embedding_dim is not None:
        config["tactile_embedding_dim"] = args.tactile_embedding_dim
    reader = DatasetReader(
        args.input_root,
        args.config,
        config=config,
        skip_invalid_episodes=str(args.skip_invalid_episodes).lower() == "true",
    )
    episodes = reader.episodes()
    if not episodes:
        raise SystemExit(f"No readable episodes found under {args.input_root}")
    if reader.invalid_episode_errors:
        print(f"Skipped {len(reader.invalid_episode_errors)} invalid episode(s).")
    _apply_episode_prompt_cutoff(
        episodes,
        cutoff=args.episode_prompt_cutoff,
        before_or_at_prompt=args.prompt_before_or_at_cutoff,
        after_prompt=args.prompt_after_cutoff,
    )
    report = write_lerobot_dataset(
        episodes,
        args.output_root,
        task_name=args.task_name,
        image_size=reader.config.image_size,
        include_tactile=include_tactile,
        camera_padding_strategy=reader.config.camera_padding_strategy,
        repo_id=args.repo_id,
        overwrite=str(args.overwrite).lower() == "true",
    )
    print(f"Converted {report['episode_count']} episodes / {report['total_frames']} frames to {args.output_root}")
    print(f"Report: {args.output_root / 'conversion_report.json'}")


def _apply_episode_prompt_cutoff(episodes, *, cutoff: str | None, before_or_at_prompt: str | None, after_prompt: str | None) -> None:
    provided = [cutoff is not None, before_or_at_prompt is not None, after_prompt is not None]
    if not any(provided):
        return
    if not all(provided):
        raise SystemExit(
            "--episode-prompt-cutoff, --prompt-before-or-at-cutoff, and --prompt-after-cutoff must be provided together"
        )
    for episode in episodes:
        episode.language_instruction = before_or_at_prompt if episode.episode_id <= cutoff else after_prompt


if __name__ == "__main__":
    main()

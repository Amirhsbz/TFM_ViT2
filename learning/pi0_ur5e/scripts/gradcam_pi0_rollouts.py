#!/usr/bin/env python3
"""Grad-CAM + action-to-image attention maps for HaptileTactilePI0Pytorch on recorded trajectory.h5 episodes.

For every frame of every episode it asks two questions of the SigLIP image tokens the VLM sees
(and, for tactile checkpoints, of the two tactile sensors' images -- see "Tactile" below):

  * Grad-CAM: which image patches does the predicted action chunk depend on? The chunk comes from
    the model's real flow-matching sampler (sample_actions, KV cache and all, fixed seed noise),
    run with gradients enabled, and Grad-CAM is taken on the 16x16 projected SigLIP tokens of each
    camera. --target picks the scalar that is differentiated:
      action  ||A_pred||^2 over the selected steps/dims, in normalized action space. Normalization
              centres every dim, so this is "what pushes the action away from the dataset-average
              action" -- the evidence behind what the policy actually does.
      demo    -||A_pred - A_demo||^2 against the episode's recorded action chunk -- the patches
              whose features pull the prediction towards what the demonstrator did.
  * Attention: how much attention the 50 action-expert query tokens put on each image patch,
    averaged over heads, layers and denoising steps, plus the share of that attention that lands
    on each modality (each camera, language, each tactile sensor, state/action tokens).

Tactile: each sensor image goes through the T3 ViT, and only its CLS token reaches the tactile
expert. Grad-CAM is taken on the 14x14 gel-patch tokens entering the ViT's last block (the last
point where they still feed that CLS token), with the same target and backward pass as the
cameras. The attention column for tactile rows is that block's CLS->patch attention: which gel
regions are pooled into the token the policy sees. Tactile rows use their own heatmap scale.

Inputs are built exactly like training data: episodes are read with the same DatasetReader that
convert_to_lerobot.py uses, images are resized with the same resize_rgb, and the policy's own
input transforms (delta actions, normalization, tokenization) are applied unchanged.

Outputs, per episode, under <output-dir>/<episode_id>/:
  gradcam.mp4               rows = cameras (+ tactile sensors); columns = input | Grad-CAM | attention
  timeline.png              attention share per modality and Grad-CAM mass per camera over time
  gradcam_data.npz          raw maps and per-frame numbers for your own analysis
  summary.json              settings, action MSE vs the recording (input sanity check), shares
  frames/frame_XXXXX.png    only with --save-frames true

Runs inside the OpenPI environment; this script installs the Haptile patches and re-launches
itself there (via uv, or <openpi-root>/.venv/bin/python if uv is not on PATH), like
eval_policy_action_mse.py. A100-40GB or larger recommended.

Example:
  python learning/pi0_ur5e/scripts/gradcam_pi0_rollouts.py \
    --openpi-root /path/to/openpi \
    --checkpoint-dir /path/to/checkpoints/pi0_ur5e_cup_tactile/<exp_name>/30000 \
    --input /path/to/rollouts_or_demos \
    --output-dir /path/to/gradcam_out \
    --prompt "Wipe the markers on the white board and put the sponge back"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

INNER_FLAG = "--_inside-openpi-env"
CAMERA_LABELS = {"base_0_rgb": "base", "left_wrist_0_rgb": "wrist", "right_wrist_0_rgb": "right_wrist(pad)"}
TACTILE_KEYS = ("tactile_left", "tactile_right")


def str2bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Grad-CAM and attention maps of a Haptile pi0 policy over trajectory.h5 episodes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path, help="Step dir holding model.safetensors and assets/.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="A trajectory.h5, an episode dir containing one, or a parent dir of episode dirs.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", required=True, help="Must match the DEFAULT_PROMPT the dataset was converted with.")
    parser.add_argument("--config-name", default="pi0_ur5e_cup_tactile")
    parser.add_argument("--asset-id", default=None, help="Norm-stats asset id; inferred from <checkpoint>/assets if unique.")
    parser.add_argument(
        "--action-mode",
        default="joint_position_gripper",
        choices=["ee_delta_6d_gripper", "ee_absolute_6d_gripper", "joint_position_gripper", "joint_delta_gripper"],
        help="Must match the --action-mode the training dataset was converted with.",
    )
    parser.add_argument(
        "--use-tactile-input",
        default="auto",
        choices=["auto", "true", "false"],
        help="auto = true iff the checkpoint has tactile_encoder weights (vision-only runs have none).",
    )
    parser.add_argument("--target", default="action", choices=["action", "demo"])
    parser.add_argument("--action-dims", default=None, help="Comma-separated action dims to explain, e.g. '6' for the gripper. Default all.")
    parser.add_argument("--horizon-steps", type=int, default=None, help="Explain only the first N steps of the chunk. Default all.")
    parser.add_argument(
        "--cam-method",
        default="gradcam",
        choices=["gradcam", "gradxact"],
        help="gradcam = channel weights from token-averaged gradients; gradxact = per-token gradient x activation.",
    )
    parser.add_argument("--cam-norm", default="frame", choices=["frame", "episode"], help="Heatmap scale shared per frame or per episode.")
    parser.add_argument("--num-steps", type=int, default=10, help="Flow-matching denoising steps, as in deployment.")
    parser.add_argument("--seed", type=int, default=0, help="Noise seed, identical for every frame so maps only change with inputs.")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--display-size", type=int, default=336, help="Pixel size of each tile in the video.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap blend strength.")
    parser.add_argument("--save-frames", default="false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(INNER_FLAG, dest="inside_openpi_env", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------------------------
# Outer process: install patches, re-launch inside OpenPI's uv environment.
# ---------------------------------------------------------------------------------------------


def openpi_python(openpi_root: Path) -> list[str]:
    uv = shutil.which("uv")
    if uv is None and (Path.home() / ".local" / "bin" / "uv").exists():
        uv = str(Path.home() / ".local" / "bin" / "uv")
    if uv is not None:
        return [uv, "run", "python"]
    venv_python = openpi_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return [str(venv_python)]
    raise SystemExit("Could not find uv or <openpi-root>/.venv/bin/python. Install uv or set PATH so uv is available.")


def launch(args) -> None:
    for installer in ("install_openpi_config.py", "install_openpi_pytorch_patch.py"):
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name(installer)), "--openpi-root", str(args.openpi_root)],
            check=True,
        )
    # The inner process runs with cwd=openpi_root, so it resolves relative paths against ours.
    env = {**os.environ, "GRADCAM_CALLER_CWD": os.getcwd()}
    command = [*openpi_python(args.openpi_root), str(Path(__file__).resolve()), *sys.argv[1:], INNER_FLAG]
    subprocess.run(command, cwd=args.openpi_root, env=env, check=True)


# ---------------------------------------------------------------------------------------------
# Inner process (inside OpenPI's environment).
# ---------------------------------------------------------------------------------------------


def infer_asset_id(checkpoint_dir: Path) -> str | None:
    norm_stats = sorted((checkpoint_dir / "assets").glob("**/norm_stats.json"))
    if len(norm_stats) != 1:
        return None
    return str(norm_stats[0].parent.relative_to(checkpoint_dir / "assets"))


def checkpoint_has_tactile(weight_path: Path) -> bool:
    import safetensors

    with safetensors.safe_open(str(weight_path), framework="pt") as f:
        return any(key.startswith("tactile_encoder.") for key in f.keys())  # noqa: SIM118


def configure_env(args) -> bool:
    """Sets the PI0_UR5E_TACTILE_* variables the tactile TrainConfig reads at import time.

    Uses setdefault, so anything exported by the caller (e.g. PI0_UR5E_TACTILE_ACTION_HORIZON or
    PI0_UR5E_TACTILE_PI05 for a non-default training run) wins.
    """
    if args.use_tactile_input == "auto":
        use_tactile = checkpoint_has_tactile(args.checkpoint_dir / "model.safetensors")
    else:
        use_tactile = args.use_tactile_input == "true"
    asset_id = args.asset_id or infer_asset_id(args.checkpoint_dir)
    if asset_id is None:
        raise SystemExit(f"Could not infer the norm-stats asset id under {args.checkpoint_dir / 'assets'}; pass --asset-id.")
    os.environ["PI0_UR5E_TACTILE_USE_TACTILE_INPUT"] = str(use_tactile).lower()
    os.environ.setdefault("PI0_UR5E_TACTILE_ASSET_ID", asset_id)
    os.environ.setdefault("PI0_UR5E_TACTILE_ACTION_FORMAT", args.action_mode)
    # The checkpoint carries the fine-tuned tactile encoder; no need to download T3 weights.
    os.environ["PI0_UR5E_TACTILE_LOAD_T3_CHECKPOINT"] = "false"
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
    return use_tactile


class AttentionRecorder:
    """Wraps Gemma's eager attention to keep the action expert's attention rows during denoising.

    Only active inside denoise_step, where the queries are the suffix tokens (state + actions) and
    the keys are [VLM prefix | tactile | suffix] -- see HaptileTactilePI0Pytorch.denoise_step.
    """

    def __init__(self, modeling_gemma, action_horizon: int):
        self._modeling_gemma = modeling_gemma
        self._original = modeling_gemma.eager_attention_forward
        self.action_horizon = action_horizon
        self.active = False
        self.sum = None
        self.count = 0

        def recording_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            output, weights = self._original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            if self.active:
                rows = weights[0, :, -self.action_horizon :, :].detach().float().mean(dim=(0, 1))
                self.sum = rows if self.sum is None else self.sum + rows
                self.count += 1
            return output, weights

        modeling_gemma.eager_attention_forward = recording_attention

    def reset(self):
        self.sum = None
        self.count = 0

    def mean(self):
        return (self.sum / max(self.count, 1)).cpu().numpy()

    def wrap_denoise_step(self, model):
        original = model.denoise_step

        def recorded_denoise_step(*args, **kwargs):
            self.active = True
            try:
                return original(*args, **kwargs)
            finally:
                self.active = False

        model.denoise_step = recorded_denoise_step


class ImageTokenTap:
    """Forward hook on the multimodal projector: captures each camera's image tokens as a fresh
    leaf requiring grad, so Grad-CAM gradients stop there instead of flowing into SigLIP."""

    def __init__(self, projector):
        self.tokens = []
        projector.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        tokens = output.detach().requires_grad_(True)
        self.tokens.append(tokens)
        return tokens

    def reset(self):
        self.tokens = []


class TactileTokenTap:
    """Forward pre-hook on the last block of the T3 tactile ViT (shared by both sensors, run left
    then right). The encoder passes only its output CLS token on to the tactile expert, so patch
    tokens are captured at the input of the last block -- the last place they still feed the CLS
    token -- as fresh leaves requiring grad. Also records that block's CLS->patch attention, i.e.
    which gel regions get pooled into the token the tactile expert sees."""

    def __init__(self, vit):
        self.tokens = []
        self.cls_attention = []
        vit.blocks[-1].register_forward_pre_hook(self._hook)

    def _hook(self, block, args):
        import torch

        tokens = args[0].detach().requires_grad_(True)
        self.tokens.append(tokens)
        with torch.no_grad():
            attn = block.attn
            x = block.norm1(tokens)
            batch, length, _ = x.shape
            qkv = attn.qkv(x).reshape(batch, length, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k = attn.q_norm(qkv[0]), attn.k_norm(qkv[1])
            weights = ((q[:, :, :1] * attn.scale) @ k.transpose(-2, -1)).softmax(dim=-1)  # [B, heads, 1, N]
            cls_to_patches = weights[0, :, 0, 1:].float().mean(dim=0)
            side = int(round(cls_to_patches.numel() ** 0.5))
            self.cls_attention.append(cls_to_patches.reshape(side, side).cpu().numpy())
        return (tokens, *args[1:])

    def reset(self):
        self.tokens = []
        self.cls_attention = []


def compute_cam(tokens, grads, method: str):
    import torch

    tokens = tokens.detach().float()
    grads = grads.float()
    if method == "gradcam":
        weights = grads.mean(dim=1, keepdim=True)
        cam = (weights * tokens).sum(dim=-1)
    else:
        cam = (grads * tokens).sum(dim=-1)
    cam = torch.relu(cam)[0]
    side = int(round(cam.numel() ** 0.5))
    return cam.reshape(side, side).cpu().numpy()


def action_chunk_for_frame(actions, t: int, horizon: int):
    """Future action chunk, repeating the last action past the episode end (LeRobot clamps the
    same way when it builds training chunks)."""
    import numpy as np

    idx = np.minimum(np.arange(t, t + horizon), len(actions) - 1)
    return actions[idx]


def heatmap_overlay(image_rgb, heat, alpha: float, size: int):
    import cv2
    import numpy as np

    image = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    heat = cv2.resize(np.clip(heat, 0.0, 1.0).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1 - alpha, colored, alpha, 0)


def render_panel(images, cams, attns, camera_names, attn_titles, header_lines, args):
    import cv2
    import numpy as np

    size = args.display_size
    rows = []
    for image, cam, attn, name, attn_title in zip(images, cams, attns, camera_names, attn_titles, strict=True):
        tiles = [
            cv2.cvtColor(cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR),
            heatmap_overlay(image, cam, args.alpha, size),
            heatmap_overlay(image, attn, args.alpha, size),
        ]
        for tile, label in zip(tiles, (name, f"{name} Grad-CAM", f"{name} {attn_title}"), strict=True):
            cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.hstack(tiles))
    body = np.vstack(rows)
    header = np.zeros((24 * len(header_lines) + 8, body.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(header_lines):
        cv2.putText(header, line, (8, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([header, body])


def plot_timeline(path: Path, frame_indices, shares, share_names, mass_groups, title: str):
    """mass_groups: list of (ylabel, [N, C] Grad-CAM mass, names) -- one subplot each, since
    SigLIP and T3 token Grad-CAMs are not on comparable scales."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping timeline.png")
        return
    fig, axes = plt.subplots(1 + len(mass_groups), 1, figsize=(11, 3.25 * (1 + len(mass_groups))), sharex=True)
    axes[0].stackplot(frame_indices, shares.T, labels=share_names, alpha=0.85)
    axes[0].set_ylabel("action-token attention share")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper left", fontsize=8, ncol=len(share_names))
    for ax, (ylabel, mass, names) in zip(axes[1:], mass_groups, strict=True):
        for i, name in enumerate(names):
            ax.plot(frame_indices, mass[:, i], label=name)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_inside(args) -> None:
    use_tactile = configure_env(args)

    import jax
    import numpy as np
    import torch

    # Import data_loader before config/policy_config to avoid the native-extension segfault
    # eval_policy_action_mse.py works around the same way.
    from openpi.training import data_loader as _unused_data_loader  # noqa: F401
    from openpi.models import model as _model
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config
    from transformers.models.gemma import modeling_gemma

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pi0_ur5e.dataset_reader import DatasetReader
    from pi0_ur5e.io_utils import resize_rgb

    cfg = openpi_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        cfg,
        args.checkpoint_dir,
        default_prompt=args.prompt,
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device=args.device,
    )
    model = policy._model  # noqa: SLF001
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    horizon = model.config.action_horizon
    action_dim = model.config.action_dim
    dims = list(range(action_dim)) if args.action_dims is None else [int(d) for d in args.action_dims.split(",")]
    steps = horizon if args.horizon_steps is None else min(args.horizon_steps, horizon)

    tap = ImageTokenTap(model.paligemma_with_expert.paligemma.model.multi_modal_projector)
    tactile_tap = TactileTokenTap(model.tactile_encoder.vit) if use_tactile else None
    recorder = AttentionRecorder(modeling_gemma, horizon)
    recorder.wrap_denoise_step(model)
    # sample_actions is @torch.no_grad(); its undecorated body gives the identical computation
    # with gradients flowing back to the image tokens.
    sample_actions = type(model).sample_actions.__wrapped__

    print(
        f"Loaded {args.config_name} from {args.checkpoint_dir} "
        f"(tactile={use_tactile}, horizon={horizon}, target={args.target}, dims={dims}, steps={steps})"
    )

    reader = DatasetReader(
        args.input,
        Path(__file__).resolve().parents[1] / "configs" / "dataset_schema.yaml",
        config={
            "action_mode": args.action_mode,
            "include_tactile": use_tactile,
            "tactile_feature_mode": "raw_image" if use_tactile else "none",
            "default_prompt": args.prompt,
        },
    )
    image_size = reader.config.image_size
    # Read episodes one at a time: DatasetReader.episodes() would decode every video up front.
    episode_paths = reader._episode_paths()  # noqa: SLF001
    if not episode_paths:
        raise SystemExit(f"No trajectory.h5 episodes found under {args.input}")
    if args.max_episodes is not None:
        episode_paths = episode_paths[: args.max_episodes]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in episode_paths:
        episode = reader._read_episode_path(path)  # noqa: SLF001
        if episode is None:
            print(f"Skipping unreadable episode {path}")
            continue
        out_dir = args.output_dir / episode.episode_id
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_indices = list(range(0, len(episode.timestamps), args.stride))
        if args.max_frames is not None:
            frame_indices = frame_indices[: args.max_frames]
        print(f"\n=== {episode.episode_id}: {len(frame_indices)} of {len(episode.timestamps)} frames -> {out_dir}")

        records = []
        camera_keys = None
        for n, t in enumerate(frame_indices):
            obs = {
                "base_rgb": resize_rgb(episode.base_rgb[t], image_size),
                "wrist_rgb": resize_rgb(episode.wrist_rgb[t], image_size),
                "state": np.asarray(episode.robot_state[t], dtype=np.float32),
                "prompt": args.prompt,
            }
            if use_tactile:
                obs["tactile_left_rgb"] = resize_rgb(episode.tactile_left_rgb[t], image_size)
                obs["tactile_right_rgb"] = resize_rgb(episode.tactile_right_rgb[t], image_size)
            demo_chunk = action_chunk_for_frame(episode.action, t, horizon).astype(np.float32)
            obs["actions"] = demo_chunk.copy()  # DeltaActions edits the array in place

            # Same steps as Policy.infer, minus no_grad.
            inputs = policy._input_transform(jax.tree.map(lambda x: x, obs))  # noqa: SLF001
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(args.device)[None, ...], inputs)
            demo_norm = inputs.pop("actions").float()
            observation = _model.Observation.from_dict(inputs)
            if camera_keys is None:
                camera_keys = [k for k in observation.images if bool(observation.image_masks[k].reshape(-1)[0])]
                if use_tactile:
                    camera_keys += list(TACTILE_KEYS)

            generator = torch.Generator(device=args.device).manual_seed(args.seed)
            noise = torch.randn((1, horizon, action_dim), generator=generator, device=args.device, dtype=torch.float32)

            tap.reset()
            if tactile_tap is not None:
                tactile_tap.reset()
            recorder.reset()
            with torch.enable_grad():
                pred_norm = sample_actions(model, args.device, observation, noise=noise, num_steps=args.num_steps).float()
                selected = pred_norm[:, :steps, dims]
                if args.target == "action":
                    score = selected.pow(2).sum()
                else:
                    score = -(selected - demo_norm[:, :steps, dims]).pow(2).sum()
                image_names = list(observation.images)
                tactile_tokens = tactile_tap.tokens if tactile_tap is not None else []
                grads = torch.autograd.grad(score, tap.tokens + tactile_tokens, allow_unused=True)

            cams = {}
            for name, tokens, grad in zip(image_names, tap.tokens, grads[: len(tap.tokens)], strict=True):
                side = int(round(tokens.shape[1] ** 0.5))
                cams[name] = np.zeros((side, side)) if grad is None else compute_cam(tokens, grad, args.cam_method)
            # Tactile: drop the CLS token, keep the 14x14 gel patches.
            if tactile_tokens:
                for name, tokens, grad in zip(TACTILE_KEYS, tactile_tokens, grads[len(tap.tokens) :], strict=True):
                    side = int(round((tokens.shape[1] - 1) ** 0.5))
                    cams[name] = (
                        np.zeros((side, side))
                        if grad is None
                        else compute_cam(tokens[:, 1:], grad[:, 1:], args.cam_method)
                    )

            # Split the action rows' attention over [images | language | tactile | suffix].
            attn = recorder.mean()
            n_img = [tokens.shape[1] for tokens in tap.tokens]
            lang_len = int(observation.tokenized_prompt.shape[1])
            prefix_len = sum(n_img) + lang_len
            suffix_len = horizon + (0 if model.pi05 else 1)
            tactile_len = attn.shape[0] - prefix_len - suffix_len
            attn_maps, shares, start = {}, {}, 0
            for name, count in zip(image_names, n_img, strict=True):
                segment = attn[start : start + count]
                side = int(round(count ** 0.5))
                attn_maps[name] = segment.reshape(side, side)
                shares[CAMERA_LABELS.get(name, name)] = float(segment.sum())
                start += count
            shares["language"] = float(attn[start : start + lang_len].sum())
            if tactile_len == len(TACTILE_KEYS):  # one CLS-pooled token per sensor, left then right
                for i, name in enumerate(TACTILE_KEYS):
                    shares[name] = float(attn[prefix_len + i])
            elif tactile_len > 0:
                shares["tactile"] = float(attn[prefix_len : prefix_len + tactile_len].sum())
            shares["state+actions"] = float(attn[prefix_len + max(tactile_len, 0) :].sum())
            if tactile_tap is not None:
                attn_maps.update(zip(TACTILE_KEYS, tactile_tap.cls_attention, strict=True))

            outputs = {
                "state": np.asarray(inputs["state"][0].detach().cpu()),
                "actions": np.asarray(pred_norm[0].detach().cpu()),
            }
            pred_actions = policy._output_transform(outputs)["actions"]  # noqa: SLF001
            n_cmp = min(len(pred_actions), len(demo_chunk))
            mse = float(np.mean((pred_actions[:n_cmp] - demo_chunk[:n_cmp, : pred_actions.shape[1]]) ** 2))

            records.append(
                {
                    "frame": t,
                    "cams": cams,
                    "attn": attn_maps,
                    "shares": shares,
                    "score": float(score.detach()),
                    "mse": mse,
                    "pred_actions": pred_actions,
                    "images": {
                        "base_0_rgb": obs["base_rgb"],
                        "left_wrist_0_rgb": obs["wrist_rgb"],
                        **(
                            {"tactile_left": obs["tactile_left_rgb"], "tactile_right": obs["tactile_right_rgb"]}
                            if use_tactile
                            else {}
                        ),
                    },
                }
            )
            if (n + 1) % 25 == 0 or n + 1 == len(frame_indices):
                print(f"  {n + 1}/{len(frame_indices)} frames  (action MSE vs recording: {mse:.5f})", flush=True)

        write_episode_outputs(out_dir, episode, records, camera_keys, args, use_tactile)


def write_episode_outputs(out_dir: Path, episode, records, camera_keys, args, use_tactile) -> None:
    import cv2
    import numpy as np

    # Cameras and tactile sensors are separate groups: their Grad-CAMs come from different token
    # spaces (SigLIP vs T3) and their attention maps measure different things, so each group gets
    # its own heatmap scale. Within a group the scale is shared, so left vs right (or base vs
    # wrist) stay comparable.
    shown = [k for k in camera_keys if k in records[0]["images"]]
    vision_keys = [k for k in shown if k not in TACTILE_KEYS]
    tactile_keys = [k for k in shown if k in TACTILE_KEYS]
    groups = [g for g in (vision_keys, tactile_keys) if g]

    def stack(field, keys):
        return np.stack([[r[field][k] for k in keys] for r in records])  # [N, C, g, g]

    def group_scales(field, keys, percentile):
        values = stack(field, keys).reshape(len(records), -1)
        if args.cam_norm == "episode":
            scale = np.full(len(records), np.percentile(values, percentile))
        else:
            scale = np.percentile(values, percentile, axis=1)
        return np.maximum(scale, 1e-12)

    # A few "attention sink" patches soak up most of the attention mass in ViT-fed transformers;
    # scaling attention to the 99th percentile keeps them from washing out the rest of the map.
    cam_scale, attn_scale = {}, {}
    for group in groups:
        group_cam_scale = group_scales("cams", group, 100)
        group_attn_scale = group_scales("attn", group, 99)
        for key in group:
            cam_scale[key] = group_cam_scale
            attn_scale[key] = group_attn_scale
    labels = [CAMERA_LABELS.get(k, k) for k in shown]
    attn_titles = ["T3 CLS attn" if k in TACTILE_KEYS else "action->image attn" for k in shown]

    share_names = list(records[0]["shares"])
    shares = np.asarray([[r["shares"][s] for s in share_names] for r in records])
    frame_indices = np.asarray([r["frame"] for r in records])
    mses = np.asarray([r["mse"] for r in records])

    fps = max(float(episode.metadata.get("hz") or 10.0) / args.stride, 1.0)
    frames_dir = out_dir / "frames"
    if str2bool(args.save_frames):
        frames_dir.mkdir(exist_ok=True)
    writer = None
    for i, record in enumerate(records):
        header = [
            f"{episode.episode_id}  frame {record['frame']}  target={args.target}  cam={args.cam_method}  "
            f"action MSE vs recording={record['mse']:.4f}",
            *textwrap.wrap(
                "attn share: " + "  ".join(f"{name} {record['shares'][name]:.2f}" for name in share_names),
                width=args.display_size * 3 // 11,
            ),
        ]
        panel = render_panel(
            [record["images"][k] for k in shown],
            [record["cams"][k] / cam_scale[k][i] for k in shown],
            [record["attn"][k] / attn_scale[k][i] for k in shown],
            labels,
            attn_titles,
            header,
            args,
        )
        if writer is None:
            height, width = panel.shape[:2]
            writer = cv2.VideoWriter(str(out_dir / "gradcam.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        writer.write(panel)
        if str2bool(args.save_frames):
            cv2.imwrite(str(frames_dir / f"frame_{record['frame']:05d}.png"), panel)
    if writer is not None:
        writer.release()

    arrays = {}
    mass_groups = []
    if vision_keys:
        arrays.update(
            camera_names=np.asarray([CAMERA_LABELS.get(k, k) for k in vision_keys]),
            gradcam=stack("cams", vision_keys),
            attention=stack("attn", vision_keys),
        )
        mass = arrays["gradcam"].reshape(len(records), len(vision_keys), -1).sum(axis=2)
        mass_groups.append(("camera Grad-CAM mass", mass, arrays["camera_names"].tolist()))
    if tactile_keys:
        arrays.update(
            tactile_names=np.asarray(tactile_keys),
            tactile_gradcam=stack("cams", tactile_keys),
            tactile_cls_attention=stack("attn", tactile_keys),
        )
        mass = arrays["tactile_gradcam"].reshape(len(records), len(tactile_keys), -1).sum(axis=2)
        mass_groups.append(("tactile Grad-CAM mass", mass, tactile_keys))
    np.savez_compressed(
        out_dir / "gradcam_data.npz",
        frame_indices=frame_indices,
        attention_share=shares,
        attention_share_names=np.asarray(share_names),
        score=np.asarray([r["score"] for r in records]),
        action_mse=mses,
        pred_actions=np.stack([r["pred_actions"] for r in records]),
        **arrays,
    )
    plot_timeline(
        out_dir / "timeline.png",
        frame_indices,
        shares,
        share_names,
        mass_groups,
        f"{episode.episode_id} ({args.target} target)",
    )
    summary = {
        "episode_id": episode.episode_id,
        "source": episode.metadata.get("source_path"),
        "checkpoint_dir": str(args.checkpoint_dir),
        "prompt": args.prompt,
        "use_tactile_input": use_tactile,
        "target": args.target,
        "cam_method": args.cam_method,
        "action_dims": args.action_dims,
        "horizon_steps": args.horizon_steps,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "frames_processed": len(records),
        "stride": args.stride,
        "action_mse_vs_recording": {"mean": float(mses.mean()), "median": float(np.median(mses)), "max": float(mses.max())},
        "mean_attention_share": {name: float(v) for name, v in zip(share_names, shares.mean(axis=0), strict=True)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir / 'gradcam.mp4'}, timeline.png, gradcam_data.npz, summary.json")
    print(f"  mean attention share: {summary['mean_attention_share']}")


def main() -> None:
    args = parse_args()
    if args.inside_openpi_env:
        caller_cwd = Path(os.environ.get("GRADCAM_CALLER_CWD", os.getcwd()))
        for name in ("checkpoint_dir", "input", "output_dir"):
            setattr(args, name, (caller_cwd / getattr(args, name)).resolve())
        run_inside(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()

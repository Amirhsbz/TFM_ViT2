"""
gradcam_offline.py
──────────────────
Offline Grad-CAM visualiser for the Diffusion Policy image encoder.

WHAT THIS SCRIPT DOES
─────────────────────
For every PNG image in a folder of saved camera frames, it:
  1. Loads the trained model from a checkpoint directory
  2. Runs each frame through the ResNet18 image encoder
  3. Computes a Grad-CAM heatmap (which pixels the encoder focused on)
  4. Saves a side-by-side image: original | heatmap overlay
  5. Optionally stitches all frames into a video

HOW TO RUN
──────────
python gradcam_offline.py \
    --ckpt_path  /path/to/checkpoint/last.ckpt \
    --frames_dir /path/to/saved_frames/ \
    --output_dir /path/to/output/ \
    --camera_idx 0 \
    --make_video  True

The checkpoint directory (the folder containing last.ckpt) must also
contain: ema_last.ckpt, args_log.txt, stats.pkl

FEEDING A ROLLOUT trajectory.h5 DIRECTLY
─────────────────────────────────────────
Rollouts recorded by this codebase are saved as trajectory.h5, with each
camera's video embedded as an mp4 byte stream under the "videos" group
(keys like "base_camera_rgb_0", "base_camera_rgb_1", ...) -- this is the
exact same file format that learning/dp/data_processing.py decodes when
loading data for Diffusion Policy training/eval.

--frames_dir now accepts, in addition to a folder of already-saved PNG/JPG
frames:
  - a path directly to a trajectory.h5 file
  - a path to a directory containing a trajectory.h5 file (e.g. the
    episode directory produced by a rollout)

In either case, the script decodes the embedded video for the camera being
visualised (--camera_idx) using the same decoder used for training
(data_processing._decode_h5_video_frames) and writes it out as
frame_00000.png, frame_00001.png, ... under
<output_dir>/extracted_frames/, then feeds those PNGs to Grad-CAM as usual.
If that folder already has frames from a previous run, extraction is
skipped unless --overwrite_frames True is passed.

BATCH MODE: PROCESSING MULTIPLE TRAJECTORIES IN ONE RUN
─────────────────────────────────────────────────────────
--frames_dir can also point at a *parent* folder containing several
trajectory subfolders (e.g. eval_wipe_board/dp_img_pos/, where each
subfolder like 0527_153346/ holds its own trajectory.h5). The script
detects this automatically: if frames_dir does not itself contain
trajectory.h5 or PNG/JPG frames, it scans one level down and treats every
subfolder that does as a separate trajectory to process. The checkpoint
is loaded only once and reused across all of them.

In batch mode (and always, for consistency), --output_dir is treated as a
parent folder too: results for each trajectory are written to
    <output_dir>/<trajectory_name>/camera_idx<camera_idx>/
so passing the same --output_dir on every run never overwrites a
different trajectory's or camera's results.

HOW TO SAVE FRAMES DURING A ROLLOUT (legacy / manual alternative)
─────────────────────────────────────────────────────────────────
In agents/dp_agent.py, inside the act() method, add these two lines
right after "obs = self._preprocess_obs(obs)" fails you can add before it:

    import cv2, os
    os.makedirs("saved_frames", exist_ok=True)
    raw_img = obs["base_camera_rgb"][0]          # first camera, HWC uint8
    cv2.imwrite(f"saved_frames/frame_{self._step:05d}.png",
                cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))
    self._step = getattr(self, "_step", 0) + 1

That is the only change needed to your deployment code. Prefer the
trajectory.h5 route above when a rollout was already recorded -- it needs
no code changes to the deployment agent at all.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

# ── Make sure the repo root is on the Python path so imports work ─────────────
# This script lives at the repo root, so __file__ already resolves correctly.
# If you move the script, adjust this line to point at the repo root.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from agents.dp_agent import BimanualDPAgent  # noqa: E402  (import after path fix)
from learning.dp import data_processing  # noqa: E402  (same h5 decoder used for training)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — GradCAM class
# ═════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Attaches hooks to the last convolutional block (layer4) of one
    ImageEncoder and produces a spatial heatmap for any input image.

    Parameters
    ----------
    img_encoder : nn.Module
        One element from agent.dp.policy.nets["img_encoder"],
        i.e. the ImageEncoder for a single camera.
    """

    def __init__(self, img_encoder):
        self.encoder = img_encoder

        # Two plain Python dicts used as mailboxes.
        # The hook functions below will deposit values into them.
        self._activations = {}
        self._gradients = {}

        # ── Install the forward hook on layer4 ───────────────────────────────
        # PyTorch calls this automatically every time layer4 finishes its
        # forward computation. It receives (module, input, output) from PyTorch.
        # We save 'output', which is the 7×7 feature map of shape [1, 512, 7, 7].
        # .detach() means "copy the values but don't track them for gradients" —
        # we just want to read them, not interfere with the backward pass.
        self.encoder.encoder.layer4.register_forward_hook(self._save_activation)

        # ── Install the backward hook on layer4 ──────────────────────────────
        # PyTorch calls this automatically when gradients flow backwards through
        # layer4. grad_output[0] is the gradient of our scalar loss with respect
        # to each cell in the 7×7 feature map — i.e. "how much did this cell
        # matter to the final output?"
        self.encoder.encoder.layer4.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # Called automatically by PyTorch during the forward pass.
        # Stores the 7×7 activation map produced by layer4.
        self._activations["layer4"] = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # Called automatically by PyTorch during the backward pass.
        # Stores the gradients flowing back through layer4.
        self._gradients["layer4"] = grad_output[0].detach()

    def generate(self, frame_tensor):
        """
        Compute a Grad-CAM heatmap for a single preprocessed image tensor.

        Parameters
        ----------
        frame_tensor : torch.Tensor, shape [1, C, H, W]
            The image after applying agent.dp.eval_transform().
            Must already be on the same device as the model (GPU).

        Returns
        -------
        cam : np.ndarray, shape [7, 7], values in [0, 1]
            The raw heatmap before upsampling to the original image size.
        """

        # Switch to train() mode so GroupNorm behaves the same way it did
        # during training. In eval() mode, GroupNorm is identical, but
        # BatchNorm (if present) would differ — we keep train() for safety.
        self.encoder.train()

        # ── Forward pass ─────────────────────────────────────────────────────
        # Run the image through the full ImageEncoder.
        # This triggers _save_activation → self._activations["layer4"] is filled.
        # 'feat' has shape [1, 32] (your image_output_size).
        feat = self.encoder(frame_tensor)

        # We need a single scalar to differentiate.
        # .norm() = the overall magnitude of the feature vector.
        # Think of it as "how strongly did this image activate the encoder?"
        # Any scalar that summarises the output works here.
        scalar_score = feat.norm()

        # ── Backward pass ─────────────────────────────────────────────────────
        # Zero out any leftover gradients from a previous call.
        self.encoder.zero_grad()

        # Compute gradients all the way back through the network.
        # When the gradient flow reaches layer4, _save_gradient fires
        # → self._gradients["layer4"] is filled.
        scalar_score.backward()

        # ── Build the CAM ─────────────────────────────────────────────────────
        acts  = self._activations["layer4"]   # shape: [1, 512, 7, 7]
        grads = self._gradients["layer4"]     # shape: [1, 512, 7, 7]

        # Average the gradients across the 7×7 spatial grid for each channel.
        # Result shape: [1, 512, 1, 1]
        # Each value = "how important was this channel, on average?"
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weight each channel's 7×7 map by its importance, then sum all 512
        # channels together. Result: one 7×7 importance map.
        cam = (weights * acts).sum(dim=1).squeeze()   # [7, 7]

        # Keep only positive contributions.
        # Negative values mean the region suppressed the output — we don't
        # care about those for visualisation purposes.
        cam = torch.relu(cam)

        # Guard against an all-zero map (e.g. a blank/dark image).
        if cam.max() > 1e-8:
            cam = cam / cam.max()   # normalise to [0, 1]

        return cam.cpu().numpy()   # return as numpy for use with OpenCV


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Image utilities
# ═════════════════════════════════════════════════════════════════════════════

def extract_frames_from_h5(h5_path, output_dir, camera_number, overwrite=False):
    """
    Decode the embedded rollout video for ONE camera out of trajectory.h5
    and save every frame as a PNG in output_dir.

    Uses data_processing._decode_h5_video_frames — the exact same decoder
    learning/dp/data_processing.py's from_h5() uses when loading
    trajectory.h5 for Diffusion Policy training/eval — so the extracted
    frames are byte-identical to what the training pipeline sees before
    its own downsample/crop.

    Parameters
    ----------
    h5_path       : path to trajectory.h5
    output_dir    : folder to write frame_00000.png, frame_00001.png, ...
    camera_number : the physical camera index used in the h5 video key,
                    i.e. the N in "base_camera_rgb_N" (NOT necessarily the
                    same as --camera_idx — see resolve_frames_dir).
    overwrite     : if False and output_dir already has PNGs, skip decoding.
    """
    import h5py

    os.makedirs(output_dir, exist_ok=True)
    existing = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".png"))
    if existing and not overwrite:
        print(
            f"Found {len(existing)} previously extracted frames in {output_dir}, "
            "reusing them (pass --overwrite_frames True to re-decode)."
        )
        return output_dir

    with h5py.File(h5_path, "r") as f:
        if "videos" not in f:
            raise ValueError(f"No 'videos' group found in {h5_path}")

        video_key = None
        for candidate in (
            f"base_camera_rgb_{camera_number}",
            f"base_rgb_{camera_number}",
        ):
            if candidate in f["videos"]:
                video_key = candidate
                break
        if video_key is None:
            raise ValueError(
                f"Could not find a base-camera video stream for camera "
                f"{camera_number} in {h5_path}. "
                f"Available streams: {list(f['videos'].keys())}"
            )

        print(f"Decoding embedded video stream '{video_key}' from {h5_path} ...")
        frames = data_processing._decode_h5_video_frames(f["videos"][video_key])

    print(f"Writing {len(frames)} frames to {output_dir} ...")
    for i, rgb_frame in enumerate(frames):
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:05d}.png"), bgr_frame)

    return output_dir


def resolve_frames_dir(frames_dir, output_dir, camera_number, overwrite_frames):
    """
    Accepts either:
      - a directory of already-saved PNG/JPG frames (used as-is), or
      - a path directly to a trajectory.h5 file, or
      - a directory containing a trajectory.h5 file (e.g. a rollout's
        episode directory)

    In the h5 cases, decodes the rollout video for `camera_number` into
    <output_dir>/extracted_frames/ and returns that path. Otherwise
    returns frames_dir unchanged.
    """
    if os.path.isfile(frames_dir) and frames_dir.lower().endswith(".h5"):
        h5_path = frames_dir
    elif os.path.isdir(frames_dir):
        maybe_h5 = os.path.join(frames_dir, "trajectory.h5")
        h5_path = maybe_h5 if os.path.exists(maybe_h5) else None
    else:
        raise FileNotFoundError(f"frames_dir does not exist: {frames_dir}")

    if h5_path is None:
        return frames_dir

    extracted_dir = os.path.join(output_dir, "extracted_frames")
    return extract_frames_from_h5(
        h5_path, extracted_dir, camera_number, overwrite=overwrite_frames
    )


def _dir_has_frames_or_h5(path):
    """True if `path` directly contains trajectory.h5 or PNG/JPG frames."""
    if os.path.exists(os.path.join(path, "trajectory.h5")):
        return True
    return any(
        f.lower().endswith((".png", ".jpg")) for f in os.listdir(path)
    )


def resolve_input_trajectories(frames_dir):
    """
    Figure out what --frames_dir points at and return a list of
    (trajectory_name, trajectory_path) pairs to process.

    Accepts:
      - a path directly to a trajectory.h5 file
        -> single trajectory, named after its parent directory
      - a single trajectory folder (contains trajectory.h5 and/or PNG/JPG
        frames directly)
        -> single trajectory, named after that folder
      - a parent folder containing multiple trajectory subfolders (each
        with its own trajectory.h5 or PNG/JPG frames)
        -> one entry per subfolder, in sorted order, named after each
        subfolder. Subfolders that contain neither (e.g. a previous run's
        output folder) are skipped.
    """
    if os.path.isfile(frames_dir):
        if not frames_dir.lower().endswith(".h5"):
            raise FileNotFoundError(
                f"frames_dir points at a file that is not a .h5: {frames_dir}"
            )
        name = os.path.basename(os.path.dirname(os.path.abspath(frames_dir)))
        return [(name, frames_dir)]

    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"frames_dir does not exist: {frames_dir}")

    if _dir_has_frames_or_h5(frames_dir):
        name = os.path.basename(os.path.normpath(frames_dir))
        return [(name, frames_dir)]

    trajectories = []
    for name in sorted(os.listdir(frames_dir)):
        path = os.path.join(frames_dir, name)
        if os.path.isdir(path) and _dir_has_frames_or_h5(path):
            trajectories.append((name, path))

    if not trajectories:
        raise FileNotFoundError(
            f"No trajectory.h5 or PNG/JPG frames found directly in "
            f"{frames_dir}, nor in any of its immediate subdirectories."
        )
    return trajectories


def load_and_preprocess(frame_path, dp, device):
    """
    Load a saved PNG frame from disk and apply the same preprocessing
    the model used during training/inference.

    Parameters
    ----------
    dp : the loaded DPAgent (agent.dp from learning/dp/pipeline.py),
         used to reuse its actual downsample + eval_transform instead of
         reimplementing them, so this always matches training exactly.

    Returns
    -------
    raw_frame   : np.ndarray [H, W, 3] uint8 RGB  — for display
    frame_tensor: torch.Tensor [1, C, H, W] float — for the model
    """
    # cv2.imread returns BGR uint8 [H, W, 3]
    bgr = cv2.imread(frame_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {frame_path}")

    # Convert BGR → RGB because the model was trained on RGB images.
    raw_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Convert numpy [H, W, 3] → torch float [3, H, W]
    tensor = torch.from_numpy(raw_frame).float()   # values still 0-255
    tensor = tensor.permute(2, 0, 1)               # [H,W,3] → [3,H,W]
    tensor = tensor.unsqueeze(0)                   # → [1, 3, H, W]

    # Mirror Agent._get_image_observation (learning/dp/pipeline.py): raw
    # 480x640 camera frames are first downsampled to 240x320 before the
    # eval_transform's CenterCrop+Normalize is applied. Skipping this step
    # (as the original offline script did) would feed the encoder a crop
    # at the wrong scale relative to what it saw during training.
    _, _, H, W = tensor.shape
    if H == 480 and W == 640 and not dp.color_jitter:
        tensor = dp.downsample(tensor)

    # Same centre-crop + normalise used at inference time.
    # From pipeline.py: CenterCrop(216, 288) then Normalize(mean=128, std=128)
    # This maps pixel values from [0,255] → roughly [-1, 1]
    tensor = dp.eval_transform(tensor)

    # Move to GPU (same device as the model weights)
    tensor = tensor.to(device)

    return raw_frame, tensor


def overlay_heatmap(raw_frame, cam_7x7, alpha=0.5):
    """
    Upsample the 7×7 CAM to the original image size and blend it
    with the original frame as a colour overlay.

    Parameters
    ----------
    raw_frame : np.ndarray [H, W, 3] uint8 RGB
    cam_7x7   : np.ndarray [7, 7]    float in [0, 1]
    alpha     : float — blend weight for the heatmap (0=invisible, 1=full)

    Returns
    -------
    overlay : np.ndarray [H, W, 3] uint8 BGR  (ready for cv2.imwrite)
    """
    H, W = raw_frame.shape[:2]

    # Upsample 7×7 → original resolution using bilinear interpolation.
    # The heatmap blurs naturally, which is fine — the regions are big enough.
    cam_full = cv2.resize(cam_7x7, (W, H), interpolation=cv2.INTER_LINEAR)

    # Scale to 0–255 so cv2.applyColorMap can colour it.
    cam_uint8 = (cam_full * 255).astype(np.uint8)

    # COLORMAP_JET: blue=cold (ignored), red=hot (focused on).
    # You can also try COLORMAP_INFERNO or COLORMAP_TURBO.
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Convert the original frame from RGB → BGR for OpenCV blending.
    frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

    # Blend: (1-alpha)*original + alpha*heatmap
    overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap_bgr, alpha, 0)

    return overlay


def make_side_by_side(frame_bgr, overlay_bgr):
    """
    Stack the original frame and the heatmap overlay side by side.
    Adds a small label to each half.
    """
    # Both images must be the same height for hstack to work.
    assert frame_bgr.shape == overlay_bgr.shape

    label_h = 30
    H, W = frame_bgr.shape[:2]

    # Create label bars
    left_label  = np.zeros((label_h, W, 3), dtype=np.uint8)
    right_label = np.zeros((label_h, W, 3), dtype=np.uint8)

    cv2.putText(left_label,  "Original",       (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    cv2.putText(right_label, "Grad-CAM",        (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    left_panel  = np.vstack([left_label,  frame_bgr])
    right_panel = np.vstack([right_label, overlay_bgr])

    return np.hstack([left_panel, right_panel])


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Main processing loop
# ═════════════════════════════════════════════════════════════════════════════

def load_agent_and_gradcam(ckpt_path, camera_idx):
    """
    Load the trained agent once and set up Grad-CAM hooks on the requested
    camera's image encoder. Returns (agent, gradcam, device, h5_camera_number)
    for reuse across as many trajectories as needed.
    """
    # BimanualDPAgent.__init__ reads args_log.txt from the checkpoint directory,
    # rebuilds the model with the exact same architecture used at training time,
    # then loads the EMA weights (ema_last.ckpt or ema_<basename>).
    print(f"Loading checkpoint from: {ckpt_path}")
    agent = BimanualDPAgent(ckpt_path=ckpt_path)
    print("Checkpoint loaded successfully.")

    # ── Grab the image encoder for the requested camera ───────────────────────
    # agent.dp          = the DPAgent (from pipeline.py)
    # agent.dp.policy   = the DiffusionPolicy object
    # .nets["img_encoder"] = ModuleList, one ImageEncoder per camera
    # [camera_idx]      = the encoder for the camera we want to inspect
    num_cameras = len(agent.dp.camera_indices)
    if camera_idx >= num_cameras:
        raise ValueError(
            f"camera_idx={camera_idx} but this checkpoint only has "
            f"{num_cameras} camera(s) (indices 0..{num_cameras-1})."
        )

    # agent.dp.camera_indices maps encoder position -> physical camera number
    # (e.g. camera_indices=[1,2] means encoder 0 looks at physical camera 1).
    # trajectory.h5 video streams are keyed by that physical camera number
    # ("base_camera_rgb_<physical_number>"), so we need this, not camera_idx
    # itself, to pick the right stream to decode.
    h5_camera_number = agent.dp.camera_indices[camera_idx]

    img_encoder = agent.dp.policy.nets["img_encoder"][camera_idx]
    print(f"Using image encoder for camera index: {camera_idx}")

    # ── Set up Grad-CAM ───────────────────────────────────────────────────────
    # This installs the hooks on layer4 of the chosen encoder.
    gradcam = GradCAM(img_encoder)

    device = agent.dp.device

    return agent, gradcam, device, h5_camera_number


def process_one_trajectory(
    agent,
    gradcam,
    device,
    h5_camera_number,
    frames_dir,
    output_dir,
    alpha,
    make_video,
    overwrite_frames,
):
    """
    Run Grad-CAM over every frame of a single trajectory (a folder of
    PNG/JPG frames, or a trajectory.h5 file/folder) and save results under
    output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Resolve the input frames ──────────────────────────────────────────────
    # frames_dir may be a folder of PNG/JPG frames, a trajectory.h5 file, or a
    # directory containing one — in the h5 cases the rollout video for this
    # camera is decoded into <output_dir>/extracted_frames/ first.
    frames_dir = resolve_frames_dir(
        frames_dir, output_dir, h5_camera_number, overwrite_frames
    )

    # ── Collect and sort all PNG frames in the input folder ───────────────────
    all_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(".png") or f.lower().endswith(".jpg")
    ])
    if len(all_files) == 0:
        raise FileNotFoundError(f"No PNG/JPG images found in: {frames_dir}")
    print(f"Found {len(all_files)} frames to process.")

    output_paths = []   # collect paths for optional video assembly

    for i, fname in enumerate(all_files):
        frame_path = os.path.join(frames_dir, fname)
        print(f"  [{i+1}/{len(all_files)}] Processing {fname} ...")

        # Load and preprocess the image
        raw_frame, frame_tensor = load_and_preprocess(
            frame_path, agent.dp, device
        )

        # Compute the Grad-CAM heatmap.
        # Returns a float32 numpy array of shape [7, 7] with values in [0, 1].
        cam = gradcam.generate(frame_tensor)

        # Upsample the 7×7 map to full image resolution and blend with original.
        overlay = overlay_heatmap(raw_frame, cam, alpha=alpha)
        frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

        # Create a side-by-side panel: original on the left, overlay on the right.
        panel = make_side_by_side(frame_bgr, overlay)

        # Save the panel image.
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"gradcam_{stem}.png")
        cv2.imwrite(out_path, panel)
        output_paths.append(out_path)

    print(f"\nSaved {len(output_paths)} Grad-CAM images to: {output_dir}")

    # ── Optional: stitch all frames into a video ──────────────────────────────
    if make_video and len(output_paths) > 0:
        sample = cv2.imread(output_paths[0])
        H, W = sample.shape[:2]

        video_path = os.path.join(output_dir, "gradcam_video.mp4")
        # mp4v codec, 10 fps — adjust fps to match your robot's control rate
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 10, (W, H))

        for path in output_paths:
            frame = cv2.imread(path)
            writer.write(frame)

        writer.release()
        print(f"Video saved to: {video_path}")


def run(
    ckpt_path,
    frames_dir,
    output_dir,
    camera_idx,
    alpha,
    make_video,
    overwrite_frames=False,
):
    """
    Main entry point: load the model once, discover one or more trajectory
    folders under frames_dir, and run Grad-CAM over each one.

    frames_dir may be:
      - a single trajectory (a trajectory.h5 file, or a folder containing
        trajectory.h5 / PNG-JPG frames directly), or
      - a parent folder containing multiple such trajectory subfolders
        (e.g. eval_wipe_board/dp_img_pos/), in which case every subfolder
        is processed in turn.

    For each trajectory named <traj_name>, results are written to
    <output_dir>/<traj_name>/camera_idx<camera_idx>/ so a single
    --output_dir can be reused across every run without collisions.
    """
    agent, gradcam, device, h5_camera_number = load_agent_and_gradcam(
        ckpt_path, camera_idx
    )

    trajectories = resolve_input_trajectories(frames_dir)
    if len(trajectories) > 1:
        print(f"Found {len(trajectories)} trajectories under {frames_dir}:")
        for traj_name, _ in trajectories:
            print(f"  - {traj_name}")

    for traj_name, traj_path in trajectories:
        traj_output_dir = os.path.join(
            output_dir, traj_name, f"camera_idx{camera_idx}"
        )
        print(f"\n=== Trajectory '{traj_name}' → {traj_output_dir} ===")
        process_one_trajectory(
            agent,
            gradcam,
            device,
            h5_camera_number,
            traj_path,
            traj_output_dir,
            alpha,
            make_video,
            overwrite_frames,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Command-line interface
# ═════════════════════════════════════════════════════════════════════════════

def boolean_string(s):
    # Helper so argparse accepts "True"/"False" strings (same as your pipeline.py)
    return s.lower() in ("true", "1", "yes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline Grad-CAM for the Diffusion Policy image encoder."
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to last.ckpt (the directory must also contain ema_last.ckpt "
             "and args_log.txt).",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="A single trajectory (folder of saved PNG/JPG frames, a "
             "trajectory.h5 rollout file, or a directory containing one -- "
             "its embedded video for --camera_idx is decoded to PNGs under "
             "<output_dir>/.../extracted_frames/ automatically), OR a "
             "parent folder containing several such trajectory subfolders "
             "(e.g. eval_wipe_board/dp_img_pos/), in which case every "
             "subfolder is processed automatically.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gradcam_output",
        help="Parent folder for results. Each trajectory's Grad-CAM images "
             "and video are written to "
             "<output_dir>/<trajectory_name>/camera_idx<camera_idx>/.",
    )
    parser.add_argument(
        "--camera_idx",
        type=int,
        default=0,
        help="Which camera encoder to visualise (0 = first camera).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Heatmap blend strength: 0.0 = invisible, 1.0 = full heatmap.",
    )
    parser.add_argument(
        "--make_video",
        type=boolean_string,
        default=True,
        help="Whether to stitch all frames into a gradcam_video.mp4.",
    )
    parser.add_argument(
        "--overwrite_frames",
        type=boolean_string,
        default=False,
        help="If --frames_dir points at a trajectory.h5 (or a directory "
             "containing one) and extracted_frames/ already has PNGs from "
             "a previous run, re-decode the video anyway instead of "
             "reusing them.",
    )

    args = parser.parse_args()

    run(
        ckpt_path=args.ckpt_path,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        camera_idx=args.camera_idx,
        alpha=args.alpha,
        make_video=args.make_video,
        overwrite_frames=args.overwrite_frames,
    )

# Raw tactile image pipeline (`tactile_feature_mode=raw_image`) — implementation notes

This document explains every change made to add a `tactile_feature_mode=raw_image` mode to the
`pi0_ur5e` data pipeline. Unlike the existing `image_embedding` mode (which squashes tactile RGB
frames through a fixed, untrained random projection and concatenates the result onto
`observation.state`), `raw_image` mode keeps `observation.state` at its plain 7-D shape and
instead writes raw tactile pixels as their own LeRobot dataset columns
(`observation.images.tactile_left_rgb` / `observation.images.tactile_right_rgb`), so a downstream
model can learn its own tactile encoder with real gradients. It's Part 1 of a larger effort — see
`openpi_patches_pytorch/docs/ftp1_tactile_expert_port.md` for the model (Part 2) that actually
consumes these raw frames.

Supersedes `docs/temporal_tactile_tokens.md` (an earlier, unimplemented design that claimed the
same `Episode` fields for a different purpose — see the banner at the top of that file).

## Data flow

```
raw tactile_left_rgb/tactile_right_rgb (h5/pkl/npz/jsonl/LeRobot source)
  -> DatasetReader._tactile_raw_from_frames() / _tactile_raw_from_arrays()   [NEW]
  -> Episode.tactile_left_rgb / Episode.tactile_right_rgb (ndarray [T,H,W,3] uint8)  [NEW fields]
  -> lerobot_writer.write_lerobot_dataset():
       observation.images.tactile_left_rgb / tactile_right_rgb  [NEW LeRobot image keys]
       observation.state stays 7-D (no concatenation)
  -> agents/pi0_agent.py Pi0Agent._policy_observation():
       forwards resized raw tactile frames as tactile_left_rgb/tactile_right_rgb  [NEW]
  -> policy_client.build_policy_observation():
       passes them through to the websocket payload unchanged  [NEW kwargs]
```

The websocket payload reaching the OpenPI-served policy is consumed by
`TeleGsyUR5eInputs.__call__` (Part 2, `openpi_patches/pi0_ur5e_cup_config.py`) — Part 1 alone does
not make the served model *use* tactile; it only makes the raw pixels available end-to-end.

## `learning/pi0_ur5e/pi0_ur5e/schema.py`

- `Episode` gains two new optional fields, `tactile_left_rgb`/`tactile_right_rgb: np.ndarray | None
  = None`, inserted right after the existing `tactile` field (before `language_instruction`).
  `Episode` is constructed positionally at every call site, so the insertion position matters —
  it's placed last among the optional fields with defaults so every existing positional call
  remains valid.
- `Episode.validate()` gets two new length checks, following the exact same pattern as the
  existing `tactile` check: if the field is present, its length must match `timestamps`.

## `learning/pi0_ur5e/pi0_ur5e/dataset_reader.py`

- `_video_keys_to_decode()`: the `tactile_feature_mode == "image_embedding"` check that decides
  whether `tactile_left_rgb`/`tactile_right_rgb` need video decoding is widened to
  `in ("image_embedding", "raw_image")`, since raw_image mode also needs those source keys
  decoded.
- `_tactile_raw_from_arrays(data)` / `_tactile_raw_from_frames(frames)` (new): return
  `(left, right)` raw stacked arrays (or path lists) for `raw_image` mode, reusing the existing
  `_stack_or_paths` helper — the same helper `image_embedding` mode already uses to collect
  per-frame tactile images before pooling them.
- `_tactile_from_arrays`/`_tactile_from_frames` (existing, embedding path): each gets an explicit
  early `if tactile_feature_mode == "raw_image": return None` guard. Without this, raw_image mode
  would fall through to returning a raw numeric `"tactile"` series into `episode.tactile` (if one
  happens to exist in the source data), which would wrongly re-trigger the
  `state = concatenate([state, episode.tactile])` path in `lerobot_writer.py`. The guard keeps
  `episode.tactile` strictly `None` in raw_image mode, so state stays 7-D.
- `_episode_from_frames`/`_episode_from_npz`: both now also call the new `_tactile_raw_from_*`
  helpers and pass the results into `Episode(...)` at the new positional slot.
- `_read_lerobot_jsonl`/`_read_lerobot_v2` were already populating `tactile_left_rgb`/
  `tactile_right_rgb` per-frame dict keys for the `image_embedding` path — no changes needed, the
  new raw-image reader functions pick those same per-frame keys up automatically.

## `learning/pi0_ur5e/pi0_ur5e/lerobot_writer.py`

- `has_raw_tactile` (new local): `True` when `include_tactile` and any episode has
  `tactile_left_rgb`/`tactile_right_rgb` set.
- `features={...}` dict passed to `LeRobotDataset.create()`: when `has_raw_tactile`, two new
  entries are added — `observation.images.tactile_left_rgb`/`tactile_right_rgb`, `dtype="image"`,
  same `image_shape` as `base_rgb`/`wrist_rgb`.
- Per-frame `add_frame({...})` call: when `has_raw_tactile`, the frame dict gains the two tactile
  image keys, populated via the existing generic `_image_from_episode` helper (handles both
  in-memory arrays and path-list sources, then resizes) — the exact same call pattern already
  used for `base_rgb`/`wrist_rgb`.
- `report["image_shape"]`: gains the two tactile keys when `has_raw_tactile`, for parity with the
  camera image keys (useful for the `meta/info.json`-style sanity check below).
- The `state = np.concatenate([state, episode.tactile...])` line is unchanged — it already only
  fires when `episode.tactile is not None`, which the dataset_reader.py guard above keeps `False`
  for raw_image mode.

## `learning/pi0_ur5e/configs/dataset_schema.yaml`

- `tactile_feature_mode` documented as accepting `raw_image` (a comment; the field itself is a
  plain `str`, no schema change needed). Field-map aliases for `tactile_left_rgb`/
  `tactile_right_rgb` already existed.

## `learning/pi0_ur5e/scripts/convert_to_lerobot.py` and `scripts/train_pi0_base.py`

- Both scripts maintain their own separate `--tactile-feature-mode` `choices=[...]` list;
  `"raw_image"` was added to both (the original design doc only mentioned the converter script).

## `agents/pi0_agent.py`

- `Pi0Agent.tactile_encoder` is unchanged — it's still only constructed for `image_embedding`
  mode, so `_state()` already leaves `observation.state` at 7-D for `raw_image` mode with no
  changes needed there.
- `_tactile_raw_images(obs)` (new): looks up `tactile_left_rgb`/`tactile_right_rgb` (or their
  `left_tactile_rgb`/`right_tactile_rgb` aliases) in the live observation dict, resizes each via
  the existing `_resize_image` helper (same resize path used for base/wrist cameras), and returns
  `(left, right)` or `None` for missing frames.
- `_policy_observation()`: when `include_tactile and tactile_feature_mode == "raw_image"`, calls
  `_tactile_raw_images` and forwards the results into `build_policy_observation(...)`.

## `learning/pi0_ur5e/pi0_ur5e/policy_client.py`

- `build_policy_observation()` gains optional `tactile_left_rgb`/`tactile_right_rgb: np.ndarray |
  None = None` keyword args, included in the returned observation dict only when not `None` — the
  websocket transport itself already supports arbitrary ndarray-valued keys, so no protocol change
  was needed.

## Verification performed

Convert a small sample dataset with `--tactile-feature-mode raw_image` and inspect
`meta/info.json`'s `features` dict for `observation.images.tactile_left_rgb`/
`tactile_right_rgb` with the expected `[H, W, 3]` shape, and confirm
`observation.state`'s shape is `[7]`. See `README.md`'s "Option C: Convert With Raw Tactile
Images" section for the exact command and expected output.

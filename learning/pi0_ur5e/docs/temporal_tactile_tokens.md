# Temporal tactile tokens for pi0 — implementation notes

This document walks through every change made to add a `tactile_feature_mode=temporal_tokens`
mode: instead of pooling tactile sensor frames into one flat embedding vector that gets
concatenated onto `observation.state`, a window of raw tactile RGB frames is encoded by a
dedicated ViT into multiple tokens and injected directly into pi0's suffix sequence, next to
the state and action tokens.

Two repositories are involved:

- **`tele-amir`** (this repo) — paths below are relative to the repo root
  (`/home/amirhosein/Projects/ur5_tele/tele-amir`).
- **`openpi`** — a sibling checkout at `/home/amirhosein/Projects/ur5_tele/openpi`. Paths below
  are relative to *that* repo's root, and are edited directly (there is no patch-injection
  mechanism for model files the way `install_openpi_config.py` patches `training/config.py`).

The feature is entirely opt-in: every new field defaults to "off" (`tactile_feature_mode="none"`,
`tactile_window_keys=()`), so nothing about this change alters behavior for existing configs,
datasets, or checkpoints unless a caller explicitly asks for `temporal_tokens`.

## Data flow, top to bottom

Reading top to bottom mirrors how a tactile frame actually travels through the system:

```
raw per-frame tactile RGB (left/right fingers)
  -> DatasetReader (tele-amir)              reads + resizes raw frames, no pooling
  -> Episode.tactile_left_rgb/right_rgb      new fields carrying [T, H, W, 3] arrays
  -> write_lerobot_dataset (tele-amir)       bakes a padded trailing window per row,
                                              writes it as a plain uint8 array feature
  -> LeRobot dataset on disk                 observation.tactile_left_rgb_window / _right_
  -> RepackTransform + TeleGsyUR5eInputs     (openpi_patches, tele-amir) parses the window
                                              into Observation.tactile_images / _masks
  -> Pi0._embed_tactile (openpi)             dedicated ViT -> per-frame patch tokens
                                              + learned time/stream position embeddings
  -> Pi0.embed_suffix (openpi)               tactile tokens inserted into the suffix,
                                              attention-mask blocks updated
  -> Gemma action-expert forward pass        unchanged
```

The sections below follow this same order: tele-amir's conversion pipeline first, then the
openpi model side, then the tests that verify each stage.

---

## 1. `learning/pi0_ur5e/pi0_ur5e/schema.py`

**Purpose.** Extends the two dataclasses that describe an episode and the reader's
configuration so a raw (unpooled) per-frame tactile stream and its windowing parameters can be
carried through the pipeline, without touching the existing pooled `tactile` field used by the
`low_dim`/`image_embedding` modes.

### `Episode` — lines 25–45

```python
25  gripper_state: np.ndarray | None = None
26  tactile: np.ndarray | None = None
27  tactile_left_rgb: np.ndarray | None = None
28  tactile_right_rgb: np.ndarray | None = None
29  language_instruction: str = "pick up the paper cup and place it on the target"
30  metadata: dict[str, Any] = field(default_factory=dict)
31
32  def validate(self) -> None:
...
42      if self.tactile_left_rgb is not None and len(self.tactile_left_rgb) != length:
43          raise ValueError(f"{self.episode_id}: tactile_left_rgb length does not match timestamps")
44      if self.tactile_right_rgb is not None and len(self.tactile_right_rgb) != length:
45          raise ValueError(f"{self.episode_id}: tactile_right_rgb length does not match timestamps")
```

- **Lines 27–28**: two new optional fields, `tactile_left_rgb`/`tactile_right_rgb`. Each holds a
  `[T, H, W, 3]` `uint8` array — the raw per-frame tactile camera images for one finger, one
  entry per timestep in the episode. They are placed immediately after the existing `tactile`
  field (line 26) and before `language_instruction`/`metadata`, which matters because `Episode`
  is a plain (non-keyword) dataclass and every call site in `dataset_reader.py` constructs it
  positionally — the new fields had to land in this exact slot.
- **Lines 42–43 / 44–45**: mirror the existing length check pattern used for `tactile` a few
  lines above (line 40–41) — if a caller supplies one of the new arrays, its length must match
  `len(self.timestamps)`, same as every other per-timestep field.

### `Pi0Ur5eConfig` — lines 56–57

```python
56  tactile_window_size: int = 4
57  tactile_image_size: tuple[int, int] = (32, 32)
```

- **Line 56**: how many trailing frames go into each tactile window (defaults to 4).
- **Line 57**: the `(width, height)` each tactile frame is resized to before being stored —
  kept small and independent of the main camera `image_size` (line 51, `(224, 224)`) since
  tactile sensor images don't need scene-camera resolution and a smaller ViT input keeps the
  dedicated tactile tower cheap.

---

## 2. `learning/pi0_ur5e/pi0_ur5e/dataset_reader.py`

**Purpose.** `DatasetReader` already had two code paths for tactile: pool into a fixed vector
(`image_embedding`) or pass through low-dimensional numeric features (`low_dim`). This adds a
third path that keeps the *raw* per-frame RGB images (no pooling) whenever
`tactile_feature_mode == "temporal_tokens"`, reusing the same frame-loading and
precontact-baseline-gating helpers the `image_embedding` path already uses.

### Imports — lines 13, 15

```python
13  from .io_utils import decode_h5_video, load_pickle, load_yaml, parse_timestamp, resize_rgb
15  from .tactile_features import _load_image_if_needed, tactile_images_to_embeddings, tactile_to_features
```

- **Line 13**: adds `resize_rgb` (single-image resize, already used elsewhere in this file for
  base/wrist cameras) — needed to resize each tactile frame individually.
- **Line 15**: adds `_load_image_if_needed`, a private helper from `tactile_features.py` that
  turns a raw array *or* an image file path into an `np.ndarray`. Frames can arrive either way
  (`_stack_or_paths`, used below, returns a list of path strings when it can't stack them into
  one array), so this helper is reused rather than reimplemented.

### `DatasetReader.__init__` — lines 55–56

```python
55  tactile_window_size=int(cfg.get("tactile_window_size", 4)),
56  tactile_image_size=tuple(cfg.get("tactile_image_size", [32, 32])),
```

Threads the two new YAML/dict config keys (added in `schema.py` §1) into the
`Pi0Ur5eConfig` the reader builds, with the same defaults as the dataclass.

### `_episode_from_npz` — lines 178–195, and `_episode_from_frames` — lines 214–239

```python
178  tactile = self._tactile_from_arrays(data)
179  tactile_left_rgb, tactile_right_rgb = self._tactile_rgb_from_arrays(data)
180  ep = Episode(
...
189      tactile_left_rgb,
190      tactile_right_rgb,
191      prompt,
192      {"source_path": str(path)},
193  )
```

```python
214  tactile = self._tactile_from_frames(frames, metadata)
215  tactile_left_rgb, tactile_right_rgb = self._tactile_rgb_from_frames(frames, metadata)
...
224  ep = Episode(
...
233      tactile_left_rgb,
234      tactile_right_rgb,
235      str(prompt),
236      meta,
237  )
```

Both are the two places an `Episode` gets constructed from raw source data (`.npz` files, and
everything else — pickled frame dirs, `.h5` trajectories, LeRobot v2/JSONL datasets — which all
funnel through `_episode_from_frames`). In each:

- The **existing** call to `_tactile_from_arrays`/`_tactile_from_frames` (line 178 / 214) is
  untouched — it still populates the pooled `tactile` field for `low_dim`/`image_embedding`
  modes, and returns `None` for `temporal_tokens` (see §"`_tactile_from_frames`" below — its
  `tactile_feature_mode == "image_embedding"` / `"low_dim"` branches simply don't match
  `"temporal_tokens"`, so it falls through to returning the raw numeric `"tactile"` series,
  which for a temporal-tokens dataset is normally absent → `None`).
- The **new** call (line 179 / 215) to `_tactile_rgb_from_arrays`/`_tactile_rgb_from_frames`
  (defined below) returns a `(left, right)` tuple, unpacked into two locals.
- Both locals are threaded into the `Episode(...)` positional constructor call in the exact slot
  `schema.py` reserved for them (right after the pooled `tactile` argument).

### `_video_keys_to_decode` — line 321

```python
319  if self.config.include_tactile:
320      tactile_fields = ["tactile"]
321      if self.config.tactile_feature_mode in ("image_embedding", "temporal_tokens"):
322          tactile_fields = ["tactile_left_rgb", "tactile_right_rgb"]
```

This method tells the `.h5` reader (`_frames_from_h5`) which embedded video streams to actually
decode (decoding every stream in a large `.h5` file is wasteful when most aren't needed). Before
this change, only `image_embedding` requested the raw `tactile_left_rgb`/`tactile_right_rgb`
streams; `temporal_tokens` needs the exact same raw streams (just kept unpooled downstream), so
it's added to the same condition on line 321.

### `_tactile_rgb_from_frames` — lines 369–383, `_tactile_rgb_from_arrays` — lines 385–393

```python
369  def _tactile_rgb_from_frames(
370      self, frames: list[dict[str, Any]], metadata: dict[str, Any] | None = None
371  ) -> tuple[np.ndarray | None, np.ndarray | None]:
372      if not self.config.include_tactile or self.config.tactile_feature_mode != "temporal_tokens":
373          return None, None
374      left = self._stack_or_paths(frames, "tactile_left_rgb")
375      right = self._stack_or_paths(frames, "tactile_right_rgb")
376      gate = self._numeric_series(frames, "contact_gate")
377      if gate is not None and not _tactile_video_already_gated(metadata):
378          left = _replace_tactile_precontact_with_baseline(left, gate)
379          right = _replace_tactile_precontact_with_baseline(right, gate)
380      return (
381          _resize_rgb_stack(left, self.config.tactile_image_size),
382          _resize_rgb_stack(right, self.config.tactile_image_size),
383      )
```

- **Line 372**: early-exit guard — this method only does anything when both `include_tactile`
  is set *and* the mode is exactly `"temporal_tokens"`; every other mode gets `(None, None)`,
  which is why `Episode.tactile_left_rgb`/`tactile_right_rgb` stay `None` for `image_embedding`
  and `low_dim` datasets (verified by `test_dataset_reader_skips_tactile_rgb_for_other_modes`,
  see §7).
- **Lines 374–375**: `_stack_or_paths` (a pre-existing helper, unchanged) collects the
  `tactile_left_rgb`/`tactile_right_rgb` value out of every frame dict and either stacks them
  into one `ndarray` or, if that fails (e.g. mixed shapes or path strings), returns a plain list.
- **Lines 376–379**: reuses the exact same "precontact baseline" gating the `image_embedding`
  path already applies (`_numeric_series`/`_tactile_video_already_gated`/
  `_replace_tactile_precontact_with_baseline` are all pre-existing, unmodified functions) — if a
  `contact_gate` signal is present and the video wasn't already pre-gated upstream, frames before
  first contact get replaced with a per-episode baseline. Keeping this identical to the
  `image_embedding` path means both tactile modes see the same preprocessing.
- **Lines 380–383**: resizes every frame in `left`/`right` down to `tactile_image_size` via the
  new `_resize_rgb_stack` helper (§ below) and returns the pair.

```python
385  def _tactile_rgb_from_arrays(self, data: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
386      if not self.config.include_tactile or self.config.tactile_feature_mode != "temporal_tokens":
387          return None, None
388      left = data["tactile_left_rgb"] if "tactile_left_rgb" in data else None
389      right = data["tactile_right_rgb"] if "tactile_right_rgb" in data else None
390      return (
391          _resize_rgb_stack(left, self.config.tactile_image_size),
392          _resize_rgb_stack(right, self.config.tactile_image_size),
393      )
```

The `.npz`-source counterpart of the function above — same guard, same resize call, but reads
directly from the `data` mapping (an `np.load` result) instead of a list of per-frame dicts, and
skips the contact-gate step since `.npz` episodes in this codebase don't carry that signal.

### `_resize_rgb_stack` — lines 505–514

```python
505  def _resize_rgb_stack(value: Any, size: tuple[int, int]) -> np.ndarray | None:
506      if value is None:
507          return None
508      length = len(value)
509      if length == 0:
510          return None
511      out = np.empty((length, size[1], size[0], 3), dtype=np.uint8)
512      for i in range(length):
513          out[i] = resize_rgb(_load_image_if_needed(value[i]), size)
514      return out
```

- **Lines 506–507 / 508–510**: `None`/empty input passes through as `None` — used by both
  callers above whenever a stream is entirely missing.
- **Line 511**: pre-allocates the output array. `size` is `(width, height)` (matching the
  existing `resize_rgb`/cv2 convention used throughout this file), so the array shape is
  `(length, size[1], size[0], 3)` = `(T, height, width, 3)`.
- **Lines 512–513**: for every frame, `_load_image_if_needed` first normalizes the input (an
  in-memory array, or a file path string) into an `ndarray`, then `resize_rgb` resizes it to the
  target `(width, height)`. This is what lets `left`/`right` be either an already-stacked
  `ndarray` or a list of path strings — both are handled uniformly per-element.

---

## 3. `learning/pi0_ur5e/pi0_ur5e/lerobot_writer.py`

**Purpose.** Writes the raw per-frame tactile arrays produced by `dataset_reader.py` into the
actual LeRobot dataset on disk as a **pre-computed trailing window** per row (the "precompute at
conversion time" design chosen instead of relying on openpi's `delta_timestamps` mechanism), so
no changes are needed to openpi's core data-loading code.

### New parameters — lines 20–22

```python
19  include_tactile: bool = False,
20  tactile_feature_mode: str = "none",
21  tactile_window_size: int = 4,
22  tactile_image_size: tuple[int, int] = (32, 32),
```

Three new keyword-only parameters (mirroring the fields threaded through in §1/§2), all
defaulted so every existing call site keeps working unchanged.

### Enable flag and accumulators — lines 44–47

```python
44  tactile_windows_enabled = include_tactile and tactile_feature_mode == "temporal_tokens"
45  tactile_left_windows_by_episode: list[np.ndarray] = []
46  tactile_right_windows_by_episode: list[np.ndarray] = []
47  tactile_window_shape = None
```

- **Line 44**: single boolean computed once, used everywhere below to gate all the new
  behavior — this is the one place that decides whether this function does anything different
  from before.
- **Lines 45–47**: per-episode accumulators for the windowed arrays (parallel to the
  pre-existing `states_by_episode`/`actions_by_episode` lists a few lines down), plus a shape
  placeholder for the conversion report.

### Per-episode window construction — lines 65–75

```python
65  if tactile_windows_enabled:
66      if episode.tactile_left_rgb is None or episode.tactile_right_rgb is None:
67          raise ValueError(
68              f"{episode.episode_id}: tactile_feature_mode=temporal_tokens requires "
69              "tactile_left_rgb/tactile_right_rgb frames"
70          )
71      left_windows = _build_tactile_windows(episode.tactile_left_rgb, tactile_window_size)
72      right_windows = _build_tactile_windows(episode.tactile_right_rgb, tactile_window_size)
73      tactile_left_windows_by_episode.append(left_windows)
74      tactile_right_windows_by_episode.append(right_windows)
75      tactile_window_shape = list(left_windows.shape[1:])
```

Inside the existing per-episode loop (the loop also builds `state`/`action` arrays, unchanged):

- **Lines 66–70**: fail loudly if the mode is enabled but the dataset reader didn't actually
  produce raw tactile frames for this episode (e.g. `include_tactile=False` was passed to the
  reader while `True` was passed here) — this is a configuration-mismatch error, not something
  that can legitimately happen from valid input, so it's a hard `raise` rather than a silent
  skip.
- **Lines 71–72**: `_build_tactile_windows` (defined below) turns each episode's flat
  `[T, H, W, 3]` array into a windowed `[T, window_size, H, W, 3]` array — one full trailing
  window per timestep.
- **Lines 73–75**: stash the per-episode windows for use in the `add_frame` loop below, and
  record the per-frame window shape (`[window_size, H, W, 3]`) for the conversion report.

### Feature schema — lines 105–112

```python
105  if tactile_windows_enabled:
106      window_shape = (tactile_window_size, tactile_image_size[1], tactile_image_size[0], 3)
107      for key in ("observation.tactile_left_rgb_window", "observation.tactile_right_rgb_window"):
108          features[key] = {
109              "dtype": "uint8",
110              "shape": window_shape,
111              "names": ["window", "height", "width", "channel"],
112          }
```

Registers the two new LeRobot dataset feature columns.

- **Line 106**: shape is `(window, height, width, channel)` — a 4-D per-row array.
- **Line 107**: the key names — deliberately `observation.tactile_left_rgb_window`, **not**
  `observation.images.tactile_left_rgb_window`. This was a real bug caught while testing: LeRobot's
  stats aggregation (`compute_stats.py`) decides whether a feature is image-like purely by
  checking whether the substring `"image"` appears in the *key name*, regardless of the
  declared `dtype`. Since this feature is deliberately `dtype="uint8"` (a plain array, not
  `"image"`/`"video"` — it doesn't need LeRobot's image/video codec pipeline for a tiny 4-D
  side-channel array), putting it under an `observation.images.*` key made the downstream stats
  validator (`_assert_type_and_shape`) demand a `(3,1,1)`-shaped reduction that this dtype never
  produces, and dataset creation failed. Renaming to drop the `images.` segment fixed it (see
  §"`pi0_ur5e_cup_config.py` — repack" for the matching change on the reader side).
- **Line 109**: `dtype="uint8"` — keeps the on-disk representation compact (vs. `float32`) and
  matches how the main camera images are stored before normalization.

### Per-frame writes — lines 140–142

```python
140  if tactile_windows_enabled:
141      frame["observation.tactile_left_rgb_window"] = tactile_left_windows_by_episode[ep_i][t]
142      frame["observation.tactile_right_rgb_window"] = tactile_right_windows_by_episode[ep_i][t]
```

Inside the existing `for t in range(len(episode.timestamps))` write loop (note line 123 changed
from a plain `for episode, state in zip(...)` to `for ep_i, (episode, state) in enumerate(zip(...))`
purely to get an index into the two accumulator lists above) — for each timestep `t`, pulls that
timestep's pre-built window out of the per-episode array and adds it to the frame dict passed to
`dataset.add_frame(...)`.

### Report field — line 160

```python
160  "tactile_window_shape": tactile_window_shape,
```

Surfaces the per-frame window shape in `conversion_report.json`, next to the pre-existing
`tactile_shape` field (line 159) used by the pooled modes.

### `_build_tactile_windows` — lines 177–190

```python
177  def _build_tactile_windows(frames: np.ndarray, window_size: int) -> np.ndarray:
178      """Stack a trailing window of `window_size` frames per timestep.
179
180      Pads the start of the episode by repeating frame 0, so windows never leak across
181      episode boundaries and every timestep gets a full-length window.
182      """
183      length = len(frames)
184      padded = np.empty((length + window_size - 1, *frames.shape[1:]), dtype=frames.dtype)
185      padded[: window_size - 1] = frames[0]
186      padded[window_size - 1 :] = frames
187      windows = np.empty((length, window_size, *frames.shape[1:]), dtype=frames.dtype)
188      for t in range(length):
189          windows[t] = padded[t : t + window_size]
190      return windows
```

- **Line 183**: `length` = number of timesteps in the episode (`T`).
- **Lines 184–186**: builds a `padded` array of length `T + window_size - 1` by repeating
  `frames[0]` for the first `window_size - 1` slots (line 185) and then copying the real
  frames after that (line 186). This is what guarantees every timestep — including `t=0` — gets
  a full window, and that the padding never reaches into a neighboring episode (since this
  function only ever sees one episode's frames at a time, called once per episode in §"Per-episode
  window construction" above).
- **Lines 187–189**: for each real timestep `t`, slices a `window_size`-length window out of
  `padded` starting at `padded[t]` — because of the `window_size - 1` left-padding, this slice
  is exactly `[frame max(0, t-window_size+1), ..., frame t]` in the original (unpadded)
  numbering, i.e. a trailing window ending at the current frame.
- **Line 190**: returns the full `[T, window_size, H, W, C]` array. Verified directly in
  `test_fake_raw_dataset_converts_with_temporal_tactile_windows` (§7): at `t=0` both window
  slots are identical (all padding), and at `t=1` the two slots hold `frame 0` then `frame 1`.

---

## 4. `learning/pi0_ur5e/scripts/convert_to_lerobot.py`

**Purpose.** Exposes the new config knobs as CLI flags and threads them through to both
`DatasetReader` and `write_lerobot_dataset`.

### New flags — lines 38, 40–41

```python
38  parser.add_argument("--tactile-feature-mode", default=None, choices=["none", "low_dim", "image_embedding", "temporal_tokens"])
40  parser.add_argument("--tactile-window-size", default=None, type=int)
41  parser.add_argument("--tactile-image-size", default=None, type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
```

- **Line 38**: the pre-existing `--tactile-feature-mode` flag's `choices` list gains
  `"temporal_tokens"` as a fourth option.
- **Line 40**: new `--tactile-window-size` integer flag.
- **Line 41**: new `--tactile-image-size` flag taking two ints (`WIDTH HEIGHT`), matching the
  `(width, height)` tuple convention `tactile_image_size` uses everywhere else.

### Threading into the reader config — lines 58–61

```python
58  if args.tactile_window_size is not None:
59      config["tactile_window_size"] = args.tactile_window_size
60  if args.tactile_image_size is not None:
61      config["tactile_image_size"] = tuple(args.tactile_image_size)
```

Same pattern as the pre-existing `--tactile-embedding-dim` handling immediately above it: only
overrides the dict key if the flag was actually passed, so unset flags fall through to
`Pi0Ur5eConfig`'s dataclass defaults.

### Threading into `write_lerobot_dataset` — lines 85–87

```python
83  image_size=reader.config.image_size,
84  include_tactile=include_tactile,
85  tactile_feature_mode=reader.config.tactile_feature_mode,
86  tactile_window_size=reader.config.tactile_window_size,
87  tactile_image_size=reader.config.tactile_image_size,
```

Reads the *resolved* config back off `reader.config` (which has already merged CLI overrides
with YAML/dataclass defaults) rather than re-reading `args` directly, so the writer always sees
the same values the reader used.

---

## 5. `learning/pi0_ur5e/openpi_patches/pi0_ur5e_cup_config.py`

**Purpose.** This file is appended into openpi's `src/openpi/training/config.py` by
`install_openpi_config.py` — it defines the `pi0_ur5e_cup` `TrainConfig` and the data
transforms that turn a raw observation dict into the `dict` shape `Observation.from_dict`
expects. This section adds: (a) parsing the LeRobot dataset's windowed tactile columns into the
`tactile_image`/`tactile_image_mask` keys `Observation.from_dict` (openpi side, §6) knows how
to read, (b) making the freeze filter aware of the new tactile module's parameter paths, and
(c) new environment-variable knobs, following the exact pattern every other tunable in this
file already uses.

### `TeleGsyUR5eInputs` — new field, lines 96–98

```python
96  # Short (post-repack) keys for windowed tactile RGB arrays, e.g.
97  # ("tactile_left_rgb_window", "tactile_right_rgb_window"). Empty tuple disables the feature.
98  tactile_window_keys: tuple = ()
```

A new dataclass field on the data-transform itself: the list of keys (already renamed by the
repack step, §"`TeleGsyLeRobotUR5eDataConfig.create`" below) this transform should look for and
forward as tactile tokens. Empty by default.

### `TeleGsyUR5eInputs.__call__` — lines 139–152

```python
139  if self.tactile_window_keys:
140      tactile_image = {}
141      tactile_image_mask = {}
142      for key in self.tactile_window_keys:
143          if key not in data:
144              continue
145          window = _tele_gsy_np.asarray(data[key])
146          if window.dtype != _tele_gsy_np.uint8:
147              window = _tele_gsy_np.clip(window, 0, 255).astype(_tele_gsy_np.uint8)
148          tactile_image[key] = window
149          tactile_image_mask[key] = _tele_gsy_np.True_
150      if tactile_image:
151          inputs["tactile_image"] = tactile_image
152          inputs["tactile_image_mask"] = tactile_image_mask
153  return inputs
```

Appended right before the pre-existing `return inputs` (this transform already builds `state`,
`image`, `image_mask`, and optionally `actions`/`prompt` above this block, all unchanged):

- **Line 139**: no-op entirely when `tactile_window_keys` is empty — the default, disabled
  state.
- **Lines 142–144**: iterates the configured key list; `continue`s past any key not present in
  the raw `data` dict for this particular row (defensive — in normal operation every configured
  key is always present, since the repack step below requires it).
- **Lines 145–147**: normalizes to a `uint8` array. The LeRobot column is already written as
  `uint8` by `lerobot_writer.py` (§3), so this is mostly a type-safety net (e.g. against the
  value arriving as some other integer/float dtype from a different loader path) rather than an
  expected code path.
- **Line 149**: `_tele_gsy_np.True_` — a scalar `True`, broadcast later; this transform doesn't
  currently have a way to signal "this sensor was missing for this row" (that would require an
  upstream per-row validity signal the dataset doesn't carry), so every present key is always
  marked valid. This mirrors how the base/wrist camera masks are set to `True` on the equivalent
  lines above (127–130) when there's no padding-strategy reason to mask them out.
- **Lines 150–152**: only attaches the two new top-level keys, `tactile_image`/
  `tactile_image_mask`, if at least one stream was actually found — these are exactly the key
  names `Observation.from_dict` (openpi side, §6) reads via `data.get("tactile_image")` /
  `data.get("tactile_image_mask")`.

### `TeleGsyLeRobotUR5eDataConfig` — new field, lines 172–174

```python
172  # Short (post-repack) keys for windowed tactile RGB arrays. Empty tuple (default) disables
173  # the feature and leaves repack/data transforms identical to today.
174  tactile_window_keys: tuple = ()
```

Same shape of field as on `TeleGsyUR5eInputs` above, but on the `DataConfigFactory` — this is
the value that actually gets set from the environment (§"module-level env knobs" below) and
passed down into `TeleGsyUR5eInputs` at construction time.

### `TeleGsyLeRobotUR5eDataConfig.create` — lines 177–190

```python
177  def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
178      repack_structure = {
179          "base_rgb": "observation.images.base_rgb",
180          "wrist_rgb": "observation.images.wrist_rgb",
181          "state": "observation.state",
182          "actions": "action",
183          "prompt": "task",
184      }
185      for key in self.tactile_window_keys:
186          # Not under observation.images.* -- these are plain uint8 array features (not
187          # dtype="image"/"video"), and LeRobot's stats aggregation assumes any key containing
188          # "image" is that dtype (see compute_stats.py's `"image" in fkey` check).
189          repack_structure[key] = f"observation.{key}"
190      repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])
```

Before this change, the repack dict (originally passed straight into `RepackTransform(...)`)
was a literal fixed at lines 178–183. Now it's built into a local `repack_structure` variable
first:

- **Lines 178–184**: identical five entries to before, unchanged.
- **Lines 185–189**: for each configured tactile key (e.g. `"tactile_left_rgb_window"`), adds
  one more repack entry mapping the *short* key `TeleGsyUR5eInputs` reads (§ above) to the
  *actual* LeRobot column name written by `lerobot_writer.py`, `observation.{key}` — matching
  the `observation.tactile_left_rgb_window` naming from §3, deliberately without an
  `images.` segment. This loop only runs at all when `tactile_window_keys` is non-empty, so a
  dataset/config with tactile disabled gets the exact same `repack_structure` as before this
  change, byte for byte.
- **Line 190**: `RepackTransform` does a hard dictionary lookup per configured key
  (`openpi/src/openpi/transforms.py`, `flat_item[k]`) — it would raise `KeyError` if a key in
  `repack_structure` isn't present in the raw LeRobot row. This is exactly why the tactile
  entries are only added conditionally: a dataset that was never converted with
  `tactile_feature_mode=temporal_tokens` has no `observation.tactile_left_rgb_window` column at
  all, so unconditionally repacking it would break every existing dataset/config.

### `TeleGsyUR5eInputs(...)` construction — line 196

```python
193  TeleGsyUR5eInputs(
194      model_type=model_config.model_type,
195      camera_padding_strategy=self.camera_padding_strategy,
196      tactile_window_keys=self.tactile_window_keys,
197  )
```

Passes the `DataConfig`-level `tactile_window_keys` down into the data-transform instance
constructed for this config.

### `_tele_gsy_freeze_filter` — lines 62, 72–89

```python
62  _TELE_GSY_PI0_UR5E_TACTILE_PARAM_REGEX = "tactile_img|tactile_proj|tactile_time_pos_emb|tactile_stream_pos_emb"
```

A single shared regex alternation naming every new nnx module attribute the tactile tower adds
on the openpi side (§8: `self.tactile_img`, `self.tactile_proj`, `self.tactile_time_pos_emb`,
`self.tactile_stream_pos_emb`) — defined once here and interpolated into all three freeze-filter
branches below so they can't drift out of sync.

```python
65  def _tele_gsy_freeze_filter(model_config):
66      freeze_mode = _tele_gsy_os.environ.get("PI0_UR5E_FREEZE_MODE", "default")
67      if freeze_mode == "vision_action_head":
...
72          trainable = _tele_gsy_nnx_utils.PathRegex(
73              f".*(PaliGemma/img|state_proj|action_in_proj|action_out_proj|{_TELE_GSY_PI0_UR5E_TACTILE_PARAM_REGEX}).*"
74          )
75          return nnx.All(nnx.Param, nnx.Not(trainable))
76      if freeze_mode == "action_head":
78          trainable = _tele_gsy_nnx_utils.PathRegex(
79              f".*(state_proj|action_in_proj|action_out_proj|{_TELE_GSY_PI0_UR5E_TACTILE_PARAM_REGEX}).*"
80          )
81          return nnx.All(nnx.Param, nnx.Not(trainable))
83      base_filter = model_config.get_freeze_filter() if _TELE_GSY_PI0_UR5E_LORA else nnx.Nothing()
84      train_resized_projection = nnx.Not(
85          _tele_gsy_nnx_utils.PathRegex(
86              f".*(state_proj|action_in_proj|action_out_proj|{_TELE_GSY_PI0_UR5E_TACTILE_PARAM_REGEX}).*"
87          )
88      )
89      return nnx.All(base_filter, train_resized_projection)
```

Three pre-existing branches — `vision_action_head` mode (lines 67–75), `action_head` mode
(lines 76–81), and the default LoRA-aware branch (lines 83–89) — each already special-cased
`state_proj`/`action_in_proj`/`action_out_proj` (the robot-specific heads that always need
retraining regardless of freeze mode, since they're resized/replaced for this robot). Each of
the three regex strings now also includes `_TELE_GSY_PI0_UR5E_TACTILE_PARAM_REGEX`, so whenever
the tactile tower exists in the model, its parameters are treated exactly like the other
robot-specific heads: always excluded from freezing (`nnx.Not(trainable)` on lines 75/81, or
folded into `train_resized_projection`'s `nnx.Not(...)` on lines 84–89), never subject to the
base LoRA freeze filter. When the tactile tower doesn't exist (default, disabled config), the
regex simply matches nothing extra — it's a pure no-op string addition.

### Module-level env knobs — lines 236–245

```python
236  _TELE_GSY_PI0_UR5E_TACTILE_FEATURE_MODE = _tele_gsy_os.environ.get("PI0_UR5E_TACTILE_FEATURE_MODE", "none")
237  _TELE_GSY_PI0_UR5E_TACTILE_TEMPORAL = _TELE_GSY_PI0_UR5E_TACTILE_FEATURE_MODE == "temporal_tokens"
238  _TELE_GSY_PI0_UR5E_TACTILE_WINDOW_SIZE = _tele_gsy_env_int("PI0_UR5E_TACTILE_WINDOW_SIZE", 4)
239  _TELE_GSY_PI0_UR5E_TACTILE_VIT_VARIANT = _tele_gsy_os.environ.get("PI0_UR5E_TACTILE_VIT_VARIANT", "Ti/8")
240  _TELE_GSY_PI0_UR5E_TACTILE_IMAGE_SIZE = _tele_gsy_env_int("PI0_UR5E_TACTILE_IMAGE_SIZE", 32)
241  # Short (post-repack) keys the temporal-tokens tactile mode expects to find in the LeRobot dataset.
242  # Empty tuple disables the feature end-to-end (data transform, model construction, freeze filter).
243  _TELE_GSY_PI0_UR5E_TACTILE_WINDOW_KEYS = (
244      ("tactile_left_rgb_window", "tactile_right_rgb_window") if _TELE_GSY_PI0_UR5E_TACTILE_TEMPORAL else ()
245  )
```

Five new module-level constants, defined using the same `_tele_gsy_env_bool`/`_tele_gsy_env_int`/
`os.environ.get` helpers every other tunable in this file uses (e.g. `PI0_UR5E_LORA`,
`PI0_UR5E_PI05` a few lines above):

- **Line 236**: `PI0_UR5E_TACTILE_FEATURE_MODE` env var, defaults to `"none"`.
- **Line 237**: convenience boolean, `True` only when the mode is exactly `"temporal_tokens"`.
- **Line 238**: `PI0_UR5E_TACTILE_WINDOW_SIZE`, defaults to `4` (matches `schema.py`'s default).
- **Line 239**: `PI0_UR5E_TACTILE_VIT_VARIANT`, defaults to `"Ti/8"` — one of the small
  big_vision ViT variant strings openpi's `siglip.py` already knows how to decode (§8).
- **Line 240**: `PI0_UR5E_TACTILE_IMAGE_SIZE`, a single int (square resolution), defaults to `32`.
- **Lines 243–245**: the actual on/off switch for the whole feature — a 2-tuple of the two
  window key names when temporal mode is on, or an empty tuple otherwise. This is the single
  value threaded into `TeleGsyLeRobotUR5eDataConfig.tactile_window_keys` (line 304, below) and
  `Pi0Config.tactile_window_keys` (lines 264/275, below); everything downstream keys off whether
  this tuple is empty.

### Threading into model construction — lines 264–267, 275–278

```python
257  pi0_config.Pi0Config(
258      paligemma_variant="gemma_2b_lora",
259      action_expert_variant="gemma_300m_lora",
260      pi05=_TELE_GSY_PI0_UR5E_PI05,
261      action_dim=_TELE_GSY_PI0_UR5E_MODEL_ACTION_DIM,
262      action_horizon=_tele_gsy_env_int("PI0_UR5E_ACTION_HORIZON", 50),
263      max_token_len=_TELE_GSY_PI0_UR5E_MAX_TOKEN_LEN,
264      tactile_window_keys=_TELE_GSY_PI0_UR5E_TACTILE_WINDOW_KEYS,
265      tactile_window_size=_TELE_GSY_PI0_UR5E_TACTILE_WINDOW_SIZE,
266      tactile_image_resolution=(_TELE_GSY_PI0_UR5E_TACTILE_IMAGE_SIZE, _TELE_GSY_PI0_UR5E_TACTILE_IMAGE_SIZE),
267      tactile_vit_variant=_TELE_GSY_PI0_UR5E_TACTILE_VIT_VARIANT,
268  )
```

The same four keyword arguments (lines 264–267) are added to **both** `pi0_config.Pi0Config(...)`
branches — the LoRA branch (lines 257–268) and the non-LoRA branch (lines 270–279) — but
deliberately **not** to the `pi0_fast.Pi0FASTConfig(...)` branch a few lines above (lines
249–254): pi0-FAST tokenizes state/actions discretely into the language stream instead of using
`Pi0.embed_suffix`, so this feature (which is specifically about injecting continuous tokens
into `embed_suffix`) doesn't apply there, matching the task's original scope.

### Threading into the data config — line 304

```python
295  data=TeleGsyLeRobotUR5eDataConfig(
...
303      use_delta_actions=_TELE_GSY_PI0_UR5E_USE_DELTA_ACTIONS,
304      tactile_window_keys=_TELE_GSY_PI0_UR5E_TACTILE_WINDOW_KEYS,
305  ),
```

Same tuple, passed to the data-side config so the repack/transform wiring in
`TeleGsyLeRobotUR5eDataConfig.create` (above) picks it up.

### Metadata — line 326

```python
326  "tactile_feature_mode": _TELE_GSY_PI0_UR5E_TACTILE_FEATURE_MODE,
```

Recorded into `policy_metadata`, next to the pre-existing `camera_padding_strategy`/
`freeze_mode` entries — purely informational, surfaced to whatever inspects a trained
checkpoint's metadata later (e.g. deployment tooling deciding how to build observations).

---

## 6. `openpi/src/openpi/models/model.py`

**Purpose.** `Observation` is the structured, pytree-compatible container every downstream
model code (prefix/suffix embedding, `compute_loss`, `sample_actions`) reads. This adds two new
optional fields to carry the windowed tactile arrays, and updates the three functions that
construct/convert `Observation` instances (`from_dict`, `to_dict`, `preprocess_observation`) to
know about them.

### New fields — lines 109–116

```python
109  # Optional windowed tactile RGB frames, keyed by sensor stream (e.g. "tactile_left_rgb_window").
110  # Each value is [*b, window, th, tw, c] in [-1, 1] float32. Opt-in: None unless a model/data config
111  # explicitly requests temporal tactile tokens. Uses its own th/tw (not h/w) since tactile frames
112  # are a different resolution than the main camera images and jaxtyping shares dim names by
113  # identifier across all fields on this class.
114  tactile_images: dict[str, at.Float[ArrayT, "*b window th tw c"]] | None = None
115  # Per-stream validity mask, same keys as tactile_images, each [*b] bool.
116  tactile_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None
```

- **Line 114**: `tactile_images`, a `dict[str, array]` just like the pre-existing `images` field
  a few lines above (line 91), but each array is 5-D (`[*b, window, th, tw, c]`) instead of 4-D.
  The dim names `th`/`tw` (not `h`/`w`) matter: this class is wrapped in `@at.typecheck`
  (line 81), which uses `jaxtyping` — dimension-name variables like `h`/`w` are shared *by name*
  across every field on the class within one type-check call. The main camera `images` field
  already binds `h`/`w` to `224` (the camera resolution); reusing those same letters here for a
  `32`-sized tactile frame produced a real `jaxtyping.TypeCheckError` (`h=224` vs. the tactile
  array's actual height) the first time this was exercised end-to-end — caught by the manual
  smoke test described in §9, and fixed by using the distinct names `th`/`tw`.
- **Line 116**: `tactile_image_masks`, one scalar bool per batch item per stream — same shape
  convention as the pre-existing `image_masks` field (line 93).
- Both default to `None` and are declared *after* every other field (following `tokenized_prompt`,
  `token_ar_mask`, etc., which are also optional/default-`None`), required by dataclass field
  ordering rules (non-default fields must come first).

### `Observation.from_dict` — lines 130–135, 144–145

```python
130  # Same uint8 -> [-1, 1] float32 conversion for the optional windowed tactile streams.
131  tactile_images = data.get("tactile_image")
132  if tactile_images is not None:
133      for key in tactile_images:
134          if tactile_images[key].dtype == np.uint8:
135              tactile_images[key] = tactile_images[key].astype(np.float32) / 255.0 * 2.0 - 1.0
```

Mirrors the pre-existing per-key `uint8 → [-1, 1] float32` conversion loop for `data["image"]`
a few lines above (lines 125–129) — `x/255*2-1` maps `[0, 255]` to `[-1, 1]`, the same
normalization every image the model sees uses.

- **Line 131**: reads the `"tactile_image"` dict out of the raw input `data` (this is exactly
  the key `TeleGsyUR5eInputs.__call__` populates, §5) — `None` if the caller didn't provide any
  (the disabled/default case).
- **Lines 132–135**: if present, converts each stream's array in place from `uint8` to
  normalized `float32`, same formula as the camera images.

```python
136  return cls(
...
144      tactile_images=tactile_images,
145      tactile_image_masks=data.get("tactile_image_mask"),
146  )
```

Passes the (possibly-`None`, possibly-converted) `tactile_images` and the raw
`data.get("tactile_image_mask")` (no conversion needed — already boolean) into the constructed
`Observation`.

### `Observation.to_dict` — lines 153–154

```python
148  def to_dict(self) -> at.PyTree[ArrayT]:
149      """Convert the Observation to a nested dict."""
150      result = dataclasses.asdict(self)
151      result["image"] = result.pop("images")
152      result["image_mask"] = result.pop("image_masks")
153      result["tactile_image"] = result.pop("tactile_images")
154      result["tactile_image_mask"] = result.pop("tactile_image_masks")
155      return result
```

The inverse of `from_dict`: `dataclasses.asdict` (line 150) turns every field into a plain dict
keyed by its Python attribute name (`tactile_images`, `tactile_image_masks`, ...); lines 153–154
rename those two keys back to the wire format (`tactile_image`, `tactile_image_mask`), exactly
matching the existing rename pattern for `images`→`image` / `image_masks`→`image_mask` on lines
151–152. When the fields are `None` (disabled case), `result["tactile_image"]` / `_mask` simply
become `None` in the dict — harmless, since nothing reads those keys downstream unless a model
config actually enabled tactile.

### `preprocess_observation` — lines 227–230

```python
219  return Observation(
220      images=out_images,
221      image_masks=out_masks,
222      state=observation.state,
223      tokenized_prompt=observation.tokenized_prompt,
224      tokenized_prompt_mask=observation.tokenized_prompt_mask,
225      token_ar_mask=observation.token_ar_mask,
226      token_loss_mask=observation.token_loss_mask,
227      # Passed through unchanged: tactile frames are not scene photos, so the crop/rotate/color-jitter
228      # augmentations above (meant for RGB cameras) would be physically wrong to apply to them.
229      tactile_images=observation.tactile_images,
230      tactile_image_masks=observation.tactile_image_masks,
231  )
```

`preprocess_observation` (the function this `return` belongs to) applies train-time image
augmentation — random crop, resize, rotate, color jitter (visible a few lines above, lines
187–206) — to the **camera** images (`out_images`, built from `observation.images` earlier in
the function). Tactile frames are carried through this function completely unchanged (lines
229–230, straight from the input `observation` rather than any locally-augmented variable),
because those augmentations model camera-specific nuisance variation (viewpoint jitter, scene
lighting) that doesn't apply to — and would corrupt the physical meaning of — a tactile sensor
reading.

---

## 7. `openpi/src/openpi/models/pi0_config.py`

**Purpose.** `Pi0Config` is the dataclass that fully parameterizes a `Pi0` model instance. This
adds the four tactile-specific knobs and updates `inputs_spec` (used both to build fake/shape
inputs for `nnx.eval_shape` and by training code to know the expected observation shape) to
include the tactile fields whenever they're configured.

### New fields — lines 35–42

```python
35  # Optional temporal tactile tokens: a window of raw tactile RGB frames per stream, encoded by a
36  # dedicated ViT into multiple tokens each and injected into the suffix alongside the state/action
37  # tokens (see Pi0.embed_suffix). Disabled by default (empty tuple) -- purely opt-in, no effect on
38  # existing configs/checkpoints.
39  tactile_window_keys: tuple[str, ...] = ()
40  tactile_window_size: int = 0
41  tactile_image_resolution: tuple[int, int] = (32, 32)  # (height, width), matches _model.IMAGE_RESOLUTION
42  tactile_vit_variant: str = "Ti/8"
```

- **Line 39**: `tactile_window_keys` — the master switch. Every other tactile-related code path
  in `pi0_config.py`/`pi0.py` checks `if self.tactile_window_keys:` / `if config.tactile_window_keys:`
  to decide whether to activate at all; empty tuple (the default) means fully disabled.
- **Line 40**: `tactile_window_size`, defaults to `0` here (vs. `4` in the tele-amir side) since
  this value is meaningless when the feature is off — it only matters once `tactile_window_keys`
  is non-empty, at which point the tele-amir config layer (§5) always supplies an explicit value.
- **Line 41**: `tactile_image_resolution` — note the comment: unlike tele-amir's
  `tactile_image_size` (a `(width, height)` tuple, matching cv2's convention), this is
  `(height, width)`, matching `_model.IMAGE_RESOLUTION`'s convention (used a few lines below at
  line 74) since it directly parameterizes a `jax.ShapeDtypeStruct`'s shape tuple. Both are `32`
  by default so the distinction is invisible unless someone configures a non-square tactile
  resolution.
- **Line 42**: `tactile_vit_variant`, a big_vision-style variant string (`"width/patch_size"`,
  e.g. `"Ti/8"`) that `openpi/src/openpi/models/siglip.py`'s `decode_variant` already knows how
  to parse — reusing the same variant vocabulary the main SigLIP tower uses (`"So400m/14"`) for
  consistency, just picking a much smaller variant since tactile images are small and this is a
  lightweight side-channel encoder.

### `inputs_spec` — lines 77–84, 101–102

```python
73  def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
74      image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
75      image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
76
77      tactile_images = None
78      tactile_image_masks = None
79      if self.tactile_window_keys:
80          tactile_spec = jax.ShapeDtypeStruct(
81              [batch_size, self.tactile_window_size, *self.tactile_image_resolution, 3], jnp.float32
82          )
83          tactile_images = {key: tactile_spec for key in self.tactile_window_keys}
84          tactile_image_masks = {key: image_mask_spec for key in self.tactile_window_keys}
```

- **Lines 77–78**: default to `None` — the disabled case, matching `Observation`'s own defaults.
- **Line 79**: only builds tactile specs when the feature is on.
- **Lines 80–82**: one shared `jax.ShapeDtypeStruct` (a shape+dtype placeholder, not real data)
  of shape `[batch, window, height, width, 3]` — every configured tactile stream uses the exact
  same shape, so building it once and reusing it (line 83) is sufficient.
- **Line 83**: `{key: tactile_spec for key in self.tactile_window_keys}` — one dict entry per
  configured key, all pointing at the same spec object (fine, since `jax.ShapeDtypeStruct` is
  immutable and these are never mutated in place).
- **Line 84**: same pattern for the per-stream mask spec, reusing the pre-existing
  `image_mask_spec` (line 75) since tactile masks have the exact same `[batch]` bool shape as
  camera image masks.

```python
86  with at.disable_typechecking():
87      observation_spec = _model.Observation(
...
98          state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
99          tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
100         tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
101         tactile_images=tactile_images,
102         tactile_image_masks=tactile_image_masks,
103     )
```

Lines 101–102 pass the (possibly-`None`) dicts built above into the `Observation(...)`
constructor call, alongside the pre-existing `images`/`image_masks`/`state`/... fields
(unchanged, lines 88–100). The whole block is already inside `at.disable_typechecking()` (line
86, pre-existing) because `jax.ShapeDtypeStruct` values aren't real arrays and would otherwise
fail `Observation`'s `@at.typecheck` decorator — this applies equally to the new tactile fields,
so no additional handling was needed here.

---

## 8. `openpi/src/openpi/models/pi0.py`

**Purpose.** This is where the actual tactile ViT tower lives, and where its output tokens get
spliced into the suffix sequence that pi0's action-expert transformer consumes. Three additions:
tower construction in `__init__`, a new `_embed_tactile` method that runs the tower and returns
tokens+mask, and edits to `embed_suffix`/`compute_loss`/`sample_actions` to call it and insert
the result at the right point with the right attention-mask bookkeeping.

### `__init__` — tower construction, lines 102–125

```python
100  self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
101
102  # Optional dedicated tactile ViT: encodes a window of raw tactile RGB frames per stream into
103  # multiple patch tokens each, injected into the suffix alongside the state/action tokens (see
104  # `embed_suffix`). Disabled (all attributes stay None) unless `tactile_window_keys` is set.
105  self.tactile_img = None
106  self.tactile_proj = None
107  self.tactile_time_pos_emb = None
108  self.tactile_stream_pos_emb = None
109  if config.tactile_window_keys:
110      tactile_vit_width = _siglip.decode_variant(config.tactile_vit_variant)["width"]
111      tactile_img = nnx_bridge.ToNNX(
112          _siglip.Module(
113              num_classes=None,
114              variant=config.tactile_vit_variant,
115              pool_type="none",
116              scan=True,
117              dtype_mm=config.dtype,
118          )
119      )
120      fake_tactile_image = jnp.zeros([1, *config.tactile_image_resolution, 3], dtype=jnp.float32)
121      tactile_img.lazy_init(fake_tactile_image, train=False, rngs=rngs)
122      self.tactile_img = tactile_img
123      self.tactile_proj = nnx.Linear(tactile_vit_width, action_expert_config.width, rngs=rngs)
124      self.tactile_time_pos_emb = nnx.Embed(config.tactile_window_size, tactile_vit_width, rngs=rngs)
125      self.tactile_stream_pos_emb = nnx.Embed(len(config.tactile_window_keys), tactile_vit_width, rngs=rngs)
```

- **Lines 105–108**: four attributes always exist on every `Pi0` instance, initialized to
  `None`. This means downstream code (`_embed_tactile`) can always check
  `if self.tactile_img is None` rather than needing `hasattr` — and it's what makes the feature
  a true no-op when disabled: no extra parameters are created, nothing changes shape.
- **Line 109**: the same master-switch check as everywhere else — only builds the tower when
  `tactile_window_keys` is non-empty.
- **Line 110**: `_siglip.decode_variant(...)["width"]` looks up the embedding width for the
  chosen tiny ViT variant (e.g. `"Ti"` → 192) — needed below both to size `tactile_proj`'s input
  and the two position-embedding tables.
- **Lines 111–119**: constructs a *second*, independent SigLIP-style ViT tower — deliberately
  separate weights from `self.PaliGemma.img` (the main camera tower built a few lines above,
  lines 81–90), since the task calls for a *dedicated* tactile encoder, not a shared one.
  - `num_classes=None` (line 113): critically different from the main tower's
    `num_classes=paligemma_config.width` (line 83) — passing a `num_classes` makes
    `siglip.Module` apply an internal `Dense` "head" that projects every patch token straight to
    that width (that's actually how the main tower gets from SigLIP's native width to Gemma's
    embedding width in one step). Passing `None` here means `self.tactile_img` returns raw
    per-patch tokens in the tactile ViT's *own* width (`tactile_vit_width`), left unprojected —
    projection to the action-expert width is instead done explicitly by `self.tactile_proj`
    (line 123) after the position embeddings are added (see `_embed_tactile` below), which
    wouldn't be possible if the projection were already baked into the ViT's own head.
  - `pool_type="none"` (line 114): the whole point of this feature — returns the full sequence
    of per-patch tokens instead of pooling them into one vector (which is what `pool_type="gap"`/
    `"map"` would do, and is conceptually what the *old* `TactileImageEncoder` /
    `image_embedding` mode did by different means).
  - `variant=config.tactile_vit_variant` (line 114): the small variant string from `pi0_config.py`.
  - `scan=True`, `dtype_mm=config.dtype` (lines 116–117): match the main tower's settings
    (`scan=True` for compilation efficiency across transformer layers; same activation dtype as
    the rest of the model).
- **Line 120**: builds a dummy all-zeros `[1, height, width, 3]` image just to trace the
  tower's parameter shapes.
- **Line 121**: `lazy_init` — same call pattern as the main tower (`img.lazy_init(...)`, line
  90) — actually allocates the tower's parameters using this fake input purely to determine
  shapes; `train=False` since this is parameter initialization, not a real forward pass.
- **Line 122**: only after successful init does `self.tactile_img` get assigned (overwriting the
  `None` from line 105) — so if this block is never entered, the attribute stays `None`.
- **Line 123**: `tactile_proj`, a plain `nnx.Linear` from the tactile ViT's native width to
  `action_expert_config.width` — this is what makes the tactile tokens compatible with the rest
  of the suffix sequence, which all lives in the action-expert's embedding space.
- **Line 124**: `tactile_time_pos_emb`, an `nnx.Embed` lookup table with one row per window
  position (`config.tactile_window_size` rows). Needed because every frame in the window is run
  through the *same* ViT weights (§"`_embed_tactile`" below) — without an explicit position
  signal, the model would have no way to tell "this token came from the frame 3 steps ago" apart
  from "this token came from the current frame".
  - **Line 125**: `tactile_stream_pos_emb`, the same idea but for *which sensor stream* a token
  came from (one row per configured key, e.g. left finger vs. right finger) — for the same
  reason: both streams share the same ViT weights, so without this the model couldn't
  distinguish a left-finger token from a right-finger token.

### `_embed_tactile` — lines 164–197

```python
164  def _embed_tactile(
165      self, obs: _model.Observation
166  ) -> tuple[at.Float[at.Array, "b t emb"], at.Bool[at.Array, "b t"]] | tuple[None, None]:
```

Purpose stated in its docstring (lines 167–174, omitted here for brevity): turns the optional
windowed tactile streams on `obs` into one flat sequence of suffix-ready tokens plus a matching
per-token validity mask, or `(None, None)` if the feature is off. Not decorated with
`@at.typecheck` (unlike most other methods in this file) because its return type is a *union* of
two structurally different tuples (`tuple[Array, Array] | tuple[None, None]`), which doesn't fit
the single-shape annotation style `at.typecheck` expects elsewhere in this file.

```python
176  if self.tactile_img is None or obs.tactile_images is None:
177      return None, None
```

Two independent disabled-cases collapse to the same early return: either this *model* was never
configured for tactile (`self.tactile_img is None`, from `__init__` above), or this particular
*observation* just doesn't carry any tactile data (`obs.tactile_images is None`, e.g. a stale
non-tactile dataset fed into a tactile-enabled model — shouldn't happen in practice, but this
keeps the method total either way).

```python
178  tokens = []
179  masks = []
180  for stream_idx, key in enumerate(sorted(obs.tactile_images)):
181      frames = obs.tactile_images[key]
182      batch_size, window = frames.shape[0], frames.shape[1]
183      flat_frames = frames.reshape((batch_size * window, *frames.shape[2:]))
184      patch_tokens, _ = self.tactile_img(flat_frames, train=False)
185      num_patches = patch_tokens.shape[1]
186      patch_tokens = patch_tokens.reshape((batch_size, window, num_patches, -1))
187      patch_tokens = patch_tokens + self.tactile_time_pos_emb(jnp.arange(window))[None, :, None, :]
188      patch_tokens = patch_tokens + self.tactile_stream_pos_emb(jnp.asarray(stream_idx))[None, None, None, :]
189      patch_tokens = self.tactile_proj(patch_tokens)
190      stream_tokens = patch_tokens.reshape((batch_size, window * num_patches, -1))
191      tokens.append(stream_tokens)
192      if obs.tactile_image_masks is not None and key in obs.tactile_image_masks:
193          stream_mask = obs.tactile_image_masks[key]
194      else:
195          stream_mask = jnp.ones((batch_size,), dtype=jnp.bool_)
196      masks.append(einops.repeat(stream_mask, "b -> b s", s=stream_tokens.shape[1]))
197  return jnp.concatenate(tokens, axis=1), jnp.concatenate(masks, axis=1)
```

One iteration of this loop body runs per configured tactile stream (e.g. once for
`tactile_left_rgb_window`, once for `tactile_right_rgb_window`):

- **Line 180**: `sorted(obs.tactile_images)` — iterates dict keys in a *fixed, deterministic*
  order (Python dict order isn't guaranteed to survive round-trips through JAX's pytree
  flatten/unflatten machinery, which sorts dict keys internally), and `enumerate` gives each
  stream a stable integer index (`stream_idx`) used below to look up its identity embedding.
- **Line 181**: pulls this stream's `[batch, window, height, width, 3]` array.
- **Line 182**: reads off `batch_size` and `window` from the array's own shape.
- **Line 183**: flattens the `(batch, window)` axes together into one leading axis of size
  `batch*window` — necessary because `self.tactile_img` (a ViT) processes a batch of
  independent 2-D images, and doesn't have a notion of "window" on its own; every frame in every
  window, across the whole batch, gets treated as one independent image for this call.
- **Line 184**: runs the ViT — same call signature as the main tower (`self.PaliGemma.img(...)`,
  seen in `embed_prefix`), `train=False` since dropout/stochastic behavior isn't wanted here
  (matches the main tower's usage too, which is always called with `train=False` regardless of
  the outer training mode — image-tower dropout isn't used in this codebase).
  Returns `(patch_tokens, aux_dict)`; `_` discards the aux dict, matching how `embed_prefix`
  ignores it too.
- **Line 185**: `num_patches`, how many spatial patches this ViT produces per image — depends on
  `tactile_image_resolution` and the variant's patch size.
- **Line 186**: un-flattens the leading axis back into separate `(batch, window)` axes, giving
  `[batch, window, num_patches, tactile_vit_width]`.
- **Line 187**: `self.tactile_time_pos_emb(jnp.arange(window))` looks up `window` rows from the
  time-embedding table, shape `[window, tactile_vit_width]`; `[None, :, None, :]` reshapes that
  to broadcast against `patch_tokens`' `[batch, window, num_patches, width]` shape — every patch
  in a given window-frame gets that frame's index embedding added, identical across the batch
  and across patches within the frame.
- **Line 188**: same idea for the stream identity — `self.tactile_stream_pos_emb(jnp.asarray(stream_idx))`
  looks up a single row (shape `[tactile_vit_width]`) since `stream_idx` is one static Python
  int per loop iteration (not a traced/batched value); `[None, None, None, :]` broadcasts it
  against every batch item, every window frame, every patch equally.
- **Line 189**: `self.tactile_proj` (the `nnx.Linear` from `__init__`, line 123) projects the
  last axis from `tactile_vit_width` to `action_expert_config.width` — applied *after* both
  position embeddings are summed in, so the projection layer sees (and can mix) position
  information along with the ViT's visual features, rather than position embeddings being
  tacked on post-projection in a separate space.
- **Line 190**: collapses `(window, num_patches)` into one token axis — from this point on, the
  "window" and "which patch within a frame" distinctions no longer matter structurally; they're
  just more tokens in the sequence (the model can still tell them apart via the position
  embeddings already baked in).
- **Line 191**: appends this stream's `[batch, window*num_patches, action_expert_width]` token
  block to the running list.
- **Lines 192–195**: per-stream validity mask — uses `obs.tactile_image_masks[key]` if present
  (the `True`/`False` scalar per batch item, as set by `TeleGsyUR5eInputs`, §5), else defaults
  to "always valid" (`jnp.ones(...)`) — the same default-to-valid convention
  `preprocess_observation` uses for camera image masks (`model.py` lines 213–217).
- **Line 196**: `einops.repeat(stream_mask, "b -> b s", ...)` broadcasts the one
  scalar-per-batch-item mask out to every token this stream produced (`s = stream_tokens.shape[1]`)
  — the exact same broadcast pattern `embed_prefix` uses for camera image masks
  (`einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])`).
- **Line 197**: after the loop has run once per stream, concatenates every stream's tokens
  together along the token axis (`axis=1`) — left-finger tokens followed by right-finger tokens,
  in `sorted(...)` key order — and likewise concatenates every stream's mask. This combined
  `(tokens, mask)` pair is what gets returned to the caller.

### `embed_suffix` — signature and insertion logic, lines 199–237

```python
199  @at.typecheck
200  def embed_suffix(
201      self,
202      obs: _model.Observation,
203      noisy_actions: _model.Actions,
204      timestep: at.Float[at.Array, " b"],
205      *,
206      tactile_tokens: at.Float[at.Array, "b t emb"] | None = None,
207      tactile_mask: at.Bool[at.Array, "b t"] | None = None,
208  ) -> tuple[
```

- **Lines 205–207**: two new *keyword-only* parameters (the `*` on line 205 forces this),
  both optional and defaulting to `None`. Deliberately **not** computed internally by calling
  `self._embed_tactile(obs)` inside `embed_suffix` itself — instead, both callers
  (`compute_loss`, `sample_actions`, below) call `_embed_tactile` themselves once and pass the
  result in. This matters specifically for `sample_actions`, where `embed_suffix` gets called
  once per denoising step (typically 10 times) — computing the tactile ViT forward pass fresh on
  every one of those calls would be pure waste, since the raw tactile pixels never change during
  denoising (see the `sample_actions` section below for how this is exploited).

```python
214  input_mask = []
215  ar_mask = []
216  tokens = []
217  context_block_opened = False
218  if not self.pi05:
219      # add a single state token
220      state_token = self.state_proj(obs.state)[:, None, :]
221      tokens.append(state_token)
222      input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
223      # image/language inputs do not attend to state or actions
224      ar_mask += [True]
225      context_block_opened = True
226
227  if tactile_tokens is not None:
228      tokens.append(tactile_tokens)
229      input_mask.append(tactile_mask)
230      n_tactile = tactile_tokens.shape[1]
231      if context_block_opened:
232          # Join the state token's block: state and tactile mutually attend.
233          ar_mask += [False] * n_tactile
234      else:
235          # No state token (pi05): the first tactile token opens the context block.
236          ar_mask += [True] + [False] * (n_tactile - 1)
237      context_block_opened = True
```

- **Line 217**: `context_block_opened`, a new local flag tracking whether *some* non-action
  suffix content (state and/or tactile) has already started an attention-mask "block" — needed
  because whether tactile tokens should *open* a new block or *join* an existing one depends on
  whether the (optional, pi0-only) state token ran first.
- **Lines 218–225**: **unchanged** pre-existing state-token logic (the state token is only
  emitted for `pi0`, not `pi05` — see `pi0_config.py`'s comment on `pi05`), with one addition:
  line 225 sets `context_block_opened = True` right after the state token's own block-opening
  `ar_mask += [True]` on line 224.
- **Line 227**: the new block — only runs when tactile tokens were actually supplied by the
  caller (`None` in every existing/disabled call site, so this is skipped identically to before
  for every config that doesn't use tactile).
- **Lines 228–229**: appends the tactile token block and its mask to the same `tokens`/
  `input_mask` lists the state token (and, further down, the action tokens) also append to —
  order matters here: tactile tokens land *after* the state token (if any) and *before* the
  action tokens appended later in this function, which is what makes them "alongside the state
  and action tokens" in the suffix, per the task's requirement.
- **Line 230**: `n_tactile`, how many tactile tokens were supplied (the full
  `window*num_patches`-per-stream, summed across streams — from `_embed_tactile`'s
  concatenation).
- **Lines 231–233**: if a state token already opened a block (`context_block_opened` is `True`
  from line 225), every tactile token gets `ar_mask=False`, meaning "join the same block as the
  token before it" — per `make_attn_mask`'s semantics (see its docstring, unchanged, lines
  19–44), tokens sharing one block can attend to each other bidirectionally. So state and
  tactile end up mutually attending — proprioception and touch context inform each other.
- **Lines 234–236**: the `pi05` case, where there's no state token — the *first* tactile token
  gets `ar_mask=True` (starts a brand new block, exactly like the state token does on line 224
  in the `pi0` case) and every subsequent tactile token gets `ar_mask=False` (joins that same
  block). Either way, by the time this `if`/`else` finishes, exactly one new block has been
  opened for "context" (state and/or tactile), and every token within it can attend to every
  other token within it.
- **Line 237**: `context_block_opened = True` again — harmless if it was already `True` from the
  state-token branch, and correctly `True` now if this was the block-opener (the `pi05`,
  no-state-token case). Nothing downstream in this function currently reads this flag again, but
  it's kept in the correct state for clarity and any future insertions between this point and
  the action tokens.
- **Everything below this block** (action token construction, lines 239–262 — unchanged) always
  starts its own new block via `ar_mask += [True] + ([False] * (self.action_horizon - 1))` on
  line 262, *regardless* of whether a context block came before it — so action tokens always end
  up attending to everything before them (prefix + state + tactile, all in earlier blocks) plus
  each other, exactly like today, just with the tactile tokens now included in "everything
  before them" when present.

No change was needed to `make_attn_mask` (lines 19–44) itself — it already builds an attention
mask generically from an `ar_mask` sequence of arbitrary length via `jnp.cumsum`, so simply
inserting more entries into the `ar_mask` list (as `embed_suffix` does above) is sufficient; the
mask-construction math didn't need to know anything specific about tactile tokens.

### `compute_loss` — lines 282–287

```python
282  # one big forward pass of prefix + suffix at once
283  prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
284  tactile_tokens, tactile_mask = self._embed_tactile(observation)
285  suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
286      observation, x_t, time, tactile_tokens=tactile_tokens, tactile_mask=tactile_mask
287  )
```

- **Line 284**: new call to `_embed_tactile`, once, right alongside the pre-existing
  `embed_prefix` call — `compute_loss` only ever calls `embed_suffix` once per training step, so
  there's no repeated-work concern here (unlike `sample_actions`); this could equally have been
  computed inside `embed_suffix`, but keeping the same "caller computes tactile tokens, passes
  them in" shape as `sample_actions` (below) avoids two different calling conventions for the
  same method.
- **Lines 285–287**: the pre-existing `embed_suffix(observation, x_t, time)` call gains the two
  new keyword arguments, threading the just-computed tokens/mask through.

### `sample_actions` — lines 322–334

```python
316  # first fill KV cache with a forward pass of the prefix
317  prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
318  prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
319  positions = jnp.cumsum(prefix_mask, axis=1) - 1
320  _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
321
322  # Raw tactile pixels don't change across denoising steps, so encode them once here rather than
323  # inside `step` (which runs `num_steps` times) -- same rationale as prefix KV-caching above.
324  tactile_tokens, tactile_mask = self._embed_tactile(observation)
325
326  def step(carry):
327      x_t, time = carry
328      suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
329          observation,
330          x_t,
331          jnp.broadcast_to(time, batch_size),
332          tactile_tokens=tactile_tokens,
333          tactile_mask=tactile_mask,
334      )
```

- **Line 320**: pre-existing — the prefix (images + language) is embedded and cached into
  `kv_cache` exactly *once*, before the denoising loop starts, specifically because re-running
  the (large, expensive) SigLIP + Gemma prefix encoder on every denoising step would be wasteful
  — the prefix doesn't depend on the diffusion timestep or the current noisy action estimate.
- **Line 324**: applies the *exact same reasoning* to the tactile tower: `_embed_tactile` is
  called once, here, before `step` is defined — not inside it. The tactile ViT forward pass
  depends only on `observation` (the raw pixels), never on `x_t`/`time`, so it's just as
  step-invariant as the prefix.
- **Lines 326–334**: `step` is the body of the `jax.lax.while_loop` that runs once per denoising
  step (line 369, `num_steps` times, typically 10). `tactile_tokens`/`tactile_mask` are captured
  by this closure from the enclosing scope (line 324) rather than recomputed — Python closure
  semantics, not anything JAX-specific — so the ViT forward pass genuinely runs once per
  `sample_actions` call, not once per denoising step, regardless of `num_steps`. `embed_suffix`
  is still called inside `step` (line 328) because the *action*-token part of the suffix
  genuinely does change every step (`x_t`, `time`) — only the tactile portion is hoisted out.

Nothing else in `sample_actions` needed to change: the rest of `step` (building
`suffix_attn_mask`, `full_attn_mask`, `positions`, and slicing `suffix_out[:, -self.action_horizon:]`
for the output) operates on whatever `suffix_tokens`/`suffix_mask`/`suffix_ar_mask` came back
from `embed_suffix`, and since tactile tokens are always placed *before* the trailing
`action_horizon` action tokens (per `embed_suffix`'s insertion order above), that final slice
still correctly grabs only the action tokens no matter how many tactile tokens were mixed in
earlier in the sequence.

---

## 9. Tests

### `learning/pi0_ur5e/tests/test_dataset_reader.py`

- **`test_dataset_reader_builds_tactile_temporal_windows`** (lines 68–101): builds a 3-frame
  synthetic episode with distinct constant-color left/right tactile frames per timestep, reads
  it with `tactile_feature_mode="temporal_tokens"`, and checks: the pooled `out.tactile` field
  stays `None` (line 95 — confirms the two tactile code paths, pooled vs. raw, don't interfere);
  `out.tactile_left_rgb`/`tactile_right_rgb` have the expected `(T, H, W, 3)` `uint8` shape
  (lines 96–98); and resizing a constant-color image preserves its exact value (lines 100–101 —
  a correctness check on `_resize_rgb_stack`'s use of `cv2`'s area interpolation, which averages
  exactly to the same constant on flat input).
- **`test_dataset_reader_skips_tactile_rgb_for_other_modes`** (lines 104–128): same setup but
  with `tactile_feature_mode="image_embedding"` — confirms the new fields stay `None` (lines
  126–127) and the pre-existing pooled behavior (`out.tactile.shape == (2, 16)`, line 128) is
  completely unaffected by this change, i.e. a regression guard on the *other* three modes.

### `learning/pi0_ur5e/tests/test_lerobot_conversion.py`

- **`test_fake_raw_dataset_converts_with_temporal_tactile_windows`** (lines 40–93): an
  end-to-end conversion test — reads synthetic per-frame tactile pkl data, writes a real LeRobot
  dataset via `write_lerobot_dataset(..., tactile_feature_mode="temporal_tokens", ...)`, then
  checks: the conversion report's `tactile_window_shape` (line 78); the on-disk feature schema's
  `dtype`/`shape` for `observation.tactile_left_rgb_window` (lines 80–82 — this is the assertion
  that would have caught the `images.`-prefix naming bug described in §3 if it had been present
  when this test was written); and, by re-opening the dataset with LeRobot's own
  `LeRobotDataset` class (lines 84–92), directly verifies `_build_tactile_windows`'
  episode-start padding behavior end-to-end — at row `t=0` both window slots are identical
  (all padding), and at `t=1` the window holds two genuinely different frames in the right order.

### `openpi/src/openpi/models/pi0_tactile_test.py`

Five tests, added directly to the openpi checkout (no equivalent file existed before), using a
`"dummy"` Gemma variant (width 64) and a `"mu/8"` tactile ViT variant so the whole suite runs on
CPU in seconds rather than requiring GPU/real weight downloads:

- **`test_tactile_disabled_by_default_matches_baseline`** (lines 39–44): confirms a
  default-constructed `Pi0Config` produces an `Observation` with `tactile_images=None` —
  the "feature is truly opt-in" guarantee, checked at the config/spec level.
- **`test_tactile_disabled_config_runs_end_to_end`** (lines 47–56): runs `compute_loss` and
  `sample_actions` on a tactile-disabled config and checks finite, correctly-shaped output — a
  regression guard that the tactile code paths added throughout `pi0.py`/`model.py` don't break
  the pre-existing, still-default behavior.
- **`test_temporal_tactile_tokens_pi0`** (lines 59–74): the main positive case — two tactile
  streams, `pi05=False` (so a state token exists). Checks the observation actually carries both
  configured streams at the right shape (lines 64–66), then runs `compute_loss`/`sample_actions`
  end-to-end and checks finite output.
- **`test_temporal_tactile_tokens_pi05_no_state_token`** (lines 77–89): same idea but
  `pi05=True` — specifically exercises the `embed_suffix` branch where tactile tokens must open
  the context block themselves (`pi0.py` lines 234–236), since there's no state token to open it
  for them.
- **`test_tactile_gradients_flow_to_all_tactile_params`** (lines 92–125): the strongest
  correctness check in this set — rather than only checking output shapes (which a
  silently-disconnected tactile block, e.g. one whose tokens get computed but never actually
  concatenated into the suffix, would still pass), this computes `jax.grad` of the training loss
  with respect to every model parameter, filters down to every parameter path under
  `tactile_img`/`tactile_proj`/`tactile_time_pos_emb`/`tactile_stream_pos_emb` (line 118, the
  exact same four prefixes named in `pi0_ur5e_cup_config.py`'s freeze-filter regex, §5), and
  asserts every single one has nonzero gradient norm (line 125). This test is what caught (during
  manual exploration before the test was written) that an all-`False`-mask synthetic observation
  makes gradients vanish for *both* the tactile tower and the pre-existing `PaliGemma.img` tower
  identically — ruling out a tactile-specific wiring bug and confirming it was purely a
  mask-construction artifact in the test input, not the model code.

All 13 tele-amir tests (`learning/pi0_ur5e/tests/`) and all 20 tests under
`openpi/src/openpi/models/` (including the 5 new tactile-specific ones) pass.

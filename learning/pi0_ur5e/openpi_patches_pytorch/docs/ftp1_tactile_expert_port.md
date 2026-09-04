# FTP1 tactile-expert port — implementation notes

Part 2 of porting FTP1's tactile-fusion strategy into Haptile: a new PyTorch model,
`HaptileTactilePI0Pytorch`, built on top of the existing `PI0Pytorch` training path, adding a
learned ViT tactile encoder + a dedicated "tactile expert" transformer branch (cross-attended by
the action expert) + FTP1's KV-cache-reuse inference strategy. See
`docs/tactile_raw_image_pipeline.md` for Part 1 (the data pipeline that feeds this model raw
tactile frames).

This is a cross-repo effort: everything is *authored* in this repo
(`learning/pi0_ur5e/openpi_patches_pytorch/`), then *installed* into `$OPENPI_ROOT`
(`/home/amirhosein/Projects/ur5_tele/openpi`) by `scripts/install_openpi_pytorch_patch.py`. A
handful of files are vendored, read-only reference material copied from `ftp1-policy`
(`/home/amirhosein/Projects/FTP_1/ftp1-policy`) at commit `076cd9a6c9e32629549655ddcf795eb86ae2f7c4`.

## Why `$OPENPI_ROOT` isn't a clean checkout

Before this work, `$OPENPI_ROOT` was already 1 commit ahead of `origin/main`
(`73e28e3 add tactile transformer encoder`) with a **JAX/Flax** tactile-fusion prototype: a
windowed, multi-frame `Observation.tactile_images`/`tactile_image_masks`, `Pi0Config.tactile_window_keys`,
and a SigLIP-tower-based `Pi0._embed_tactile`. It's a different architecture (temporal window →
shared SigLIP tower → suffix tokens) for a different purpose, JAX-only, and not wired into the
`pi0_ur5e_cup` TrainConfig. `HaptileTactilePI0Pytorch`/`HaptileTactileConfig` are a **new, separate**
model class that coexists with it without modifying it — the two `Observation` field pairs
(`tactile_images`/`tactile_image_masks` vs. `tactile_left_image`/`tactile_right_image`) are
intentionally distinct.

## Pre-flight: environment fixes in `$OPENPI_ROOT` (not tracked by git, done once per machine)

1. **`transformers_replace` (adaRMS patch)**: `$OPENPI_ROOT/.venv`'s installed `transformers==4.53.2`
   had no `modeling_gemma._gated_residual` and `GemmaRMSNorm.forward` took no `cond` argument —
   meaning even plain `PI0Pytorch` could not construct (`transformers.models.siglip.check`
   sentinel raises). Fixed by copying `ftp1-policy`'s
   `src/openpi/models_pytorch/transformers_replace/models/{gemma,paligemma,siglip}/*` over the
   installed `transformers` package's matching files. Originals backed up to
   `$OPENPI_ROOT/.venv_transformers_replace_backup/` before overwriting. Verified: plain
   `PI0Pytorch(Pi0Config(pi05=True, ...))` now constructs (3.6B params) — this also fixes the
   pre-existing pi0.5 PyTorch training path, which was broken by the same missing patch
   independent of tactile work.
2. **`timm`**: `HaptileTactileEncoder`'s `ViTEncoder` (from `t3_tactile_encoder.py`) depends on
   `timm`, not previously a dependency of `$OPENPI_ROOT`. Added `timm==1.0.27` (matching
   `ftp1-policy`'s pin) to `pyproject.toml` and installed it via a bootstrapped `pip`
   (`python -m ensurepip`, since `uv` itself isn't on `PATH` in this environment).
   **`uv.lock` was not regenerated** — run `uv lock` on a machine with `uv` installed to keep it
   in sync with `pyproject.toml`.

## `Observation` extension (`$OPENPI_ROOT/src/openpi/models/model.py`, direct edit)

Added `tactile_left_image`/`tactile_right_image: Float[..., "*b h w c"] | None = None` — one
unwindowed frame per camera, as opposed to `tactile_images`' multi-frame dict. Threaded through:
- `from_dict`: same uint8 → `[-1, 1]` float32 conversion as `images`, **including the
  `torch.uint8` → channels-first-permute branch** (a bug found and fixed during verification:
  the first version of this code only handled the numpy-uint8 case, leaving `torch.uint8` tactile
  tensors un-permuted while `images` became channels-first — jaxtyping's shared `"h w c"` pattern
  letters across both fields then failed a shape-consistency check, and even without that check
  the two tensors would have reached `HaptileTactileEncoder`/`get_image_features` in inconsistent
  layouts).
- `to_dict`: no rename needed (unlike `images`/`tactile_images`, whose dict keys differ from
  their field names) — `tactile_left_image`/`tactile_right_image` already match.
- `preprocess_observation` (JAX): passed through unchanged, same as `tactile_images` — tactile
  frames shouldn't get the RGB-camera crop/rotate/color-jitter augmentations.

`src/openpi/models_pytorch/preprocessing_pytorch.py`: `preprocess_observation_pytorch` /
`SimpleProcessedObservation` previously dropped all tactile fields silently for the PyTorch
backend. Now passes `tactile_left_image`/`tactile_right_image` through unchanged (`getattr`
with a `None` default, defensive against older `Observation` instances).

## Vendored FTP1 files (`openpi_patches_pytorch/_vendor/`, installed verbatim/near-verbatim)

See `_vendor/README.md` for exact provenance. Installed into
`$OPENPI_ROOT/src/openpi/models_pytorch/`:
- **`ftp1_attention_masks.py`** — verbatim. Block-structured attention-layout builders:
  `build_prefix_attention_layout`, `build_tactile_attention_layout`,
  `build_expert_attention_layout` (used for the full training-time joint attention over
  prefix+tactile+suffix), `build_action_denoise_layout` (used per denoising step, attending the
  action tokens back to the cached prefix+tactile KV without recomputing them).
- **`ftp1_gemma_pytorch.py`** — verbatim. `FTP1PaliGemmaWithExpertModel`: a 3-expert
  (VLM/tactile/action) generalization of `gemma_pytorch.PaliGemmaWithExpertModel`, confirmed to
  behave as the plain 2-expert model when `use_tactile_input=False`. Requires the
  `transformers_replace` patch (uses `modeling_gemma._gated_residual` and `GemmaRMSNorm`'s
  `cond`/`active_tail_tokens` kwargs directly).
- **`t3_tactile_encoder.py`** — one import changed. Originally imported
  `T3_PRETRAINED_SENSOR_NAME_MAP`/`T3_PRETRAINED_TACTILE_ENCODER_CHECKPOINTS_BASE_URL` from
  `ftp1_model_config.py` (the full FTP1 model config, out of scope). Those two constants were
  relocated to `haptile_tactile_encoder.py`; `t3_tactile_encoder.py`'s two
  `_load_t3_pretrained_checkpoint` methods now import them **lazily** (inside the method, not at
  module top level) to avoid a circular import — `haptile_tactile_encoder.py` imports
  `ViTEncoder` from this module, so a top-level import back would cycle.

## New Haptile files (`openpi_patches_pytorch/`, installed as-is)

### `haptile_tactile_encoder.py` — the (a) encoder

`HaptileTactileEncoder(nn.Module)`: one shared `ViTEncoder` (weights tied across both cameras,
called twice), `encoder_depth=3`, `embed_dim=768`/`num_heads=12`/`mlp_ratio=4`,
`patch_size=16`/`image_size=224` (matches the 224×224 tactile frames Part 1's `lerobot_writer.py`
writes) — these dimensions were originally labeled "T3-large defaults" in this file's comments;
that label was wrong (see below) but the dimensions themselves are correct, now confirmed against
a real downloaded checkpoint rather than an assumed label. Takes the CLS token (index 0), projects
via `LayerNorm → Linear → GELU → Linear`, adds a learned `nn.Embedding(2, token_dim)` left/right
tag. `_to_channels_first` defensively normalizes either `(B,H,W,C)` or `(B,C,H,W)` input, since
`Observation`'s tactile image layout depends on how it was constructed (see the `from_dict` note
above) — this mirrors the same `is_channels_first` detection pattern `preprocessing_pytorch.py`
already uses for the main camera images. Also hosts the two relocated T3-checkpoint constants.

`load_t3_tactile_checkpoint: bool = True` (on `HaptileTactileConfig`, threaded through to
`HaptileTactileEncoder`'s `load_t3_pretrained_checkpoint`/`sensor_name`/
`cache_t3_pretrained_checkpoint_dir` kwargs) — fine-tunes the tactile ViT encoder from a
pretrained T3 checkpoint instead of random init. Two real bugs found and fixed getting this
working, verified end-to-end (download + `load_state_dict` + forward pass, on the real sensor
config):

- **Wrong size class.** `T3_PRETRAINED_TACTILE_ENCODER_CHECKPOINTS_BASE_URL` pointed at
  `t3_large`, which is `embed_dim=1024`/`depth=6` — nothing close to this encoder's
  `embed_dim=768`/`depth=3`, and `ViTEncoder.load()`'s `self.load_state_dict(checkpoint)` is
  strict, so this failed immediately with `size mismatch` errors on every block, not silently.
  Downloaded and inspected tensor shapes for all four published size classes directly
  (`t3_tiny`=192/3, `t3_small`=384/3, `t3_medium`=768/3, `t3_large`=1024/6) — `t3_medium` is the
  one that actually matches. Fixed by pointing the base URL at `t3_medium` instead.
- **Sensor variant.** `sensor_name` (default `"gs_tag"`, unchanged) selects which of several
  released sensor-specific checkpoints to load — this needs to match the physical sensor, not
  just "GelSight" as a brand: `gs_tag` is the marker/dot-pattern gel (shear/slip tracked via
  marker displacement), `gs_black` is the plain/markerless black-gel variant (photometric-stereo
  surface reconstruction). Confirmed against the actual hardware (visible dots/markers on the gel
  pad) that `gs_tag` is correct — this wasn't independently verified before, just inherited as
  whatever the default happened to be.

`sensor_name`/checkpoint provenance: `https://huggingface.co/datasets/alanz-mit/FoundationTactile`
— a third-party dataset repo, not a Physical Intelligence asset; its continued availability isn't
under this project's control.

### `haptile_tactile_pytorch.py` — the (b)+(c) model

`HaptileTactilePI0Pytorch(nn.Module)`. Structured almost identically to `PI0Pytorch`
(`embed_prefix`, `action_in_proj`/`action_out_proj`, `sample_noise`/`sample_time`, gradient
checkpointing helpers are line-for-line the same); the differences:

- **pi0 vs pi0.5, selected by `config.pi05`** (default `False`, i.e. plain pi0 — see the `pi05`
  field's own comment in `haptile_tactile_config.py` for why). `embed_suffix`, `__init__`'s
  `time_mlp_in`/`time_mlp_out` vs `state_proj`/`action_time_mlp_in`/`action_time_mlp_out`
  submodule construction, and `use_adarms` all branch on `self.pi05`, matching `PI0Pytorch` line
  for line: pi05 embeds action+time only, via adaRMS conditioning on the action-expert branch, no
  separate state token; non-pi05 embeds `state` as its own leading suffix token and fuses
  action+time by concatenation through an MLP instead (`adarms_cond=None` in that case).
  `state`/`denoise_step` are threaded through `forward`/`sample_actions` accordingly (previously
  discarded as `_state` when this model only supported pi05). Originally this model only
  implemented the pi05 branch, matching the *unset-env-var default* of the existing
  `pi0_ur5e_cup` config — but that default doesn't reflect how this project actually runs
  training: every task's `TrainConfig` (T-shirt folding included) passes `--pi05 false`. Fixed to
  match real usage, with `pi05=False` as this config's own default too.
- **`embed_tactile(tactile_left, tactile_right)`**: runs `HaptileTactileEncoder` once (checkpointed
  like the other embed_* methods); returns `(None, None)` if tactile is disabled or the frames are
  missing. `forward`/`sample_actions` raise if `config.use_tactile_input=True` but the observation
  has no tactile frames, since `FTP1PaliGemmaWithExpertModel`'s internal 3-branch wiring is fixed
  at construction time and always expects exactly 3 `inputs_embeds` in that case.
- **`FTP1PaliGemmaWithExpertModel` in place of the 2-branch model**, constructed with
  `use_tactile_input=<config flag>`.
- **`build_expert_attention_layout`/`build_action_denoise_layout`** (from `ftp1_attention_masks.py`)
  in place of the plain `make_att_2d_masks` calls, in `forward`/`denoise_step` respectively —
  verified mathematically (and via the ablation test below) that
  `build_expert_attention_layout(tactile_pad_masks=None)` produces a bit-identical mask/position-id
  layout to `PI0Pytorch`'s plain 2-region prefix+suffix cumsum-based mask.
- **`sample_actions`**: replicates `FTP1Pytorch.sample_actions`'s two-stage KV-cache build
  (referenced directly from `ftp1-policy`'s `ftp1_pytorch.py`, adapted to drop FTP1-specific
  `state_input_mode`/`tactile_function_areas`/`cached_static_suffix` machinery not needed here):
  the VLM prefix and the tactile tokens are each forwarded once through separate single-branch
  calls into `FTP1PaliGemmaWithExpertModel`, producing two per-layer K/V caches concatenated
  (`torch.cat(..., dim=2)`, the sequence axis) into one `DynamicCache`, reused unchanged across
  every Euler denoising step. **Tactile tokens and the VLM prefix are each computed exactly once
  per action-chunk prediction, not once per denoising step** — verified directly by
  call-count instrumentation (see Verification below).
- **`_match_tactile_dtype`**: casts the tactile encoder's output to the tactile-expert branch's
  dtype (bf16 by default) before feeding it into `FTP1PaliGemmaWithExpertModel` — needed because
  `HaptileTactileEncoder` is a plain `nn.Module` outside `paligemma_with_expert`, so it isn't
  auto-cast by that model's own `to_bfloat16_for_selected_params`.

### `haptile_tactile_config.py` — the config

`HaptileTactileConfig(_model.BaseModelConfig)`: mirrors `Pi0Config`'s real field set, plus
`tactile_expert_variant: str = "gemma_300m"` and `use_tactile_input: bool = True`.
**`tactile_expert_variant` defaults to `"gemma_300m"`, not FTP1's own `"gemma_small"`** — that
variant only exists in `ftp1-policy`'s own (unported) `gemma.py`; plain `openpi`'s `gemma.py`
only defines `dummy`/`gemma_300m`(`_lora`)/`gemma_2b`(`_lora`). There's a second, harder
constraint discovered during verification: `FTP1PaliGemmaWithExpertModel` applies **one shared
rotary embedding**, sized for the VLM's `head_dim`, across all three branches — so every expert's
`head_dim` must match the VLM's. Only `gemma_2b`(`_lora`)/`gemma_300m`(`_lora`) have `head_dim=256`;
`dummy` (`head_dim=16`) does not, and cannot be used for *any* of the three variants in a working
config (confirmed by the `RuntimeError: size of tensor... must match... 16... 256` hit while
testing with `dummy` before switching to real variants). `model_type` returns
`_model.ModelType.PI05` if `self.pi05` else `_model.ModelType.PI0` (see the `pi05` field, below).
`create`/`inputs_spec`/`get_freeze_filter` raise `NotImplementedError`
(PyTorch-only model, mirroring `ftp1_model_config.FTP1ModelConfig`'s no-op stub pattern, but
raising rather than silently returning `None`). `load_pytorch` constructs
`HaptileTactilePI0Pytorch` instead of the base class's hardcoded `PI0Pytorch`.

**`pi05: bool = False`** — mirrors `Pi0Config.pi05`, selecting the pi0 vs pi0.5
transform/embedding convention (see `haptile_tactile_pytorch.py`'s section above). Defaults to
`False`: every task's `TrainConfig` in this repo, T-shirt folding included, passes `--pi05 false`
to `train_pi0_base.sh` — pi0.5 was never this project's actual convention. It was initially
hardcoded `True` here anyway, justified in code only by "matches the existing `pi0_ur5e_cup`
config" — true only of that config's *unset-env-var default*, not of any config it has actually
been run with. `PI0_UR5E_TACTILE_PI05` env var (default `false`) controls it in
`haptile_train_config_patch.py`, and the `run_train_pi0_haptile_tactile.sh` sbatch script exposes
it as a plain `PI05=` shell variable (there's no `train_pi0_base.sh`-style flag parser for this
training path).

**`discrete_state_input: bool | None = None`** — not read by the model itself; read by
`ModelTransformFactory`'s PI05 branch (`$OPENPI_ROOT/src/openpi/training/config.py`), which
decides whether `TokenizePrompt` folds `state` into the tokenized prompt text. Resolved from
`pi05` in `__post_init__` if left unset, exactly like `Pi0Config` does
(`discrete_state_input = pi05`). Originally hardcoded `True` unconditionally — found missing
during the end-to-end dry run below, back when this model only supported pi05: with the model
pi05-only but this field left at its dataclass default of `False`, the model would have been
silently blind to robot state, with no error to signal it. Now that `embed_suffix` supports both
branches (see above), `discrete_state_input` must track `pi05` exactly — `True` when pi05 (state
enters via the tokenized prompt only, `embed_suffix` never embeds it), `False` when non-pi05
(state enters via `embed_suffix`'s own `state_proj` suffix token instead, so it must *not* also
be duplicated into the prompt).

## Config-splice edits (`$OPENPI_ROOT/src/openpi/training/config.py`, via installer scripts)

Rather than redefining `TeleGsyUR5eInputs`/`TeleGsyLeRobotUR5eDataConfig` a second time inside a
new patch file, the **existing** classes (Haptile's own
`openpi_patches/pi0_ur5e_cup_config.py`, installed by the existing `install_openpi_config.py`)
were extended at their one source of truth:

- `TeleGsyUR5eInputs.__call__`: reads `data["tactile_left_rgb"]`/`data["tactile_right_rgb"]` when
  present, places them at `inputs["tactile_left_image"]`/`inputs["tactile_right_image"]` — deliberately
  **outside** `image`/`image_mask`, so they're never routed into the SigLIP vision tower.
- `TeleGsyLeRobotUR5eDataConfig` gained a new field, **`include_tactile_images: bool = False`**.
  This wasn't in the original design doc's plan, but is necessary: `_transforms.RepackTransform`
  raises `KeyError` on a missing source column, and the existing non-tactile `pi0_ur5e_cup`
  dataset has no `observation.images.tactile_left_rgb` column — unconditionally adding the
  tactile repack mapping would have broken the existing config. The tactile-specific `TrainConfig`
  sets it `True`; the original `pi0_ur5e_cup` config is unaffected (defaults `False`).

Both changes are additive/conditional — confirmed by a direct regression check that the original
`pi0_ur5e_cup` config still loads and its `TeleGsyLeRobotUR5eDataConfig.include_tactile_images`
defaults to `False` with no tactile keys appearing in its repack/transform output.

### `ModelTransformFactory` PI05 assert (direct edit, not a splice block)

Separately, `ModelTransformFactory.__call__`'s `PI05` case (same file, but core `openpi` code —
not part of Haptile's `TELE_GSY_*` blocks) had `assert isinstance(model_config, pi0_config.Pi0Config)`
before building the transform group. `HaptileTactileConfig` is a sibling `_model.BaseModelConfig`
subclass, not a `Pi0Config`, so this assert fired unconditionally before the tactile TrainConfig's
transform pipeline could even be constructed — found by the same end-to-end dry run that found the
`discrete_state_input` bug above (this one failed first, before `discrete_state_input` was
reachable at all). Relaxed to accept any `model_config` that exposes `discrete_state_input`, rather
than narrowing to `Pi0Config` specifically. Not installed by any script in this repo (no existing
splice block covers arbitrary core-function edits like this one) — applied directly to
`$OPENPI_ROOT` and committed there in its own git history (`$OPENPI_ROOT` is a git checkout with
local, unpushed commits — see "Why `$OPENPI_ROOT` isn't a clean checkout" above). If you set up a
fresh `$OPENPI_ROOT` from scratch (e.g. on a different machine), this edit needs to be re-applied
by hand unless you carry over that commit. (The `PI0` case of the same `match` statement needed
no such fix — it never asserted `isinstance(model_config, pi0_config.Pi0Config)` in the first
place, just reads `model_config.max_token_len`/`action_dim`, so it already worked generically for
any `BaseModelConfig`. This matters now that `HaptileTactileConfig.pi05` defaults to `False`: the
`PI0` branch is the one actually exercised by default.)

One automatic side effect worth knowing about, not a bug: `DataConfigFactory.create_base_config`
sets `use_quantile_norm=model_config.model_type != ModelType.PI0` — so with `pi05=False` (the
default), norm stats use non-quantile normalization, matching every other task's `pi0_ur5e_cup`
config (which also defaults non-pi05 in practice); with `pi05=True` it switches to quantile
normalization instead. This falls out of `model_type` automatically, no separate wiring needed.

`haptile_train_config_patch.py`: a second, independent marker-delimited block
(`# BEGIN/END TELE_GSY_PI0_UR5E_CUP_TACTILE`), appended by `install_openpi_pytorch_patch.py` at
the same `_CONFIGS` anchor `install_openpi_config.py` uses. Adds `TrainConfig(name=
"pi0_ur5e_cup_tactile", model=HaptileTactileConfig(...), data=TeleGsyLeRobotUR5eDataConfig(...,
include_tactile_images=True), ...)`. No `weight_loader` is set — the existing
`TeleGsyShapeTolerantCheckpointWeightLoader` shape-tolerantly restores a **JAX** params pytree,
which doesn't correspond to this PyTorch model's `state_dict` naming at all; use
`--pytorch_weight_path` (an existing `TrainConfig` field, consumed by `train_pytorch.py`-style
scripts) to seed from a pretrained PyTorch checkpoint instead, if desired.

**Installer bug found and fixed** (in both `install_openpi_pytorch_patch.py` and the pre-existing
`install_openpi_config.py`): the "insert before anchor" fallback path did
`text.replace(ANCHOR, patch + "\n" + ANCHOR, 1)` — replacing the *first* occurrence of the anchor
string. But each installed block's own header comment quotes that exact anchor text ("Paste this
block ... immediately before: `if len(...)`"), and that comment sits *earlier* in the file than
the real, executable anchor line. Once one block was already installed, a second installer run
matched the comment instead of the real anchor, corrupting the file. Fixed by using the *last*
occurrence (`text.rindex(ANCHOR)`) instead.

## `scripts/install_openpi_pytorch_patch.py`

Copies the 3 vendored files + 3 new Haptile files into `$OPENPI_ROOT/src/openpi/models_pytorch/`,
then splices `haptile_train_config_patch.py` into `training/config.py`. Run
`install_openpi_config.py` first (or re-run it after editing `pi0_ur5e_cup_config.py`) — the
tactile block reuses classes only `install_openpi_config.py`'s block defines.

## `scripts/train_haptile_tactile_pytorch.py`

Fork of `$OPENPI_ROOT/scripts/train_pytorch.py`. Identical in every respect except the
model-construction block: no `Pi0Config` fallback-conversion path (that model has no tactile
branch), just a `TypeError` guard confirming `config.model` is a `HaptileTactileConfig`, then
`HaptileTactilePI0Pytorch(model_cfg).to(device)`.

## Verification performed

All run against the real installed model (`gemma_2b` VLM + `gemma_300m` action/tactile experts —
the smallest variant combination with matching `head_dim=256`, since `dummy` doesn't satisfy the
shared-rotary-embedding constraint noted above), on CPU:

- **Encoder unit test**: `HaptileTactileEncoder` on dummy `(B,H,W,C)` tensors → correct
  `(B,2,token_dim)` output shape; `.backward()` reaches ViT parameters (28/~ nonzero grads).
- **Identity check**: swapping left/right tactile inputs changes the output tokens
  (max diff 0.0765, not order-invariant) — confirms the learned side embedding matters.
- **Training forward/backward**: `HaptileTactilePI0Pytorch.forward()` returns a finite,
  correctly-shaped loss; `.backward()` reaches `tactile_encoder` parameters (47 nonzero grads).
- **Cache-call-count check**: instrumented `embed_tactile` and the VLM-prefix-only
  `paligemma_with_expert.forward` call — each invoked **exactly once** per `sample_actions` call
  with `num_steps=3`, confirming the (c) inference-cost property directly.
- **KV-cache correctness check**: cached `sample_actions` vs. a naive version that fully
  recomputes the prefix+tactile cache from scratch on every denoising step — **max diff
  0.00000000** (bit-exact on CPU float32).
- **Ablation check**: `use_tactile_input=False` constructs `FTP1PaliGemmaWithExpertModel` with no
  `gemma_tactile_expert` and runs `sample_actions` cleanly via the plain 2-branch path. (Bit-exact
  numerical equivalence to standalone `PI0Pytorch` given identical seeds was not executed directly
  — that would require weight-transplanting between two independently-initialized ~3B-parameter
  models — but is supported by two things: `build_expert_attention_layout(tactile_pad_masks=None)`
  is provably equivalent, term by term, to `PI0Pytorch`'s cumsum-based mask/position-id
  construction; and `FTP1PaliGemmaWithExpertModel`'s own 2-branch code path is structurally
  identical to `gemma_pytorch.PaliGemmaWithExpertModel`'s.)
- **Transform pipeline check**: a real LeRobot row (from a dataset converted with Part 1's
  `raw_image` mode) run through `RepackTransform` (with `include_tactile_images=True`) then
  `TeleGsyUR5eInputs.__call__` → confirmed `tactile_left_image`/`tactile_right_image` present in
  the output and **not** present in `image`/`image_mask`; regression-checked that the
  non-tactile repack path is unaffected.
- **`Observation.from_dict` layout check**: confirmed the `torch.uint8` tactile branch fix
  produces the same channels-first layout as the main camera `images`, resolving the bug noted
  above.
- **Data pipeline check** (Part 1, referenced here since it feeds this model): raw pkl → `Episode`
  → LeRobot dataset → reread round trip, confirming `observation.images.tactile_left_rgb`/
  `tactile_right_rgb` present with the expected shape and `observation.state` staying 7-D.
- **End-to-end training dry run** (a later session, after the checks above): converted a real
  2-episode/16-frame dataset with Part 1's `raw_image` mode
  (`local/pi0_ur5e_cup_tactile_dryrun`), ran `scripts/compute_norm_stats.py --config-name
  pi0_ur5e_cup_tactile` against it, then launched
  `scripts/train_haptile_tactile_pytorch.py pi0_ur5e_cup_tactile` for real (not a unit test) —
  config resolution, dataset loading with `include_tactile_images=True`, the full repack →
  `TeleGsyUR5eInputs` → `TokenizePrompt`(`discrete_state_input=True`) → `HaptileTactilePI0Pytorch`
  construction (gradient checkpointing enabled) → training loop all assembled and ran with no
  errors, through the start of the first forward/backward pass (manually stopped there — CPU
  bf16 compute for a 3.6B-param model is slow, and confirming assembly didn't require completing
  the step). This is what surfaced the missing `discrete_state_input` field documented above,
  and confirmed it fixed the issue once added. Two environment-only blockers, unrelated to the
  tactile code, had to be worked around to get this far: no norm stats existed yet for the
  dry-run dataset (fixed by the `compute_norm_stats.py` run above), and this machine's GPU
  (RTX PRO 2000 Blackwell, `sm_120`) isn't supported by the installed PyTorch/JAX builds (only
  `sm_50`–`sm_90`), which crashes on kernel launch unless both are forced to CPU
  (`CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`).

- **pi0/pi0.5 switch check** (a later session, after discovering the `pi05=True` default didn't
  match this project's actual usage): for both `pi05=False` and `pi05=True`, on CPU, with the
  real installed model — confirmed `HaptileTactileConfig.model_type`/`discrete_state_input`
  resolve correctly; confirmed the right submodule set gets constructed (`state_proj`/
  `action_time_mlp_in`/`action_time_mlp_out` for pi0, `time_mlp_in`/`time_mlp_out` for pi0.5, never
  both); `forward()` returns a finite, correctly-shaped loss for both; `.backward()` reaches
  `state_proj`/`action_time_mlp_in` for pi0 and `time_mlp_in` for pi0.5 (nonzero gradients, in
  addition to `tactile_encoder` for both); `sample_actions()` (the inference/denoising path,
  which needed `state` threaded through `denoise_step` for the new pi0 case) returns a finite,
  correctly-shaped action chunk for both. pi0.5 was a pure regression check — pre-existing,
  previously-verified code path, gated by branches that already existed — and came back
  unaffected.
- **T3 pretrained tactile-encoder load check**: isolated `HaptileTactileEncoder` construction
  with `load_t3_pretrained_checkpoint=True, sensor_name="gs_tag"` — real download from the
  HuggingFace dataset repo, `load_state_dict` (strict, no `strict=False`) succeeds with no key or
  shape errors, loaded weight statistics are non-degenerate (`std≈0.024`, not zero/default-init),
  and a forward pass on dummy tactile images returns a finite, correctly-shaped `(B, 2, token_dim)`
  output. Repeated through the full `HaptileTactileConfig` → `HaptileTactilePI0Pytorch`
  construction path (not just the isolated encoder) with identical weight statistics, confirming
  the config fields actually reach the encoder. This is what caught the `t3_large`-vs-`t3_medium`
  size-class bug documented above — the first attempt failed with `RuntimeError: size mismatch`
  on every block before the base URL was fixed.

**Still not executed**: a *completed* multi-step training run (checkpoint save/load included) and
a serving smoke test through `create_trained_policy`/`Policy.infer()` — the dry run above
confirms the pieces assemble and run, not that loss goes down or that a checkpoint round-trips.
Both would also need a working GPU (or a lot of patience on CPU) and, for a meaningful loss
trend, a larger real dataset than the 2-episode smoke-test one used here.

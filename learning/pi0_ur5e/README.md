# pi0_base LoRA Fine-Tuning for UR5e

This guide documents the end-to-end workflow for fine-tuning OpenPI `pi0_base` with LoRA on UR5e data:

- convert raw trajectories to a LeRobot dataset,
- train `pi0_base` with or without tactile image embeddings,
- serve the trained policy,
- deploy it through `run_env.py`.

The pi0 input is always:

```text
base RGB image + wrist RGB image + observation.state + prompt
```

Tactile images are not passed to pi0 as extra image slots. When tactile is enabled, the left and right tactile RGB frames are encoded into one fixed-size embedding and appended to `observation.state`.

## Paths

The commands below use this example layout:

```text
/home/rpl/yongqiang/tele-gsy   # this repository
/home/rpl/yongqiang/openpi     # OpenPI checkout
/path/to/raw_dataset           # raw recorded trajectories
turn_cleanser_bottle           # example task name
```

Run conversion from the OpenPI environment so LeRobot is available:

```bash
cd /home/rpl/yongqiang/openpi
```

Run training and deployment helper scripts from this repository:

```bash
cd /home/rpl/yongqiang/tele-gsy
```

## Raw Data Requirements

The converter reads episode folders, `trajectory.h5` files, `.npz` episodes, or existing LeRobot-style data. The default schema looks for these fields:

```text
base_rgb / base_camera_rgb_1 / third_view_camera_rgb
wrist_rgb / base_camera_rgb_0 / wrist_camera_rgb
joint_positions or ee_pos_quat
gripper_position
action / control
language_instruction / prompt / task
```

For tactile embedding, the raw data must also contain left and right tactile RGB frames under one of these names:

```text
tactile_left_rgb, tactile_right_rgb
left_tactile_rgb, right_tactile_rgb
observation.images.tactile_left_rgb, observation.images.tactile_right_rgb
```

If your dataset uses different names, update `learning/pi0_ur5e/configs/dataset_schema.yaml` before conversion.

## Option A: Convert Without Tactile

Use this path for the no-tactile baseline. The resulting `observation.state` is normally 7D:

```text
[joint_0, ..., joint_5, gripper]
```

Convert:

```bash
cd /home/rpl/yongqiang/openpi
uv run python /home/rpl/yongqiang/tele-gsy/learning/pi0_ur5e/scripts/convert_to_lerobot.py \
  --input-root /path/to/raw_dataset \
  --output-root /home/rpl/yongqiang/tele-gsy/outputs/turn_cleanser_bottle_lerobot_no_tactile \
  --config /home/rpl/yongqiang/tele-gsy/learning/pi0_ur5e/configs/dataset_schema.yaml \
  --task-name turn_cleanser_bottle \
  --repo-id local/pi0_ur5e_turn_cleanser_bottle_no_tactile \
  --default-prompt "turn the cleanser bottle" \
  --action-mode joint_position_gripper \
  --include-tactile false \
  --overwrite true
```

Check the converted state dimension:

```bash
python - <<'CHECK'
import json
from pathlib import Path
root = Path("/home/rpl/yongqiang/tele-gsy/outputs/turn_cleanser_bottle_lerobot_no_tactile")
info = json.loads((root / "meta" / "info.json").read_text())
print("state shape:", info["features"]["observation.state"]["shape"])
print("action shape:", info["features"]["action"]["shape"])
CHECK
```

Expected output:

```text
state shape: [7]
action shape: [7]
```

## Option B: Convert With Tactile Embedding

Use this path when training should condition on the tactile RGB images. The converter encodes the two tactile images into one embedding:

```text
left tactile RGB + right tactile RGB -> tactile_embedding_dim
```

For `--tactile-embedding-dim 128`, the final state is normally:

```text
robot_state(7) + tactile_embedding(128) = observation.state(135)
```

Convert:

```bash
cd /home/rpl/yongqiang/openpi
uv run python /home/rpl/yongqiang/tele-gsy/learning/pi0_ur5e/scripts/convert_to_lerobot.py \
  --input-root /path/to/raw_dataset \
  --output-root /home/rpl/yongqiang/tele-gsy/outputs/turn_cleanser_bottle_lerobot_tactile_emb \
  --config /home/rpl/yongqiang/tele-gsy/learning/pi0_ur5e/configs/dataset_schema.yaml \
  --task-name turn_cleanser_bottle \
  --repo-id local/pi0_ur5e_turn_cleanser_bottle_tactile_emb \
  --default-prompt "turn the cleanser bottle" \
  --action-mode joint_position_gripper \
  --include-tactile true \
  --tactile-feature-mode image_embedding \
  --tactile-embedding-dim 128 \
  --overwrite true
```

Check the converted state and tactile dimensions:

```bash
python - <<'CHECK'
import json
from pathlib import Path
root = Path("/home/rpl/yongqiang/tele-gsy/outputs/turn_cleanser_bottle_lerobot_tactile_emb")
info = json.loads((root / "meta" / "info.json").read_text())
print("state shape:", info["features"]["observation.state"]["shape"])
print("action shape:", info["features"]["action"]["shape"])
report = json.loads((root / "conversion_report.json").read_text())
print("tactile_shape:", report.get("tactile_shape"))
CHECK
```

Expected output for a 128D tactile embedding:

```text
state shape: [135]
action shape: [7]
tactile_shape: [N, 128]
```

If the state is still `[37]` or another low-dimensional value, the dataset was not converted with `--tactile-feature-mode image_embedding`, or the tactile RGB fields were not detected.

## Train pi0_base With LoRA

The training helper:

- injects the `pi0_ur5e_cup` config into the OpenPI checkout,
- links the LeRobot dataset into a local `HF_LEROBOT_HOME`,
- reads `observation.state.shape` from `meta/info.json`,
- runs `compute_norm_stats.py`,
- runs OpenPI training.

For `pi0_base`, use:

```text
--model-family pi0 --pi05 false --lora true
```

The `--use-delta-actions true` flag tells the OpenPI data transform to train on delta joint actions while keeping the converted dataset action format as `joint_position_gripper`.

### Train Without Tactile

```bash
cd /home/rpl/yongqiang/tele-gsy
bash learning/pi0_ur5e/scripts/train_pi0_base.sh \
  --dataset-root /home/rpl/yongqiang/tele-gsy/outputs/turn_cleanser_bottle_lerobot_no_tactile \
  --output-dir /home/rpl/yongqiang/tele-gsy/outputs/pi0_turn_cleanser_bottle_no_tactile_lora \
  --openpi-root /home/rpl/yongqiang/openpi \
  --repo-id local/pi0_ur5e_turn_cleanser_bottle_no_tactile \
  --exp-name turn_cleanser_bottle_pi0_base_no_tactile_lora \
  --steps 30000 \
  --batch-size 16 \
  --model-family pi0 \
  --pi05 false \
  --lora true \
  --camera-padding-strategy zeros \
  --use-delta-actions true \
  --include-tactile false \
  --default-prompt "turn the cleanser bottle"
```

### Train With Tactile Embedding

```bash
cd /home/rpl/yongqiang/tele-gsy
bash learning/pi0_ur5e/scripts/train_pi0_base.sh \
  --dataset-root /home/rpl/yongqiang/tele-gsy/outputs/turn_cleanser_bottle_lerobot_tactile_emb \
  --output-dir /home/rpl/yongqiang/tele-gsy/outputs/pi0_turn_cleanser_bottle_tactile_emb_lora \
  --openpi-root /home/rpl/yongqiang/openpi \
  --repo-id local/pi0_ur5e_turn_cleanser_bottle_tactile_emb \
  --exp-name turn_cleanser_bottle_pi0_base_tactile_emb_lora \
  --steps 30000 \
  --batch-size 16 \
  --model-family pi0 \
  --pi05 false \
  --lora true \
  --camera-padding-strategy zeros \
  --use-delta-actions true \
  --include-tactile true \
  --default-prompt "turn the cleanser bottle"
```

`--include-tactile true` documents the run, but the training input dimension is determined by the converted dataset. A tactile-embedding dataset must already have the tactile embedding appended to `observation.state`.

## Training Outputs

Training writes logs and checkpoints under the selected output directory:

```text
<output-dir>/logs/compute_norm_stats.log
<output-dir>/logs/train.log
<output-dir>/checkpoints/pi0_ur5e_cup/<exp-name>/<step>/
<output-dir>/conversion_report.json
```

Use the checkpoint step directory when serving the policy. The exact final step folder depends on OpenPI checkpoint numbering. Use `ls` to pick the checkpoint directory you want to serve.

## Serve A Trained Policy

Run the policy server on the GPU machine. Use the same dataset variant that was used for training because `serve_policy.py` reads its `observation.state` dimension and sets `PI0_UR5E_STATE_DIM`.

### Serve No-Tactile Checkpoint

```bash
cd /home/rpl/yongqiang/tele-gsy
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
python learning/pi0_ur5e/scripts/serve_policy.py \
  --openpi-root /home/rpl/yongqiang/openpi \
  --config-name pi0_ur5e_cup \
  --checkpoint-dir /home/rpl/yongqiang/tele-gsy/outputs/pi0_turn_cleanser_bottle_no_tactile_lora/checkpoints/pi0_ur5e_cup/turn_cleanser_bottle_pi0_base_no_tactile_lora/<step> \
  --dataset-root /home/rpl/yongqiang/tele-gsy/outputs/turn_cleanser_bottle_lerobot_no_tactile \
  --model-family pi0 \
  --use-delta-actions true \
  --camera-padding-strategy zeros \
  --default-prompt "turn the cleanser bottle" \
  --port 8000
```

### Serve Tactile-Embedding Checkpoint

```bash
cd /home/rpl/yongqiang/tele-gsy
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
python learning/pi0_ur5e/scripts/serve_policy.py \
  --openpi-root /home/rpl/yongqiang/openpi \
  --config-name pi0_ur5e_cup \
  --checkpoint-dir /home/rpl/yongqiang/tele-gsy/outputs/pi0_turn_cleanser_bottle_tactile_emb_lora/checkpoints/pi0_ur5e_cup/turn_cleanser_bottle_pi0_base_tactile_emb_lora/<step> \
  --dataset-root /home/rpl/yongqiang/tele-gsy/outputs/turn_cleanser_bottle_lerobot_tactile_emb \
  --model-family pi0 \
  --use-delta-actions true \
  --camera-padding-strategy zeros \
  --default-prompt "turn the cleanser bottle" \
  --port 8000
```

## Deploy Through run_env.py

Run these commands on the robot/control machine after the policy server is running.

Use `--agent pi0` for joint-position or delta-joint policies. Use `--agent pi0_eef` only for datasets converted with `--action-mode ee_delta_6d_gripper`.

### Deploy No-Tactile Policy

```bash
python run_env.py \
  --agent pi0 \
  --safe \
  --safe-max-joint-delta 0.03 \
  --safe-max-hand-delta 0.03 \
  --pi0-policy-host <gpu_computer_ip> \
  --pi0-policy-port 8000 \
  --pi0-prompt "turn the cleanser bottle" \
  --pi0-state-dim 7 \
  --pi0-action-chunk-size 6 \
  --pi0-include-tactile false
```

### Deploy Tactile-Embedding Policy

For `--tactile-embedding-dim 128`, use state dimension `135`:

```bash
python run_env.py \
  --agent pi0 \
  --safe \
  --safe-max-joint-delta 0.03 \
  --safe-max-hand-delta 0.03 \
  --use-tactile true \
  --pi0-policy-host <gpu_computer_ip> \
  --pi0-policy-port 8000 \
  --pi0-prompt "turn the cleanser bottle" \
  --pi0-state-dim 135 \
  --pi0-include-tactile \
  --pi0-tactile-feature-mode image_embedding \
  --pi0-tactile-embedding-dim 16 \
  --pi0-action-chunk-size 6
```

The deployment state dimension must match the training dataset:

```text
no tactile:             pi0_state_dim = 7
128D tactile embedding: pi0_state_dim = 135
64D tactile embedding:  pi0_state_dim = 71
```

If tactile was used in training but missing during deployment, `Pi0Agent` will pad or encode missing tactile inputs differently from training, and policy behavior will not match the trained distribution.

## Verify Before Running The Robot

Check the dataset state dimension:

```bash
python - <<'CHECK'
import json
from pathlib import Path
root = Path("/home/rpl/yongqiang/tele-gsy/outputs/turn_cleanser_bottle_lerobot_tactile_emb")
info = json.loads((root / "meta" / "info.json").read_text())
print("state shape:", info["features"]["observation.state"]["shape"])
print("action shape:", info["features"]["action"]["shape"])
report = root / "conversion_report.json"
if report.exists():
    print("tactile_shape:", json.loads(report.read_text()).get("tactile_shape"))
CHECK
```

Check that the policy server responds before connecting the real robot:

```bash
python learning/pi0_ur5e/scripts/query_policy_server.py \
  --host <gpu_computer_ip> \
  --port 8000 \
  --state-dim <7-or-135>
```

For tactile deployment, also confirm that `run_env.py` sees both tactile camera streams and that observations contain `tactile_left_rgb` and `tactile_right_rgb`.

## Important Consistency Rules

- Train and deploy with the same dataset variant: no-tactile with no-tactile, tactile-embedding with tactile-embedding.
- Keep `--model-family pi0 --pi05 false` when the target is `pi0_base`.
- Keep `--lora true` for LoRA fine-tuning.
- Keep `--camera-padding-strategy` the same for training and serving.
- Keep `--use-delta-actions` the same for training and serving.
- Do not change `--tactile-embedding-dim` between conversion and deployment.
- Use `--agent pi0` for the joint-space commands shown here.

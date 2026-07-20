#!/bin/bash -l

#SBATCH --job-name=convert_pi0_tactile
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/grp/luo/shiyi/project/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/grp/luo/shiyi/project/tele-gsy/script_results/%x_%j.err

set -e

PROJECT_ROOT=/scratch/grp/luo/shiyi/project/tele-gsy
OPENPI_ROOT=/scratch/grp/luo/shiyi/project/openpi
DATASET_NAME=turn_cleanser_water_bottle_gated_tactile_crop
OUTPUT_NAME=turn_cleanser_water_bottle_gated_tactile_crop_lerobot_tactile_emb
REPO_ID=local/pi0_ur5e_turn_cleanser_water_bottle_gated_tactile_crop
DEFAULT_PROMPT="Pick up the cleanser bottle, tilt it over either the yellow bowl or the blue bowl as if pouring, then place it back in its original position"
YELLOW_BOWL_PROMPT="Pick up the cleanser bottle, tilt it over the yellow bowl as if pouring, then place it back in its original position"
BLUE_BOWL_PROMPT="Pick up the cleanser bottle, tilt it over the blue bowl as if pouring, then place it back in its original position"
PROMPT_CUTOFF_EPISODE=0519_181949
TACTILE_EMBEDDING_DIM=16

INPUT_ROOT=${PROJECT_ROOT}/shared/data/bc_data/${DATASET_NAME}
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/${OUTPUT_NAME}
CONFIG_PATH=${PROJECT_ROOT}/learning/pi0_ur5e/configs/dataset_schema.yaml
CONVERT_SCRIPT=${PROJECT_ROOT}/learning/pi0_ur5e/scripts/convert_to_lerobot.py

cd "${OPENPI_ROOT}"
source /users/k25070928/miniconda3/etc/profile.d/conda.sh
conda activate tele

echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "UV path: $(which uv)"
echo "Input root: ${INPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Tactile embedding dim: ${TACTILE_EMBEDDING_DIM}"
echo "================================"

echo "Checking GPU with nvidia-smi:"
nvidia-smi

echo "Converting raw trajectories to LeRobot/OpenPI format with tactile embedding:"
uv run python "${CONVERT_SCRIPT}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --config "${CONFIG_PATH}" \
  --task-name "${DATASET_NAME}" \
  --repo-id "${REPO_ID}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --episode-prompt-cutoff "${PROMPT_CUTOFF_EPISODE}" \
  --prompt-before-or-at-cutoff "${YELLOW_BOWL_PROMPT}" \
  --prompt-after-cutoff "${BLUE_BOWL_PROMPT}" \
  --action-mode joint_position_gripper \
  --include-tactile true \
  --tactile-feature-mode image_embedding \
  --tactile-embedding-dim "${TACTILE_EMBEDDING_DIM}" \
  --overwrite true

echo "Conversion finished."

echo "Checking converted dataset shapes:"
OUTPUT_ROOT="${OUTPUT_ROOT}" uv run python - <<'CHECK'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
info = json.loads((root / "meta" / "info.json").read_text())
print("state shape:", info["features"]["observation.state"]["shape"])
print("action shape:", info["features"]["action"]["shape"])
report_path = root / "conversion_report.json"
if report_path.exists():
    report = json.loads(report_path.read_text())
    print("tactile_shape:", report.get("tactile_shape"))
CHECK

#!/bin/bash -l

#SBATCH --job-name=convert_pi0_haptile_tactile_disposableCup_move
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.err

#
# Part 1 of the new tactile-expert pipeline: converts a raw dataset to LeRobot format with
# --tactile-feature-mode raw_image (writes tactile pixels as their own image columns, keeps
# observation.state 7-D) instead of the older --tactile-feature-mode image_embedding used by
# run_convert_pi0_lerobot_tactile_emb.sh (which squashes tactile through a fixed random
# projection into observation.state). See learning/pi0_ur5e/docs/tactile_raw_image_pipeline.md
# for the full data-flow explanation, and README.md's "Option C" section for the reference
# command this script is based on.
#
# This raw_image dataset is NOT interchangeable with an image_embedding one -- it feeds the new
# HaptileTactilePI0Pytorch model (run_train_pi0_haptile_tactile.sh), not the older
# image-embedding-in-state training path (run_train_pi0_tactile_emb.sh-style scripts).

set -e

PROJECT_ROOT=/scratch/grp/luo/Amir/TFM_ViT2               # e.g. /scratch/grp/luo/<you>/project/tele-gsy
OPENPI_ROOT=/scratch/users/k2691893/projects/openpi               # e.g. /scratch/grp/luo/<you>/project/openpi
DATASET_NAME=grab_03_disposableCup_move              # e.g. fold_Tshirt
OUTPUT_NAME=${DATASET_NAME}_lerobot_tactile_raw
REPO_ID=local/pi0_ur5e_${DATASET_NAME}_tactile_raw
DEFAULT_PROMPT="Lift the disposable paper cup and place it on the other side of the table"          # e.g. "Fold the t-shirt in half"

INPUT_ROOT=/scratch/grp/luo/shiyi/project/tele-gsy/shared/data/bc_data/${DATASET_NAME}
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/${OUTPUT_NAME}
CONFIG_PATH=${PROJECT_ROOT}/learning/pi0_ur5e/configs/dataset_schema.yaml
CONVERT_SCRIPT=${PROJECT_ROOT}/learning/pi0_ur5e/scripts/convert_to_lerobot.py

cd "${OPENPI_ROOT}"
source /scratch/users/k2691893/miniconda3/etc/profile.d/conda.sh   # CHANGE_ME: your conda.sh path
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
echo "================================"

echo "Checking GPU with nvidia-smi:"
nvidia-smi

echo "Converting raw trajectories to LeRobot/OpenPI format with raw tactile images:"
uv run python "${CONVERT_SCRIPT}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --config "${CONFIG_PATH}" \
  --task-name "${DATASET_NAME}" \
  --repo-id "${REPO_ID}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --action-mode joint_position_gripper \
  --include-tactile true \
  --tactile-feature-mode raw_image \
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
for key in ("observation.images.tactile_left_rgb", "observation.images.tactile_right_rgb"):
    feat = info["features"].get(key)
    print(f"{key}:", feat["shape"] if feat else "MISSING")
CHECK

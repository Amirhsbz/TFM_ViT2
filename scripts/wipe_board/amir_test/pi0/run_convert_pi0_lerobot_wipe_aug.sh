#!/bin/bash -l

#SBATCH --job-name=convert_pi0_wipe_aug
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.err

set -e

PROJECT_ROOT=/scratch/grp/luo/shiyi/project/tele-gsy-worktree-amir
OPENPI_ROOT=/scratch/grp/luo/shiyi/project/openpi
SHARED_ROOT=/scratch/grp/luo/shiyi/project/tele-gsy
DATASET_NAME=wipe_board_aug
OUTPUT_NAME=wipe_board_aug_lerobot_no_tactile
REPO_ID=local/pi0_ur5e_wipe_board_aug_no_tactile
DEFAULT_PROMPT="Grab the sponge, wipe the markers on the white board and put the sponge back"

INPUT_ROOT=${SHARED_ROOT}/shared/data/bc_data/${DATASET_NAME}
OUTPUT_ROOT=/scratch/users/k2691893/projects/tele-gsy/source/data/${DATASET_NAME}/${OUTPUT_NAME}
CONFIG_PATH=${PROJECT_ROOT}/learning/pi0_ur5e/configs/dataset_schema.yaml
CONVERT_SCRIPT=${PROJECT_ROOT}/learning/pi0_ur5e/scripts/convert_to_lerobot.py

cd "${OPENPI_ROOT}"
source /scratch/users/k2691893/miniconda3/etc/profile.d/conda.sh
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

echo "Converting raw trajectories to LeRobot/OpenPI format:"
uv run python "${CONVERT_SCRIPT}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --config "${CONFIG_PATH}" \
  --task-name "${DATASET_NAME}" \
  --repo-id "${REPO_ID}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --action-mode joint_position_gripper \
  --include-tactile false \
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
CHECK

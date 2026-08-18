#!/bin/bash -l

set -e

PROJECT_ROOT=/home/amirhosein/Projects/ur5_tele/tele-amir
OPENPI_ROOT=/home/amirhosein/Projects/ur5_tele/openpi
DATASET_NAME=wipe_board
LEROBOT_REPO_ID=local/pi0_ur5e_wipe_board_no_tactile
EXP_NAME=wipe_board_pi0_base_no_tactile_lora
DEFAULT_PROMPT="Grab the sponge, wipe the markers on the white board and put the sponge back"
DRY_RUN=true
WANDB=true

DATASET_ROOT=${PROJECT_ROOT}/outputs/${DATASET_NAME}_lerobot_no_tactile
OUTPUT_DIR=${PROJECT_ROOT}/outputs/pi0_${DATASET_NAME}_no_tactile_lora

cd "${PROJECT_ROOT}"
source /home/amirhosein/miniconda3/etc/profile.d/conda.sh
conda activate tele

echo "================================"
echo "Running on node: $HOSTNAME"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "UV path: $(which uv)"
echo "Dataset root: ${DATASET_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Dry run: ${DRY_RUN}"
echo "WandB enabled: ${WANDB}"
echo "================================"

echo "Checking GPU with nvidia-smi:"
nvidia-smi

echo "Launching pi0 LoRA training helper:"
bash learning/pi0_ur5e/scripts/train_pi0_base.sh \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --openpi-root "${OPENPI_ROOT}" \
  --repo-id "${LEROBOT_REPO_ID}" \
  --exp-name "${EXP_NAME}" \
  --steps 30000 \
  --batch-size 16 \
  --model-family pi0 \
  --pi05 false \
  --lora true \
  --camera-padding-strategy zeros \
  --use-delta-actions true \
  --include-tactile false \
  --wandb "${WANDB}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --dry-run "${DRY_RUN}"

echo "Job finished."

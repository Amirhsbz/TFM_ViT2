#!/bin/bash -l

#SBATCH --job-name=train_pi0_tactile
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|a100_80g"
#SBATCH --exclude=erc-hpc-comp031,erc-hpc-comp035
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/grp/luo/shiyi/project/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/grp/luo/shiyi/project/tele-gsy/script_results/%x_%j.err

set -e

PROJECT_ROOT=/scratch/grp/luo/shiyi/project/tele-gsy
OPENPI_ROOT=/scratch/grp/luo/shiyi/project/openpi
DATASET_NAME=turn_cleanser_water_bottle_gated_tactile
LEROBOT_DATASET_NAME=turn_cleanser_water_bottle_gated_tactile_lerobot_tactile_emb_two_prompt
LEROBOT_REPO_ID=local/pi0_ur5e_turn_cleanser_water_bottle_gated_tactile_tactile_emb
EXP_NAME=turn_cleanser_water_bottle_gated_tactile_pi0_base_tactile_emb_two_prompt_lora
DEFAULT_PROMPT="Pick up the cleanser bottle, tilt it over either the yellow bowl or the blue bowl as if pouring, then place it back in its original position"
DRY_RUN=false
WANDB=true
KEEP_PERIOD=10000

DATASET_ROOT=${PROJECT_ROOT}/outputs/${LEROBOT_DATASET_NAME}
OUTPUT_DIR=${PROJECT_ROOT}/outputs/pi0_${DATASET_NAME}_tactile_emb_two_prompt_lora

cd "${PROJECT_ROOT}"
source /users/k25070928/miniconda3/etc/profile.d/conda.sh
conda activate tele

echo "================================"
echo "Job ID: $SLURM_JOB_ID"
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

echo "Launching pi0 LoRA training helper with tactile embedding:"
bash learning/pi0_ur5e/scripts/train_pi0_base.sh \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --openpi-root "${OPENPI_ROOT}" \
  --repo-id "${LEROBOT_REPO_ID}" \
  --exp-name "${EXP_NAME}" \
  --steps 30000 \
  --batch-size 16 \
  --keep-period "${KEEP_PERIOD}" \
  --model-family pi0 \
  --pi05 false \
  --lora true \
  --camera-padding-strategy zeros \
  --use-delta-actions true \
  --include-tactile true \
  --wandb "${WANDB}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --dry-run "${DRY_RUN}"

echo "Job finished."

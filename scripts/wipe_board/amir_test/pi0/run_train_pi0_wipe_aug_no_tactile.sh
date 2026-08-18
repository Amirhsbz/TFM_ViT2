#!/bin/bash -l

#SBATCH --job-name=train_pi0_wipe_aug_no_tactile
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40g|h200|a100_80g|l40s"
#SBATCH --exclude=erc-hpc-comp031,erc-hpc-comp035
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.err

set -e

PROJECT_ROOT=/scratch/grp/luo/shiyi/project/tele-gsy-worktree-amir
OPENPI_ROOT=/scratch/grp/luo/shiyi/project/openpi
DATASET_NAME=wipe_board_aug
LEROBOT_REPO_ID=local/pi0_ur5e_wipe_board_aug_no_tactile
EXP_NAME=wipe_board_aug_pi0_base_no_tactile_lora
DEFAULT_PROMPT="Grab the sponge, wipe the markers on the white board and put the sponge back"
DRY_RUN=false
WANDB=true

DATASET_ROOT=/scratch/users/k2691893/projects/tele-gsy/source/data/${DATASET_NAME}/${DATASET_NAME}_lerobot_no_tactile
OUTPUT_DIR=/scratch/users/k2691893/projects/tele-gsy/source/data/${DATASET_NAME}/pi0_${DATASET_NAME}_no_tactile_lora

cd "${PROJECT_ROOT}"
source /scratch/users/k2691893/miniconda3/etc/profile.d/conda.sh
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

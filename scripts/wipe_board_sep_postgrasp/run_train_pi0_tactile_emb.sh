#!/bin/bash -l

#SBATCH --job-name=train_pi0_tactile_wipe_board_sep_postgrasp_telegsy
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40g|h200|a100_80g|l40s"
#SBATCH --exclude=erc-hpc-comp031,erc-hpc-comp035
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.err

set -e

PROJECT_ROOT=/scratch/grp/luo/Amir/TFM_ViT2  
OPENPI_ROOT=/scratch/users/k2691893/projects/openpi 
DATASET_NAME=wipe_board_sep_postgrasp
LEROBOT_REPO_ID=local/pi0_ur5e_${DATASET_NAME}_tactile_raw_train
EXP_NAME=wipe_board_pi0_base_tactile_emb_lora_sep_postgrasp_telegsy
DEFAULT_PROMPT="Wipe the markers on the white board and put the sponge back"
DRY_RUN=false
WANDB=true
KEEP_PERIOD=10000

DATASET_ROOT=/scratch/grp/luo/Amir/TFM_ViT2/outputs/${DATASET_NAME}_lerobot_tactile_raw_train
OUTPUT_DIR=${PROJECT_ROOT}/outputs/pi0_${DATASET_NAME}_tactile_emb_lora_telegsy

# cd "${PROJECT_ROOT}"
# source /users/k25070928/miniconda3/etc/profile.d/conda.sh
# conda activate tele

cd "${PROJECT_ROOT}"
source /scratch/users/k2691893/miniconda3/etc/profile.d/conda.sh   # CHANGE_ME: your conda.sh path
conda activate tele
export OPENPI_DATA_HOME=/scratch/users/k2691893/.cache/openpi 

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

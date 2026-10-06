#!/bin/bash -l

#SBATCH --job-name=gradcam_pi0_haptile_wipe_board_sep_postgrasp
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40g|h200|a100_80g|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.err

#
# Grad-CAM + action->image attention maps of a trained HaptileTactilePI0Pytorch checkpoint over
# trajectory.h5 episodes (held-out demos or recorded rollouts). See the docstring of
# learning/pi0_ur5e/scripts/gradcam_pi0_rollouts.py for what each output shows.
#
# Point CHECKPOINT_DIR at the vision-only run's checkpoint to compare the two models; tactile vs
# vision-only is detected from the checkpoint itself.

set -e

PROJECT_ROOT=/scratch/grp/luo/Amir/TFM_ViT2
OPENPI_ROOT=/scratch/users/k2691893/projects/openpi
DATASET_NAME=wipe_board_sep_postgrasp
DEFAULT_PROMPT="Wipe the markers on the white board and put the sponge back"   # must match the conversion prompt
EXP_NAME=${DATASET_NAME}_haptile_tactileexpert_vit
STEP=30000

CHECKPOINT_DIR=${PROJECT_ROOT}/outputs/pi0_${DATASET_NAME}_haptile_tactile_train/checkpoints/pi0_ur5e_cup_tactile/${EXP_NAME}/${STEP}
INPUT_ROOT=/scratch/grp/luo/shiyi/project/tele-gsy/data_split/${DATASET_NAME}_test   # or a folder of rollouts
OUTPUT_DIR=${PROJECT_ROOT}/outputs/gradcam/${EXP_NAME}_${STEP}

TARGET=action          # action = what drives the predicted chunk; demo = what pulls it towards the recording
STRIDE=1               # process every Nth frame
MAX_EPISODES=5         # empty for all

cd "${OPENPI_ROOT}"
source /scratch/users/k2691893/miniconda3/etc/profile.d/conda.sh   # CHANGE_ME: your conda.sh path
conda activate tele

echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Input: ${INPUT_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Target: ${TARGET}  stride: ${STRIDE}  max episodes: ${MAX_EPISODES:-all}"
echo "================================"
nvidia-smi

EXTRA_ARGS=()
if [ -n "${MAX_EPISODES}" ]; then
  EXTRA_ARGS+=(--max-episodes "${MAX_EPISODES}")
fi

python "${PROJECT_ROOT}/learning/pi0_ur5e/scripts/gradcam_pi0_rollouts.py" \
  --openpi-root "${OPENPI_ROOT}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --input "${INPUT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --prompt "${DEFAULT_PROMPT}" \
  --target "${TARGET}" \
  --stride "${STRIDE}" \
  "${EXTRA_ARGS[@]}"

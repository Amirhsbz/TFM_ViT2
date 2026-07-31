#!/bin/bash -l

#SBATCH --job-name=train_dp_wipe_vonly_noproj
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40g|h200|a100_80g|l40s"
#SBATCH --exclude=erc-hpc-comp031,erc-hpc-comp035
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.err

set -e

PROJECT_ROOT=/scratch/grp/luo/shiyi/project/tele-gsy-worktree-amir
DATASET_NAME=wipe_board
REPRESENTATION_TYPE=img-pos
WANDB_EXP_NAME=wipe_board_dp_img_pos

DATA_PATH=/scratch/grp/luo/shiyi/project/tele-gsy/data_split/${DATASET_NAME}
MODEL_SAVE_PATH=/scratch/users/k2691893/projects/tele-gsy/source/data/${DATASET_NAME}/dp_img_pos

cd "${PROJECT_ROOT}"
source /scratch/users/k2691893/miniconda3/etc/profile.d/conda.sh
conda activate tele
echo "python: $(which python)"
python -c "import numpy; print('numpy:', numpy.__version__, numpy.__file__)"
conda env list


echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "================================"

echo "Checking GPU with nvidia-smi:"
nvidia-smi

echo "Training DP model:"
python learning/dp/pipelineNoProj.py \
  --data_path "${DATA_PATH}" \
  --model_save_path "${MODEL_SAVE_PATH}" \
  --representation_type "${REPRESENTATION_TYPE}" \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --touch_dim 60 \
  --obs_horizon 1 \
  --pred_horizon 16 \
  --action_horizon 8 \
  --batch_size 32 \
  --epochs 300 \
  --num_workers 16 \
  --use_train_test_split True \
  --use_memmap_cache True \
  --eval_freq 10 \
  --save_freq 100 \
  --num_diffusion_iters 100 \
  --load_img False \
  --use_wandb True \
  --wandb_entity_name masondavidblack-icl \
  --wandb_project_name tele-gsy \
  --wandb_exp_name "${WANDB_EXP_NAME}"

echo "Job finished."

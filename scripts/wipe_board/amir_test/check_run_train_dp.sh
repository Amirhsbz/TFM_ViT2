#!/bin/bash
# Local dry run for learning/dp/pipeline.py.
#
# Purpose: catch code errors (bad args, typos, broken imports, shape
# mismatches, etc.) BEFORE submitting run_train_dp.sh via sbatch, so you're
# not waiting hours on a node allocation just to see a stack trace from the
# first few seconds of the job.
#
# Requires data_split/wipe_board_{train,test} to already exist locally
# (run check_run_split_data.sh first). Runs pipeline.py for 1 short epoch
# with wandb disabled, exercising the real data-loading + training-loop +
# eval + checkpoint-save code paths.
#
# Usage:
#   scripts/wipe_board/amir_test/check_run_train_dp.sh                     # default representation_type=img-pos
#   scripts/wipe_board/amir_test/check_run_train_dp.sh img-tactile_img-pos # override representation_type
#
# Caveats:
#   - Only catches errors that trigger with this data + these args; it is
#     not a substitute for the real run.
#   - --use_memmap_cache is left False here (the real job uses True) to
#     avoid depending on check_run_prepare_cache.sh having been run first.
#   - Needs a local CUDA GPU (pipeline.py unconditionally calls
#     torch.cuda.set_device via --gpu). This machine has one, so it's fine.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_DATA_PATH="${REPO_ROOT}/data_split/wipe_board"
SAMPLE_ROOT="${REPO_ROOT}/data/amir_data"

REPRESENTATION_TYPE="${1:-img-pos}"

if [[ ! -d "${LOCAL_DATA_PATH}_train" ]]; then
  echo "Expected ${LOCAL_DATA_PATH}_train to exist. Run check_run_split_data.sh first." >&2
  exit 1
fi

echo "================================"
echo "Dry run: representation_type=${REPRESENTATION_TYPE}"
echo "data_path=${LOCAL_DATA_PATH}"
echo "================================"

cd "${REPO_ROOT}"
python learning/dp/pipeline.py \
  --data_path "${LOCAL_DATA_PATH}" \
  --model_save_path "${SAMPLE_ROOT}/dryrun_ckpts" \
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
  --use_wandb False

echo "================================"
echo "Dry run passed."
echo "================================"

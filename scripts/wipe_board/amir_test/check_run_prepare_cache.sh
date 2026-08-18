#!/bin/bash
# Local dry run for the memmap cache prep step (mirrors run_prepare_cache.sh).
#
# Purpose: catch cache-prep errors locally before submitting the real
# prepare_cache job to the HPC. Requires data_split/wipe_board_{train,test}
# to already exist (run check_run_split_data.sh first).
#
# Usage:
#   scripts/wipe_board/amir_test/check_run_prepare_cache.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOCAL_DATA_PATH="${REPO_ROOT}/data_split/wipe_board"
SAMPLE_ROOT="${REPO_ROOT}/data/amir_data"

if [[ ! -d "${LOCAL_DATA_PATH}_train" ]]; then
  echo "Expected ${LOCAL_DATA_PATH}_train to exist. Run check_run_split_data.sh first." >&2
  exit 1
fi

echo "================================"
echo "Preparing cache from ${LOCAL_DATA_PATH}_{train,test}"
echo "================================"

cd "${REPO_ROOT}"
python learning/dp/pipeline.py \
  --data_path "${LOCAL_DATA_PATH}" \
  --model_save_path "${SAMPLE_ROOT}/cache_prepare_dummy" \
  --use_train_test_split True \
  --representation_type img-pos \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --batch_size 32 \
  --num_workers 2 \
  --obs_horizon 1 \
  --pred_horizon 16 \
  --action_horizon 8 \
  --num_diffusion_iters 100 \
  --use_memmap_cache True \
  --load_img False \
  --gpu 0 \
  --prepare_cache_only True

echo "================================"
echo "Cache preparation dry run passed."
echo "================================"

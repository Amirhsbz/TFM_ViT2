#!/bin/bash
# Local dry run for workflow/split_data.py (mirrors run_split_data.sh).
#
# Purpose: split the local raw sample data already copied into
# shared/data/bc-data/wipe_board into data_split/wipe_board_{train,test},
# so check_run_prepare_cache.sh and check_run_train_dp.sh have a real
# train/test split to run against without touching the HPC.
#
# Usage:
#   scripts/wipe_board/amir_test/check_run_split_data.sh
#
# Safe to re-run: split_data.py is a no-op if data_split/wipe_board_train
# already exists.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAW_DATA_PATH="${REPO_ROOT}/shared/data/bc-data"
OUTPUT_PATH="${REPO_ROOT}/data_split"

if [[ ! -d "${RAW_DATA_PATH}/wipe_board" ]]; then
  echo "Expected raw sample data at ${RAW_DATA_PATH}/wipe_board but it doesn't exist." >&2
  exit 1
fi

echo "================================"
echo "Splitting data: ${RAW_DATA_PATH}/wipe_board -> ${OUTPUT_PATH}/wipe_board_{train,test}"
echo "================================"

cd "${REPO_ROOT}"
python workflow/split_data.py \
  --base_path "${RAW_DATA_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --data_name wipe_board

echo "================================"
echo "Split finished."
echo "================================"

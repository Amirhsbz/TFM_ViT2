#!/bin/bash -l

#SBATCH --job-name=pi0_emb_analysis
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/grp/luo/shiyi/project/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/grp/luo/shiyi/project/tele-gsy/script_results/%x_%j.err

set -e

PROJECT_ROOT=/scratch/grp/luo/shiyi/project/tele-gsy
OPENPI_ROOT=/scratch/grp/luo/shiyi/project/openpi
DATASET_NAME=put_bottle_upright
OUTPUT_NAME=put_bottle_upright_lerobot_tactile_emb

DATASET_ROOT=${PROJECT_ROOT}/outputs/${OUTPUT_NAME}
ANALYSIS_SCRIPT=${PROJECT_ROOT}/Model_analysis/pi0_embedding_analysis.py

cd "${OPENPI_ROOT}"
source /users/k25070928/miniconda3/etc/profile.d/conda.sh
conda activate tele

echo "================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on node: ${HOSTNAME}"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "UV path: $(which uv)"
echo "Project root: ${PROJECT_ROOT}"
echo "OpenPI root: ${OPENPI_ROOT}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Analysis script: ${ANALYSIS_SCRIPT}"
echo "================================"

echo "Running Pi0 tactile embedding analysis:"
uv run python "${ANALYSIS_SCRIPT}" \
  --dataset-root "${DATASET_ROOT}"

echo "Analysis finished."

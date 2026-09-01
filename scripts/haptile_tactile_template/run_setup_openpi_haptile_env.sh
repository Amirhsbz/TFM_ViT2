#!/bin/bash -l

#SBATCH --job-name=setup_openpi_haptile_env
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=CHANGE_ME/script_results/%x_%j.out
#SBATCH --error=CHANGE_ME/script_results/%x_%j.err
# CHANGE_ME: add --account/--partition/--qos directives if your cluster requires them.
#
# One-time (per fresh $OPENPI_ROOT checkout) setup for HaptileTactilePI0Pytorch -- installs the
# tactile-expert model into $OPENPI_ROOT and patches its transformers package. Every step here is
# idempotent, so it's also safe to re-run against an already-set-up checkout.
#
# If $OPENPI_ROOT already has the Haptile port committed in its own local git history (as it does
# on the machine this was authored on -- see openpi_patches_pytorch/docs/ftp1_tactile_expert_port.md,
# "Why $OPENPI_ROOT isn't a clean checkout"), and you copied/pulled that exact history to HPC,
# steps 1 and 4 are redundant but harmless. Steps 2-3 (the venv-local transformers_replace patch
# and the timm/uv.lock fix) are NOT captured by git and must run on every machine regardless.
#
# If your cluster's compute nodes have no internet access (needed for `uv lock`/`uv sync` to
# fetch timm), run this on the login node instead of via sbatch, or point uv at a pre-populated
# cache/mirror.

set -e

PROJECT_ROOT=CHANGE_ME          # e.g. /scratch/grp/luo/<you>/project/tele-gsy
OPENPI_ROOT=CHANGE_ME           # e.g. /scratch/grp/luo/<you>/project/openpi
FTP1_POLICY_ROOT=CHANGE_ME      # e.g. /scratch/grp/luo/<you>/project/ftp1-policy
CONDA_ENV=tele

cd "${OPENPI_ROOT}"
source /users/CHANGE_ME/miniconda3/etc/profile.d/conda.sh   # CHANGE_ME: your conda.sh path
conda activate "${CONDA_ENV}"

echo "================================"
echo "Job ID: ${SLURM_JOB_ID:-<interactive>}"
echo "Running on node: $HOSTNAME"
echo "OpenPI root: ${OPENPI_ROOT}"
echo "FTP1 policy root: ${FTP1_POLICY_ROOT}"
echo "Python: $(which python)"
echo "UV: $(which uv)"
echo "================================"

echo "Step 1/4: installing Haptile's config splice blocks into ${OPENPI_ROOT}..."
python "${PROJECT_ROOT}/learning/pi0_ur5e/scripts/install_openpi_config.py" --openpi-root "${OPENPI_ROOT}"
python "${PROJECT_ROOT}/learning/pi0_ur5e/scripts/install_openpi_pytorch_patch.py" --openpi-root "${OPENPI_ROOT}"

echo "Step 2/4: locking + syncing the openpi env (picks up timm==1.0.27 from pyproject.toml --"
echo "the committed uv.lock does not have it yet, so a plain 'uv sync' without 'uv lock' first"
echo "would not install it, and could even remove it if it's already present some other way):"
uv lock
uv sync

echo "Step 3/4: applying the transformers_replace (adaRMS) patch..."
SITE_PACKAGES="$(uv run python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')"
BACKUP_DIR="${OPENPI_ROOT}/.venv_transformers_replace_backup"
SRC_DIR="${FTP1_POLICY_ROOT}/src/openpi/models_pytorch/transformers_replace/models"
for rel in gemma/configuration_gemma.py gemma/modeling_gemma.py paligemma/modeling_paligemma.py siglip/modeling_siglip.py siglip/check.py; do
  dst="${SITE_PACKAGES}/models/${rel}"
  if [[ -f "${dst}" && ! -f "${BACKUP_DIR}/models/${rel}" ]]; then
    mkdir -p "$(dirname "${BACKUP_DIR}/models/${rel}")"
    cp "${dst}" "${BACKUP_DIR}/models/${rel}"
  fi
  mkdir -p "$(dirname "${dst}")"
  cp "${SRC_DIR}/${rel}" "${dst}"
  echo "  patched ${rel}"
done

echo "Step 4/4: verifying the model constructs..."
uv run python - <<'PY'
from openpi.models_pytorch.haptile_tactile_config import HaptileTactileConfig
from openpi.models_pytorch.haptile_tactile_pytorch import HaptileTactilePI0Pytorch

cfg = HaptileTactileConfig(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    tactile_expert_variant="gemma_300m",
    action_dim=7,
    action_horizon=50,
)
model = HaptileTactilePI0Pytorch(cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"HaptileTactilePI0Pytorch constructed OK -- {n_params / 1e9:.2f}B params")
PY

echo "Setup finished."

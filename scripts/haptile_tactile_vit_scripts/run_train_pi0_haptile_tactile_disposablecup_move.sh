#!/bin/bash -l

#SBATCH --job-name=train_pi0_haptile_tactile_disposableCup_move
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40g|h200|a100_80g|l40s"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.err
# CHANGE_ME: add --account/--qos/--exclude directives to match your cluster's current
# conventions (don't copy run_train_pi0_tactile_emb.sh's --exclude node list blindly -- check
# whether those nodes are still flaky before reusing it).
#
# Part 2 of the new tactile-expert pipeline: trains HaptileTactilePI0Pytorch (learned ViT tactile
# encoder + dedicated tactile-expert transformer branch + FTP1 KV-cache-reuse inference) on a
# dataset converted by run_convert_pi0_lerobot_tactile_raw.sh. Unlike the older
# run_train_pi0_tactile_emb.sh, there's no train_pi0_base.sh-style wrapper for this model yet --
# this script does by hand what that wrapper automated: installing the config splice, linking the
# dataset into a local HF_LEROBOT_HOME, computing norm stats, then launching training. See
# learning/pi0_ur5e/openpi_patches_pytorch/docs/ftp1_tactile_expert_port.md for the full design.
#
# Run run_setup_openpi_haptile_env.sh once per fresh $OPENPI_ROOT checkout before the first job
# that uses this script (installs timm + the transformers_replace patch; without it the model
# can't even construct).
#
# Note on LoRA: unlike run_train_pi0_tactile_emb.sh's --lora flag, there's no LoRA toggle here --
# it's implied by whether PYTORCH_WEIGHT_PATH is set below. paligemma_variant="gemma_2b_lora" and
# action_expert_variant="gemma_300m_lora" are hardcoded in haptile_train_config_patch.py (only
# TACTILE_EXPERT_VARIANT is env-overridable; to change the other two, edit that patch file), but
# the LoRA freeze+adapter split only actually gets applied when a pretrained checkpoint is loaded
# (freezing a randomly-initialized backbone would badly undertrain it) -- see
# learning/pi0_ur5e/openpi_patches_pytorch/docs/ftp1_tactile_expert_port.md's "LoRA +
# pretrained-weight loading" section for how to produce a checkpoint to point PYTORCH_WEIGHT_PATH
# at (examples/convert_jax_model_to_pytorch.py, run from $OPENPI_ROOT).
#
# Note on PI05 below: matches run_train_pi0_tactile_emb.sh's --pi05 flag, just as an env var
# instead of a bash flag (there's no train_pi0_base.sh-style flag parser for this script).
# Defaults to false (plain pi0) -- this project's actual convention across every task, T-shirt
# folding included; pi0.5 is opt-in via PI05=true.
#
# Note on VISION_TOWER_MODE below: for small datasets (a handful of demos per task), fully
# fine-tuning the ~400M-param pretrained vision tower for thousands of steps risks catastrophic
# forgetting/overfitting -- "lora"/"frozen" trade some vision adaptation capacity for protection
# against that. Only valid together with PYTORCH_WEIGHT_PATH being set (adapting/freezing a
# randomly-initialized vision tower would never learn anything useful) -- the training script
# errors clearly if you set this without also setting PYTORCH_WEIGHT_PATH.

set -e

PROJECT_ROOT=/scratch/grp/luo/Amir/TFM_ViT2                # e.g. /scratch/grp/luo/<you>/project/tele-gsy
OPENPI_ROOT=/scratch/users/k2691893/projects/openpi                  # e.g. /scratch/grp/luo/<you>/project/openpi
DATASET_NAME=grab_03_disposableCup_move                # must match run_convert_pi0_lerobot_tactile_raw.sh's DATASET_NAME
DEFAULT_PROMPT="Lift the disposable paper cup and place it on the other side of the table"            # must match run_convert_pi0_lerobot_tactile_raw.sh's DEFAULT_PROMPT
EXP_NAME=${DATASET_NAME}_haptile_tactileexpert_vit                    # e.g. ${DATASET_NAME}_haptile_tactile

LEROBOT_REPO_ID=local/pi0_ur5e_${DATASET_NAME}_tactile_raw
DATASET_ROOT=${PROJECT_ROOT}/outputs/${DATASET_NAME}_lerobot_tactile_raw
OUTPUT_DIR=${PROJECT_ROOT}/outputs/pi0_${DATASET_NAME}_haptile_tactile

WANDB=true
OVERWRITE=true                        # set RESUME=true (and OVERWRITE irrelevant) to resume a crashed run
RESUME=false
STEPS=30000
BATCH_SIZE=16
SAVE_INTERVAL=3000                    # checkpoint every N steps; train_haptile_tactile_pytorch.py
                                       # prunes old ones automatically (keeps the last couple in
                                       # full for --resume, plus model-only keep_period milestones)
                                       # so this no longer grows disk usage without bound
TACTILE_EXPERT_VARIANT=gemma_300m     # only field env-overridable here -- see note above
PI05=false                            # plain pi0 (this project's convention); true = pi0.5 -- see note above
LOAD_T3_CHECKPOINT=true               # fine-tune the tactile ViT encoder from a pretrained T3
                                       # checkpoint rather than random init -- verified working
T3_SENSOR_NAME=gs_tag                 # marker/dot-pattern GelSight gel -- must match your actual
                                       # sensor hardware, not just the "GelSight" brand name
T3_CACHE_DIR=${PROJECT_ROOT}/shared/t3_cache  # shared across experiments/EXP_NAMEs, not scoped to
                                       # OUTPUT_DIR -- downloaded once (~84MB), reused by every run
                                       # instead of every new experiment re-downloading its own copy
PYTORCH_WEIGHT_PATH=/scratch/users/k2691893/projects/openpi/openpi-assets/checkpoints/pi0_base_pytorch             # e.g. ~/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch
                                       # -- seeds the VLM/action-expert backbone from a pretrained
                                       # checkpoint and enables LoRA on it (see note above); leave
                                       # empty to train the whole backbone from scratch instead
VISION_TOWER_MODE=lora                 # "full" (default, matches JAX precedent) / "lora" / "frozen"
                                       # -- only valid if PYTORCH_WEIGHT_PATH is set, see note above
VISION_LORA_RANK=16                    # only used if VISION_TOWER_MODE=lora
VISION_LORA_ALPHA=16.0                 # only used if VISION_TOWER_MODE=lora

cd "${OPENPI_ROOT}"
source /scratch/users/k2691893/miniconda3/etc/profile.d/conda.sh   # CHANGE_ME: your conda.sh path
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
echo "WandB enabled: ${WANDB}"
echo "PI05: ${PI05}"
echo "Pretrained weight path (LoRA if set, full training from scratch if empty): ${PYTORCH_WEIGHT_PATH:-<none>}"
echo "T3 tactile encoder checkpoint: load=${LOAD_T3_CHECKPOINT} sensor=${T3_SENSOR_NAME} cache_dir=${T3_CACHE_DIR}"
echo "Vision tower mode: ${VISION_TOWER_MODE} (rank=${VISION_LORA_RANK} alpha=${VISION_LORA_ALPHA} if lora)"
echo "================================"

echo "Checking GPU with nvidia-smi:"
nvidia-smi

echo "(Re-)installing Haptile's config splice blocks (idempotent, safe if already installed):"
python "${PROJECT_ROOT}/learning/pi0_ur5e/scripts/install_openpi_config.py" --openpi-root "${OPENPI_ROOT}"
python "${PROJECT_ROOT}/learning/pi0_ur5e/scripts/install_openpi_pytorch_patch.py" --openpi-root "${OPENPI_ROOT}"

echo "Linking the converted LeRobot dataset into a local HF_LEROBOT_HOME:"
mkdir -p "${OUTPUT_DIR}/logs"
LEROBOT_HOME_DIR="${OUTPUT_DIR}/lerobot_home"
LINK_PATH="${LEROBOT_HOME_DIR}/${LEROBOT_REPO_ID}"
mkdir -p "$(dirname "${LINK_PATH}")"
rm -f "${LINK_PATH}"
ln -s "${DATASET_ROOT}" "${LINK_PATH}"
export HF_LEROBOT_HOME="${LEROBOT_HOME_DIR}"

export PI0_UR5E_TACTILE_LEROBOT_REPO_ID="${LEROBOT_REPO_ID}"
export PI0_UR5E_TACTILE_ASSET_ID="${LEROBOT_REPO_ID}"
export PI0_UR5E_TACTILE_TRAIN_STEPS="${STEPS}"
export PI0_UR5E_TACTILE_BATCH_SIZE="${BATCH_SIZE}"
export PI0_UR5E_TACTILE_SAVE_INTERVAL="${SAVE_INTERVAL}"
export PI0_UR5E_TACTILE_EXPERT_VARIANT="${TACTILE_EXPERT_VARIANT}"
export PI0_UR5E_TACTILE_PI05="${PI05}"
export PI0_UR5E_TACTILE_LOAD_T3_CHECKPOINT="${LOAD_T3_CHECKPOINT}"
export PI0_UR5E_TACTILE_T3_SENSOR_NAME="${T3_SENSOR_NAME}"
export PI0_UR5E_TACTILE_T3_CACHE_DIR="${T3_CACHE_DIR}"
export PI0_UR5E_TACTILE_VISION_TOWER_MODE="${VISION_TOWER_MODE}"
export PI0_UR5E_TACTILE_VISION_LORA_RANK="${VISION_LORA_RANK}"
export PI0_UR5E_TACTILE_VISION_LORA_ALPHA="${VISION_LORA_ALPHA}"
export PI0_UR5E_TACTILE_ASSETS_BASE_DIR="${OUTPUT_DIR}/assets"
export PI0_UR5E_TACTILE_CHECKPOINT_BASE_DIR="${OUTPUT_DIR}/checkpoints"
export PI0_UR5E_DEFAULT_PROMPT="${DEFAULT_PROMPT}"
if [[ "${WANDB}" != "true" ]]; then
  export WANDB_MODE=disabled
fi

echo "Computing normalization stats (openpi's own compute_norm_stats.py, not the repo-local one --"
echo "the data loader below only reads stats written by this exact script):"
uv run python scripts/compute_norm_stats.py --config-name pi0_ur5e_cup_tactile \
  2>&1 | tee "${OUTPUT_DIR}/logs/compute_norm_stats.log"

TRAIN_CMD=(uv run python "${PROJECT_ROOT}/learning/pi0_ur5e/scripts/train_haptile_tactile_pytorch.py" \
  pi0_ur5e_cup_tactile --exp_name "${EXP_NAME}")
if [[ "${RESUME}" == "true" ]]; then
  TRAIN_CMD+=(--resume)
elif [[ "${OVERWRITE}" == "true" ]]; then
  TRAIN_CMD+=(--overwrite)
fi
if [[ "${WANDB}" != "true" ]]; then
  TRAIN_CMD+=(--no-wandb_enabled)
fi
if [[ -n "${PYTORCH_WEIGHT_PATH}" ]]; then
  TRAIN_CMD+=(--pytorch_weight_path "${PYTORCH_WEIGHT_PATH}")
fi

echo "Launching HaptileTactilePI0Pytorch training:"
printf '%q ' "${TRAIN_CMD[@]}"; printf '\n'
"${TRAIN_CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/logs/train.log"

echo "Job finished."

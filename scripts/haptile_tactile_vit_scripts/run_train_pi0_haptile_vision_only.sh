#!/bin/bash -l

#SBATCH --job-name=train_pi0_haptile_vision_only
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40g|h200|a100_80g|l40s"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%j.err
# CHANGE_ME: add --account/--qos/--exclude directives to match your cluster's current
# conventions (don't copy run_train_pi0_tactile_emb.sh's --exclude node list blindly -- check
# whether those nodes are still flaky before reusing it).
#
# Vision-only variant of run_train_pi0_haptile_tactile.sh: trains HaptileTactilePI0Pytorch with
# use_tactile_input=False, i.e. no tactile ViT encoder and no tactile-expert transformer branch --
# just the VLM + action-expert backbone, same as run_train_pi0_haptile_tactile.sh minus tactile.
# This is a structurally smaller model (FTP1PaliGemmaWithExpertModel runs with 2 experts instead
# of 3), not just a model that ignores tactile data -- see
# learning/pi0_ur5e/openpi_patches_pytorch/docs/ftp1_tactile_expert_port.md for the full design.
#
# No separate data conversion needed: this reuses the same tactile-raw dataset produced by
# run_convert_pi0_lerobot_tactile_raw.sh (DATASET_ROOT below). The data config still loads
# tactile images from disk (include_tactile_images=True is hardcoded there), but the model never
# constructs a tactile encoder/expert to consume them when use_tactile_input=False, so they're
# simply discarded after loading -- harmless, just a bit of unnecessary I/O.
#
# Run run_setup_openpi_haptile_env.sh once per fresh $OPENPI_ROOT checkout before the first job
# that uses this script (installs timm + peft + the transformers_replace patch; without it the
# model can't even construct).
#
# Note on LoRA: same as run_train_pi0_haptile_tactile.sh -- there's no LoRA toggle here, it's
# implied by whether PYTORCH_WEIGHT_PATH is set below. paligemma_variant="gemma_2b_lora" and
# action_expert_variant="gemma_300m_lora" are hardcoded in haptile_train_config_patch.py, but the
# LoRA freeze+adapter split only actually gets applied when a pretrained checkpoint is loaded
# (freezing a randomly-initialized backbone would badly undertrain it) -- see
# learning/pi0_ur5e/openpi_patches_pytorch/docs/ftp1_tactile_expert_port.md's "LoRA +
# pretrained-weight loading" section for how to produce a checkpoint to point PYTORCH_WEIGHT_PATH
# at (examples/convert_jax_model_to_pytorch.py, run from $OPENPI_ROOT).
#
# Note on PI05 below: matches run_train_pi0_haptile_tactile.sh's PI05 env var. Defaults to false
# (plain pi0) -- this project's actual convention across every task.
#
# Note on VISION_TOWER_MODE below: for small datasets (a handful of demos per task), fully
# fine-tuning the ~400M-param pretrained vision tower for thousands of steps risks catastrophic
# forgetting/overfitting -- "lora"/"frozen" trade some vision adaptation capacity for protection
# against that. Only valid together with PYTORCH_WEIGHT_PATH being set (adapting/freezing a
# randomly-initialized vision tower would never learn anything useful) -- the training script
# errors clearly if you set this without also setting PYTORCH_WEIGHT_PATH.
#
# Tactile-specific variables from run_train_pi0_haptile_tactile.sh (TACTILE_EXPERT_VARIANT,
# LOAD_T3_CHECKPOINT, T3_SENSOR_NAME, T3_CACHE_DIR) are intentionally omitted below: none of them
# have any effect when use_tactile_input=False (the tactile expert/encoder are never
# constructed), so setting them here would just be misleading.

set -e

PROJECT_ROOT=/scratch/grp/luo/Amir/TFM_ViT2                # e.g. /scratch/grp/luo/<you>/project/tele-gsy
OPENPI_ROOT=/scratch/users/k2691893/projects/openpi                  # e.g. /scratch/grp/luo/<you>/project/openpi
DATASET_NAME=wipe_board                # must match run_convert_pi0_lerobot_tactile_raw.sh's DATASET_NAME
DEFAULT_PROMPT="Grab the sponge, wipe the markers on the white board and put the sponge back"            # must match run_convert_pi0_lerobot_tactile_raw.sh's DEFAULT_PROMPT
EXP_NAME=${DATASET_NAME}_haptile_vision_only                    # e.g. ${DATASET_NAME}_haptile_vision_only

LEROBOT_REPO_ID=local/pi0_ur5e_${DATASET_NAME}_tactile_raw
DATASET_ROOT=${PROJECT_ROOT}/outputs/${DATASET_NAME}_lerobot_tactile_raw
OUTPUT_DIR=${PROJECT_ROOT}/outputs/pi0_${DATASET_NAME}_haptile_vision_only
# ^ deliberately a different OUTPUT_DIR from run_train_pi0_haptile_tactile.sh's (which is
# pi0_${DATASET_NAME}_haptile_tactile) so the two experiments' checkpoints/logs/assets never
# collide even when run against the same DATASET_NAME.

WANDB=true
OVERWRITE=true                        # set RESUME=true (and OVERWRITE irrelevant) to resume a crashed run
RESUME=false
STEPS=30000
BATCH_SIZE=16
PI05=false                            # plain pi0 (this project's convention); true = pi0.5 -- see note above
PYTORCH_WEIGHT_PATH=/scratch/users/k2691893/projects/openpi/openpi-assets/checkpoints/pi0_base_pytorch                # e.g. ~/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch
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
echo "Tactile input: disabled (vision-only run -- no tactile encoder/expert constructed)"
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
export PI0_UR5E_TACTILE_PI05="${PI05}"
export PI0_UR5E_TACTILE_USE_TACTILE_INPUT=false
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
# Any extra arguments passed to this script (sbatch/srun/bash ... -- --num_train_steps 3) are
# forwarded straight to train_haptile_tactile_pytorch.py, appended last so they override the
# flags set above -- e.g. for a quick unattended dry run before a full sbatch submission:
#   srun --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=00:30:00 \
#     bash run_train_pi0_haptile_vision_only.sh --num_train_steps 3
TRAIN_CMD+=("$@")

echo "Launching HaptileTactilePI0Pytorch training (vision-only):"
printf '%q ' "${TRAIN_CMD[@]}"; printf '\n'
"${TRAIN_CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/logs/train.log"

echo "Job finished."

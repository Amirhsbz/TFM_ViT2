#!/bin/bash -l

#SBATCH --job-name=convert_pi0_tactile_markeroverlay
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%A_%a.out
#SBATCH --error=/scratch/users/k2691893/projects/tele-gsy/script_results/%x_%A_%a.err
# No --gres: marker tracking (OpenCV) and convert_to_lerobot.py are CPU-only. Add
# "#SBATCH --gres=gpu:1" back if your partition requires a GPU request.
#
# Batch version of run_convert_pi0_lerobot_tactile_raw_*.sh that converts several of Shiyi's
# datasets in one go AND swaps the raw tactile images for marker-tracking overlays first.
#
# USAGE (on the login node -- do NOT sbatch this file yourself, it submits itself):
#   bash run_convert_pi0_lerobot_tactile_markeroverlay_batch.sh                 # all TASKS below
#   bash run_convert_pi0_lerobot_tactile_markeroverlay_batch.sh peg_in_hole wipe_board_sep_train
#                                                                               # only these (matched
#                                                                               # against the source dir's basename)
#   bash run_convert_pi0_lerobot_tactile_markeroverlay_batch.sh --list          # show what would run
#   bash run_convert_pi0_lerobot_tactile_markeroverlay_batch.sh --local [tasks] # NO sbatch: run the
#                                     # selected datasets one after another in THIS shell -- use it
#                                     # inside an interactive allocation, never on the login node:
#     srun -p luo_gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00 --pty bash -l
#     bash run_convert_pi0_lerobot_tactile_markeroverlay_batch.sh --local
#   (start tmux/screen on the login node first so a dropped SSH doesn't kill the session)
#   --local writes one log per dataset to SLURM_LOG_DIR and keeps going if one dataset fails.
#
# Default (no --local): it submits ONE SLURM job array (one array element per dataset, up to MAX_PARALLEL at once).
# Each element does:
#   1. copy   ${SHIYI_ROOT}/<src>  ->  ${PROJECT_ROOT}/shared/data/bc_data/<task>_markeroverlay[_split]
#             (Shiyi's original is never touched -- the overlay tool edits trajectory.h5 in place)
#   2. overlay  Data_analysis/batch_replace_tactile_videos_with_marker_overlay.py on the copy
#   3. verify   every trajectory.h5 has both tactile streams tagged marker_tracking_overlay=True
#   4. convert  convert_to_lerobot.py --tactile-feature-mode raw_image (same flags as the
#               per-task run_convert_pi0_lerobot_tactile_raw_*.sh scripts)
#
# NAMING: source basename "<task>" or "<task>_train"/"<task>_test" becomes
#   DATASET_NAME = <task>_markeroverlay
#   output dir   = ${PROJECT_ROOT}/outputs/<task>_markeroverlay_lerobot_tactile_raw[_train|_test]
#   repo id      = local/pi0_ur5e_<task>_markeroverlay_tactile_raw[_train|_test]
# which is exactly the pattern the existing run_train_pi0_haptile_*.sh scripts derive from
# DATASET_NAME -- so to train on it, copy the task's train script and only change
#   DATASET_NAME=<task>_markeroverlay
#
# Every step is safe to re-run: the copy is skipped once complete, already-overlaid streams are
# skipped by the overlay tool, and the conversion overwrites its own output. If a job died in the
# middle of the overlay step, delete that copy dir and re-run to start the task from a fresh copy.

set -euo pipefail

PROJECT_ROOT=/scratch/grp/luo/Amir/TFM_ViT2
OPENPI_ROOT=/scratch/users/k2691893/projects/openpi
CONDA_SH=/scratch/users/k2691893/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=tele
SHIYI_ROOT=/scratch/grp/luo/shiyi/project/tele-gsy
COPY_ROOT=${PROJECT_ROOT}/shared/data/bc_data      # where the overlay copies are written
SUFFIX=_markeroverlay
MAX_PARALLEL=4                                      # array elements running at once
SLURM_LOG_DIR=/scratch/users/k2691893/projects/tele-gsy/script_results

CONFIG_PATH=${PROJECT_ROOT}/learning/pi0_ur5e/configs/dataset_schema.yaml
CONVERT_SCRIPT=${PROJECT_ROOT}/learning/pi0_ur5e/scripts/convert_to_lerobot.py
OVERLAY_SCRIPT=${PROJECT_ROOT}/Data_analysis/batch_replace_tactile_videos_with_marker_overlay.py

# One line per dataset:  "<source path relative to SHIYI_ROOT>|<language prompt>"
# Comment a line out to skip it. Prompts are copied from the per-task convert scripts.
TASKS=(
  # "shared/data/bc_data/move_disposable_cup_nov|Lift the disposable paper cup and place it on the red plate"
  # "shared/data/bc_data/turn_cleanser_bottle_nov|Pick up the cleanser bottle, tilt it over the blue bowl as if pouring, then place it back"
  "shared/data/bc_data/turn_cleanser_bottle_nov_postgrasp|Pick up the cleanser bottle, tilt it over the blue bowl as if pouring, then place it back"
  # "shared/data/bc_data/wipe_board_sep_postgrasp|Wipe the markers on the white board and put the sponge back"
  # CHANGE_ME: run_convert_pi0_lerobot_tactile_raw_Orange_picking.sh uses the wipe-board prompt
  # (looks like a copy-paste slip) -- fill in the real orange prompt before enabling this:
  # "shared/data/bc_data/Orange_picking|<orange picking prompt>"
)

# ---------------------------------------------------------------------------------------------
# Name derivation shared by the launcher and the workers.
# Sets: SRC_REL PROMPT SRC BASE TASK SPLIT DATASET_NAME COPY_DIR OUTPUT_ROOT REPO_ID
# ---------------------------------------------------------------------------------------------
resolve_entry() {
  local entry="$1"
  SRC_REL="${entry%%|*}"
  PROMPT="${entry#*|}"
  SRC_REL="${SRC_REL%/}"
  SRC="${SHIYI_ROOT}/${SRC_REL}"
  BASE="$(basename "${SRC_REL}")"
  SPLIT=""
  TASK="${BASE}"
  case "${BASE}" in
    *_train) SPLIT=train; TASK="${BASE%_train}" ;;
    *_test)  SPLIT=test;  TASK="${BASE%_test}" ;;
  esac
  DATASET_NAME="${TASK}${SUFFIX}"
  local split_sfx="${SPLIT:+_${SPLIT}}"
  COPY_DIR="${COPY_ROOT}/${DATASET_NAME}${split_sfx}"
  OUTPUT_ROOT="${PROJECT_ROOT}/outputs/${DATASET_NAME}_lerobot_tactile_raw${split_sfx}"
  REPO_ID="local/pi0_ur5e_${DATASET_NAME}_tactile_raw${split_sfx}"
}

# ---------------------------------------------------------------------------------------------
# One dataset: copy -> overlay -> verify -> convert. Always called in a subshell (cd / set -e /
# conda activate stay contained), from either a SLURM array element or the --local loop.
# ---------------------------------------------------------------------------------------------
run_one() {
  set -euo pipefail
  resolve_entry "$1"

source "${CONDA_SH}" || { echo "ERROR: cannot source ${CONDA_SH}" >&2; exit 1; }
conda activate "${CONDA_ENV}" || { echo "ERROR: conda activate ${CONDA_ENV} failed" >&2; exit 1; }

echo "================================"
echo "Job: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-local} on ${HOSTNAME}"
echo "Source:       ${SRC}"
echo "Overlay copy: ${COPY_DIR}"
echo "Output root:  ${OUTPUT_ROOT}"
echo "Repo id:      ${REPO_ID}"
echo "Task name:    ${DATASET_NAME}${SPLIT:+ (split=${SPLIT})}"
echo "Prompt:       ${PROMPT}"
echo "================================"

[[ -d "${SRC}" ]] || { echo "ERROR: source ${SRC} does not exist" >&2; exit 1; }
SRC_REAL="$(realpath "${SRC}")"
COPY_REAL="$(realpath -m "${COPY_DIR}")"
case "${COPY_REAL}/" in
  "${SRC_REAL}/"*|"$(realpath "${SHIYI_ROOT}")/"*)
    echo "ERROR: refusing to write overlays inside Shiyi's tree (${COPY_REAL})" >&2; exit 1 ;;
esac

# 1. Copy (resumable; .copy_complete marks a finished copy so later re-runs never rsync the raw
#    videos back over already-overlaid h5 files).
if [[ -f "${COPY_DIR}/.copy_complete" ]]; then
  echo "[1/4] Copy already complete, skipping."
else
  echo "[1/4] Copying source dataset..."
  mkdir -p "${COPY_DIR}" || exit 1
  rsync -a --info=stats1 "${SRC}/" "${COPY_DIR}/" \
    || { echo "ERROR: copy failed -- NOT marking it complete" >&2; exit 1; }
  date > "${COPY_DIR}/.copy_complete" || exit 1
fi

# 2. Overlay (in place on the copy; skips streams already tagged).
echo "[2/4] Replacing tactile videos with marker-tracking overlays..."
cd "${PROJECT_ROOT}" || exit 1
PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python "${OVERLAY_SCRIPT}" "${COPY_DIR}" \
  || { echo "ERROR: overlay step failed -- not converting" >&2; exit 1; }
# (every step below also has an explicit `|| exit 1`, so a failure can never fall through to the
# conversion even if `set -e` is not in effect)

# 3. Verify -- never convert a dataset that still has raw tactile frames in it.
echo "[3/4] Verifying overlay tags..."
COPY_DIR="${COPY_DIR}" python - <<'CHECK'
import os, sys
from pathlib import Path
import h5py

root = Path(os.environ["COPY_DIR"])
files = sorted(root.glob("*/trajectory.h5"))
bad = []
for p in files:
    with h5py.File(p, "r") as f:
        videos = f["videos"] if "videos" in f else {}
        for k in ("tactile_left_rgb", "tactile_right_rgb"):
            ds = videos[k] if k in videos else None
            if ds is None or not bool(ds.attrs.get("marker_tracking_overlay", False)):
                bad.append(f"{p.parent.name}:{k}")
print(f"{len(files)} trajectories, {len(bad)} untagged streams")
if not files or bad:
    print("Untagged:", *bad[:20], sep="\n  ")
    sys.exit(1)
CHECK
[[ $? -eq 0 ]] || { echo "ERROR: overlay verification failed -- not converting" >&2; exit 1; }

# 4. Convert (same flags as the per-task run_convert_pi0_lerobot_tactile_raw_*.sh scripts).
echo "[4/4] Converting to LeRobot format..."
cd "${OPENPI_ROOT}" || exit 1
uv run python "${CONVERT_SCRIPT}" \
  --input-root "${COPY_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --config "${CONFIG_PATH}" \
  --task-name "${DATASET_NAME}" \
  --repo-id "${REPO_ID}" \
  --default-prompt "${PROMPT}" \
  --action-mode joint_position_gripper \
  --include-tactile true \
  --tactile-feature-mode raw_image \
  --overwrite true \
  || { echo "ERROR: conversion failed" >&2; exit 1; }

OUTPUT_ROOT="${OUTPUT_ROOT}" uv run python - <<'CHECK'
import json, os
from pathlib import Path
info = json.loads((Path(os.environ["OUTPUT_ROOT"]) / "meta" / "info.json").read_text())
print("state shape:", info["features"]["observation.state"]["shape"])
print("action shape:", info["features"]["action"]["shape"])
print("episodes:", info.get("total_episodes"), "frames:", info.get("total_frames"))
for key in ("observation.images.tactile_left_rgb", "observation.images.tactile_right_rgb"):
    feat = info["features"].get(key)
    print(f"{key}:", feat["shape"] if feat else "MISSING")
CHECK

echo "Done: train with DATASET_NAME=${DATASET_NAME}"
}

# ---------------------------------------------------------------------------------------------
# Launcher mode: not inside an array job -> pick the selected entries and submit the array.
# ---------------------------------------------------------------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  LIST_ONLY=false
  LOCAL=false
  WANTED=()
  for arg in ${@+"$@"}; do
    case "${arg}" in
      --list) LIST_ONLY=true ;;
      --local) LOCAL=true ;;
      -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
      *) WANTED+=("${arg}") ;;
    esac
  done

  SELECTED=()
  for entry in "${TASKS[@]}"; do
    resolve_entry "${entry}"
    if (( ${#WANTED[@]} == 0 )); then
      SELECTED+=("${entry}")
    else
      for w in ${WANTED[@]+"${WANTED[@]}"}; do
        if [[ "${w}" == "${BASE}" ]]; then SELECTED+=("${entry}"); break; fi
      done
    fi
  done
  for w in ${WANTED[@]+"${WANTED[@]}"}; do
    found=false
    for entry in "${SELECTED[@]}"; do resolve_entry "${entry}"; [[ "${BASE}" == "${w}" ]] && found=true; done
    if [[ "${found}" != true ]]; then
      echo "ERROR: '${w}' is not in TASKS (match is on the source dir's basename)." >&2
      exit 1
    fi
  done
  if (( ${#SELECTED[@]} == 0 )); then
    echo "ERROR: no tasks selected." >&2
    exit 1
  fi

  echo "Selected ${#SELECTED[@]} dataset(s):"
  for i in "${!SELECTED[@]}"; do
    resolve_entry "${SELECTED[$i]}"
    printf '  [%d] %s\n      copy   -> %s\n      output -> %s\n      train with DATASET_NAME=%s\n' \
      "$i" "${SRC}" "${COPY_DIR}" "${OUTPUT_ROOT}" "${DATASET_NAME}"
  done
  [[ "${LIST_ONLY}" == true ]] && exit 0

  mkdir -p "${SLURM_LOG_DIR}"

  if [[ "${LOCAL}" == true ]]; then
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
      echo "WARNING: not inside a SLURM allocation -- this would run the whole conversion on the" >&2
      echo "         login node. Get a node first, e.g.:" >&2
      echo "         srun -p luo_gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00 --pty bash -l" >&2
      read -r -p "Continue anyway? [y/N] " ans
      [[ "${ans}" == [yY]* ]] || exit 1
    fi
    STAMP="$(date +%Y%m%d_%H%M%S)"
    FAILED=()
    for i in "${!SELECTED[@]}"; do
      resolve_entry "${SELECTED[$i]}"
      LOG="${SLURM_LOG_DIR}/convert_pi0_tactile_markeroverlay_local_${STAMP}_${i}_${BASE}.log"
      echo ""
      echo "######## [$((i + 1))/${#SELECTED[@]}] ${BASE}  (log: ${LOG})"
      # NOT `if ( run_one ... ); then`: bash disables `set -e` for everything inside an if
      # condition (even if re-enabled inside), so a failed step would fall through to the next one.
      set +e
      ( run_one "${SELECTED[$i]}" ) 2>&1 | tee "${LOG}"
      rc=${PIPESTATUS[0]}
      set -e
      if (( rc == 0 )); then
        echo "######## ${BASE}: OK"
      else
        echo "######## ${BASE}: FAILED (see ${LOG})"
        FAILED+=("${BASE}")
      fi
    done
    echo ""
    echo "Finished $(( ${#SELECTED[@]} - ${#FAILED[@]} ))/${#SELECTED[@]} dataset(s)."
    if (( ${#FAILED[@]} > 0 )); then
      echo "Failed: ${FAILED[*]}"
      echo "Re-run just those with: bash $0 --local ${FAILED[*]}"
      exit 1
    fi
    exit 0
  fi

  # Hand the selection to the array elements through a file (prompts contain spaces/commas,
  # which don't survive sbatch --export).
  SELECTION_FILE="${SLURM_LOG_DIR}/markeroverlay_selection_$(date +%Y%m%d_%H%M%S)_$$.txt"
  printf '%s\n' "${SELECTED[@]}" > "${SELECTION_FILE}"
  last=$(( ${#SELECTED[@]} - 1 ))
  sbatch --array="0-${last}%${MAX_PARALLEL}" \
    --export="ALL,MARKEROVERLAY_SELECTION_FILE=${SELECTION_FILE}" \
    "$0"
  echo "Selection saved to ${SELECTION_FILE}"
  exit 0
fi

# ---------------------------------------------------------------------------------------------
# Array-element mode: one array element = one dataset.
# ---------------------------------------------------------------------------------------------
: "${MARKEROVERLAY_SELECTION_FILE:?submit via 'bash $0', not sbatch directly}"
mapfile -t SELECTED < "${MARKEROVERLAY_SELECTION_FILE}"
( run_one "${SELECTED[${SLURM_ARRAY_TASK_ID}]}" )

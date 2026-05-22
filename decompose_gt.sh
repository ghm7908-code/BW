#!/bin/bash
#SBATCH --job-name=decomp-gt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# If your cluster requires an explicit CPU partition/account, uncomment and edit:
##SBATCH --partition=GEOG-HPC-CPU
##SBATCH --account=your-account

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/geogfs1/home/u3666068/BuildingWorld}"
CONDA_ENV="${CONDA_ENV:-BWformer}"
DATA_ROOT="${DATA_ROOT:-/geogfs1/groups/hkurs/u3666068mgh/BuildingWorld/Tokyo}"
WIREFRAME_SUBDIR="${WIREFRAME_SUBDIR:-wireframe/wireframe}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-decomp_gt_v1}"
SUMMARY_NAME="${SUMMARY_NAME:-decomp_gt_v1_summary.jsonl}"
FAILED_NAME="${FAILED_NAME:-decomp_gt_v1_failed.txt}"
NAMES_FILE="${NAMES_FILE:-}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
SAVE_DEBUG_OBJ="${SAVE_DEBUG_OBJ:-0}"
THETA_VERTICAL_DEG="${THETA_VERTICAL_DEG:-10.0}"
EPS_XY_RATIO="${EPS_XY_RATIO:-0.002}"
EPS_Z_RATIO="${EPS_Z_RATIO:-0.002}"
MIN_VERTICAL_SPAN_RATIO="${MIN_VERTICAL_SPAN_RATIO:-0.01}"
MIN_SUPPORT_SPAN_RATIO="${MIN_SUPPORT_SPAN_RATIO:-0.05}"

mkdir -p "${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"

set +u
if [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "Could not find conda initialization script." >&2
  exit 1
fi
set -u

conda activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Project dir: ${PROJECT_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Wireframe subdir: ${WIREFRAME_SUBDIR}"
echo "Output subdir: ${OUTPUT_SUBDIR}"
echo "Conda env: ${CONDA_ENV}"
echo "Limit: ${LIMIT}"
echo "Overwrite: ${OVERWRITE}"
echo "Save debug obj: ${SAVE_DEBUG_OBJ}"

CMD=(
  python decompose_wireframe_gt.py
  --dataset_root "${DATA_ROOT}"
  --wireframe_subdir "${WIREFRAME_SUBDIR}"
  --output_subdir "${OUTPUT_SUBDIR}"
  --summary_name "${SUMMARY_NAME}"
  --failed_name "${FAILED_NAME}"
  --limit "${LIMIT}"
  --theta_vertical_deg "${THETA_VERTICAL_DEG}"
  --eps_xy_ratio "${EPS_XY_RATIO}"
  --eps_z_ratio "${EPS_Z_RATIO}"
  --min_vertical_span_ratio "${MIN_VERTICAL_SPAN_RATIO}"
  --min_support_span_ratio "${MIN_SUPPORT_SPAN_RATIO}"
)

if [ -n "${NAMES_FILE}" ]; then
  CMD+=(--names_file "${NAMES_FILE}")
fi

if [ "${OVERWRITE}" = "1" ]; then
  CMD+=(--overwrite)
fi

if [ "${SAVE_DEBUG_OBJ}" = "1" ]; then
  CMD+=(--save_debug_obj)
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"

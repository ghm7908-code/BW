#!/bin/bash
#SBATCH --job-name=bwformer-tallin-roof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=GEOG-HPC-GPU

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/geogfs1/home/u3666068/BuildingWorld}"
CONDA_ENV="${CONDA_ENV:-BWformer}"
RAW_ROOT="${RAW_ROOT:-/geogfs1/groups/hkurs/u3666068mgh/Tallin}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
TEST_SPLIT="${TEST_SPLIT:-test}"
PC_SUBDIR="${PC_SUBDIR:-xyz}"
GT_SUBDIR="${GT_SUBDIR:-gt}"
FALLBACK_GT_SUBDIR="${FALLBACK_GT_SUBDIR:-wireframe}"

IMAGE_SIZE="${IMAGE_SIZE:-256}"
PROJECTION_MODE="${PROJECTION_MODE:-standard}"
TOP_BAND_PX="${TOP_BAND_PX:-8.0}"
BLUR_KERNEL="${BLUR_KERNEL:-5}"
HEIGHT_GAMMA="${HEIGHT_GAMMA:-0.55}"
VALID_FLOOR="${VALID_FLOOR:-0.18}"
OVERWRITE_PROJ="${OVERWRITE_PROJ:-1}"
OVERWRITE_PREPARE="${OVERWRITE_PREPARE:-0}"
BUILD_OP="${BUILD_OP:-0}"

EXP_NAME="${EXP_NAME:-tallin_roof_prior_scratch}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${RAW_ROOT}/bwformer_trainval_${IMAGE_SIZE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RAW_ROOT}/checkpoints/${EXP_NAME}}"
LOG_DIR="${LOG_DIR:-${RAW_ROOT}/tensorboard/${EXP_NAME}}"
RESUME="${RESUME:-}"
DEVICE="${DEVICE:-cuda:0}"

NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-650}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
LR_DROP="${LR_DROP:-600}"
MAX_CORNER_NUM="${MAX_CORNER_NUM:-150}"
CORNER_LIMIT="${CORNER_LIMIT:-150}"
CORNER_TO_EDGE_MULTIPLIER="${CORNER_TO_EDGE_MULTIPLIER:-3}"
LAMBDA_CORNER="${LAMBDA_CORNER:-0.05}"
LAMBDA_ROOF="${LAMBDA_ROOF:-0.05}"
SAVE_EVERY="${SAVE_EVERY:-20}"
INFER_TIMES="${INFER_TIMES:-3}"
CORNER_THRESH="${CORNER_THRESH:-0.01}"

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}" "${LOG_DIR}"

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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Project dir: ${PROJECT_DIR}"
echo "Raw root: ${RAW_ROOT}"
echo "Processed train/val root: ${PROCESSED_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "TensorBoard root: ${LOG_DIR}"
if [ -n "${RESUME}" ]; then
  echo "Resume checkpoint: ${RESUME}"
fi
echo "Projection mode: ${PROJECTION_MODE}"

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"

if [ "${BUILD_OP}" = "1" ]; then
  pushd models/ops >/dev/null
  pip install -v .
  popd >/dev/null
fi

resolve_gt_subdir() {
  local split_root="$1"
  if [ -d "${split_root}/${GT_SUBDIR}" ]; then
    echo "${GT_SUBDIR}"
  elif [ -d "${split_root}/${FALLBACK_GT_SUBDIR}" ]; then
    echo "${FALLBACK_GT_SUBDIR}"
  else
    echo "Could not find GT directory under ${split_root}. Tried ${GT_SUBDIR} and ${FALLBACK_GT_SUBDIR}." >&2
    exit 1
  fi
}

project_eval_split() {
  local split="$1"
  local split_root="${RAW_ROOT}/${split}"
  local split_gt_subdir
  split_gt_subdir="$(resolve_gt_subdir "${split_root}")"

  if [ ! -d "${split_root}/${PC_SUBDIR}" ]; then
    echo "Point cloud directory does not exist: ${split_root}/${PC_SUBDIR}" >&2
    exit 1
  fi

  local proj_args=(
    --dataset_root "${split_root}"
    --pc_subdir "${PC_SUBDIR}"
    --wireframe_subdir "${split_gt_subdir}"
    --rgb_subdir rgb
    --annot_subdir annot
    --vis_subdir vis
    --image_size "${IMAGE_SIZE}"
    --projection_mode "${PROJECTION_MODE}"
    --top_band_px "${TOP_BAND_PX}"
    --blur_kernel "${BLUR_KERNEL}"
    --height_gamma "${HEIGHT_GAMMA}"
    --valid_floor "${VALID_FLOOR}"
  )
  if [ "${OVERWRITE_PROJ}" = "1" ]; then
    proj_args+=(--overwrite)
  fi

  echo ""
  echo "[proj] ${split}: ${split_root}"
  python proj.py "${proj_args[@]}"
}

evaluate_split() {
  local split="$1"
  local checkpoint_path="$2"
  local split_root="${RAW_ROOT}/${split}"
  local split_gt_subdir
  split_gt_subdir="$(resolve_gt_subdir "${split_root}")"

  local names_file="${split_root}/all_list.txt"
  local result_dir="${split_root}/pred_wireframe_${EXP_NAME}"

  if [ ! -f "${names_file}" ]; then
    echo "Missing projected names file: ${names_file}" >&2
    exit 1
  fi

  mkdir -p "${result_dir}"

  echo ""
  echo "[infer] ${split}: ${checkpoint_path}"
  python infer.py \
    --checkpoint_path "${checkpoint_path}" \
    --data_path "${split_root}" \
    --test_list "${names_file}" \
    --pc_root "${split_root}/${PC_SUBDIR}" \
    --result_dir "${result_dir}" \
    --image_size "${IMAGE_SIZE}" \
    --infer_times "${INFER_TIMES}" \
    --num_workers "${NUM_WORKERS}" \
    --corner_thresh "${CORNER_THRESH}" \
    --device "${DEVICE}"

  echo ""
  echo "[eval] ${split}"
  python evaluate_wireframe.py \
    --pred_dir "${result_dir}" \
    --gt_dir "${split_root}/${split_gt_subdir}" \
    --names_file "${names_file}" \
    --output_json "${result_dir}/wireframe_eval.json" \
    --output_csv "${result_dir}/wireframe_eval.csv"
}

echo ""
echo "[1/4] Project val and test splits"
project_eval_split "${VAL_SPLIT}"
project_eval_split "${TEST_SPLIT}"

prepare_args=(
  --raw_root "${RAW_ROOT}"
  --output_root "${PROCESSED_ROOT}"
  --train_split "${TRAIN_SPLIT}"
  --val_split "${VAL_SPLIT}"
  --pc_subdir "${PC_SUBDIR}"
  --wireframe_subdir "${GT_SUBDIR}"
  --fallback_wireframe_subdir "${FALLBACK_GT_SUBDIR}"
  --image_size "${IMAGE_SIZE}"
  --projection_mode "${PROJECTION_MODE}"
  --top_band_px "${TOP_BAND_PX}"
  --blur_kernel "${BLUR_KERNEL}"
  --height_gamma "${HEIGHT_GAMMA}"
  --valid_floor "${VALID_FLOOR}"
)
if [ "${OVERWRITE_PREPARE}" = "1" ]; then
  prepare_args+=(--overwrite)
fi

echo ""
echo "[2/4] Prepare train/val unified training root"
python prepare_tallinn_trainval_for_bwformer.py "${prepare_args[@]}"

train_list="${PROCESSED_ROOT}/train_list.txt"
val_list="${PROCESSED_ROOT}/valid_list.txt"
if [ ! -f "${train_list}" ] || [ ! -f "${val_list}" ]; then
  echo "Missing split files under ${PROCESSED_ROOT}" >&2
  exit 1
fi

echo ""
train_args=(
  --data_path "${PROCESSED_ROOT}"
  --train_list "${train_list}"
  --val_list "${val_list}"
  --run_validation
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${LOG_DIR}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --val_batch_size "${VAL_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --lr_drop "${LR_DROP}"
  --image_size "${IMAGE_SIZE}"
  --max_corner_num "${MAX_CORNER_NUM}"
  --corner_limit "${CORNER_LIMIT}"
  --corner_to_edge_multiplier "${CORNER_TO_EDGE_MULTIPLIER}"
  --lambda_corner "${LAMBDA_CORNER}"
  --lambda_roof "${LAMBDA_ROOF}"
  --save_every "${SAVE_EVERY}"
)
if [ -n "${RESUME}" ]; then
  if [ ! -f "${RESUME}" ]; then
    echo "Resume checkpoint not found: ${RESUME}" >&2
    exit 1
  fi
  train_args+=(--resume "${RESUME}")
  echo "[3/4] Resume training"
else
  echo "[3/4] Train from scratch"
fi

python train.py \
  "${train_args[@]}"

checkpoint_best="${OUTPUT_DIR}/checkpoint_best.pth"
checkpoint_last="${OUTPUT_DIR}/checkpoint.pth"
if [ -f "${checkpoint_best}" ]; then
  eval_ckpt="${checkpoint_best}"
elif [ -f "${checkpoint_last}" ]; then
  eval_ckpt="${checkpoint_last}"
else
  echo "No checkpoint found under ${OUTPUT_DIR}" >&2
  exit 1
fi

echo ""
echo "[4/4] Evaluate trained checkpoint on val and test"
evaluate_split "${VAL_SPLIT}" "${eval_ckpt}"
evaluate_split "${TEST_SPLIT}" "${eval_ckpt}"

echo ""
echo "Done"
echo "Checkpoint: ${eval_ckpt}"
echo "Val predictions: ${RAW_ROOT}/${VAL_SPLIT}/pred_wireframe_${EXP_NAME}"
echo "Test predictions: ${RAW_ROOT}/${TEST_SPLIT}/pred_wireframe_${EXP_NAME}"

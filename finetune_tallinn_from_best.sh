#!/bin/bash
#SBATCH --job-name=bwformer-tallinn-finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=GEOG-HPC-GPU

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/geogfs1/home/u3666068/BuildingWorld}"
CONDA_ENV="${CONDA_ENV:-BWformer}"
RAW_ROOT="${RAW_ROOT:-/geogfs1/groups/hkurs/u3666068mgh/Tallin}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
PC_SUBDIR="${PC_SUBDIR:-xyz}"
WIREFRAME_SUBDIR="${WIREFRAME_SUBDIR:-wireframe}"
FALLBACK_WIREFRAME_SUBDIR="${FALLBACK_WIREFRAME_SUBDIR:-gt}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${RAW_ROOT}/bwformer_trainval_256}"
WEIGHTS_CKPT="${WEIGHTS_CKPT:-${PROJECT_DIR}/checkpoints/checkpoint_best.pth}"
EXP_NAME="${EXP_NAME:-tallinn_finetune_from_best}"
OUTPUT_DIR="${OUTPUT_DIR:-${RAW_ROOT}/checkpoints/${EXP_NAME}}"
LOG_DIR="${LOG_DIR:-${RAW_ROOT}/tensorboard/${EXP_NAME}}"

# Finetuning knobs
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
EPOCHS="${EPOCHS:-30}"
LR_DROP="${LR_DROP:-20}"
MAX_CORNER_NUM="${MAX_CORNER_NUM:-150}"
CORNER_LIMIT="${CORNER_LIMIT:-128}"
CORNER_TO_EDGE_MULTIPLIER="${CORNER_TO_EDGE_MULTIPLIER:-2}"
LAMBDA_CORNER="${LAMBDA_CORNER:-0.01}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-10}"
SAVE_EVERY="${SAVE_EVERY:-5}"
DEVICE="${DEVICE:-cuda:0}"
PREPARE_DATA="${PREPARE_DATA:-1}"
OVERWRITE_PREPARE="${OVERWRITE_PREPARE:-0}"
BUILD_OP="${BUILD_OP:-0}"

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
echo "Processed root: ${PROCESSED_ROOT}"
echo "Weights checkpoint: ${WEIGHTS_CKPT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "TensorBoard root: ${LOG_DIR}"
echo "Conda env: ${CONDA_ENV}"

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"

if [ ! -f "${WEIGHTS_CKPT}" ]; then
  echo "Weights checkpoint does not exist: ${WEIGHTS_CKPT}" >&2
  exit 1
fi

if [ "${BUILD_OP}" = "1" ]; then
  pushd models/ops >/dev/null
  pip install -v .
  popd >/dev/null
fi

if [ "${PREPARE_DATA}" = "1" ]; then
  PREPARE_ARGS=(
    --raw_root "${RAW_ROOT}"
    --output_root "${PROCESSED_ROOT}"
    --train_split "${TRAIN_SPLIT}"
    --val_split "${VAL_SPLIT}"
    --pc_subdir "${PC_SUBDIR}"
    --wireframe_subdir "${WIREFRAME_SUBDIR}"
    --fallback_wireframe_subdir "${FALLBACK_WIREFRAME_SUBDIR}"
    --image_size "${IMAGE_SIZE}"
  )
  if [ "${OVERWRITE_PREPARE}" = "1" ]; then
    PREPARE_ARGS+=(--overwrite)
  fi

  echo ""
  echo "[1/2] Prepare unified BWFormer train/val root"
  python prepare_tallinn_trainval_for_bwformer.py "${PREPARE_ARGS[@]}"
fi

TRAIN_LIST="${PROCESSED_ROOT}/train_list.txt"
VAL_LIST="${PROCESSED_ROOT}/valid_list.txt"

if [ ! -f "${TRAIN_LIST}" ] || [ ! -f "${VAL_LIST}" ]; then
  echo "Missing split files under ${PROCESSED_ROOT}. Expected ${TRAIN_LIST} and ${VAL_LIST}." >&2
  exit 1
fi

CKPT_MAX_CORNER_NUM="$(
  CHECKPOINT_PATH="${WEIGHTS_CKPT}" python - <<'PY'
import argparse
import os
import torch

path = os.environ["CHECKPOINT_PATH"]
try:
    ckpt = torch.load(path, map_location="cpu")
except Exception:
    from torch.serialization import safe_globals
    with safe_globals([argparse.Namespace]):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

ckpt_args = ckpt.get("args", None)
value = getattr(ckpt_args, "max_corner_num", 150)
print(int(value))
PY
)"

if [ "${MAX_CORNER_NUM}" != "${CKPT_MAX_CORNER_NUM}" ]; then
  echo "MAX_CORNER_NUM=${MAX_CORNER_NUM} does not match checkpoint max_corner_num=${CKPT_MAX_CORNER_NUM}." >&2
  echo "Please keep MAX_CORNER_NUM=${CKPT_MAX_CORNER_NUM} when finetuning from this checkpoint." >&2
  exit 1
fi

echo ""
echo "[2/2] Finetune BWFormer from pretrained weights"
echo "Epochs: ${EPOCHS}"
echo "Learning rate: ${LR}"
echo "Weight decay: ${WEIGHT_DECAY}"
echo "Batch size: ${BATCH_SIZE}"
echo "Validation batch size: ${VAL_BATCH_SIZE}"
echo "Max corner num: ${MAX_CORNER_NUM}"
echo "Corner limit: ${CORNER_LIMIT}"
echo "Corner-to-edge multiplier: ${CORNER_TO_EDGE_MULTIPLIER}"
echo "Lambda corner: ${LAMBDA_CORNER}"
echo "Freeze backbone epochs: ${FREEZE_BACKBONE_EPOCHS}"

python train.py \
  --data_path "${PROCESSED_ROOT}" \
  --train_list "${TRAIN_LIST}" \
  --val_list "${VAL_LIST}" \
  --run_validation \
  --load_weights "${WEIGHTS_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --lr_drop "${LR_DROP}" \
  --image_size "${IMAGE_SIZE}" \
  --max_corner_num "${MAX_CORNER_NUM}" \
  --corner_limit "${CORNER_LIMIT}" \
  --corner_to_edge_multiplier "${CORNER_TO_EDGE_MULTIPLIER}" \
  --lambda_corner "${LAMBDA_CORNER}" \
  --freeze_backbone_epochs "${FREEZE_BACKBONE_EPOCHS}" \
  --save_every "${SAVE_EVERY}"

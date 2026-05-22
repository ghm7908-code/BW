#!/bin/bash
#SBATCH --job-name=bwformer-tallin-train
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

IMAGE_SIZE="${IMAGE_SIZE:-256}"
EXP_NAME="${EXP_NAME:-tallin_roof_prior_scratch}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${RAW_ROOT}/bwformer_trainval_${IMAGE_SIZE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RAW_ROOT}/checkpoints/${EXP_NAME}}"
LOG_DIR="${LOG_DIR:-${RAW_ROOT}/tensorboard/${EXP_NAME}}"
RESUME="${RESUME:-}"
LOAD_WEIGHTS="${LOAD_WEIGHTS:-}"
DEVICE="${DEVICE:-cuda:0}"

NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
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
RUN_VALIDATION="${RUN_VALIDATION:-1}"
VAL_EVERY="${VAL_EVERY:-10}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
PRINT_FREQ="${PRINT_FREQ:-200}"

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

train_list="${TRAIN_LIST:-${PROCESSED_ROOT}/train_list.txt}"
val_list="${VAL_LIST:-${PROCESSED_ROOT}/valid_list.txt}"

if [ ! -d "${PROCESSED_ROOT}" ]; then
  echo "Processed root not found: ${PROCESSED_ROOT}" >&2
  echo "Run the preprocessing script once before using this train-only script." >&2
  exit 1
fi

if [ ! -f "${train_list}" ]; then
  echo "Missing train list: ${train_list}" >&2
  exit 1
fi

if [ "${RUN_VALIDATION}" = "1" ] && [ ! -f "${val_list}" ]; then
  echo "Missing validation list: ${val_list}" >&2
  exit 1
fi

if [ -n "${RESUME}" ] && [ -n "${LOAD_WEIGHTS}" ]; then
  echo "Use either RESUME or LOAD_WEIGHTS, not both." >&2
  exit 1
fi

if [ -n "${RESUME}" ] && [ ! -f "${RESUME}" ]; then
  echo "Resume checkpoint not found: ${RESUME}" >&2
  exit 1
fi

if [ -n "${LOAD_WEIGHTS}" ] && [ ! -f "${LOAD_WEIGHTS}" ]; then
  echo "Weights checkpoint not found: ${LOAD_WEIGHTS}" >&2
  exit 1
fi

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Project dir: ${PROJECT_DIR}"
echo "Processed root: ${PROCESSED_ROOT}"
echo "Train list: ${train_list}"
echo "Validation list: ${val_list}"
echo "Output dir: ${OUTPUT_DIR}"
echo "TensorBoard root: ${LOG_DIR}"
echo "Run validation: ${RUN_VALIDATION}"
echo "Validation interval: ${VAL_EVERY}"
echo "Max train samples: ${MAX_TRAIN_SAMPLES}"
echo "Max validation samples: ${MAX_VAL_SAMPLES}"
echo "Sample seed: ${SAMPLE_SEED}"
if [ -n "${RESUME}" ]; then
  echo "Resume checkpoint: ${RESUME}"
fi
if [ -n "${LOAD_WEIGHTS}" ]; then
  echo "Load weights: ${LOAD_WEIGHTS}"
fi

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"

train_help="$(python train.py -h 2>&1 || true)"

train_args=(
  --data_path "${PROCESSED_ROOT}"
  --train_list "${train_list}"
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${LOG_DIR}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --val_batch_size "${VAL_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --print_freq "${PRINT_FREQ}"
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

if echo "${train_help}" | grep -q -- "--val_every"; then
  train_args+=(--val_every "${VAL_EVERY}")
else
  echo "train.py does not support --val_every; skipping validation interval control."
fi

if echo "${train_help}" | grep -q -- "--max_train_samples"; then
  train_args+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
else
  echo "train.py does not support --max_train_samples; using all training samples."
fi

if echo "${train_help}" | grep -q -- "--max_val_samples"; then
  train_args+=(--max_val_samples "${MAX_VAL_SAMPLES}")
else
  echo "train.py does not support --max_val_samples; using all validation samples."
fi

if echo "${train_help}" | grep -q -- "--sample_seed"; then
  train_args+=(--sample_seed "${SAMPLE_SEED}")
fi

if [ "${RUN_VALIDATION}" = "1" ]; then
  train_args+=(--val_list "${val_list}" --run_validation)
fi

if [ -n "${RESUME}" ]; then
  train_args+=(--resume "${RESUME}")
fi

if [ -n "${LOAD_WEIGHTS}" ]; then
  train_args+=(--load_weights "${LOAD_WEIGHTS}")
fi

echo ""
echo "[train-only] Start training"
python train.py "${train_args[@]}"

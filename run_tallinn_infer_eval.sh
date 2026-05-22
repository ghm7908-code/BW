#!/bin/bash
#SBATCH --job-name=bwformer-tallinn-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=GEOG-HPC-GPU

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/geogfs1/home/u3666068/BuildingWorld}"
CONDA_ENV="${CONDA_ENV:-BWformer}"
DATASET_ROOT="${DATASET_ROOT:-/geogfs1/groups/hkurs/u3666068mgh/Tallin}"
TEST_SPLIT="${TEST_SPLIT:-test}"
TEST_SPLIT_ROOT="${TEST_SPLIT_ROOT:-${DATASET_ROOT}/${TEST_SPLIT}}"
PC_SUBDIR="${PC_SUBDIR:-xyz}"
GT_SUBDIR="${GT_SUBDIR:-wireframe}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_DIR}/checkpoints/checkpoint_best.pth}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
INFER_TIMES="${INFER_TIMES:-3}"
CORNER_THRESH="${CORNER_THRESH:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda:0}"
RESULT_DIR="${RESULT_DIR:-${TEST_SPLIT_ROOT}/pred_wireframe_bwformer}"
EVAL_JSON="${EVAL_JSON:-${RESULT_DIR}/wireframe_eval.json}"
EVAL_CSV="${EVAL_CSV:-${RESULT_DIR}/wireframe_eval.csv}"
BUILD_OP="${BUILD_OP:-0}"
OVERWRITE_PROJ="${OVERWRITE_PROJ:-1}"

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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Project dir: ${PROJECT_DIR}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Test split root: ${TEST_SPLIT_ROOT}"
echo "Checkpoint: ${CHECKPOINT_PATH}"

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"

if [ ! -d "${TEST_SPLIT_ROOT}" ]; then
  echo "Test split directory does not exist: ${TEST_SPLIT_ROOT}" >&2
  exit 1
fi

if [ ! -d "${TEST_SPLIT_ROOT}/${GT_SUBDIR}" ]; then
  if [ -d "${TEST_SPLIT_ROOT}/gt" ]; then
    GT_SUBDIR="gt"
  elif [ -d "${TEST_SPLIT_ROOT}/wireframe" ]; then
    GT_SUBDIR="wireframe"
  else
    echo "Could not find GT wireframe directory under ${TEST_SPLIT_ROOT}. Tried ${GT_SUBDIR}, gt, wireframe." >&2
    exit 1
  fi
fi

if [ ! -d "${TEST_SPLIT_ROOT}/${PC_SUBDIR}" ]; then
  echo "Point cloud directory does not exist: ${TEST_SPLIT_ROOT}/${PC_SUBDIR}" >&2
  exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "Checkpoint file does not exist: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [ "${BUILD_OP}" = "1" ]; then
  pushd models/ops >/dev/null
  pip install -v .
  popd >/dev/null
fi

PROJ_ARGS=(
  --dataset_root "${TEST_SPLIT_ROOT}"
  --pc_subdir "${PC_SUBDIR}"
  --wireframe_subdir "${GT_SUBDIR}"
  --rgb_subdir rgb
  --annot_subdir annot
  --vis_subdir vis
  --image_size "${IMAGE_SIZE}"
)

if [ "${OVERWRITE_PROJ}" = "1" ]; then
  PROJ_ARGS+=(--overwrite)
fi

echo ""
echo "[1/3] Project supervised test split into BWFormer RGB inputs"
python proj.py "${PROJ_ARGS[@]}"

NAMES_FILE="${TEST_SPLIT_ROOT}/all_list.txt"
if [ ! -f "${NAMES_FILE}" ]; then
  echo "Projection did not generate ${NAMES_FILE}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo ""
echo "[2/3] Run BWFormer inference"
python infer.py \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --data_path "${TEST_SPLIT_ROOT}" \
  --test_list "${NAMES_FILE}" \
  --pc_root "${TEST_SPLIT_ROOT}/${PC_SUBDIR}" \
  --result_dir "${RESULT_DIR}" \
  --image_size "${IMAGE_SIZE}" \
  --infer_times "${INFER_TIMES}" \
  --num_workers "${NUM_WORKERS}" \
  --corner_thresh "${CORNER_THRESH}" \
  --device "${DEVICE}"

echo ""
echo "[3/3] Evaluate predicted wireframes"
python evaluate_wireframe.py \
  --pred_dir "${RESULT_DIR}" \
  --gt_dir "${TEST_SPLIT_ROOT}/${GT_SUBDIR}" \
  --names_file "${NAMES_FILE}" \
  --output_json "${EVAL_JSON}" \
  --output_csv "${EVAL_CSV}"

echo ""
echo "Done"
echo "Predictions: ${RESULT_DIR}"
echo "Evaluation JSON: ${EVAL_JSON}"
echo "Evaluation CSV: ${EVAL_CSV}"

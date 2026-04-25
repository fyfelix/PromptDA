#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL_PATH=${1:-"ckpts/model.ckpt"}
DEFAULT_DATASET_PATH=${PROMPTDA_DATASET_PATH:-"data/HAMMER/test.jsonl"}
DATASET_PATH=${2:-"${DEFAULT_DATASET_PATH}"}
CAMERA_TYPE=${3:-"d435"}
ENCODER=${PROMPTDA_ENCODER:-"vitl"}
PYTHON_BIN=${PYTHON_BIN:-python3}
RESIZE_METHOD=${PROMPTDA_RESIZE_METHOD:-upper_bound}

# Supports the LingBot-style call:
#   bash scripts/run_bs.sh ckpt data/HAMMER/test.jsonl d435 --align
# and the CDM-style call:
#   bash scripts/run_bs.sh ckpt vitl d435 upper_bound false true
if [[ $# -ge 2 && "${2}" =~ ^(auto|vits|vitb|vitl|vitg)$ ]]; then
    DATASET_PATH="${DEFAULT_DATASET_PATH}"
    if [[ "${2}" != "auto" ]]; then
        ENCODER="${2}"
    fi
    CAMERA_TYPE=${3:-"d435"}
    RESIZE_METHOD=${4:-"${RESIZE_METHOD}"}
    if [[ $# -ge 5 && "${5}" =~ ^(auto|true|false)$ ]]; then
        EVAL_EXTRA_ARGS=("${@:7}")
    else
        EVAL_EXTRA_ARGS=("${@:5}")
    fi
else
    if [[ $# -ge 4 && "${4}" =~ ^(vits|vitb|vitl|vitg)$ ]]; then
        ENCODER="${4}"
        EVAL_EXTRA_ARGS=("${@:5}")
    else
        EVAL_EXTRA_ARGS=("${@:4}")
    fi
fi

echo "Using Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Camera Type: ${CAMERA_TYPE}"
echo "Encoder: ${ENCODER}"
echo "Python: ${PYTHON_BIN}"
echo "Resize Method: ${RESIZE_METHOD}"

MODEL_NAME=$(basename "${MODEL_PATH}")
MODEL_STUB="${MODEL_NAME%%.*}"
DIRNAME=$(dirname "${MODEL_PATH}")

OUTPUT_DIR=${PROMPTDA_OUTPUT_DIR:-"${DIRNAME}/eval_hammer_${MODEL_STUB}_${CAMERA_TYPE}"}
echo "Output Directory: ${OUTPUT_DIR}"

BS=${PROMPTDA_BS:-16}
NUM_WORKERS=${PROMPTDA_NUM_WORKERS:-8}
DEPTH_SCALE=${PROMPTDA_DEPTH_SCALE:-1000.0}
MIN_DEPTH=${PROMPTDA_MIN_DEPTH:-0.1}
MAX_DEPTH=${PROMPTDA_MAX_DEPTH:-6.0}
INPUT_SIZE=${PROMPTDA_INPUT_SIZE:-1008}
PREDICTION_RESIZE_MODE=${PROMPTDA_PREDICTION_RESIZE_MODE:-bilinear}

echo "Batch Size: ${BS}"
echo "Workers: ${NUM_WORKERS}"
echo "Depth Scale: ${DEPTH_SCALE}"
echo "Eval Depth Range Defaults: [${MIN_DEPTH}, ${MAX_DEPTH}]"
echo "Raw Prompt Depth Max: ${MAX_DEPTH}"
echo "PromptDA Input Size: ${INPUT_SIZE}"
echo "Prediction Resize Mode: ${PREDICTION_RESIZE_MODE}"

echo "[1/2] Running PromptDA batch inference..."
time "${PYTHON_BIN}" scripts/infer_dataset_bs.py \
    --model-path "${MODEL_PATH}" \
    --dataset "${DATASET_PATH}" \
    --raw-type "${CAMERA_TYPE}" \
    --output "${OUTPUT_DIR}" \
    --encoder "${ENCODER}" \
    --depth-scale "${DEPTH_SCALE}" \
    --min-depth "${MIN_DEPTH}" \
    --max-depth "${MAX_DEPTH}" \
    --input-size "${INPUT_SIZE}" \
    --resize-method "${RESIZE_METHOD}" \
    --prediction-resize-mode "${PREDICTION_RESIZE_MODE}" \
    --batch-size "${BS}" \
    --num-workers "${NUM_WORKERS}"

echo "[2/2] Evaluating benchmark metrics..."
time "${PYTHON_BIN}" scripts/eval_mp.py \
    --dataset "${DATASET_PATH}" \
    --output "${OUTPUT_DIR}" \
    --raw-type "${CAMERA_TYPE}" \
    --depth-scale "${DEPTH_SCALE}" \
    --min-depth "${MIN_DEPTH}" \
    --max-depth "${MAX_DEPTH}" \
    --batch-size "${BS}" \
    --num-workers "${NUM_WORKERS}" \
    "${EVAL_EXTRA_ARGS[@]}"

echo "Evaluation completed. Results saved in ${OUTPUT_DIR}"

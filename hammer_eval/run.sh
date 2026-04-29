#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

MODEL_PATH="ckpts/promptda_vitl.ckpt"
ENCODER="vitl"
DATASET="data/HAMMER/test.jsonl"
RAW_TYPE="d435"
OUTPUT=""
DEVICE="auto"
DEPTH_SCALE="1000.0"
MAX_DEPTH="6.0"
MAX_SIZE="1008"
SAMPLE_CHECK_LIMIT="16"
BATCH_SIZE="1"
EVAL_BATCH_SIZE="32"
LIMIT="0"
DRY_RUN="false"
CLEANUP_NPY="false"

usage() {
    cat <<'USAGE'
Usage:
  bash hammer_eval/run.sh [model_path] [raw_type] [options]

Options:
  --model-path PATH          Local PromptDA checkpoint.
  --encoder vits|vitb|vitl|vitg
                             PromptDA encoder.
  --dataset PATH             HAMMER jsonl path.
  --raw-type d435|l515|tof   Raw depth sensor field.
  --output DIR               Prediction/evaluation output directory.
  --device auto|cuda|cpu|mps Inference and metric device.
  --depth-scale FLOAT        Depth PNG scale factor.
  --max-depth FLOAT          Maximum valid raw depth in meters.
  --max-size N               Maximum model input side before 14x resizing.
  --batch-size N             Inference batch size for same-shaped model inputs.
  --eval-batch-size N        Metric evaluation batch size.
  --sample-check-limit N     Number of samples checked in dry-run.
  --limit N                  Debug sample limit; 0 means all.
  --dry-run                  Validate only; do not load model, infer, or evaluate.
  --cleanup-npy              Remove generated .npy predictions after evaluation.
  -h, --help                 Show this help.
USAGE
}

POSITIONAL_INDEX=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --encoder)
            ENCODER="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --raw-type)
            RAW_TYPE="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --depth-scale)
            DEPTH_SCALE="$2"
            shift 2
            ;;
        --max-depth)
            MAX_DEPTH="$2"
            shift 2
            ;;
        --max-size)
            MAX_SIZE="$2"
            shift 2
            ;;
        --sample-check-limit)
            SAMPLE_CHECK_LIMIT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --eval-batch-size)
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --cleanup-npy)
            CLEANUP_NPY="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ "${POSITIONAL_INDEX}" -eq 0 ]]; then
                MODEL_PATH="$1"
            elif [[ "${POSITIONAL_INDEX}" -eq 1 ]]; then
                RAW_TYPE="$1"
            else
                echo "Unexpected positional argument: $1" >&2
                usage >&2
                exit 2
            fi
            POSITIONAL_INDEX=$((POSITIONAL_INDEX + 1))
            shift
            ;;
    esac
done

if [[ -z "${OUTPUT}" ]]; then
    MODEL_NAME="$(basename "${MODEL_PATH}")"
    MODEL_STEM="${MODEL_NAME%.*}"
    OUTPUT="hammer_eval/outputs/hammer_${MODEL_STEM}_data_${RAW_TYPE}"
fi

cd "${PROJECT_ROOT}"

COMMON_ARGS=(
    --model-path "${MODEL_PATH}"
    --dataset "${DATASET}"
    --raw-type "${RAW_TYPE}"
    --device "${DEVICE}"
    --depth-scale "${DEPTH_SCALE}"
    --sample-check-limit "${SAMPLE_CHECK_LIMIT}"
    --limit "${LIMIT}"
)

INFER_ARGS=(
    "${COMMON_ARGS[@]}"
    --encoder "${ENCODER}"
    --output "${OUTPUT}"
    --batch-size "${BATCH_SIZE}"
    --max-depth "${MAX_DEPTH}"
    --max-size "${MAX_SIZE}"
)

EVAL_ARGS=(
    "${COMMON_ARGS[@]}"
    --output "${OUTPUT}"
    --batch-size "${EVAL_BATCH_SIZE}"
)

if [[ "${DRY_RUN}" == "true" ]]; then
    "${PYTHON_BIN}" hammer_eval/infer.py "${INFER_ARGS[@]}" --dry-run
    echo "Dry-run complete; skipped inference and evaluation."
    exit 0
fi

"${PYTHON_BIN}" hammer_eval/infer.py "${INFER_ARGS[@]}"
"${PYTHON_BIN}" hammer_eval/evaluate.py "${EVAL_ARGS[@]}"

if [[ "${CLEANUP_NPY}" == "true" ]]; then
    find "${OUTPUT}" -maxdepth 1 -type f -name '*.npy' -delete
fi

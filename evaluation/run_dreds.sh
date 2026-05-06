#!/usr/bin/env bash

set -euo pipefail

export OPENCV_IO_ENABLE_OPENEXR=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage:
  bash evaluation/run_dreds.sh <checkpoint_or_hf_model_id> [encoder=vitl] [variant=all] [cleanup_npy=false]

Arguments:
  variant               catknown | catnovel | all. Default: all

Environment overrides:
  DREDS_KNOWN_JSONL     DREDS catknown JSONL. Default: data/DREDS/test_std_catknown.jsonl
  DREDS_NOVEL_JSONL     DREDS catnovel JSONL. Default: data/DREDS/test_std_catnovel.jsonl
  OUTPUT_DIR            Prediction/evaluation output directory for one variant only.
  OUTPUT_ROOT           Root directory for default per-variant outputs. Default: checkpoint directory
  INPUT_SIZE            PromptDA max RGB side length. Default: 1008
  BATCH_SIZE            Path batch size; PromptDA inference runs one sample at a time. Default: 1
  NUM_WORKERS           DataLoader workers. Default: 0
  MAX_SAMPLES           Maximum number of samples to run. 0 means all. Default: 0
  SAVE_VIS              Save optional visualization images when true. Default: true
  CLAMP_PREDICTION      Clamp saved predictions to the dataset depth range. Default: false
  PYTHON_BIN            Python executable. Default: python3

DREDS uses EXR floating-point depth in meters. raw-type is passed as d435 only
to satisfy the shared Python CLI and is ignored by the DREDS dataset loader.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

model_path="$1"
encoder="${2:-vitl}"
variant="${3:-all}"
cleanup_npy="${4:-false}"
camera_type="d435"

dreds_known_jsonl="${DREDS_KNOWN_JSONL:-data/DREDS/test_std_catknown.jsonl}"
dreds_novel_jsonl="${DREDS_NOVEL_JSONL:-data/DREDS/test_std_catnovel.jsonl}"
input_size="${INPUT_SIZE:-1008}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
max_samples="${MAX_SAMPLES:-0}"
save_vis="${SAVE_VIS:-true}"
clamp_prediction="${CLAMP_PREDICTION:-false}"

model_name="$(basename "${model_path}")"
model_stub="${model_name%%.*}"
model_dir="$(dirname "${model_path}")"
output_root="${OUTPUT_ROOT:-${model_dir}}"

if [[ "${variant}" == "all" && -n "${OUTPUT_DIR:-}" ]]; then
    echo "OUTPUT_DIR can only be used with variant=catknown or variant=catnovel; use OUTPUT_ROOT for variant=all." >&2
    exit 2
fi

save_vis_arg=()
if [[ "${save_vis}" == "true" ]]; then
    save_vis_arg=(--save-vis)
fi

run_one_variant() {
    local label="$1"
    local jsonl_path="$2"
    local output_dir="${OUTPUT_DIR:-${output_root}/dreds_${label}_${model_stub}}"

    echo "[${label}] model path/id: ${model_path}"
    echo "[${label}] fixed model class: promptda.promptda.PromptDA"
    echo "[${label}] encoder: ${encoder}"
    echo "[${label}] dataset path: ${jsonl_path}"
    echo "[${label}] input size: ${input_size}"
    echo "[${label}] batch size: ${batch_size}"
    echo "[${label}] max samples: ${max_samples}"
    echo "[${label}] output dir: ${output_dir}"
    echo "[${label}] save vis: ${save_vis}"
    echo "[${label}] clamp prediction: ${clamp_prediction}"
    echo "[${label}] cleanup npy: ${cleanup_npy}"

    "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
        --encoder "${encoder}" \
        --model-path "${model_path}" \
        --dataset "${jsonl_path}" \
        --raw-type "${camera_type}" \
        --output "${output_dir}" \
        --input-size "${input_size}" \
        --batch-size "${batch_size}" \
        --num-workers "${num_workers}" \
        --max-samples "${max_samples}" \
        --clamp-prediction "${clamp_prediction}" \
        "${save_vis_arg[@]}"

    echo "[${label}] evaluating PromptDA predictions on DREDS"
    time "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" \
        --encoder "${encoder}" \
        --model-path "${model_path}" \
        --dataset "${jsonl_path}" \
        --output "${output_dir}" \
        --raw-type "${camera_type}" \
        --input-size "${input_size}" \
        --max-samples "${max_samples}"

    if [[ "${cleanup_npy}" == "true" ]]; then
        echo "[${label}] cleanup_npy is enabled, removing generated .npy files under ${output_dir}/predictions"
        if [[ -d "${output_dir}/predictions" ]]; then
            find "${output_dir}/predictions" -maxdepth 1 -type f -name '*.npy' -delete
        fi
    fi
}

case "${variant}" in
    catknown)
        run_one_variant catknown "${dreds_known_jsonl}"
        ;;
    catnovel)
        run_one_variant catnovel "${dreds_novel_jsonl}"
        ;;
    all)
        run_one_variant catknown "${dreds_known_jsonl}"
        run_one_variant catnovel "${dreds_novel_jsonl}"
        ;;
    *)
        echo "unknown DREDS variant: ${variant} (expected: catknown | catnovel | all)" >&2
        exit 1
        ;;
esac

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage:
  bash evaluation/run_bs_transpose.sh <checkpoint_or_hf_model_id> [encoder=vitl] [camera_type=l515] [cleanup_npy=true]

Arguments:
  camera_type          l515 only. TRansPose is fixed to L515 raw depth.

Environment overrides:
  DATASET_PATH          TRansPose JSONL path. Default: data/TRansPose/sequences/dc_testset.jsonl
  OUTPUT_DIR            Prediction/evaluation output directory. Default: <checkpoint_dir>/transpose_<dataset_stub>_<checkpoint_stub>_data_l515
  INPUT_SIZE            PromptDA max RGB side length. Default: 1008
  BATCH_SIZE            Path batch size; PromptDA inference runs one sample at a time. Default: 16
  NUM_WORKERS           DataLoader workers. Default: 4
  MAX_SAMPLES           Maximum number of samples to run. 0 means all. Default: 0
  SAVE_VIS              Save TRansPose 3x2 visualizations when true. Default: false
  INTRINSICS_PATH       Camera intrinsics text file. Default: data/TRansPose/sequences/intrinsics.txt
  CLAMP_PREDICTION      Clamp saved predictions to the dataset depth range. Default: false
  PYTHON_BIN            Python executable. Default: python3
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
camera_type="${3:-l515}"
cleanup_npy="${4:-true}"

if [[ "${camera_type}" != "l515" ]]; then
    echo "TRansPose dataset only supports l515 raw type, got ${camera_type}" >&2
    exit 1
fi

dataset_path="${DATASET_PATH:-data/TRansPose/sequences/dc_testset.jsonl}"
input_size="${INPUT_SIZE:-1008}"
batch_size="${BATCH_SIZE:-16}"
num_workers="${NUM_WORKERS:-4}"
max_samples="${MAX_SAMPLES:-0}"
save_vis="${SAVE_VIS:-false}"
intrinsics_path="${INTRINSICS_PATH:-data/TRansPose/sequences/intrinsics.txt}"
clamp_prediction="${CLAMP_PREDICTION:-false}"

model_name="$(basename "${model_path}")"
model_stub="${model_name%%.*}"
model_dir="$(dirname "${model_path}")"
dataset_name="$(basename "${dataset_path}")"
dataset_stub="${dataset_name%%.*}"
output_dir="${OUTPUT_DIR:-${model_dir}/transpose_${dataset_stub}_${model_stub}_data_${camera_type}}"

save_vis_arg=()
if [[ "${save_vis}" == "true" ]]; then
    save_vis_arg=(--save-vis)
fi

echo "model path/id: ${model_path}"
echo "fixed model class: promptda.promptda.PromptDA"
echo "encoder: ${encoder}"
echo "dataset path: ${dataset_path}"
echo "camera type: ${camera_type}"
echo "input size: ${input_size}"
echo "batch size: ${batch_size}"
echo "max samples: ${max_samples}"
echo "output dir: ${output_dir}"
echo "save vis: ${save_vis}"
echo "intrinsics path: ${intrinsics_path}"
echo "clamp prediction: ${clamp_prediction}"
echo "cleanup npy: ${cleanup_npy}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
    --encoder "${encoder}" \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --raw-type "${camera_type}" \
    --output "${output_dir}" \
    --input-size "${input_size}" \
    --batch-size "${batch_size}" \
    --num-workers "${num_workers}" \
    --max-samples "${max_samples}" \
    --clamp-prediction "${clamp_prediction}" \
    --intrinsics-path "${intrinsics_path}" \
    "${save_vis_arg[@]}"

echo "evaluating PromptDA predictions on TRansPose"
time "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" \
    --encoder "${encoder}" \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --output "${output_dir}" \
    --raw-type "${camera_type}" \
    --input-size "${input_size}" \
    --max-samples "${max_samples}"

if [[ "${cleanup_npy}" == "true" ]]; then
    echo "cleanup_npy is enabled, removing generated .npy files under ${output_dir}/predictions"
    if [[ -d "${output_dir}/predictions" ]]; then
        find "${output_dir}/predictions" -maxdepth 1 -type f -name '*.npy' -delete
    fi
fi

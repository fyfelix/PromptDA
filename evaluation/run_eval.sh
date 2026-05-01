#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage:
  ./evaluation/run_eval.sh [checkpoint_or_hf_model_id=ckpt/promptda_vitl.ckpt] [encoder=vitl] [raw_type=d435] [cleanup_npy=false]

Environment overrides:
  DATASET_PATH          HAMMER JSONL path. Default: data/HAMMER/test.jsonl
  OUTPUT_DIR            Prediction/evaluation output directory. Default: evaluation/output
  INPUT_SIZE            PromptDA max RGB side length. Default: 1008
  BATCH_SIZE            Path batch size; PromptDA runs one sample at a time. Default: 1
  NUM_WORKERS           DataLoader workers. Default: 0
  SAVE_VIS              Save optional visualization images. Default: true
  CLAMP_PREDICTION      Clamp saved predictions to HAMMER depth range. Default: false
  FILL_PROMPT_DEPTH_HOLES
                        Fill invalid prompt depth pixels before inference. Default: true
  PYTHON_BIN            Python executable. Default: python3

This wrapper is adapted for PromptDA:
  infer.py loads promptda.promptda.PromptDA
  HAMMER raw depth is used as PromptDA prompt_depth
  Prompt depth holes are filled by nearest valid depth by default
  infer.py writes per-sample metric-depth .npy files in meters
  eval.py reads those .npy files and writes fixed CSV/JSON metrics
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

model_path="${1:-ckpt/promptda_vitl.ckpt}"
encoder="${2:-vitl}"
raw_type="${3:-d435}"
cleanup_npy="${4:-false}"

dataset_path="${DATASET_PATH:-data/HAMMER/test.jsonl}"
input_size="${INPUT_SIZE:-1008}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
save_vis="${SAVE_VIS:-true}"
clamp_prediction="${CLAMP_PREDICTION:-false}"
fill_prompt_depth_holes="${FILL_PROMPT_DEPTH_HOLES:-true}"
output_dir="${OUTPUT_DIR:-evaluation/output}"

echo "model path/id: ${model_path}"
echo "fixed model class: promptda.promptda.PromptDA"
echo "encoder: ${encoder}"
echo "dataset path: ${dataset_path}"
echo "raw type: ${raw_type}"
echo "input size: ${input_size}"
echo "batch size: ${batch_size}"
echo "output dir: ${output_dir}"
echo "save vis: ${save_vis}"
echo "clamp prediction: ${clamp_prediction}"
echo "fill prompt depth holes: ${fill_prompt_depth_holes}"
echo "cleanup npy: ${cleanup_npy}"

save_vis_args=()
if [[ "${save_vis}" == "true" ]]; then
    save_vis_args+=(--save-vis)
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
    --encoder "${encoder}" \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --raw-type "${raw_type}" \
    --output "${output_dir}" \
    --input-size "${input_size}" \
    --batch-size "${batch_size}" \
    --num-workers "${num_workers}" \
    --clamp-prediction "${clamp_prediction}" \
    --fill-prompt-depth-holes "${fill_prompt_depth_holes}" \
    "${save_vis_args[@]}"

echo "evaluating PromptDA predictions on HAMMER"
time "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" \
    --encoder "${encoder}" \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --output "${output_dir}" \
    --raw-type "${raw_type}"

if [[ "${cleanup_npy}" == "true" ]]; then
    echo "cleanup_npy is enabled, removing generated .npy files under ${output_dir}"
    find "${output_dir}" -maxdepth 1 -type f -name '*.npy' -delete
fi

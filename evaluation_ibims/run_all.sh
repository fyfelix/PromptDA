#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage:
  ./evaluation_ibims/run_all.sh [checkpoint_or_hf_model_id=ckpts/promptda_vitl.ckpt] [encoder=vitl]

Environment overrides:
  IBIMS_ROOT          iBims dataset root. Default: data/ibims1
  OUTPUT_DIR          Base output directory. Default: evaluation_ibims/output
  RUN_DIR             Existing or new run directory. Overrides OUTPUT_DIR timestamping.
  LEVELS              Difficulty levels. Default: easy medium hard extreme
  INPUT_SIZE          PromptDA max RGB side length. Default: 1008
  MAX_SAMPLES         Maximum samples per level. 0 means all. Default: 0
  CLAMP_PREDICTION    Clamp finite positive predictions to manifest depth range. Default: false
  SKIP_INFER          Skip inference and use existing predictions. Default: false
  SKIP_EVAL           Skip official evaluation. Default: false
  PYTHON_BIN          Python executable from the active conda environment. Default: python

This wrapper expects an activated conda environment with PromptDA dependencies.
It consumes existing synthetic iBims manifests under:
  <IBIMS_ROOT>/ibims1_synthetic_raw_depth/manifests/ibims_<level>.jsonl
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Activate the target conda environment or set PYTHON_BIN." >&2
    exit 1
fi

model_path="${1:-ckpts/promptda_vitl.ckpt}"
encoder="${2:-vitl}"

ibims_root="${IBIMS_ROOT:-data/ibims1}"
output_dir="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
run_dir="${RUN_DIR:-}"
levels_string="${LEVELS:-easy medium hard extreme}"
input_size="${INPUT_SIZE:-1008}"
max_samples="${MAX_SAMPLES:-0}"
clamp_prediction="${CLAMP_PREDICTION:-false}"
skip_infer="${SKIP_INFER:-false}"
skip_eval="${SKIP_EVAL:-false}"

read -r -a levels <<< "${levels_string}"

echo "python: ${PYTHON_BIN}"
echo "model path/id: ${model_path}"
echo "fixed model class: promptda.promptda.PromptDA"
echo "encoder: ${encoder}"
echo "iBims root: ${ibims_root}"
echo "levels: ${levels_string}"
echo "input size: ${input_size}"
echo "max samples: ${max_samples}"
echo "output dir: ${output_dir}"
echo "run dir: ${run_dir:-<timestamped>}"
echo "clamp prediction: ${clamp_prediction}"
echo "skip infer: ${skip_infer}"
echo "skip eval: ${skip_eval}"

args=(
    "${SCRIPT_DIR}/run_all.py"
    --model-path "${model_path}"
    --encoder "${encoder}"
    --ibims-root "${ibims_root}"
    --output-root "${output_dir}"
    --levels "${levels[@]}"
    --input-size "${input_size}"
    --max-samples "${max_samples}"
)

if [[ -n "${run_dir}" ]]; then
    args+=(--run-dir "${run_dir}")
fi
if [[ "${clamp_prediction}" == "true" ]]; then
    args+=(--clamp-prediction)
fi
if [[ "${skip_infer}" == "true" ]]; then
    args+=(--skip-infer)
fi
if [[ "${skip_eval}" == "true" ]]; then
    args+=(--skip-eval)
fi

exec "${PYTHON_BIN}" "${args[@]}"

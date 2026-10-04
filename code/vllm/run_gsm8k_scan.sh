#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
VLLM_DIR="$ROOT_DIR/vllm"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export VLLM_USE_V1=0
export PYTHONPATH="$VLLM_DIR${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
JUDI_THRESHOLDS=(0.10 0.30 0.40 0.50 0.90)

# All Python benchmark options (e.g. --end 3) apply to every experiment.
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        exec "$PYTHON" -m judi_vllm.bench_gsm8k run --help
    fi
done

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/vllm_gsm8k/$(date +%Y%m%d_%H%M%S_%N)}"
mkdir -p "$(dirname "$OUTPUT_DIR")"
# Refuse to mix old results into a new scan.
mkdir "$OUTPUT_DIR"
printf 'GPU: %s\nResults: %s\n' "$CUDA_VISIBLE_DEVICES" "$OUTPUT_DIR"

"$PYTHON" -u -m judi_vllm.bench_gsm8k run "$@" \
    --method standard --output-dir "$OUTPUT_DIR/standard" \
    2>&1 | tee "$OUTPUT_DIR/standard.log"

for threshold in "${JUDI_THRESHOLDS[@]}"; do
    "$PYTHON" -u -m judi_vllm.bench_gsm8k run "$@" \
        --method judi --threshold "$threshold" \
        --output-dir "$OUTPUT_DIR/judi_$threshold" \
        2>&1 | tee "$OUTPUT_DIR/judi_$threshold.log"
done

"$PYTHON" -m judi_vllm.bench_gsm8k summarize "$OUTPUT_DIR"

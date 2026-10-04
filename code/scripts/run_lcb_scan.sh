#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
PROFILE="${PROFILE:-llama3-8b-1b}"
DTYPE="bfloat16"
CONFIDENCE_THRESHOLD=0.9
SAMPLES_START=0
SAMPLES_END=3
DRAFT_LENGTH=16
MAX_NEW_TOKENS=2048
TIMEOUT=6

# LCB-specific scans from the AJ-V2 Llama 1B/8B experiments.
JUDI_THRESHOLDS=(-1 0.1 0.3 0.6 0.9 1.1)
AUTOJUDGE_THRESHOLDS=(-1 0.005 0.01 0.015 0.02 0.025)

OUTPUT_ROOT="$ROOT_DIR/outputs/lcb/$PROFILE"

run_method() {
    local method="$1"
    local output_dir="$2"
    shift 2
    local -a thresholds=("$@")

    if ((${#thresholds[@]} == 0)); then
        printf 'No thresholds configured for %s.\n' "$method" >&2
        return 2
    fi

    rm -rf "$output_dir"
    mkdir -p "$output_dir"
    echo "Starting $method LCB scan on GPU $GPU."
    CUDA_VISIBLE_DEVICES="$GPU" bash "$SCRIPT_DIR/run_lcb_core.sh" "$method" "$PROFILE" \
        --dtype "$DTYPE" \
        --confidence-threshold "$CONFIDENCE_THRESHOLD" \
        --draft-length "$DRAFT_LENGTH" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --timeout "$TIMEOUT" \
        --thresholds "${thresholds[@]}" \
        --start "$SAMPLES_START" \
        --end "$SAMPLES_END" \
        --output-dir "$output_dir" >"$output_dir.log" 2>&1

    python - "$output_dir" "$method" "$GPU" "$PROFILE" "$CONFIDENCE_THRESHOLD" \
        "${thresholds[@]}" <<'PY_SUMMARY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
method, gpu, profile, confidence_threshold = sys.argv[2:6]
thresholds = [float(value) for value in sys.argv[6:]]
prefix = "autojudge" if method == "AutoJudge" else "judi"
threshold_field = "threshold"
rows = []
for path in output_dir.glob(f"{prefix}_threshold_*/summary.json"):
    rows.append(json.loads(path.read_text(encoding="utf-8")))
rows.sort(key=lambda row: float(row[threshold_field]))
with (output_dir / "scan_summary.jsonl").open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
(output_dir / "scan_config.json").write_text(json.dumps({
    "dataset": "lcb",
    "method": method,
    "profile": profile,
    "thresholds": thresholds,
    "confidence_threshold": float(confidence_threshold),
    "samples": int(rows[0].get("samples", 0)) if rows else 0,
    "gpus": [int(gpu)],
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Completed {len(rows)} {method} threshold runs; summaries: {output_dir / 'scan_summary.jsonl'}")
PY_SUMMARY
}

run_method JuDi "$OUTPUT_ROOT/JuDi" "${JUDI_THRESHOLDS[@]}"
run_method AutoJudge "$OUTPUT_ROOT/AutoJudge" "${AUTOJUDGE_THRESHOLDS[@]}"

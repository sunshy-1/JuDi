#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${1:-AutoJudge}"
PROFILE="${2:-llama3-8b-1b}"
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi

case "$METHOD" in
  AutoJudge|JuDi) ;;
  *)
    echo "Usage: $0 {AutoJudge|JuDi} [profile] [runner options...]" >&2
    echo "Without --threshold/--judi-threshold/--thresholds, the configured default scan is run." >&2
    exit 2
    ;;
esac

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON:-python}"
exec "$PYTHON_BIN" -m code_judi.run_gsm8k --method "$METHOD" --profile "$PROFILE" "$@"

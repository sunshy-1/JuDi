#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"
bash vllm/run_gsm8k_scan.sh "$@"

#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"
bash scripts/run_gsm8k_scan.sh
bash scripts/run_lcb_scan.sh

#!/usr/bin/env bash
# POD B1-smollm2 — SmolLM2-135M on WikiText-2 (core-6, PSO auto, n=3). Journaled; re-run to resume.
set -euo pipefail
METHODS=${1:-${METHODS:-}}
source "$(dirname "$0")/pod_common.sh"
pod_provision
pod_preflight B1_smollm2
cd "$REPO"
echo "[POD B1-smollm2] $(date -u) starting"
python -u wikitext_benchmarks.py --model smollm2 ${METHODS:+--methods "$METHODS"} --quiet 2>&1 | tee -a logs/pod_B1_smollm2.log
echo "[POD B1-smollm2] $(date -u) done"

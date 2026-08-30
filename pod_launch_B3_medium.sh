#!/usr/bin/env bash
# POD B3-medium — 'medium' (18M) scale point, optional method subset (roster split across pods).
#   bash pod_launch_B3_medium.sh "ElasticNet"      bash pod_launch_B3_medium.sh "Baseline,L2,WD-tuned,Tau(alpha=0)"
# Journaled (results/journal/gpt2_medium_wikitext_standardized_results.jsonl); assemble with pod_collect.sh.
set -euo pipefail
METHODS=${1:-${METHODS:-}}
source "$(dirname "$0")/pod_common.sh"
pod_provision
pod_preflight B3_medium
cd "$REPO"
echo "[POD B3-medium] $(date -u) starting (${METHODS:-full core-6})"
python -u gpt2_wikitext_standardized.py --scale medium --instrument ${METHODS:+--methods "$METHODS"} --quiet 2>&1 | tee -a "logs/pod_B3_medium.log"
echo "[POD B3-medium] $(date -u) done"

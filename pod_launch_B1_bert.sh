#!/usr/bin/env bash
# POD B1-bert — BERT-tiny on SST-2 (10 methods, PSO light, n=5). Journaled; re-run to resume.
set -euo pipefail
source "$(dirname "$0")/pod_common.sh"
pod_provision
pod_preflight B1_bert
cd "$REPO"
echo "[POD B1-bert] $(date -u) starting"
python -u bert_sst2_standardized.py --quiet 2>&1 | tee -a logs/pod_B1_bert.log
echo "[POD B1-bert] $(date -u) done"

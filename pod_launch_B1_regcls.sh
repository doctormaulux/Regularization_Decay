#!/usr/bin/env bash
# POD B1-regcls — sin / Friedman regression + MNIST / CIFAR-10 CNN (10 methods, PSO light, n=5)
# Split off pod_launch_B1_custom.sh (BERT-tiny and SmolLM2 run on their own pods) to shorten
# the wall-clock. Journaled; re-run to resume.
set -euo pipefail
source "$(dirname "$0")/pod_common.sh"
pod_provision
pod_preflight B1_regcls
cd "$REPO"
run() { echo "[POD B1-regcls] $(date -u) $*"; "$@" 2>&1 | tee -a logs/pod_B1_regcls.log; }
run python -u regression_benchmarks.py --model all --quiet
run python -u classification_benchmarks.py --model all --quiet
echo "[POD B1-regcls] $(date -u) done"

#!/usr/bin/env bash
# ============================================================================
# POD B1 — the cheap Table-1 benchmarks + SmolLM2, corrected early stopping
# ----------------------------------------------------------------------------
# sin / Friedman regression (10 methods, PSO light, n=5)   ~1.4 h + ~3.6 h
# MNIST FC / CIFAR-10 CNN (10 methods, PSO light, n=5)     ~10 h + ~9 h
# BERT-tiny SST-2 (10 methods, PSO light, n=5)             ~26 h
# SmolLM2-135M WikiText-2 (core-6, PSO auto, n=3)          ~36 h
# Total ~86 h. ViT-mini (86 h on its own) runs on a separate pod: pod_launch_B1_vit.sh.
# Outputs: results/{sin_regression,complex_regression,mnist,cifar,bert_sst2,smollm2_wikitext}_standardized_results.{csv,json}
# SAFE TO STOP AT ANY TIME (each benchmark is journaled; re-run to resume).
# ============================================================================
set -euo pipefail
source "$(dirname "$0")/pod_common.sh"
pod_provision
pod_preflight B1_custom
cd "$REPO"
run() { echo "[POD B1] $(date -u) $*"; "$@" 2>&1 | tee -a logs/pod_B1_custom.log; }
run python -u regression_benchmarks.py --model all --quiet
run python -u classification_benchmarks.py --model all --quiet
run python -u bert_sst2_standardized.py --quiet
run python -u wikitext_benchmarks.py --model smollm2 --quiet
echo "[POD B1] $(date -u) done"

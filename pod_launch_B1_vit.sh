#!/usr/bin/env bash
# ============================================================================
# POD B1-ViT — ViT-mini on CIFAR-10 (10 methods, PSO light, n=5), corrected early stopping
# ----------------------------------------------------------------------------
# ~86 h on its own (150-epoch ceiling, patience 25). Output: results/vit_cifar_standardized_results.{csv,json}
# ============================================================================
set -euo pipefail
# Optional first argument (or METHODS env): comma-separated subset of the roster, to split
# the ~40-min-per-run ViT benchmark across pods (assemble with pod_assemble_vit.sh).
METHODS=${1:-${METHODS:-}}
source "$(dirname "$0")/pod_common.sh"
pod_provision
pod_preflight B1_vit
cd "$REPO"
echo "[POD B1-ViT] $(date -u) starting"
python -u vit_cifar_standardized.py ${METHODS:+--methods "$METHODS"} --quiet 2>&1 | tee -a logs/pod_B1_vit.log
echo "[POD B1-ViT] $(date -u) done"

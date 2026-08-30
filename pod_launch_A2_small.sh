#!/usr/bin/env bash
# ============================================================================
# POD A2 — 'small' (7.4M) reference configuration, full 10-method roster, n = 5
# ----------------------------------------------------------------------------
# Regenerates the Table-1 GPT-2 row and the 7M point of the scale sweep with the
# corrected early stopping and the uniform PSO budget (12 evals per 1-D search,
# 40 per >=2-D search; the 2026-06 reference used 'standard' = 80). Instrumented.
# Cost: 5 + 4x(12+5) + 5x(40+5) = 298 runs x ~17 min = ~84 h (~3.5 days).
# Output: results/gpt2_wikitext_standardized_results.{csv,json}
# SAFE TO STOP AT ANY TIME (journaled; re-run to resume).
# ============================================================================
set -euo pipefail
# Optional first argument (or METHODS env): comma-separated subset of the roster.
METHODS=${1:-${METHODS:-}}
source "$(dirname "$0")/pod_common.sh"
pod_provision
pod_preflight A2_small
cd "$REPO"
echo "[POD A2] $(date -u) starting"
# METHODS="A,B" splits the roster across pods (shared journal name; assemble with
# pod_assemble_A2.sh). Default: the full 10-method roster on one pod.
python -u gpt2_wikitext_standardized.py --scale small --instrument \
    ${METHODS:+--methods "$METHODS"} --quiet 2>&1 | tee -a "logs/pod_A2_small${METHODS:+_split}.log"
echo "[POD A2] $(date -u) done"

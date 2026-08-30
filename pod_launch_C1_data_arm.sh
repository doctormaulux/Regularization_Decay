#!/usr/bin/env bash
# ============================================================================
# POD C1 — data-quantity arm (REVIEWER-11) at the 'large' scale, AFTER A1 is assembled
# ----------------------------------------------------------------------------
# usage: bash pod_launch_C1_data_arm.sh results/gpt2_large_wikitext_standardized_results.json
#        (the ASSEMBLED A1 file: its best_hyperparams are the fixed hyperparameters)
# 25% and 50% of the training split with optimizer steps held constant (48 and 24
# epochs; the LR schedule spans the same step budget as the full-data run, so the
# 2026-07 schedule confound cannot recur); Baseline / Tau(alpha=0) / tau(w); n = 5;
# instrumented. The 100% point is the A1 run itself.
# Cost: 15 runs x ~23 min + 15 runs x ~31 min = ~14 h.
# Outputs: results/gpt2_large_wikitext_standardized_results_data{25,50}.{csv,json}
# ============================================================================
set -euo pipefail
HP=${1:?path to the assembled A1 results JSON (best_hyperparams)}
source "$(dirname "$0")/pod_common.sh"
pod_provision
pod_preflight C1
cd "$REPO"
for frac in 0.25 0.5; do
  echo "[POD C1] $(date -u) fraction $frac"
  python -u gpt2_wikitext_standardized.py --scale large --data-fraction "$frac" --seeds 5 \
      --instrument --hp-from "$HP" --quiet 2>&1 | tee -a "logs/pod_C1_data${frac}.log"
done
echo "[POD C1] $(date -u) done"

#!/usr/bin/env bash
# ============================================================================
# POD A1-en_l2 — 'large' (66M) from-scratch GPT-2 on WikiText-2, method group: ElasticNet,L2
# ----------------------------------------------------------------------------
# Part of the A1 run (plan of 2026-08-29):
# core-6 + Tau(AdamW-scope) at the 'large' scale, n = 10 seeds for every method,
# uniform PSO budget (12 evals per 1-D search, 40 per >=2-D search), instrumented.
# The 7-method roster is split across four pods BY METHOD; the four journals are
# concatenated afterwards and the roster is assembled with pod_assemble_A1.sh
# (every run is then served from the journal, nothing retrains).
# Cost of this group: (40+10) + (12+10) = 72 runs x ~45 min = ~54 h
#
# SAFE TO STOP AT ANY TIME: every finished run is written to results/ and journaled;
# re-running this script resumes and repeats at most the run that was in flight.
# Pull with:  bash pod_status.sh --pull     (pods.conf: "en_l2 <ip> <port> gpt2_large_wikitext_standardized_results")
# ============================================================================
set -euo pipefail
# Optional first argument: subset of this group (e.g. "ElasticNet" alone) to split it further.
METHODS=${1:-"ElasticNet,L2"}
source "$(dirname "$0")/pod_common.sh"
pod_provision
pod_preflight "A1_en_l2"
cd "$REPO"
echo "[POD A1_en_l2] $(date -u) starting"
python -u gpt2_wikitext_standardized.py --scale large --seeds 10 --instrument \
    --methods "$METHODS" --quiet 2>&1 | tee -a "logs/pod_A1_en_l2.log"
echo "[POD A1_en_l2] $(date -u) done -> results/gpt2_large_wikitext_standardized_results.{csv,json} (this group only), results/journal/, results/instrumentation/"

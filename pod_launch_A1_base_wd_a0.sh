#!/usr/bin/env bash
# ============================================================================
# POD A1-base_wd_a0 — 'large' (66M) from-scratch GPT-2 on WikiText-2, method group: Baseline,WD-tuned,Tau(alpha=0)
# ----------------------------------------------------------------------------
# Part of the A1 run (plan of 2026-08-29):
# core-6 + Tau(AdamW-scope) at the 'large' scale, n = 10 seeds for every method,
# uniform PSO budget (12 evals per 1-D search, 40 per >=2-D search), instrumented.
# The 7-method roster is split across four pods BY METHOD; the four journals are
# concatenated afterwards and the roster is assembled with pod_assemble_A1.sh
# (every run is then served from the journal, nothing retrains).
# Cost of this group: 10 + (12+10) + (12+10) = 54 runs x ~45 min = ~41 h
# GATE P0: the Baseline group is run as a SEPARATE first invocation, because
# run_benchmark tunes every method (step 1) before any final run (step 2) - with a single
# invocation the 10 Baseline finals would only start after the 24 PSO evaluations of
# WD-tuned and Tau(alpha=0). Both invocations share the CSV name, hence the journal, so the
# second one resumes with nothing lost. Read the Baseline test PPL from the journal
# (results/journal/..., kind=eval, method=Baseline) and compare it with the clean
# tau(w)/Tau(alpha=0) numbers on disk (59.07 / 59.40, n=3): if the Baseline is below
# ~62 PPL the over-capacity story at 66M is gone - stop the other pods and rethink.
#
# SAFE TO STOP AT ANY TIME: every finished run is written to results/ and journaled;
# re-running this script resumes and repeats at most the run that was in flight.
# Pull with:  bash pod_status.sh --pull     (pods.conf: "base_wd_a0 <ip> <port> gpt2_large_wikitext_standardized_results")
# ============================================================================
set -euo pipefail
source "$(dirname "$0")/pod_common.sh"
pod_provision
pod_preflight "A1_base_wd_a0"
cd "$REPO"
echo "[POD A1_base_wd_a0] $(date -u) starting"
python -u gpt2_wikitext_standardized.py --scale large --seeds 10 --instrument \
    --methods "Baseline" --quiet 2>&1 | tee -a "logs/pod_A1_base_wd_a0.log"
echo "[POD A1_base_wd_a0] $(date -u) Baseline finals done (gate P0 readable from the journal)"
python -u gpt2_wikitext_standardized.py --scale large --seeds 10 --instrument \
    --methods "WD-tuned,Tau(alpha=0)" --quiet 2>&1 | tee -a "logs/pod_A1_base_wd_a0.log"
echo "[POD A1_base_wd_a0] $(date -u) done -> results/gpt2_large_wikitext_standardized_results.{csv,json} (this group only), results/journal/, results/instrumentation/"

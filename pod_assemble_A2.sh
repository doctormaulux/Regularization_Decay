#!/usr/bin/env bash
# ============================================================================
# Assemble the A2 (small, 10-method) roster from the per-pod journals (runs on CPU, trains nothing)
# ----------------------------------------------------------------------------
# usage: bash pod_assemble_A2.sh results/journal/gpt2_wikitext_standardized_results.A2_small.jsonl \
#                                results/journal/gpt2_wikitext_standardized_results.A2_small_2d.jsonl ...
# (the tagged copies that `pod_status.sh --pull` writes). The journals are concatenated
# into the canonical journal name and the full 10-method roster is re-run: every PSO
# winner and every final run is served from the journal, so the command only writes
# the assembled results/gpt2_wikitext_standardized_results.{csv,json} and prints
# the statistics. It then verifies the instrumentation trajectories report best epochs.
# ============================================================================
set -euo pipefail
[ "$#" -ge 1 ] || { echo "usage: $0 <journal.jsonl> [...]"; exit 1; }
J=results/journal/gpt2_wikitext_standardized_results.jsonl
mkdir -p results/journal
cat "$@" > "$J"
echo "[ASSEMBLE] $(wc -l < "$J") journal records from $# file(s)"
python -u gpt2_wikitext_standardized.py --scale small --instrument \
    --methods "Baseline,L1,L2,ElasticNet,SCAD,MCP,LSP,WD-tuned,Tau(alpha=0),Tau(w)"
python analysis/audit_early_stopping.py --instrumentation results/instrumentation --min-ok 1
echo "[ASSEMBLE] done -> results/gpt2_wikitext_standardized_results.{csv,json}; copy to 'new results/' for the paper build"

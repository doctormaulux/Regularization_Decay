#!/usr/bin/env bash
# Collect the regeneration results from every pod in pods.conf and assemble the split rosters.
#
#   bash pod_collect.sh            # pull everything, assemble, audit, copy to "new results/"
#   bash pod_collect.sh --pull-only
#
# Steps
#  1. pod_status.sh --pull   -> new results/incoming/<tag>/*.csv|json, results/journal/<stem>.<tag>.jsonl,
#                               results/instrumentation/*.json (+ pso/)
#  2. for the split rosters (large, small, vit): concatenate the per-pod journals and re-run the
#     benchmark with the full roster - everything is served from the journal, nothing retrains -
#     which writes the assembled results/<stem>.{csv,json}
#  3. single-pod benchmarks are copied as they are from new results/incoming/<tag>/
#  4. audit: every instrumentation trajectory must report its best epoch
#  5. copy the assembled CSV/JSON into "new results/" (what paper/build_paper.py reads)
set -euo pipefail
PY=${PY:-python}
command -v "$PY" >/dev/null || PY=python3
bash pod_status.sh --pull
[[ "${1:-}" == "--pull-only" ]] && exit 0

assemble() {  # <stem> <script + args...>
  local stem=$1; shift
  local parts=(results/journal/${stem}.*.jsonl)
  [ -e "${parts[0]}" ] || { echo "[COLLECT] no per-pod journals for $stem"; return 0; }
  cat "${parts[@]}" > "results/journal/${stem}.jsonl"
  echo "[COLLECT] $stem: $(wc -l < results/journal/${stem}.jsonl) journal records from ${#parts[@]} pod journal(s)"
  "$PY" -u "$@"
}
assemble gpt2_large_wikitext_standardized_results gpt2_wikitext_standardized.py --scale large --seeds 10 --instrument \
    --methods "Baseline,L2,ElasticNet,WD-tuned,Tau(AdamW-scope),Tau(alpha=0),Tau(w)"
assemble gpt2_wikitext_standardized_results gpt2_wikitext_standardized.py --scale small --instrument
assemble vit_cifar_standardized_results vit_cifar_standardized.py

# Single-pod benchmarks: take the pod's own CSV/JSON.
for f in "new results"/incoming/*/*_standardized_results.{csv,json}; do
  [ -e "$f" ] || continue
  base=$(basename "$f")
  case "$base" in
    gpt2_large_wikitext_standardized_results.*|gpt2_wikitext_standardized_results.*|vit_cifar_standardized_results.*) continue;;
  esac
  cp "$f" "results/$base"
done

"$PY" analysis/audit_early_stopping.py --instrumentation results/instrumentation --min-ok 1
for f in results/*_standardized_results.csv results/*_standardized_results.json; do
  [ -e "$f" ] && cp "$f" "new results/"
done
echo "[COLLECT] done. Regenerated files now in new results/:"
ls -la "new results"/*_standardized_results.csv | awk '{print "  " $6, $7, $8, $9}'

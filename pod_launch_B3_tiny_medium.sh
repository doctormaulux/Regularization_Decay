#!/usr/bin/env bash
# ============================================================================
# POD B3 — 'tiny' (2.1M) and 'medium' (18M) scale points, core-6, n = 3, instrumented
# ----------------------------------------------------------------------------
# Both scale points are run again so that the whole sweep shares one pipeline
# regenerated so the whole sweep shares one pipeline (corrected restore, canonical
# (rho, delta) form, uniform 12/40 PSO budget) and so the medium mechanism figures no
# longer average a PSO probe at seed 42.
# Cost: tiny ~134 runs x ~5 min = ~11 h; medium ~134 runs x ~17 min = ~38 h.
# Output: results/gpt2_{tiny,medium}_wikitext_standardized_results.{csv,json}
# ============================================================================
set -euo pipefail
source "$(dirname "$0")/pod_common.sh"
pod_provision
pod_preflight B3
cd "$REPO"
for scale in tiny medium; do
  echo "[POD B3] $(date -u) starting $scale"
  python -u gpt2_wikitext_standardized.py --scale "$scale" --instrument --quiet 2>&1 | tee -a "logs/pod_B3_${scale}.log"
done
echo "[POD B3] $(date -u) done"

#!/usr/bin/env bash
# ============================================================================
# PREFLIGHT — prove the pipeline reports best-epoch numbers BEFORE spending GPU-hours
# ----------------------------------------------------------------------------
# 1. pytest tests/test_early_stopping.py  (unit + end-to-end restore assertions)
# 2. a real, short run of the whole GPT-2 pipeline (PSO -> finals -> journal ->
#    instrumentation) at the tiny scale on 0.5% of the training split: the model
#    overfits (calibrated 2026-08-29 on CPU: best epoch ~41, val rising afterwards),
#    early stopping fires (patience 4) and the two runtime guards (parameter
#    fingerprint, re-evaluated validation metric) are exercised on a run that really
#    had to be restored. ~13 short runs; ~15-25 min on a pod GPU
# 3. analysis/audit_early_stopping.py on the trajectories it produced: every final run
#    that stopped past its best epoch must report the best epoch (--min-ok 1)
#
# Runs in an ISOLATED directory (/workspace/preflight), never in the repo: a smoke
# journal in results/journal/ would otherwise be replayed by a real tiny run later.
# Exit code != 0 means: do not launch.
# ============================================================================
set -euo pipefail
REPO=${REPO:-/workspace/Regularization_Decay}
PF=${PF:-/workspace/preflight}
cd "$REPO"

python -m pytest tests/test_early_stopping.py -q

rm -rf "$PF"; mkdir -p "$PF"
cp experiment_utils.py gpt2_wikitext_standardized.py "$PF/"
cp -r analysis "$PF/"
cd "$PF"
python -u gpt2_wikitext_standardized.py \
    --scale tiny --methods "Baseline,WD-tuned" --pso-budget light \
    --epochs 80 --schedule-epochs 200 --patience 4 --seeds 2 \
    --train-fraction 0.005 --instrument --quiet
python analysis/audit_early_stopping.py --instrumentation results/instrumentation --min-ok 1
ls results/instrumentation/pso/*.json >/dev/null 2>&1 \
    || { echo "[PREFLIGHT] no PSO probe trajectories under results/instrumentation/pso/"; exit 1; }
test -f results/journal/gpt2_tiny_wikitext_standardized_results.jsonl \
    || { echo "[PREFLIGHT] journal not written"; exit 1; }
cd "$REPO"; rm -rf "$PF"
echo "[PREFLIGHT] OK $(date -u)"

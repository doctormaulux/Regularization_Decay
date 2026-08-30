#!/usr/bin/env bash
# Shared helpers for the pod_launch_*.sh scripts. Source it, do not run it.
#
#   source pod_common.sh
#   pod_provision            # idempotent pip installs + directories
#   pod_preflight <tag>      # tests + real tiny run with the restore guards live; aborts on failure
#
# Every launch script is SAFE TO STOP AT ANY TIME: each finished run is written to
# results/ immediately and journaled to results/journal/. Re-running the same script
# resumes and repeats at most the single run that was in flight.

REPO=${REPO:-/workspace/Regularization_Decay}

pod_provision() {
  cd "$REPO"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  python -c "import torch" 2>/dev/null || pip install -q torch
  python -c "import transformers, datasets, scipy, pytest" 2>/dev/null \
    || pip install -q "transformers==4.46.3" datasets scipy pytest
  mkdir -p results/journal results/instrumentation logs
}

pod_preflight() {
  local tag=${1:-preflight}
  cd "$REPO"
  if [ "${SKIP_PREFLIGHT:-0}" = "1" ]; then
    echo "[PREFLIGHT:$tag] skipped (SKIP_PREFLIGHT=1: already passed on this pod, see logs/preflight_${tag}.log)"
    return 0
  fi
  echo "[PREFLIGHT:$tag] $(date -u) tests + tiny end-to-end run with the restore guards live"
  bash pod_preflight.sh 2>&1 | tee -a "logs/preflight_${tag}.log"
  # pipefail is on in the callers: a failing preflight aborts the launch.
}

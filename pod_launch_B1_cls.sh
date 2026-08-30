#!/usr/bin/env bash
# POD B1-cls — one classification benchmark (mnist | cifar), 10 methods, PSO light, n=5.
#   bash pod_launch_B1_cls.sh mnist     bash pod_launch_B1_cls.sh cifar
# Split off pod_launch_B1_regcls.sh so MNIST and CIFAR-10 run in parallel. Journaled; re-run to resume.
set -euo pipefail
MODEL=${1:?mnist|cifar}
source "$(dirname "$0")/pod_common.sh"
pod_provision
pod_preflight "B1_$MODEL"
cd "$REPO"
echo "[POD B1-$MODEL] $(date -u) starting"
python -u classification_benchmarks.py --model "$MODEL" --quiet 2>&1 | tee -a "logs/pod_B1_$MODEL.log"
echo "[POD B1-$MODEL] $(date -u) done"

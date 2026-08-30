#!/usr/bin/env bash
# Upload the repo tarball to a pod and launch one pod_launch_*.sh inside tmux.
#
#   bash pod_bootstrap.sh <ip> <port> <repo.tgz> <launch-script> [<extra args...>]
#   e.g. bash pod_bootstrap.sh 1.2.3.4 10001 /tmp/repo_4643eaa.tgz pod_launch_A1_base_wd_a0.sh
#
# The launch script runs pod_preflight.sh first and aborts on failure; watch it with
#   ssh -p <port> root@<ip> 'tail -f /workspace/Regularization_Decay/logs/pod_<tag>_full.log'
set -euo pipefail
IP=${1:?ip}; PORT=${2:?port}; TGZ=${3:?repo tarball}; LAUNCH=${4:?launch script}; shift 4
SO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30"
TAG=${LAUNCH#pod_launch_}; TAG=${TAG%.sh}
# BOOT_TAG overrides the log name tag so it matches the pods.conf tag when one launch
# script serves several pods (e.g. a roster split, or a replacement pod).
TAG=${BOOT_TAG:-$TAG}
# Extra arguments are shell-quoted for the remote tmux command: method names contain
# parentheses ("Tau(w)"), which otherwise break the remote bash with a syntax error.
ARGS=""; for a in "$@"; do ARGS+=" $(printf '%q' "$a")"; done

ok=0
for i in 1 2 3 4 5 6 7 8 9 10; do
  if ssh $SO -o BatchMode=yes -p "$PORT" "root@$IP" 'echo ssh-ok'; then ok=1; break; fi
  echo "[BOOT] ssh not ready (attempt $i), retrying in 30 s"; sleep 30
done
[ "$ok" -eq 1 ] || { echo "[BOOT] FAILED: cannot ssh to root@$IP:$PORT (key not injected? image still pulling?)"; exit 1; }
scp $SO -P "$PORT" "$TGZ" "root@$IP:/workspace/repo.tgz"
ssh $SO -p "$PORT" "root@$IP" "set -e; cd /workspace && rm -rf Regularization_Decay && tar -xzf repo.tgz && rm repo.tgz \
  && cd Regularization_Decay && mkdir -p logs && (command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)) \
  && tmux new-session -d -s run \"bash $LAUNCH$ARGS 2>&1 | tee -a logs/pod_${TAG}_full.log\" \
  && echo '[BOOT] launched in tmux session run' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
echo "[BOOT] follow: ssh -p $PORT root@$IP 'tail -f /workspace/Regularization_Decay/logs/pod_${TAG}_full.log'"

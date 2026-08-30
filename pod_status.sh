#!/usr/bin/env bash
# Status of every pod in pods.conf, optionally pulling whatever exists right now.
#
#   bash pod_status.sh          # status only
#   bash pod_status.sh --pull   # + download partial results, journals (tagged), instrumentation
#   bash pod_status.sh --pull A1_tau   # one pod only
#
# pods.conf lines: "<tag> <ip> <port> <results stem>". Safe at any time; never interrupts a run.
set -uo pipefail
CONF=${CONF:-pods.conf}
[ -f "$CONF" ] || { echo "no $CONF (see pods.conf.example)"; exit 1; }
PULL=0; [[ "${1:-}" == "--pull" ]] && PULL=1
ONLY=${2:-}          # optional tag: act on that pod only (e.g. --pull A1_tau)
SO="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
R=/workspace/Regularization_Decay
# macOS has no GNU `timeout`: fall back to running the command directly (ConnectTimeout still applies).
if command -v timeout >/dev/null 2>&1; then T() { timeout "$@"; }; else T() { shift; "$@"; }; fi
while read -r tag ip port stem; do
  [[ -z "${tag:-}" || "$tag" == \#* ]] && continue
  [[ -n "$ONLY" && "$tag" != "$ONLY" ]] && continue
  echo "════ pod $tag ($ip:$port) — $stem ════"
  T 40 ssh -n $SO -p "$port" "root@$ip" "
    cd $R 2>/dev/null || { echo '  unreachable / wiped'; exit 0; }
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  gpu: /'
    tmux ls 2>/dev/null | sed 's/^/  tmux: /' || echo '  tmux: no session (run finished or died)'
    j=results/journal/${stem}.jsonl
    if [ -f \"\$j\" ]; then
      echo \"  journal: \$(grep -c '\"kind\": \"pso\"' \$j) PSO evals, \$(grep -c '\"kind\": \"best_hp\"' \$j) winners, \$(grep -c '\"kind\": \"eval\"' \$j) final runs, \$(du -h \$j | cut -f1)\"
    else
      echo '  journal: not created yet (first run still in flight)'
    fi
    c=results/${stem}.csv
    [ -f \"\$c\" ] && { echo '  partial CSV (method, mean, std):'; sed -n '2,\$p' \"\$c\" | cut -d, -f1,2,3 | sed 's/^/    /'; }
    grep -cE 'Traceback|RuntimeError|OutOfMemory|\[PSO\]\[FAIL' logs/pod_${tag}.log 2>/dev/null | sed 's/^/  failures logged: /'
    grep -h 'PREFLIGHT\] OK' logs/preflight_${tag}.log 2>/dev/null | tail -1 | sed 's/^/  /'
  " 2>/dev/null
  if [ "$PULL" -eq 1 ]; then
    mkdir -p "new results/incoming/$tag" results/journal results/instrumentation/pso
    # Every result file and every journal the pod has produced (a pod may run several
    # benchmarks); journals are stored under <stem>.<tag>.jsonl so split rosters can be
    # concatenated later by pod_collect.sh.
    T 120 scp $SO -P "$port" "root@$ip:$R/results/*_standardized_results.*" "new results/incoming/$tag/" 2>/dev/null \
      && echo "  ↓ pulled results -> new results/incoming/$tag/ ($(ls "new results/incoming/$tag" | wc -l | tr -d ' ') files)" || echo "  ↓ nothing to pull yet"
    mkdir -p "results/journal/incoming_$tag"
    T 120 scp $SO -P "$port" "root@$ip:$R/results/journal/*.jsonl" "results/journal/incoming_$tag/" 2>/dev/null \
      && for j in results/journal/incoming_$tag/*.jsonl; do [ -e "$j" ] && mv "$j" "results/journal/$(basename "${j%.jsonl}").${tag}.jsonl"; done \
      && echo "  ↓ pulled journal(s) -> results/journal/*.${tag}.jsonl" || true
    rmdir "results/journal/incoming_$tag" 2>/dev/null || true
    T 300 scp $SO -P "$port" "root@$ip:$R/results/instrumentation/*.json" results/instrumentation/ 2>/dev/null \
      && echo "  ↓ pulled instrumentation" || true
    T 300 scp $SO -P "$port" "root@$ip:$R/results/instrumentation/pso/*.json" results/instrumentation/pso/ 2>/dev/null \
      && echo "  ↓ pulled PSO probe trajectories" || true
  fi
  echo
done < "$CONF"
cat <<'TXT'
Launch (on the pod, inside tmux so it survives the SSH session):
  tmux new-session -d -s run "bash pod_launch_<X>.sh 2>&1 | tee -a logs/pod_<X>_full.log"
Stop a run:      tmux kill-session -t run      Resume: launch the same script again (journaled)
Stop billing:    DELETE https://rest.runpod.io/v1/pods/<pod-id>  — /workspace is WIPED: pull first!
TXT

#!/usr/bin/env bash
# Finish a pod that has completed its run: pull results, verify the expected number of
# final runs per method in its journal, DELETE the pod via the RunPod API and drop it from
# pods.conf. Requires RUNPOD_API_KEY.
#   bash pod_finish.sh <tag> <journal stem> '<json {"method": n_finals, ...}>'
set -uo pipefail
tag=$1; stem=$2; exp=$3
line=$(grep "^$tag " pods.conf | tail -1); ip=$(awk '{print $2}' <<<"$line"); port=$(awk '{print $3}' <<<"$line")
[ -n "$ip" ] || { echo "$tag not in pods.conf"; exit 1; }
state=$(ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR -p "$port" "root@$ip" 'tmux has-session -t run 2>/dev/null && echo RUN || echo END' 2>/dev/null)
[ "$state" = "END" ] || { echo "$tag: still running (or unreachable: '$state')"; exit 2; }
bash pod_status.sh --pull "$tag" 2>&1 | grep "↓"
python3 - "$tag" "$stem" "$exp" <<'PY'
import json, os, re, io, sys, urllib.request
tag, stem, exp = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
conf = io.open('pods.conf', encoding='utf-8').read()
pid = re.search(rf'^# {tag} id=(\S+)', conf, re.M).group(1)
counts = {}
for line in open(f'results/journal/{stem}_standardized_results.{tag}.jsonl'):
    try: r = json.loads(line)
    except Exception: continue
    if r.get('kind') == 'eval': counts[r['method']] = counts.get(r['method'], 0) + 1
ok = all(counts.get(m, 0) >= n for m, n in exp.items()) and os.path.exists(f'new results/incoming/{tag}/{stem}_standardized_results.csv')
print(tag, {m: counts.get(m, 0) for m in exp}, 'OK' if ok else 'INCOMPLETE')
if not ok: sys.exit(3)
req = urllib.request.Request(f'https://rest.runpod.io/v1/pods/{pid}', method='DELETE', headers={'Authorization': f'Bearer {os.environ["RUNPOD_API_KEY"]}'})
try:
    with urllib.request.urlopen(req, timeout=60) as r: st = r.status
except urllib.error.HTTPError as e: st = e.code
print(f'DELETE {tag} {pid} -> {st}')
if st in (200, 204):
    keep = [l for l in conf.splitlines() if not (l.startswith(f'{tag} ') or l.startswith(f'# {tag} id='))]
    keep.append(f'# DONE + pulled + deleted: {tag}={pid}')
    io.open('pods.conf', 'w', encoding='utf-8').write('\n'.join(keep) + '\n')
PY

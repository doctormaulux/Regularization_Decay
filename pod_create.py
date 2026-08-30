#!/usr/bin/env python3
"""Create RunPod GPU pods for the regeneration runs and record them in pods.conf.

    python pod_create.py --tag A1_base_wd_a0 --stem gpt2_large_wikitext_standardized_results \
        [--gpu "NVIDIA RTX A4500" --gpu "NVIDIA GeForce RTX 3090"] [--cloud COMMUNITY] [--dry-run]

Reads the API key from $RUNPOD_API_KEY. Creates the pod (REST POST /v1/pods), waits until
it has a public IP and an SSH port mapping, prints the ssh command and appends the line
"<tag> <ip> <port> <stem>" to pods.conf (git-ignored). Never prints the key.
"""
import argparse
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request

API = 'https://rest.runpod.io/v1'
IMAGE = 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04'
DEFAULT_GPUS = ['NVIDIA RTX A4500', 'NVIDIA GeForce RTX 3090', 'NVIDIA RTX A5000']


def _req(method, path, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={'Authorization': f'Bearer {key}',
                                          'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b'null')
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'null')
    except (urllib.error.URLError, TimeoutError, OSError) as e:   # transient network error
        return 0, {'error': str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--stem', required=True, help='results stem for pods.conf')
    ap.add_argument('--gpu', action='append', help='gpuTypeId (repeatable, priority order)')
    ap.add_argument('--cloud', default='COMMUNITY', choices=['COMMUNITY', 'SECURE'])
    ap.add_argument('--disk', type=int, default=40, help='containerDiskInGb')
    ap.add_argument('--volume', type=int, default=30, help='volumeInGb (persists /workspace)')
    ap.add_argument('--image', default=IMAGE)
    ap.add_argument('--conf', default='pods.conf')
    ap.add_argument('--pubkey', default=os.path.expanduser('~/.ssh/id_ed25519.pub'),
                    help='SSH public key file injected as PUBLIC_KEY (RunPod adds it to '
                         'authorized_keys at start; pods created via the API get NO key '
                         'otherwise)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    try:
        pubkey = open(args.pubkey, encoding='utf-8').read().strip()
    except OSError as exc:
        sys.exit(f'cannot read {args.pubkey}: {exc}')

    key = os.environ.get('RUNPOD_API_KEY')
    if not key:
        sys.exit('set RUNPOD_API_KEY')
    body = {
        'name': f'regdecay-{args.tag}',
        'cloudType': args.cloud,
        'computeType': 'GPU',
        'imageName': args.image,
        'gpuTypeIds': args.gpu or DEFAULT_GPUS,
        'gpuTypePriority': 'custom',
        'gpuCount': 1,
        'containerDiskInGb': args.disk,
        'volumeInGb': args.volume,
        'volumeMountPath': '/workspace',
        'ports': ['22/tcp', '8888/http'],
        'supportPublicIp': True,
        'minVCPUPerGPU': 4,
        'minRAMPerGPU': 16,
        'allowedCudaVersions': ['12.4', '12.5', '12.6', '12.7', '12.8', '12.9', '13.0'],
        # PUBLIC_KEY -> /root/.ssh/authorized_keys; JUPYTER_PASSWORD locks the 8888 proxy.
        'env': {'PUBLIC_KEY': pubkey, 'JUPYTER_PASSWORD': secrets.token_urlsafe(18)},
    }
    shown = dict(body, env={'PUBLIC_KEY': pubkey[:20] + '...', 'JUPYTER_PASSWORD': '<random>'})
    print('[CREATE]', json.dumps(shown))
    if args.dry_run:
        return
    st, pod = _req('POST', '/pods', key, body)
    if st not in (200, 201):
        sys.exit(f'[CREATE] HTTP {st}: {json.dumps(pod)[:500]}')
    pid = pod['id']
    print(f'[CREATE] pod {pid} ({pod.get("machine", {}).get("gpuTypeId")}) '
          f'costPerHr={pod.get("costPerHr")}')

    ip = port = None
    for _ in range(60):                       # up to ~10 min
        time.sleep(10)
        st, p = _req('GET', f'/pods/{pid}', key)
        if st != 200 or not isinstance(p, dict):
            print(f'[WAIT] HTTP {st} (retrying)')
            continue
        ip = p.get('publicIp')
        pm = p.get('portMappings') or {}
        port = pm.get('22')
        print(f'[WAIT] status={p.get("desiredStatus")} ip={ip} ssh_port={port} '
              f'gpu={p.get("machine", {}).get("gpuTypeId")}')
        if ip and port:
            break
    if not (ip and port):
        sys.exit(f'[WAIT] pod {pid} has no public IP/SSH port yet; check the console')
    line = f'{args.tag} {ip} {port} {args.stem}'
    with open(args.conf, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')
        fh.write(f'# {args.tag} id={pid} gpu={p.get("machine", {}).get("gpuTypeId")} '
                 f'costPerHr={p.get("costPerHr")} created={time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}\n')
    print(f'[OK] pods.conf += "{line}"   pod id {pid}')
    print(f'[OK] ssh -p {port} root@{ip}')


if __name__ == '__main__':
    main()

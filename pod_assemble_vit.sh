#!/usr/bin/env bash
# Assemble the ViT-mini roster from per-pod journals (runs on CPU, trains nothing):
#   bash pod_assemble_vit.sh results/journal/vit_cifar_standardized_results.B1_vit*.jsonl
set -euo pipefail
[ "$#" -ge 1 ] || { echo "usage: $0 <journal.jsonl> [...]"; exit 1; }
J=results/journal/vit_cifar_standardized_results.jsonl
mkdir -p results/journal
cat "$@" > "$J"
echo "[ASSEMBLE] $(wc -l < "$J") journal records from $# file(s)"
python -u vit_cifar_standardized.py
echo "[ASSEMBLE] done -> results/vit_cifar_standardized_results.{csv,json}"

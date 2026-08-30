#!/usr/bin/env bash
# The pod fleet of the 2026-08-29 regeneration: which launch script ran on which pod.
# A1_base_wd_a0 was created and launched first (its preflight validated the GPU path);
# the others were created with pod_create.py and started with pod_bootstrap.sh, one by
# one, in this order. Re-running the loop is idempotent: a tag already present in
# pods.conf is not re-created (only bootstrapped again, which resumes from the journal).
#
#   RUNPOD_API_KEY=... bash pod_fleet_2026-08-29.sh <repo.tgz>
set -uo pipefail
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
TGZ=${1:?repo tarball}
GPUS=(--gpu "NVIDIA GeForce RTX 4090" --gpu "NVIDIA RTX A5000" --gpu "NVIDIA RTX A4500" --gpu "NVIDIA GeForce RTX 3090")
FLEET=(
  "A1_base_wd_a0|gpt2_large_wikitext_standardized_results|pod_launch_A1_base_wd_a0.sh"
  "A1_tau|gpt2_large_wikitext_standardized_results|pod_launch_A1_tau.sh"
  "A1_adamw|gpt2_large_wikitext_standardized_results|pod_launch_A1_adamw.sh"
  "A1_en_l2|gpt2_large_wikitext_standardized_results|pod_launch_A1_en_l2.sh"
  "A2_small|gpt2_wikitext_standardized_results|pod_launch_A2_small.sh"
  "B1_custom|smollm2_wikitext_standardized_results|pod_launch_B1_custom.sh"
  "B1_vit|vit_cifar_standardized_results|pod_launch_B1_vit.sh"
  "B3|gpt2_medium_wikitext_standardized_results|pod_launch_B3_tiny_medium.sh"
)
for entry in "${FLEET[@]}"; do
  IFS='|' read -r tag stem script <<<"$entry"
  echo "=================== $tag  ($(date -u +%H:%M:%S) UTC) ==================="
  if grep -q "^$tag " pods.conf 2>/dev/null; then
    echo "[FLEET] $tag already in pods.conf - not re-created"
  else
    python3 pod_create.py --tag "$tag" --stem "$stem" "${GPUS[@]}" \
      || { echo "[FLEET] CREATE FAILED for $tag"; continue; }
  fi
  line=$(grep "^$tag " pods.conf | tail -1)
  ip=$(awk '{print $2}' <<<"$line"); port=$(awk '{print $3}' <<<"$line")
  [ -n "$ip" ] && [ -n "$port" ] || { echo "[FLEET] no ip/port for $tag"; continue; }
  bash pod_bootstrap.sh "$ip" "$port" "$TGZ" "$script" \
    || echo "[FLEET] BOOTSTRAP FAILED for $tag ($ip:$port)"
done
echo "=================== fleet done $(date -u) ==================="

#!/usr/bin/env bash
set -euo pipefail
readonly project="/home/bjtc/Sophix/lingbot-vla"
readonly output="$project/output/package_a2d_lora_4090"
readonly report="$output/hourly_health.log"
mkdir -p "$output"
{
  printf '\n===== %s =====\n' "$(date --iso-8601=seconds)"
  printf 'training_service='; systemctl --user is-active lingbot-vla-package-a2d.service || true
  printf '%s\n' '-- latest training record --'
  tr '\r' '\n' < "$output/train.log" 2>/dev/null | tail -n 2 || true
  printf '%s\n' '-- GPU --'
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1 || true
  printf '%s\n' '-- memory and disk --'
  free -h | tail -n 2 || true
  df -h "$project" | tail -n 1 || true
  printf '%s\n' '-- kernel safety events in preceding 65 minutes --'
  journalctl -k -S '-65 min' --no-pager 2>&1 | grep -Ei 'bad page|nvrm|xid|oom|out of memory|ext4|i/o error|segfault' || true
} >> "$report"

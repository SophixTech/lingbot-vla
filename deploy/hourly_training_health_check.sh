#!/usr/bin/env bash
# Read-only, low-priority health snapshot for an active LingBot-VLA training run.
# This never starts/stops training and never invokes CUDA/Isaac Sim/model inference.
set -euo pipefail

readonly project="/home/bjtc/Sophix/lingbot-vla"
readonly output="$project/output/ruantong_a2d_parts_lora_4090"
readonly train_unit="lingbot-vla-ruantong-a2d.service"
readonly guard_unit="lingbot-vla-ruantong-a2d-guard.service"
readonly report="$output/hourly_health.log"

mkdir -p "$output"

{
    printf '\n===== %s =====\n' "$(date --iso-8601=seconds)"
    printf 'training_service='; systemctl --user is-active "$train_unit" || true
    printf 'guard_service='; systemctl --user is-active "$guard_unit" || true
    printf '%s\n' '-- latest training record --'
    tr '\r' '\n' < "$output/train.log" | tail -n 1 || true
    printf '%s\n' '-- GPU --'
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader 2>&1 || true
    printf '%s\n' '-- memory and disk --'
    free -h | tail -n 2 || true
    df -h "$project" | tail -n 1 || true
    printf '%s\n' '-- kernel safety events in preceding 65 minutes --'
    /usr/bin/journalctl -k -S '-65 min' --no-pager 2>&1 | \
        /usr/bin/grep -Ei 'bad page|nvrm|xid|oom|out of memory|ext4|i/o error|segfault' || true
} >> "$report"

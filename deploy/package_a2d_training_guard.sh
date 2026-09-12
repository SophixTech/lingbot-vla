#!/usr/bin/env bash
set -euo pipefail

readonly service="lingbot-vla-package-a2d.service"
readonly output="/home/bjtc/Sophix/lingbot-vla/output/package_a2d_lora_4090"
readonly report="$output/guard.log"
readonly train_log="$output/train.log"
readonly danger_pattern='bad page|page state|nvrm|xid|gpu has fallen off|out of memory|oom-killer|ext4|i/o error|corruption'

# Do nothing after normal completion. The service is deliberately stopped only
# while training is active and a new host-level safety event is observed.
if [[ "$(systemctl --user is-active "$service" 2>/dev/null || true)" != "active" ]]; then
  exit 0
fi

# A live systemd unit alone is not sufficient evidence of progress. Allow a
# generous window for checkpoint serialization, then stop a stalled run.
if [[ -f "$train_log" ]]; then
  now="$(date +%s)"
  log_mtime="$(stat -c %Y "$train_log")"
  if (( now - log_mtime > 1800 )); then
    mkdir -p "$output"
    printf '\n===== %s =====\nNo training-log progress for %s seconds. Stopping %s.\n' \
      "$(date --iso-8601=seconds)" "$((now - log_mtime))" "$service" >> "$report"
    systemctl --user stop "$service"
    exit 0
  fi
fi

if journalctl -k -S '-45 seconds' --no-pager 2>/dev/null | grep -Eqi "$danger_pattern"; then
  mkdir -p "$output"
  {
    printf '\n===== %s =====\n' "$(date --iso-8601=seconds)"
    printf 'Safety event detected. Stopping %s.\n' "$service"
    journalctl -k -S '-45 seconds' --no-pager | grep -Ei "$danger_pattern" || true
  } >> "$report"
  systemctl --user stop "$service"
  exit 0
fi

# A user-space evaluator can segfault independently of the training process.
# Record it for diagnosis, but do not stop a healthy GPU training job; kernel,
# driver, memory, and filesystem faults above remain fatal safety events.
if journalctl -k -S '-45 seconds' --no-pager 2>/dev/null | grep -Eqi 'segfault.*(python|libarrow)|python\[[0-9]+\].*segfault'; then
  mkdir -p "$output"
  {
    printf '\n===== %s =====\n' "$(date --iso-8601=seconds)"
    printf 'User-space evaluator segfault observed; training left running.\n'
    journalctl -k -S '-45 seconds' --no-pager | grep -Ei 'segfault.*(python|libarrow)|python\[[0-9]+\].*segfault' || true
  } >> "$report"
fi

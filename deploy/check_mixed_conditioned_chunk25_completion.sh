#!/usr/bin/env bash
set -u

OUT=/home/bjtc/Sophix/lingbot-vla/output/separated_model/mixed_conditioned_chunk25
CKPT="$OUT/checkpoints/global_step_10000"
REPORT="$OUT/completion_check.log"
UNIT=mixed-conditioned-chunk25-train.service

timestamp=$(date '+%F %T %Z')
service_status=$(systemctl --user is-active "$UNIT" 2>/dev/null || true)
latest_step=$(rg -o 'Step: [0-9]+/10000' "$OUT/train.log" 2>/dev/null | tail -n 1 | sed -E 's/.*Step: ([0-9]+)\/10000/\1/' || true)
checkpoint_status=incomplete
if test -f "$CKPT/extra_state/extra_state_rank_0.pt" \
  && test -f "$CKPT/model/.metadata" \
  && test -f "$CKPT/hf_ckpt/model.safetensors.index.json"; then
  checkpoint_status=complete
fi

printf '%s service=%s step=%s final_checkpoint=%s disk=%s\n' \
  "$timestamp" "${service_status:-unknown}" "${latest_step:-unknown}" "$checkpoint_status" \
  "$(df -h /home/bjtc | awk 'NR==2 {print $5 "," $4 " free"}')" >> "$REPORT"

#!/usr/bin/env bash
set -u

ROOT=/home/bjtc/Sophix/lingbot-vla
OUT="$ROOT/output/separated_model/mixed_conditioned"
CKPTS="$OUT/checkpoints"
LOG="$OUT/supervisor.log"
TRAIN_UNIT=mixed-conditioned-train-persistent.service
LEGACY_TRAIN_UNIT=mixed-conditioned-train.service

valid_checkpoint() {
  local d="$1"
  test -f "$d/extra_state/extra_state_rank_0.pt" \
    && test -f "$d/model/.metadata" \
    && test -f "$d/hf_ckpt/model.safetensors.index.json"
}

latest_valid_checkpoint() {
  local d step best_step=-1 best=''
  for d in "$CKPTS"/global_step_*; do
    test -d "$d" || continue
    step=${d##*_}
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    valid_checkpoint "$d" || continue
    if (( step > best_step )); then
      best_step=$step
      best=$d
    fi
  done
  test -n "$best" && printf '%s\n' "$best"
}

check_once() {
  local now status step ckpt newest_log log_age now_epoch log_epoch active_unit
  now=$(date '+%F %T %Z')
  active_unit=''
  if [[ "$(systemctl --user is-active "$TRAIN_UNIT" 2>/dev/null || true)" == active ]]; then
    active_unit="$TRAIN_UNIT"
  elif [[ "$(systemctl --user is-active "$LEGACY_TRAIN_UNIT" 2>/dev/null || true)" == active ]]; then
    active_unit="$LEGACY_TRAIN_UNIT"
  fi
  status=inactive
  [[ -n "$active_unit" ]] && status=active
  newest_log=$(find "$OUT" -maxdepth 1 -type f -name 'resume*.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-)
  step=$(test -n "$newest_log" && rg -o 'Step: [0-9]+/10000' "$newest_log" 2>/dev/null \
    | tail -n 1 | sed -E 's/.*Step: ([0-9]+)\/10000/\1/' || true)
  printf '%s status=%s step=%s disk=%s\n' "$now" "${status:-unknown}" "${step:-unknown}" "$(df -h /home/bjtc | awk 'NR==2{print $5 "," $4 " free"}')" >> "$LOG"

  if [[ "$status" == active ]]; then
    now_epoch=$(date +%s)
    log_epoch=$(test -n "$newest_log" && stat -c %Y "$newest_log" 2>/dev/null || echo 0)
    log_age=$((now_epoch - log_epoch))
    if (( log_age < 900 )); then
      return 0
    fi
    printf '%s active but log stale (%ss); restarting\n' "$now" "$log_age" >> "$LOG"
  fi
  if valid_checkpoint "$CKPTS/global_step_10000"; then
    printf '%s final checkpoint valid; supervisor stopping\n' "$now" >> "$LOG"
    return 1
  fi

  ckpt=$(latest_valid_checkpoint || true)
  if [[ -z "$ckpt" ]]; then
    printf '%s no valid checkpoint; no restart\n' "$now" >> "$LOG"
    return 0
  fi
  printf '%s restarting from %s\n' "$now" "$ckpt" >> "$LOG"
  [[ -z "$active_unit" ]] || systemctl --user stop "$active_unit" >/dev/null 2>&1 || true
  systemctl --user reset-failed "$TRAIN_UNIT" >/dev/null 2>&1 || true
  systemctl --user start "$TRAIN_UNIT" >> "$LOG" 2>&1 || true
  return 0
}

mkdir -p "$OUT"
while check_once; do
  sleep 3600
done
exit 0

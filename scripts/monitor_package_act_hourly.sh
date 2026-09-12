#!/usr/bin/env bash
# Read-only hourly monitor for the offline package ACT job.
set -euo pipefail

readonly OUTPUT_DIR="/home/bjtc/Sophix/lingbot-vla/output/package_act_4090"
readonly METRICS="$OUTPUT_DIR/metrics.jsonl"
readonly LOG="$OUTPUT_DIR/hourly_monitor.log"
readonly PLOTTER="/home/bjtc/Sophix/lingbot-vla/scripts/plot_package_act_metrics.py"

while pgrep -f '/lingbot-vla/scripts/train_package_act.py.*output/package_act_4090' >/dev/null; do
  for _ in $(seq 1 60); do
    sleep 60
    if ! pgrep -f '/lingbot-vla/scripts/train_package_act.py.*output/package_act_4090' >/dev/null; then
      break
    fi
  done

  {
    date --iso-8601=seconds
    pgrep -af '/lingbot-vla/scripts/train_package_act.py.*output/package_act_4090' || true
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
    tail -n 4 "$METRICS" 2>/dev/null || true
    find "$OUTPUT_DIR" -maxdepth 1 -name 'checkpoint_*.pt' -printf '%f\n' | sort | tail -n 3
  } >>"$LOG"
  /home/bjtc/Sophix/lingbot-vla/.venv/bin/python "$PLOTTER" "$OUTPUT_DIR" >>"$LOG" 2>&1 || true
done

{
  date --iso-8601=seconds
  echo 'Training process is no longer running.'
  tail -n 8 "$METRICS" 2>/dev/null || true
} >>"$LOG"

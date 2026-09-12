#!/usr/bin/env bash
set -euo pipefail

readonly repo="/home/bjtc/Sophix/lingbot-vla"
readonly output="$repo/output/interpolation_vla_num_denoising_step_20"
readonly report="$output/hourly_health.log"
readonly train_unit="lingbot-vla-interpolation-step20-train.service"

while true; do
  {
    date '+=== %F %T %z ==='
    ps -eo pid,etime,%cpu,%mem,stat,args | grep '[t]rain_lingbotvla.py.*interpolation_vla_num_denoising_step_20' || true
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    if [[ -s "$output/checkpoints/loss.jsonl" ]]; then
      tail -1 "$output/checkpoints/loss.jsonl"
      "$repo/.venv/bin/python" "$repo/scripts/plot_interpolation_vla_metrics.py" "$output" || true
    fi
    if [[ -d "$output/checkpoints/global_step_10000/hf_ckpt" ]]; then
      touch "$output/training_complete"
      echo 'global_step_10000/hf_ckpt found; monitor exiting successfully.'
      exit 0
    fi
    if [[ "$(systemctl --user is-active "$train_unit" 2>/dev/null || true)" != "active" ]]; then
      systemctl --user show "$train_unit" -p ActiveState -p SubState -p ExecMainCode -p ExecMainStatus || true
      touch "$output/training_failed"
      echo 'Training service is inactive before global_step_10000; monitor exiting with failure.'
      exit 1
    fi
  } >> "$report" 2>&1
  sleep 3600
done

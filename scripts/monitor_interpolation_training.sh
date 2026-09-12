#!/usr/bin/env bash
set -u
repo=/home/bjtc/Sophix/lingbot-vla
vla=$repo/output/interpolation_vla
act=$repo/output/interpolation_ACT
log=$repo/output/interpolation_training_hourly.log
while true; do
  {
    date '+=== %F %T %z ==='
    ps -eo pid,etime,%cpu,%mem,stat,cmd | grep -E 'train_lingbotvla|train_package_act' | grep -v grep || true
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    if [[ -s $vla/checkpoints/loss.jsonl ]]; then
      tail -1 $vla/checkpoints/loss.jsonl
      "$repo/.venv/bin/python" "$repo/scripts/plot_interpolation_vla_metrics.py" "$vla" || true
    fi
    if [[ -s $act/metrics.jsonl ]]; then
      tail -3 $act/metrics.jsonl
      "$repo/.venv/bin/python" "$repo/scripts/plot_package_act_metrics.py" "$act" || true
    fi
  } >> "$log" 2>&1
  sleep 3600
done

#!/usr/bin/env bash
set -euo pipefail
repo=/home/bjtc/Sophix/lingbot-vla
mkdir -p "$repo/output/interpolation_ACT"
while pgrep -f '[t]asks/vla/train_lingbotvla.py.*interpolation_vla' >/dev/null; do sleep 60; done
if [[ ! -d "$repo/output/interpolation_vla/checkpoints/global_step_10000" ]]; then
  echo "VLA did not produce global_step_10000; refusing to start ACT" >&2
  exit 1
fi
if [[ -f "$repo/output/interpolation_ACT/checkpoint_010000.pt" ]]; then exit 0; fi
nohup "$repo/scripts/run_interpolation_act.sh" </dev/null >"$repo/output/interpolation_ACT/launcher.log" 2>&1 &
echo $! > "$repo/output/interpolation_ACT/launcher.pid"

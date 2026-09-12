#!/usr/bin/env bash
set -euo pipefail

readonly repo=/home/bjtc/Sophix/lingbot-vla
readonly vla_output="$repo/output/interpolation_vla"
readonly act_output="$repo/output/interpolation_ACT"

if [[ ! -d "$vla_output/checkpoints/global_step_10000" ]]; then
  "$repo/scripts/run_interpolation_vla.sh"
fi

if [[ ! -d "$vla_output/checkpoints/global_step_10000" ]]; then
  echo "VLA exited without global_step_10000" >&2
  exit 1
fi

if [[ ! -f "$act_output/checkpoint_010000.pt" ]]; then
  # ACT may have been deliberately started in parallel with VLA. Wait for it
  # instead of launching a second writer against the same output directory.
  while pgrep -f '[t]rain_package_act.py.*output/interpolation_ACT' >/dev/null; do
    sleep 60
  done
  if [[ ! -f "$act_output/checkpoint_010000.pt" ]]; then
    "$repo/scripts/run_interpolation_act.sh"
  fi
fi

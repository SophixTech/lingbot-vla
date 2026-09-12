#!/usr/bin/env bash
set -euo pipefail

readonly project="/home/bjtc/Sophix/lingbot-vla"
readonly output="$project/output/package_a2d_lora_4090"
readonly loss_file="$output/checkpoints/loss.jsonl"
readonly curve_file="$output/package_training_loss_current.png"

[[ -s "$loss_file" ]] || exit 0

MPLCONFIGDIR=/tmp/package_lingbot_mpl \
  "$project/.venv/bin/python" "$project/scripts/export_training_curve.py" \
  "$loss_file" "$curve_file" --window 100

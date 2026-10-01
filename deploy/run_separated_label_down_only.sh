#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/bjtc/Sophix/lingbot-vla
OUT="$ROOT/output/separated_model/label_down"
LOCK="$ROOT/output/separated_model/.label_down_training.lock"

exec 9>"$LOCK"
flock -n 9 || {
  echo "label-down training is already running"
  exit 2
}

export PATH="$ROOT/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
cd "$OUT"

exec bash -o pipefail "$ROOT/train.sh" \
  "$ROOT/tasks/vla/train_lingbotvla.py" \
  "$ROOT/configs/vla/separated_label_down.yaml"

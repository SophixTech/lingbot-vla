#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/bjtc/Sophix/lingbot-vla
OUT="$ROOT/output/few_data_mixed"
LOCK="$ROOT/output/.few_data_mixed_training.lock"

mkdir -p "$OUT"
exec 9>"$LOCK"
flock -n 9 || {
  echo "few-data mixed training is already running"
  exit 2
}

export PATH="$ROOT/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
cd "$OUT"

exec bash -o pipefail "$ROOT/train.sh" \
  "$ROOT/tasks/vla/train_lingbotvla.py" \
  "$ROOT/configs/vla/mixed_few_data.yaml"

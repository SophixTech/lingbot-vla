#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/bjtc/Sophix/lingbot-vla
OUT="$ROOT/output/separated_model/mixed_conditioned"
LOCK="$ROOT/output/separated_model/.mixed_conditioned_training.lock"

mkdir -p "$OUT"
exec 9>"$LOCK"
flock -n 9 || {
  echo "mixed-conditioned training is already running"
  exit 2
}

export PATH="$ROOT/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
cd "$OUT"

exec bash -o pipefail "$ROOT/train.sh" \
  "$ROOT/tasks/vla/train_lingbotvla.py" \
  "$ROOT/configs/vla/mixed_conditioned.yaml"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/bjtc/Sophix/lingbot-vla
LOG="$ROOT/output/separated_model/test_metrics_eval.log"
mkdir -p "$ROOT/output/separated_model"
exec >>"$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] evaluation waiter started"

while systemctl --user is-active --quiet lingbot-vla-separated-label-down.service; do
  echo "[$(date --iso-8601=seconds)] label-down still active; waiting"
  sleep 60
done

echo "[$(date --iso-8601=seconds)] label-down inactive; evaluating latest complete checkpoints"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/evaluate_separated_test_metrics.py"

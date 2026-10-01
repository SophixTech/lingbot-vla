#!/usr/bin/env bash
set -u

ROOT=/home/bjtc/Sophix/lingbot-vla
UP="$ROOT/output/separated_model/label_up"
DOWN="$ROOT/output/separated_model/label_down"
LOG="$ROOT/output/separated_model/check_0300.log"
LOCK="$ROOT/output/separated_model/.check_0300.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] check started"

last_step="$($ROOT/.venv/bin/python - "$UP/checkpoints/loss.jsonl" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print(-1)
    raise SystemExit
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
print(rows[-1].get('step', -1) if rows else -1)
PY
)"
hf="$UP/checkpoints/global_step_10000/hf_ckpt"
if [[ "$last_step" != 10000 || ! -f "$hf/config.json" || ! -f "$hf/lingbotvla_cli.yaml" || ! -f "$hf/model.safetensors.index.json" ]]; then
  echo "[$(date --iso-8601=seconds)] label_up NOT ready: last_step=$last_step hf_ckpt=$hf"
  exit 0
fi
echo "[$(date --iso-8601=seconds)] label_up ready: last_step=10000 and hf_ckpt metadata present"

if pgrep -f "$ROOT/tasks/vla/train_lingbotvla.py.*separated_label_down.yaml" >/dev/null; then
  echo "[$(date --iso-8601=seconds)] label_down already running; no duplicate start"
  exit 0
fi
if [[ -f "$DOWN/checkpoints/loss.jsonl" ]]; then
  echo "[$(date --iso-8601=seconds)] label_down has a loss log; no duplicate start"
  exit 0
fi

echo "[$(date --iso-8601=seconds)] starting label_down"
cd "$DOWN"
export PATH="$ROOT/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
exec bash "$ROOT/train.sh" \
  "$ROOT/tasks/vla/train_lingbotvla.py" \
  "$ROOT/configs/vla/separated_label_down.yaml"

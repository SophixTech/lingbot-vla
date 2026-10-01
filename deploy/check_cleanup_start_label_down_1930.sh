#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/bjtc/Sophix/lingbot-vla
UP="$ROOT/output/separated_model/label_up"
DOWN="$ROOT/output/separated_model/label_down"
LOG="$ROOT/output/separated_model/check_cleanup_start_1930.log"
LOCK="$ROOT/output/separated_model/.check_cleanup_start_1930.lock"

exec 9>"$LOCK"
flock -n 9 || exit 0
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] check started"

if systemctl --user is-active --quiet lingbot-vla-separated-label-up.service; then
  echo "[$(date --iso-8601=seconds)] label_up still active; no deletion or label_down start"
  exit 0
fi

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
required=(
  "$UP/checkpoints/global_step_10000"
  "$UP/checkpoints/global_step_10000/model"
  "$UP/checkpoints/global_step_10000/optimizer"
  "$UP/checkpoints/global_step_10000/extra_state"
  "$hf/config.json"
  "$hf/lingbotvla_cli.yaml"
  "$hf/model.safetensors.index.json"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "[$(date --iso-8601=seconds)] label_up incomplete: missing $path; no deletion or label_down start"
    exit 0
  fi
done
if [[ "$last_step" != 10000 ]]; then
  echo "[$(date --iso-8601=seconds)] label_up incomplete: last_step=$last_step; no deletion or label_down start"
  exit 0
fi

echo "[$(date --iso-8601=seconds)] label_up verified at step 10000; preserving global_step_1000 and global_step_10000"
shopt -s nullglob
for path in "$UP/checkpoints"/global_step_*; do
  base="${path##*/}"
  case "$base" in
    global_step_1000|global_step_10000) ;;
    global_step_[0-9]*)
      echo "[$(date --iso-8601=seconds)] deleting $path"
      rm -rf -- "$path"
      ;;
  esac
done
shopt -u nullglob

if systemctl --user is-active --quiet lingbot-vla-separated-label-down.service; then
  echo "[$(date --iso-8601=seconds)] label_down already active; no duplicate start"
  exit 0
fi
if [[ -f "$DOWN/checkpoints/loss.jsonl" ]]; then
  echo "[$(date --iso-8601=seconds)] label_down already has a loss log; no duplicate start"
  exit 0
fi

echo "[$(date --iso-8601=seconds)] starting label_down service"
systemctl --user start lingbot-vla-separated-label-down.service
echo "[$(date --iso-8601=seconds)] label_down service start requested"

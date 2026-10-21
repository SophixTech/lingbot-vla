#!/usr/bin/env bash
# Run explicitly after activating the training Python environment.
# Optional overrides: bash log/run_training_check.sh --train.micro_batch_size=1
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
CHECK_STEPS=${CHECK_STEPS:-20}
CONFIG=configs/vla/package_multi_posttrain.yaml
OUTPUT_ROOT=$(python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml
print(Path(yaml.safe_load(Path(sys.argv[1]).read_text())["train"]["output_dir"]).resolve())
PY
)
mkdir -p "$OUTPUT_ROOT"
RUN_DIR=$(mktemp -d "$OUTPUT_ROOT/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
RUN_NAME=${RUN_DIR##*/}
OUTPUT_DIR="$RUN_DIR"
GPU_MONITOR_PID=''

cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "$GPU_MONITOR_PID" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  date -Is > "$RUN_DIR/end_time.txt"
  printf '%s\n' "$status" > "$RUN_DIR/exit_code.txt"
  if [[ -f "$RUN_DIR/train.log" ]]; then
    grep -E 'Step [0-9]+/.*StepTime' "$RUN_DIR/train.log" > "$RUN_DIR/step_metrics.log" || true
  fi
  echo "Exit code: $status; logs: $RUN_DIR"
  exit "$status"
}
trap cleanup EXIT
date -Is > "$RUN_DIR/start_time.txt"
echo "Logs: $RUN_DIR"
echo "Checkpoints: $OUTPUT_DIR"
printf '%s\n' "$OUTPUT_DIR" > "$RUN_DIR/output_dir.txt"
cp "$CONFIG" "$RUN_DIR/config_source.yaml"

{
  date -Is
  command -v python
  python --version
  git rev-parse HEAD
  git status --short
  printf 'CUDA_VISIBLE_DEVICES=%s\nNPROC_PER_NODE=%s\n' "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE"
  nvidia-smi
} > "$RUN_DIR/environment.log" 2>&1

python - "$CONFIG" "$NPROC_PER_NODE" <<'PY' 2>&1 | tee "$RUN_DIR/preflight.log"
import sys
from pathlib import Path
import torch
import yaml
import lingbotvla

module = Path(lingbotvla.__file__).resolve()
print('Code:', module)
assert module.is_relative_to(Path.cwd()), f'Wrong repository: {module}'
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() >= int(sys.argv[2]), 'Not enough visible GPUs'
for i in range(int(sys.argv[2])):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_memory / 1024**3:.1f} GiB')
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
for section, keys in {
    'model': ['model_path', 'config_path', 'tokenizer_path'],
    'data': ['train_path', 'ecot_annotations_path', 'norm_stats_file', 'robot_config_root'],
}.items():
    for key in keys:
        path = Path(cfg[section][key])
        assert path.exists(), f'Missing {section}.{key}: {path}'
        print(f'{section}.{key}: {path}')
annotations = Path(cfg['data']['ecot_annotations_path'])
assert (annotations / 'annotations.jsonl').is_file()
assert (annotations / 'episodes').is_dir()
print('Preflight passed')
PY

# Sample all GPUs: CUDA_VISIBLE_DEVICES may use UUIDs or remapped indices.
# The CSV UUID/index and environment.log identify the devices used by this run.
nvidia-smi \
  --query-gpu=timestamp,index,uuid,name,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 1 > "$RUN_DIR/gpu_metrics.csv" 2> "$RUN_DIR/gpu_monitor.err" &
GPU_MONITOR_PID=$!

TRAIN_CMD=(python -m torch.distributed.run --standalone --nnodes=1
  "--nproc-per-node=$NPROC_PER_NODE"
  tasks/vla/train_lingbotvla.py "$CONFIG"
  "--train.max_steps=$CHECK_STEPS" "--train.save_steps=$CHECK_STEPS"
  --train.enable_resume=false "$@" "--train.output_dir=$OUTPUT_DIR")
printf '%q ' "${TRAIN_CMD[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
"${TRAIN_CMD[@]}" 2>&1 | tee "$RUN_DIR/train.log"

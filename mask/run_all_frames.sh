#!/usr/bin/env bash
set -euo pipefail
# Run from any directory; defaults assume DATA/ and model/ are siblings of lingbot-vla/.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(dirname -- "$repo_root")"
# Do not inherit the previously suggested dependency overlay or install anything.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
python_bin=""
candidates=("${MASK_PYTHON:-}" "$workspace/lingbot-vla-v2/.venv/bin/python" "$repo_root/.venv/bin/python" "$repo_root/.venv-mask-full/bin/python")
for candidate in "${candidates[@]}"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" -c 'import torch, cv2, numpy; from PIL import Image; from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, SamModel, SamProcessor' >/dev/null 2>&1; then
    python_bin="$candidate"
    break
  fi
  echo "Skipping incompatible Python: $candidate" >&2
done
if [[ -z "$python_bin" ]]; then
  echo 'No compatible existing environment found. This script will not install packages.' >&2
  echo 'Check that lingbot-vla-v2/.venv is mounted, or set MASK_PYTHON to a compatible existing Python.' >&2
  exit 1
fi
echo "Using existing Python: $python_bin"
data_root="${MASK_DATA_ROOT:-$workspace/DATA/Dmetdata}"
grounder="${MASK_GROUNDER:-$workspace/model/Qwen3-VL-4B-Instruct}"
sam="${MASK_SAM:-$workspace/model/sam-vit-base}"
mode="${1:-smoke}"
extra=()
case "$mode" in
  check) exec "$python_bin" -c 'import torch, transformers; print("PyTorch:", torch.__version__); print("Transformers:", transformers.__version__); print("Visible GPUs:", torch.cuda.device_count())' ;;
  smoke) extra=(--max-videos-per-dataset 1 --max-frames-per-video 8) ;;
  all) ;;
  *) echo 'Usage: bash mask/run_all_frames.sh [check|smoke|all]' >&2; exit 2 ;;
esac
output="${MASK_OUTPUT:-$repo_root/mask/data-1023-1027-allframes-$mode}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1
exec "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="${MASK_GPUS:-2}" \
  "$repo_root/mask/build_mask_all_frames.py" \
  --datasets \
  "$data_root/1023-data-v2-uppick" \
  "$data_root/1024-data-v3.2-uppick" \
  "$data_root/1025-data-v3.3-uppick-regrasp" \
  "$data_root/1026-data-v3.4-uppick-side" \
  "$data_root/1027-data-v3.4-uppick-side" \
  --grounder "$grounder" --sam "$sam" --output "$output" \
  --batch-size "${MASK_BATCH_SIZE:-4}" "${extra[@]}"

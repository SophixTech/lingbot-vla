#!/usr/bin/env bash
set -euo pipefail
readonly repo="/home/bjtc/Sophix/lingbot-vla"
readonly log="$repo/output/logs/interpolation_ACT/train.log"
mkdir -p "$(dirname "$log")"
cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
exec .venv/bin/python scripts/train_package_act.py \
  --dataset /home/bjtc/Sophix/datasets/package_interpolation_lerobot_act224 \
  --output "$repo/output/interpolation_ACT" \
  --frame-cache /home/bjtc/Sophix/datasets/package_interpolation_lerobot_act224_frames \
  --steps 10000 --batch-size 8 --workers 6 --loader-timeout 600 \
  --prefetch-factor 2 --video-backend torchcodec --eval-every 1000 \
  --save-every 1000 --eval-batches 100 --log-every 100 >> "$log" 2>&1

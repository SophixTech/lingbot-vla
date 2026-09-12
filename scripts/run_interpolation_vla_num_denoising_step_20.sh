#!/usr/bin/env bash
set -euo pipefail

readonly repo="/home/bjtc/Sophix/lingbot-vla"
readonly config="$repo/configs/vla/interpolation_vla_num_denoising_step_20.yaml"
readonly output="$repo/output/interpolation_vla_num_denoising_step_20"
readonly norm="/home/bjtc/Sophix/datasets/package_interpolation_lerobot/norm_stats_interpolation_vla.json"
readonly log="$output/train.log"

mkdir -p "$output"
exec 9>"$output/train.lock"
flock -n 9 || { echo "A step20 training run is already active." >&2; exit 1; }

if [[ ! -f "$norm" ]]; then
  echo "Missing required normalization statistics: $norm" >&2
  exit 1
fi

cd "$repo"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_CACHE=/tmp/interpolation_lingbot_step20_hf_cache
export XDG_CACHE_HOME=/tmp/interpolation_lingbot_step20_xdg_cache
export CUDA_VISIBLE_DEVICES=0
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=62512

exec .venv/bin/python tasks/vla/train_lingbotvla.py "$config" >> "$log" 2>&1

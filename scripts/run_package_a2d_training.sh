#!/usr/bin/env bash
set -euo pipefail

readonly repo="/home/bjtc/Sophix/lingbot-vla"
readonly config="$repo/configs/vla/package_a2d_lora_4090.yaml"
readonly log="$repo/output/package_a2d_lora_4090/train.log"
readonly norm="/home/bjtc/Sophix/datasets/a2d_bag_record_v3/norm_stats_package_a2d.json"
mkdir -p "$(dirname "$log")"
cd "$repo"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_CACHE=/tmp/package_lingbot_hf_cache
export XDG_CACHE_HOME=/tmp/package_lingbot_xdg_cache
export CUDA_VISIBLE_DEVICES=0
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=62510

if [[ ! -f "$norm" ]]; then
  .venv/bin/python scripts/compute_norm.py "$config"
fi

exec .venv/bin/python tasks/vla/train_lingbotvla.py "$config" >> "$log" 2>&1

#!/usr/bin/env bash
set -euo pipefail
readonly project="/home/bjtc/Sophix/lingbot-vla"
readonly step="${1:?checkpoint step required}"
readonly output="$project/output/package_a2d_lora_4090/open_loop_holdout_step${step}"
readonly metrics_path="$output/per_episode_metrics.tsv"
# Fixed samples from the holdout split (episodes 2204-2447) make comparisons
# across checkpoints paired and reproducible.
readonly holdout_episodes=(
  2204 2215 2225 2236 2246 2257 2267 2278 2289 2299 2310 2320
  2331 2341 2352 2362 2373 2383 2394 2405 2415 2426 2436 2447
)
mkdir -p "$output"
cd "$project"
export CUDA_VISIBLE_DEVICES=""
export QWEN25_PATH="/home/bjtc/Sophix/models/Qwen2.5-VL-3B-Instruct-tokenizer"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_DATASETS_CACHE="/tmp/package_a2d_holdout_hf_cache"
export XDG_CACHE_HOME="/tmp/package_a2d_holdout_xdg_cache"
export TOKENIZERS_PARALLELISM=false
export PYARROW_NUM_THREADS=1
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"

# libarrow can segfault after repeated dataset/video reads in one long-lived
# evaluator. Keep every episode in a separate process so a native-library
# failure cannot corrupt prior metrics or the active GPU training process.
for episode in "${holdout_episodes[@]}"; do
  episode_metrics="$output/.episode_${episode}.tsv"
  if [[ ! -s "$episode_metrics" ]]; then
    rm -f "$episode_metrics"
    .venv/bin/python scripts/open_loop_eval.py \
      --model_path "$project/output/package_a2d_lora_4090/checkpoints/global_step_${step}/hf_ckpt" \
      --robo_name package_a2d \
      --norm_path /home/bjtc/Sophix/datasets/a2d_bag_record_v3/norm_stats_package_a2d.json \
      --data_path /home/bjtc/Sophix/datasets/a2d_bag_record_v3 \
      --traj_ids "$episode" \
      --use_length 16 --max-infer-time 10 --num_denoising_step 10 --device cpu \
      --metrics-path "$episode_metrics" \
      --save_plot_path "$output"
  fi
done

printf 'episode_index\tmse\tmae\n' > "$metrics_path"
for episode in "${holdout_episodes[@]}"; do
  sed '1d' "$output/.episode_${episode}.tsv" >> "$metrics_path"
done

#!/usr/bin/env bash
# Low-priority, CPU-only open-loop evaluation on episodes never used in training.
set -euo pipefail

readonly project="/home/bjtc/Sophix/lingbot-vla"
readonly output="${project}/output/ruantong_a2d_parts_lora_4090/open_loop_holdout_step16000"

mkdir -p "$output"
cd "$project"

export CUDA_VISIBLE_DEVICES=""
export QWEN25_PATH="/home/bjtc/Sophix/models/Qwen2.5-VL-3B-Instruct-tokenizer"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"

exec .venv/bin/python scripts/open_loop_eval.py \
    --model_path "$project/output/ruantong_a2d_parts_lora_4090/checkpoints/global_step_16000/hf_ckpt" \
    --robo_name ruantong_a2d_parts \
    --norm_path /home/bjtc/Sophix/datasets/agibot_g1_parts/norm_stats_ruantong_a2d_parts.json \
    --data_path /home/bjtc/Sophix/datasets/agibot_g1_parts \
    --traj_ids 743 753 763 773 783 793 803 813 823 \
    --use_length 16 \
    --max-infer-time 1 \
    --num_denoising_step 10 \
    --device cpu \
    --save_plot_path "$output"

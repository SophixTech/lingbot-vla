#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/bjtc/Sophix/lingbot-vla
OUT="$ROOT/output/separated_model/mixed_conditioned_chunk25"
CKPTS="$OUT/checkpoints"

valid_checkpoint() {
  local d="$1"
  test -f "$d/extra_state/extra_state_rank_0.pt" \
    && test -f "$d/model/.metadata" \
    && test -f "$d/hf_ckpt/model.safetensors.index.json"
}

latest_valid_checkpoint() {
  local d step best_step=-1 best=''
  for d in "$CKPTS"/global_step_*; do
    test -d "$d" || continue
    step=${d##*_}
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    valid_checkpoint "$d" || continue
    if (( step > best_step )); then
      best_step=$step
      best=$d
    fi
  done
  test -n "$best" || return 1
  printf '%s\n' "$best"
}

mkdir -p "$OUT"
if valid_checkpoint "$CKPTS/global_step_10000"; then
  exit 0
fi

cd "$ROOT"
args=(
  --standalone
  --nnodes=1
  --nproc-per-node=1
  tasks/vla/train_lingbotvla.py
  configs/vla/mixed_conditioned_chunk25.yaml
)
if checkpoint=$(latest_valid_checkpoint); then
  args+=("--train.load_checkpoint_path=$checkpoint")
fi

exec "$ROOT/.venv/bin/torchrun" "${args[@]}" >> "$OUT/train.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail

readonly project="/home/bjtc/Sophix/lingbot-vla"
readonly output="$project/output/package_a2d_lora_4090"
readonly metrics="$output/holdout_metrics_v2.tsv"
readonly selection="$output/package_checkpoint_selection.txt"
readonly checkpoints="$output/checkpoints"
readonly config="$project/configs/vla/package_a2d_lora_4090.yaml"
readonly extension_log="$output/training_extension.log"

mkdir -p "$output"

if [[ "$(systemctl --user is-active lingbot-vla-package-a2d.service 2>/dev/null || true)" == "active" ]]; then
  exit 0
fi
# The monitor is idempotent: it evaluates only an unevaluated checkpoint.
"$project/deploy/package_a2d_training_monitor.sh" || true

# Do not publish a final selection until the newest complete checkpoint has a
# fixed-cohort result. A long CPU evaluation may still own the monitor lock.
latest_step="$(find "$checkpoints" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null \
  | sed 's/^global_step_//' | awk '/^[0-9]+$/' | sort -n | tail -n 1)"
if [[ -z "$latest_step" ]] || [[ ! -f "$metrics" ]] \
  || ! awk -F '\t' -v step="$latest_step" '$2 == step { found=1 } END { exit !found }' "$metrics"; then
  exit 0
fi

# Continue only after a normal max-step completion. Safety-guard stops and
# abnormal exits require human investigation instead of automatic restart.
if ! grep -Fq "Reached max_steps=${latest_step}, stopping training." "$output/train.log"; then
  exit 0
fi

stale_three=0
if (( latest_step >= 4000 )) \
  && tail -n 3 "$metrics" | awk -F '\t' 'NR > 0 { count++; if ($5 != "not_improved") bad=1 } END { exit !(count == 3 && !bad) }'; then
  stale_three=1
fi

if (( stale_three == 0 )); then
  next_step="$((latest_step + 1000))"
  sed -i -E "s/^(  max_steps: )[0-9]+$/\\1${next_step}/" "$config"
  {
    printf '%s latest_step=%s decision=continue next_max_steps=%s\n' \
      "$(date --iso-8601=seconds)" "$latest_step" "$next_step"
  } >> "$extension_log"
  systemctl --user start lingbot-vla-package-a2d.service
  exit 0
fi

if [[ -f "$output/checkpoints/loss.jsonl" ]]; then
  MPLCONFIGDIR=/tmp/package_lingbot_mpl \
    "$project/.venv/bin/python" "$project/scripts/export_training_curve.py" \
    "$output/checkpoints/loss.jsonl" "$output/package_training_loss.png" --window 100 || true
fi

{
  printf 'generated_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'service_result=%s\n' "$(systemctl --user show lingbot-vla-package-a2d.service -p Result --value 2>/dev/null || true)"
  if [[ -f "$metrics" ]]; then
    best_numeric="$(awk -F '\t' 'NR == 2 || (NR > 2 && $4 + 0 < best + 0) { best=$0 } END { print best }' "$metrics")"
    last_supported="$(awk -F '\t' '$5 == "improved" { supported=$0 } END { print supported }' "$metrics")"
    if [[ -n "$best_numeric" ]]; then
      printf 'best_numeric_holdout_record=%s\n' "$best_numeric"
      printf 'last_statistically_supported_improvement=%s\n' "$last_supported"
      printf 'holdout_protocol=fixed 24-episode paired open-loop evaluation with 10 chunks per episode; checkpoint improvements require a 1%% MAE threshold and positive 95%% paired-bootstrap interval\n'
    fi
  else
    printf 'best_holdout_record=unavailable\n'
  fi
  printf 'loss_curve=%s\n' "$output/package_training_loss.png"
} > "$selection"

#!/usr/bin/env bash
set -euo pipefail

readonly project="/home/bjtc/Sophix/lingbot-vla"
readonly output="$project/output/package_a2d_lora_4090"
readonly checkpoints="$output/checkpoints"
readonly metrics="$output/holdout_metrics_v2.tsv"
readonly monitor_log="$output/hourly_monitor.log"
readonly train_service="lingbot-vla-package-a2d.service"
readonly eval_lock="$output/.holdout_eval.lock"

mkdir -p "$output"
# A long CPU evaluation must not be duplicated by the hourly timer or a
# bootstrap comparison run.
exec 9>"$eval_lock"
flock -n 9 || exit 0
"$project/deploy/package_a2d_hourly_health_check.sh"

latest_step="$(find "$checkpoints" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null \
  | sed 's/^global_step_//' | awk '/^[0-9]+$/' | sort -n | tail -n 1)"
[[ -n "$latest_step" ]] || exit 0
[[ -d "$checkpoints/global_step_${latest_step}/hf_ckpt" ]] || exit 0

if [[ -f "$metrics" ]] && awk -F '\t' -v step="$latest_step" '$2 == step { found=1 } END { exit !found }' "$metrics"; then
  exit 0
fi

eval_log="$output/holdout_step${latest_step}.log"
{
  printf '\n===== %s checkpoint=%s =====\n' "$(date --iso-8601=seconds)" "$latest_step"
  # CPU evaluation keeps the active CUDA training process isolated.
  # Ten 16-action chunks per episode take roughly four minutes on the
  # CPU-only evaluator; allow the full fixed 24-episode cohort to finish.
  if ! timeout 130m "$project/deploy/package_a2d_holdout_eval.sh" "$latest_step" >> "$eval_log" 2>&1; then
    printf 'holdout evaluation failed or timed out; training remains active. See %s\n' "$eval_log"
    exit 0
  fi

  per_episode="$output/open_loop_holdout_step${latest_step}/per_episode_metrics.tsv"
  if [[ ! -s "$per_episode" ]]; then
    printf 'holdout evaluation completed without per-episode metrics; training remains active.\n'
    exit 0
  fi

  if [[ ! -f "$metrics" ]]; then
    printf 'timestamp\tstep\tn_episodes\tmae\tdecision\trelative_improvement\tci95_low\tci95_high\n' > "$metrics"
  fi
  decision_record="$("$project/.venv/bin/python" "$project/deploy/package_a2d_holdout_decision.py" \
    --step "$latest_step" --metrics "$per_episode" --history "$metrics")"
  IFS=$'\t' read -r n_episodes mae decision relative ci_low ci_high <<< "$decision_record"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$latest_step" "$n_episodes" "$mae" "$decision" "$relative" "$ci_low" "$ci_high" >> "$metrics"
  printf 'holdout n=%s mae=%s decision=%s relative=%s ci95=[%s,%s]\n' \
    "$n_episodes" "$mae" "$decision" "$relative" "$ci_low" "$ci_high"

  # Stop only after meaningful training progress and three paired-bootstrap
  # stale checkpoints; otherwise max_steps is the completion condition.
  stale_three=0
  if tail -n 3 "$metrics" | awk -F '\t' 'NR > 0 { count++; if ($5 != "not_improved") bad=1 } END { exit !(count == 3 && !bad) }'; then
    stale_three=1
  fi
  if (( latest_step >= 4000 && stale_three == 1 )) \
    && [[ "$(systemctl --user is-active "$train_service" 2>/dev/null || true)" == "active" ]]; then
    printf 'Early-stop criterion met at step %s: three consecutive holdout checkpoints lacked 1%% MAE improvement.\n' "$latest_step"
    systemctl --user stop "$train_service"
  fi
} >> "$monitor_log" 2>&1

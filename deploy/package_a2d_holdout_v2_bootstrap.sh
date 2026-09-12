#!/usr/bin/env bash
set -euo pipefail

readonly project="/home/bjtc/Sophix/lingbot-vla"
readonly output="$project/output/package_a2d_lora_4090"
readonly metrics="$output/holdout_metrics_v2.tsv"
readonly eval_lock="$output/.holdout_eval.lock"

mkdir -p "$output"
exec 9>"$eval_lock"
flock -n 9 || exit 0

record_step() {
  local step="$1"
  local per_episode="$output/open_loop_holdout_step${step}/per_episode_metrics.tsv"
  local record
  [[ -s "$per_episode" ]]
  if [[ ! -f "$metrics" ]]; then
    printf 'timestamp\tstep\tn_episodes\tmae\tdecision\trelative_improvement\tci95_low\tci95_high\n' > "$metrics"
  fi
  if awk -F '\t' -v step="$step" '$2 == step { found=1 } END { exit !found }' "$metrics"; then
    return
  fi
  record="$("$project/.venv/bin/python" "$project/deploy/package_a2d_holdout_decision.py" \
    --step "$step" --metrics "$per_episode" --history "$metrics")"
  local n mae decision relative ci_low ci_high
  IFS=$'\t' read -r n mae decision relative ci_low ci_high <<< "$record"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$step" "$n" "$mae" "$decision" "$relative" "$ci_low" "$ci_high" >> "$metrics"
}

# The 1000-step service is already evaluating the fixed cohort. Wait for that
# artifact instead of starting a duplicate model process.
baseline="$output/open_loop_holdout_step1000/per_episode_metrics.tsv"
for _ in $(seq 1 300); do
  [[ -s "$baseline" ]] && break
  sleep 10
done
[[ -s "$baseline" ]] || { echo "Timed out waiting for fixed-cohort 1000-step baseline" >&2; exit 1; }
record_step 1000

timeout 50m "$project/deploy/package_a2d_holdout_eval.sh" 2000
record_step 2000

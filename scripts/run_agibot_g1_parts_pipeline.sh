#!/usr/bin/env bash
# Wait for the selected raw videos, then run the complete single-task LoRA pipeline.
set -euo pipefail

REPO=/home/bjtc/Sophix/lingbot-vla
RAW=/home/bjtc/Sophix/datasets/agibot_g1_parts_raw
V3=/home/bjtc/Sophix/datasets/agibot_g1_parts
CONFIG="$REPO/configs/vla/agibot_g1_parts_lora_4090.yaml"
CHECK_INTERVAL_SECONDS=1800
CAMERAS=(
  observation.images.cam_high_rgb
  observation.images.cam_left_wrist_rgb
  observation.images.cam_right_wrist_rgb
)

count_videos() {
  local camera="$1"
  local complete=0 file
  while IFS= read -r -d '' file; do
    # aria2 creates the final .mp4 name before it is complete. A non-empty
    # file is finished only after its sibling control file has disappeared.
    if [[ -s "$file" && ! -e "$file.aria2" ]]; then
      ((complete += 1))
    fi
  done < <(find "$RAW/videos/chunk-000/$camera" -maxdepth 1 -type f -name '*.mp4' -print0 2>/dev/null)
  printf '%s\n' "$complete"
}

raw_is_complete() {
  [[ "$(find "$RAW/data" -type f -name '*.parquet' | wc -l)" -eq 824 ]] || return 1
  local camera
  for camera in "${CAMERAS[@]}"; do
    [[ "$(count_videos "$camera")" -eq 824 ]] || return 1
  done
}

while ! raw_is_complete; do
  printf '%s waiting: parquet=%s top=%s left=%s right=%s\n' \
    "$(date '+%F %T')" \
    "$(find "$RAW/data" -type f -name '*.parquet' | wc -l)" \
    "$(count_videos observation.images.cam_high_rgb)" \
    "$(count_videos observation.images.cam_left_wrist_rgb)" \
    "$(count_videos observation.images.cam_right_wrist_rgb)"
  # Use short sleeps so stopping this process never waits longer than one minute.
  for _ in $(seq 1 $((CHECK_INTERVAL_SECONDS / 60))); do sleep 60; done
done

printf '%s raw dataset complete; converting LeRobot v2.1 to v3\n' "$(date '+%F %T')"
if [[ -e "$V3" ]]; then
  if [[ -f "$V3/meta/info.json" && -d "$V3/data" && -d "$V3/videos" ]]; then
    echo "Reusing existing converted v3 dataset: $V3"
  else
    echo "Existing v3 dataset is incomplete; refusing to overwrite: $V3" >&2
    exit 2
  fi
else
  cd "$REPO"
  .venv/bin/python scripts/prepare_agibot_g1_parts_v30.py --raw "$RAW" --output "$V3"
fi

cd "$REPO"

SINGLE_RANK_ENV=(
  env
  CUDA_VISIBLE_DEVICES=0
  RANK=0
  WORLD_SIZE=1
  LOCAL_RANK=0
  LOCAL_WORLD_SIZE=1
  MASTER_ADDR=127.0.0.1
  MASTER_PORT=62500
)

printf '%s computing train-split normalization statistics\n' "$(date '+%F %T')"
if [[ -f "$V3/norm_stats.json" ]]; then
  echo "Reusing existing normalization statistics: $V3/norm_stats.json"
else
  "${SINGLE_RANK_ENV[@]}" .venv/bin/python scripts/compute_norm.py "$CONFIG"
fi

printf '%s starting LoRA post-training\n' "$(date '+%F %T')"
"${SINGLE_RANK_ENV[@]}" .venv/bin/python tasks/vla/train_lingbotvla.py "$CONFIG"

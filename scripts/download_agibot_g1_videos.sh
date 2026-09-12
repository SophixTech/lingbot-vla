#!/usr/bin/env bash
# Download only missing AgiBot G1 videos. Intended to run under systemd --user
# so it survives the interactive controller session.
set -euo pipefail

ROOT=/home/bjtc/Sophix
RAW="$ROOT/datasets/agibot_g1_parts_raw"
ARIA="$ROOT/.tools/aria2/runtime/usr/bin/aria2c"
LOG_DIR="$ROOT/lingbot-vla/.cache"
BASE_URL='https://modelscope.cn/datasets/RoboCOIN/AgiBot-g1_picks_up_parts_b/resolve/master/videos/chunk-000'
CAMERAS=(
  observation.images.cam_high_rgb
  observation.images.cam_left_wrist_rgb
  observation.images.cam_right_wrist_rgb
)

mkdir -p "$LOG_DIR"
declare -a pids=()

for camera in "${CAMERAS[@]}"; do
  destination="$RAW/videos/chunk-000/$camera"
  list="$LOG_DIR/${camera}.missing.urls"
  : > "$list"
  for episode in $(seq 0 823); do
    filename=$(printf 'episode_%06d.mp4' "$episode")
    # A clip is complete only when it is nonempty and aria2 no longer has its
    # sibling control file. Retrying partial files makes service restarts safe.
    if [[ ! -s "$destination/$filename" || -e "$destination/$filename.aria2" ]]; then
      printf '%s/%s/%s\n' "$BASE_URL" "$camera" "$filename" >> "$list"
    fi
  done
  missing=$(wc -l < "$list")
  printf '%s: %s videos still missing\n' "$camera" "$missing"
  if ((missing == 0)); then
    continue
  fi
  # The top-view clips are larger and this CDN has intermittently terminated
  # TLS handshakes under high concurrency. Keep that stream conservative;
  # wrist streams may safely use more parallel requests.
  concurrent_downloads=6
  if [[ "$camera" == 'observation.images.cam_high_rgb' ]]; then
    concurrent_downloads=2
  fi
  "$ARIA" \
    --all-proxy=http://127.0.0.1:7897 \
    --continue=true --auto-file-renaming=false --file-allocation=none \
    --max-concurrent-downloads="$concurrent_downloads" --max-connection-per-server=4 \
    --split=4 --min-split-size=8M \
    --max-tries=0 --retry-wait=5 --timeout=60 --connect-timeout=30 \
    --auto-save-interval=60 --summary-interval=60 \
    --dir "$destination" --input-file "$list" \
    > "$LOG_DIR/${camera}.aria2.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

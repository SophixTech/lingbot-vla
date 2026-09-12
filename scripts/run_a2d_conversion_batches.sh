#!/usr/bin/env bash
set -u

ROOT=/home/bjtc/Sophix
PYTHON="$ROOT/lingbot-vla/.venv/bin/python"
SCRIPT="$ROOT/lingbot-vla/scripts/batch_convert_a2d_raw_to_lerobot.py"
VERIFY="$ROOT/lingbot-vla/scripts/verify_a2d_lerobot_output.py"
INPUT="$ROOT/data/record"
OUTPUT="$ROOT/datasets/a2d_bag_record_v3"
LOG="$OUTPUT/conversion.batches.log"
export PYTHONPATH=/tmp/a2d-h5-inspect${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$OUTPUT"
while [[ ! -f "$OUTPUT/conversion_report.json" ]]; do
    printf '[%s] starting conversion batch\n' "$(date --iso-8601=seconds)" >> "$LOG"
    "$PYTHON" "$SCRIPT" \
        --input-root "$INPUT" \
        --output "$OUTPUT" \
        --mode convert \
        --tolerance-ms 17 \
        --min-frames 50 \
        --checkpoint-episodes 25 \
        --max-new-output-episodes 100 \
        --resume >> "$LOG" 2>&1
    code=$?
    printf '[%s] batch exit=%s\n' "$(date --iso-8601=seconds)" "$code" >> "$LOG"
    if [[ $code -ne 0 ]]; then
        exit "$code"
    fi
done
printf '[%s] conversion_report.json detected; conversion finished\n' "$(date --iso-8601=seconds)" >> "$LOG"
printf '[%s] starting full output verification\n' "$(date --iso-8601=seconds)" >> "$LOG"
"$PYTHON" "$VERIFY" >> "$LOG" 2>&1
printf '[%s] full output verification passed\n' "$(date --iso-8601=seconds)" >> "$LOG"

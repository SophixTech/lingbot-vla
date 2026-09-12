#!/usr/bin/env bash
# Safely rotate completed LingBot-VLA checkpoint directories.
# Default mode is read-only.  Pass --apply only after reviewing the printed plan.
set -euo pipefail

CHECKPOINT_ROOT="/home/bjtc/Sophix/lingbot-vla/output/ruantong_a2d_parts_lora_4090/checkpoints"
KEEP=3
GRACE_MINUTES=30
APPLY=0

usage() {
    echo "Usage: $0 [--apply] [--keep N] [--grace-minutes N] [--checkpoint-root PATH]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply) APPLY=1 ;;
        --keep) KEEP="$2"; shift ;;
        --grace-minutes) GRACE_MINUTES="$2"; shift ;;
        --checkpoint-root) CHECKPOINT_ROOT="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

[[ "$KEEP" =~ ^[1-9][0-9]*$ ]] || { echo "--keep must be a positive integer" >&2; exit 2; }
[[ "$GRACE_MINUTES" =~ ^[0-9]+$ ]] || { echo "--grace-minutes must be a non-negative integer" >&2; exit 2; }
[[ -d "$CHECKPOINT_ROOT" ]] || { echo "Checkpoint root not found: $CHECKPOINT_ROOT" >&2; exit 1; }

mapfile -t entries < <(
    find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'global_step_[0-9]*' \
        -printf '%f\t%T@\t%p\n' \
    | awk -F '\t' '$1 ~ /^global_step_[0-9]+$/ { sub(/^global_step_/, "", $1); print $1 "\t" $2 "\t" $3 }' \
    | sort -n -k1,1
)

total="${#entries[@]}"
if (( total <= KEEP )); then
    echo "Nothing to rotate: ${total} checkpoint(s), retention is ${KEEP}."
    exit 0
fi

now="$(date +%s)"
delete_count=$((total - KEEP))
candidate_count=0
skipped_count=0
action="PLAN"
if (( APPLY )); then
    action="DELETE"
fi

for ((i = 0; i < delete_count; i++)); do
    IFS=$'\t' read -r step mtime path <<< "${entries[$i]}"
    age_minutes=$(( (now - ${mtime%.*}) / 60 ))

    # A completed checkpoint must contain all three independently written payloads.
    if (( age_minutes < GRACE_MINUTES )) || [[ ! -d "$path/model" || ! -d "$path/hf_ckpt" || ! -d "$path/optimizer" ]]; then
        echo "SKIP  step=${step} age=${age_minutes}m path=${path}"
        ((skipped_count += 1))
        continue
    fi

    size="$(du -sh "$path" | awk '{print $1}')"
    echo "${action} step=${step} age=${age_minutes}m size=${size} path=${path}"
    ((candidate_count += 1))
    if (( APPLY )); then
        rm -rf --one-file-system -- "$path"
    fi
done

if (( APPLY )); then
    echo "Rotation completed: removed ${candidate_count}; skipped ${skipped_count}; retained newest ${KEEP}."
else
    echo "Dry run: ${candidate_count} completed checkpoint(s) are eligible; ${skipped_count} skipped. Re-run with --apply to delete only the listed paths."
fi

#!/usr/bin/env bash
set -euo pipefail

# Edit paths for your dataset
VIDEO_PATH="${VIDEO_PATH:-/date/xrd/code/Segment-then-Splat/datasets/anli/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/date/xrd/code/Segment-then-Splat/datasets/anli/output}"
BATCH_SIZE="${BATCH_SIZE:-40}"
DETECT_STRIDE="${DETECT_STRIDE:-10}"
POINTS_PER_SIDE="${POINTS_PER_SIDE:-16}"
# Space-separated levels, e.g. "large" or "large middle"
LEVELS="${LEVELS:-large}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for level in $LEVELS; do
  echo "========== level: ${level} =========="
  python auto-mask-fast.py \
    --video_path "$VIDEO_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --detect_stride "$DETECT_STRIDE" \
    --points_per_side "$POINTS_PER_SIDE" \
    --level "$level" \
    --skip_vis

  python visulization.py \
    --video_path "$VIDEO_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --level "$level"
done

echo "All done."

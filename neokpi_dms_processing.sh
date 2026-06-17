#!/bin/bash

NEOKPI_DIR="/Users/batakalaashok/neokpi"
RUN_DIR="DATA/neokpi_run_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RUN_DIR"

echo "Processing DMS videos from $NEOKPI_DIR"
echo "Output folder: $RUN_DIR"
echo ""

for alert_dir in "$NEOKPI_DIR"/*/; do
    alert_id=$(basename "$alert_dir")
    video="$alert_dir/8.mp4"
    if [ ! -f "$video" ]; then
        echo "  [SKIP] $alert_id: no 8.mp4"
        continue
    fi
    output="$RUN_DIR/${alert_id}_result.mp4"
    echo "  [$alert_id] Processing..."
    python run_on_video.py "$video" --output "$output" --no-display || echo "  [$alert_id] FAILED"
    echo "  [$alert_id] Done -> $output"
done

echo ""
echo "All done. Results in: $RUN_DIR"

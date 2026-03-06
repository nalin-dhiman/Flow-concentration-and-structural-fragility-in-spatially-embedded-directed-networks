#!/bin/bash
set -e


CODE_DIR="../code"
OUT_DIR="../results/analysis"

mkdir -p "$OUT_DIR"
echo "Running Backbone Analysis (4_backbone.py)..."

python "$CODE_DIR/analysis/4_backbone.py" \
    --edge_rankings "../results/empirical/per_seed_metrics/REAL.parquet"

echo "Backbone analysis complete. (STUB for testing CLI logic)"

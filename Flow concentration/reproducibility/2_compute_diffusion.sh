#!/bin/bash
set -e

# Uses output from Step 1
PROC_DIR="../data/processed/v1_submission/canonical"
CODE_DIR="../code"
OUT_DIR="../results/empirical"

mkdir -p "$OUT_DIR"

echo "Running Empirical Diffusion (2_compute_diffusion.py)..."

python "$CODE_DIR/core/2_compute_diffusion.py" \
    --null "REAL" \
    --seed 0 \
    --out "$OUT_DIR" \
    --nodes "$PROC_DIR/nodes.parquet" \
    --edges "$PROC_DIR/edges.parquet" \
    --targets "../data/targets_fixed.parquet" \
    --max_minutes_per_config 60
    
echo "Empirical diffusion complete. Artifacts in $OUT_DIR/per_seed_metrics/REAL.parquet"

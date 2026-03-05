#!/bin/bash
set -e

CODE_DIR="../code"
OUT_DIR="../results/analysis"

echo "Running Perturbation Analysis (5_perturbation.py)..."

python "$CODE_DIR/analysis/5_perturbation.py" \
    --edge_rankings "../results/empirical/per_seed_metrics/REAL.parquet"

echo "Perturbation analysis complete. (STUB for testing CLI logic)"

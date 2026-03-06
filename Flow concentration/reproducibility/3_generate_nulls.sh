#!/bin/bash
set -e

PROC_DIR="../data/processed/v1_submission/canonical"
CODE_DIR="../code"
OUT_DIR="../results/nulls"

mkdir -p "$OUT_DIR"

SEEDS=(1 2 3)
NULLS=("N0" "N1" "N2")

echo "Running Null Ensemble Generation (3_generate_nulls.py)..."

for n in "${NULLS[@]}"; do
    for s in "${SEEDS[@]}"; do
        echo "Running $n Seed $s..."
        python "$CODE_DIR/analysis/3_generate_nulls.py" \
            --null "$n" \
            --seed "$s" \
            --out "$OUT_DIR" \
            --nodes "$PROC_DIR/nodes.parquet" \
            --edges "$PROC_DIR/edges.parquet" \
            --targets "../data/targets_fixed.parquet" \
            --max_minutes_per_config 60
    done
done
    
echo "Null generation complete. Artifacts in $OUT_DIR/per_seed_metrics/"

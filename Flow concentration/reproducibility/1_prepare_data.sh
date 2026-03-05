#!/bin/bash
set -e

# Data URLs (Assume user places them in ../data/raw manually later or downloads)
# For the purpose of the pipeline, we assume they exist in ../data/raw/

RAW_DIR="../data/raw"
PROC_DIR="../data/processed"
CODE_DIR="../code"

mkdir -p "$PROC_DIR"

echo "Running Data Preparation (1_prepare_data.py)..."

python "$CODE_DIR/core/1_prepare_data.py" \
    --raw_dir "$RAW_DIR" \
    --out_root "$PROC_DIR" \
    --version "v1_submission" \
    --weight_map "linear" \
    --min_synapses 1 \
    --remove_autapses 1 \
    --drop_missing_positions 1 \
    --seed 42 \
    --force
    
echo "Data preparation complete. Artifacts in $PROC_DIR/v1_submission/"

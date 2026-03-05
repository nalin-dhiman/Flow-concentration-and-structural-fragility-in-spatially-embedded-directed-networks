#!/bin/bash
set -e

echo "====================================================="
echo "  Nature Communications Submission Reproduction Pipeline"
echo "====================================================="

echo "Step 1: Data Preparation"
bash 1_prepare_data.sh

echo "Step 2: Empirical Diffusion Calculation"
bash 2_compute_diffusion.sh

echo "Step 3: Null Ensemble Generation"
bash 3_generate_nulls.sh

echo "Step 4: Backbone Analysis"
bash 4_backbone.sh

echo "Step 5: Perturbation Analysis"
bash 5_perturbation.sh

echo "All steps completed successfully."
echo "Results and figures should now be available in the relative directories."



## Directory Structure
- `code/`: Contains the modular reconstruction scripts.
- `data/`: Where raw connection datasets should be placed, and where parsed canonical formats are created.
- `figures/`: The output destination for reproduction scripts containing vector PDFs.
- `manuscript/`: Contains the main and supplementary TeX files.
- `manifest/`: File manifests and checksums for verification of package integrity.
- `reproducibility/`: Bash runner scripts for re-executing the full pipeline.
- `results/`: Intermediate analytic outputs such as empirical network statistics and null ensembles.

## Execution Order
To reproduce the work from beginning to end, run the bash scripts located in `reproducibility/` sequentially:

1. `bash reproducibility/1_prepare_data.sh`
   - Processes raw data into nodes and edges parquets.
2. `bash reproducibility/2_compute_diffusion.sh`
   - Executes the core MFPT diffusion computations.
3. `bash reproducibility/3_generate_nulls.sh`
   - Generates and analyzes strict graph equivalents.
4. `bash reproducibility/4_backbone.sh`
   - Computes statistical PageRank and reachability logic. 
5. `bash reproducibility/5_perturbation.sh`
   - Synthesizes node disruption impacts.
6. Alternatively, just run `bash reproducibility/run_all.sh` to execute the whole pipeline at once.

## Important Note
Before running, you must ensure the primary dataset (`nodes.parquet`, `edges.parquet`, `targets_fixed.parquet`) is available in the `data/` folder, as they were excluded from this minimised archive due to file size constraints. As these files are very large, you have to process them through the entire original flyEM dataset. You must use HPC for this. 

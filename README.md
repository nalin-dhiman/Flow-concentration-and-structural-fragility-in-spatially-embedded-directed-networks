# Flow Concentration and Structural Fragility in Spatially Embedded Directed Networks

This repository is a clean code/data release for the flow-concentration project.
It contains reusable analysis code, curated graph data, result tables, and
rendered figures. It deliberately includes only code, derived data, and figure
outputs.

## Repository Layout

```text
src/                         Core Python modules
scripts/                     Reusable analysis and plotting drivers
tests/                       Lightweight unit tests
data/canonical/              Canonical directed weighted graph
data/targets/                Target and compartment annotation tables
data/results/tables/         Curated result and robustness CSV tables
data/results/generative/     Generative-model summary tables
figures/main/                Rendered main analysis figures
figures/supplementary/       Rendered supplementary analysis figures
figures/diagnostics/         Additional diagnostic plots
docs/                        Data and reproducibility notes
```

## What Is Included

- Canonical graph files:
  - `nodes.parquet`
  - `edges.parquet`
  - `distance_edges.parquet`
  - `adjacency_csr.npz`
  - checksums and graph summaries
- Core modules for:
  - canonical graph construction
  - transport metrics
  - constrained null ensembles
  - Pareto/regime diagnostics
  - backbone ranking and perturbation utilities
- Curated CSV outputs for:
  - key metrics
  - null-sampling audits
  - target sensitivity
  - restart-probability sensitivity
  - additive versus multiplicative cost sensitivity
  - perturbation summaries
  - generative-model summaries
- Rendered figures used for visual inspection and reproducibility checks.

## Clean-Release Boundary

The repository excludes article-build files, editorial material, scratch logs,
cache directories, obsolete duplicate project copies, and zipped archives.

## Quick Start

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the smoke tests:

```bash
pytest -q
```

Inspect the canonical graph:

```bash
python scripts/summarize_canonical.py
```

## Data Notes

The canonical graph is stored as a directed weighted edge list plus node table.
Edges use synaptic-contact weights (`s_ij`) and spatial distances (`d_ij`).
The sparse adjacency matrix is provided for analyses that need fast matrix
operations.

Some data files are large because they contain a full dense biological network
projection. The largest tracked files are below the GitHub hard file-size limit
but may trigger GitHub's large-file warning. For long-term archival releases,
Zenodo, OSF, or Git LFS would be more appropriate.

## Reproducibility

The repository is designed for transparent inspection and partial reruns rather
than as a one-command HPC workflow. The original full analysis involved multiple
long-running constrained null simulations. Curated outputs are therefore included
alongside the reusable code so that readers can inspect both the algorithms and
the resulting tables.

Start with:

- `docs/DATA.md`
- `docs/REPRODUCIBILITY.md`
- `src/metrics/compute_metrics.py`
- `src/nulls/`
- `src/attribution/`

## Citation

If you use this repository, cite the associated project by title and repository
URL. A formal archival DOI can be added here after release.

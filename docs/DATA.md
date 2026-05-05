# Data Description

## Canonical Graph

`data/canonical/` contains the canonical directed weighted graph used by the
analysis.

| File | Description |
| --- | --- |
| `nodes.parquet` | Node table with body identifiers, geometry, and derived node summaries. |
| `edges.parquet` | Directed edge table with `pre`, `post`, synaptic weight `s_ij`, and distance `d_ij`. |
| `distance_edges.parquet` | Edge-distance table used for spatial diagnostics. |
| `adjacency_csr.npz` | Sparse adjacency matrix for matrix-based computation. |
| `summaries.json` | Basic graph summary statistics. |
| `checksums.json` | Checksums for canonical build integrity. |
| `data_manifest.yaml` | Source/build metadata retained from the canonical build. |

## Targets and Annotations

`data/targets/` contains compact target and compartment tables used by the
target-conditioned transport analyses.

| File | Description |
| --- | --- |
| `descending_neurons.csv` | In-graph target identifiers used for target-conditioned diffusion. |
| `neuron_compartment_membership.csv` | Coarse compartment membership table. |
| `t4_t5_neurons.csv` | Additional cell-type annotation table used in enrichment controls. |

## Result Tables

`data/results/tables/` contains curated CSV outputs for key metrics, robustness
checks, null audits, and perturbation summaries. These files are intentionally
CSV rather than manuscript tables so they can be read directly by code.

## Data Provenance

The repository packages derived graph and analysis tables for reproducibility.
Users should consult the original connectome data providers for upstream data
licensing, redistribution terms, and citation requirements.


# Reproducibility Notes

This release prioritizes a clean, inspectable project layout over preserving
every scratch script and intermediate log from development.

## Recommended Checks

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Inspect graph summaries:

```bash
python scripts/summarize_canonical.py
```

## Re-running Analysis Components

The reusable modules are in `src/`:

- `src/metrics/` computes transport and graph observables.
- `src/nulls/` implements constrained null models.
- `src/attribution/` contains current ranking, target utilities, and ablation controls.
- `src/pareto/` contains scalarization and Pareto/regime-distance helpers.

The full null-ensemble run is computationally expensive. Curated outputs from
the retained runs are included under `data/results/` so that the main numerical
claims can be audited without rerunning every Markov-chain sample.

## Clean-Release Policy

The following were intentionally excluded:

- article-build and editorial artifacts
- temporary logs and cache files
- duplicate backup project trees
- zipped archives

This keeps the public repository focused on code, data, and figure outputs.

#!/usr/bin/env python3
"""Print a compact summary of the released canonical graph."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data" / "canonical"


def main() -> None:
    nodes = pd.read_parquet(CANON / "nodes.parquet")
    edges = pd.read_parquet(CANON / "edges.parquet")
    adjacency = sparse.load_npz(CANON / "adjacency_csr.npz")

    print("Canonical graph")
    print(f"  nodes: {len(nodes):,}")
    print(f"  edges: {len(edges):,}")
    print(f"  adjacency shape: {adjacency.shape}")
    print(f"  adjacency nnz: {adjacency.nnz:,}")

    if "s_ij" in edges:
        print(f"  total edge weight: {edges['s_ij'].sum():,.0f}")
    if "d_ij" in edges:
        print(f"  median edge distance: {edges['d_ij'].median():.3f}")

    summary_path = CANON / "summaries.json"
    if summary_path.exists():
        print("\nBuild summary")
        summary = json.loads(summary_path.read_text())
        for key, value in summary.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()


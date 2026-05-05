
import argparse
import sys
import shutil
import time
import json
import pandas as pd
import numpy as np
import scipy.sparse as sp
import logging
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.io_utils import setup_logging, close_logging, write_manifest
from src.metrics.matrix_utils import normalize_transition_matrix, solve_absorbing_fpt

def parse_args():
    parser = argparse.ArgumentParser(description="FlyEM Optic Lobe Metrics Engine")
    parser.add_argument("--canonical_dir", type=Path, required=True, help="Path to canonical artifacts")
    parser.add_argument("--out_root", type=Path, required=True, help="Root for metrics output")
    parser.add_argument("--version", type=str, required=True, help="Version tag (e.g. v3_a)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing version")
    return parser.parse_args()

def load_canonical(canonical_dir: Path):
    """Loads nodes, edges, distances, and adjacency."""
    logging.info(f"Loading canonical artifacts from {canonical_dir}")
    nodes = pd.read_parquet(canonical_dir / "nodes.parquet")
    edges = pd.read_parquet(canonical_dir / "edges.parquet")
    dist_edges = pd.read_parquet(canonical_dir / "distance_edges.parquet")
    adj = sp.load_npz(canonical_dir / "adjacency_csr.npz")
    
    # Merge distance into edges if not already present (it might be separate)
    # edges usually has weights w_ij and counts s_ij. distance_edges has d_ij.
    # We join on (pre, post)
    if 'd_ij' not in edges.columns:
        logging.info("Merging distances into edges...")
        edges = edges.merge(dist_edges, on=['pre', 'post'], how='left')
    
    return nodes, edges, adj

def compute_energy(edges: pd.DataFrame, eta: float) -> float:
    """Computes total wiring energy: E = sum(s_ij * d_ij^eta)."""
    # edges must have s_ij and d_ij
    mask = edges['d_ij'].notna()
    e_wire = (edges.loc[mask, 's_ij'] * (edges.loc[mask, 'd_ij']**eta)).sum()
    return float(e_wire)

def build_conductance_matrix(edges: pd.DataFrame, gamma: float, shape: tuple) -> sp.csr_matrix:
    """Constructs sparse conductance matrix C_ij = s_ij * exp(-gamma * d_ij)."""
    mask = edges['d_ij'].notna()
    pre = edges.loc[mask, 'pre_idx']
    post = edges.loc[mask, 'post_idx']
    s_ij = edges.loc[mask, 's_ij'].values
    d_ij = edges.loc[mask, 'd_ij'].values
    
    if gamma > 0:
        c_ij = s_ij * np.exp(-gamma * d_ij)
    else:
        c_ij = s_ij
        
    return sp.csr_matrix((c_ij, (pre, post)), shape=shape)

def get_stratified_targets(nodes: pd.DataFrame, n_targets: int = 500, seed: int = 42) -> np.ndarray:
    """Selects stratified targets based on in-degree."""
    np.random.seed(seed)
    
    # Stratify by in-degree deciles
    if 'in_degree' not in nodes.columns:
         return np.random.choice(nodes.index, n_targets, replace=False)
         
    nodes['decile'] = pd.qcut(nodes['in_degree'].rank(method='first'), 10, labels=False)
    targets = []
    
    per_decile = n_targets // 10
    for d in range(10):
        cands = nodes[nodes['decile'] == d].index.values
        if len(cands) > per_decile:
            targets.extend(np.random.choice(cands, per_decile, replace=False))
        else:
            targets.extend(cands)
            
    return np.unique(targets)

def main():
    args = parse_args()
    
    # Setup
    out_dir = args.out_root / args.version
    if out_dir.exists():
        if args.force:
            shutil.rmtree(out_dir)
        else:
            print(f"Error: {out_dir} exists. Use --force.")
            sys.exit(1)
            
    out_dir.mkdir(parents=True)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir()
    sweeps_dir = metrics_dir / "sweeps"
    sweeps_dir.mkdir()
    reports_dir = out_dir / "reports"
    reports_dir.mkdir()
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir()
    logs_dir = out_dir / "logs"
    logs_dir.mkdir()
    
    setup_logging(logs_dir / "metrics.log")
    logging.info(f"Starting metrics build {args.version}")
    
    # Load
    nodes, edges, adj = load_canonical(args.canonical_dir)
    
    # Map node indices for sparse matrix construction
    node_to_idx = {bid: i for i, bid in enumerate(nodes['bodyId'])}
    nodes['idx'] = nodes['bodyId'].map(node_to_idx)
    edges['pre_idx'] = edges['pre'].map(node_to_idx)
    edges['post_idx'] = edges['post'].map(node_to_idx)
    
    # Parameters
    etas = [1.0, 1.25, 1.5, 2.0]
    gammas = [0.0, 1e-6, 5e-6, 1e-5]
    
    sweep_results = []
    
    # Base computations for Energy
    logging.info("Computing Energy Terms...")
    total_synapses = edges['s_ij'].sum()
    
    for eta in etas:
        e_wire = compute_energy(edges, eta)
        logging.info(f"Energy (eta={eta}): {e_wire:.2e}")
        sweep_results.append({
            "metric": "energy",
            "eta": eta,
            "gamma": 0.0, # Energy independent of gamma in this formulation
            "value": e_wire
        })

    # Latency / FPT
    logging.info("Computing Latency (FPT)...")
    
    # Restrict to Largest SCC for FPT well-posedness
    logging.info("Extracting Largest SCC for FPT analysis...")
    n_components, labels = sp.csgraph.connected_components(adj, directed=True, connection='strong')
    _, counts = np.unique(labels, return_counts=True)
    largest_label = np.argmax(counts)
    scc_mask = (labels == largest_label)
    scc_indices = np.where(scc_mask)[0]
    logging.info(f"Largest SCC size: {len(scc_indices)}")
    
    # Create map from original index to SCC-local index
    # We need to subset the adjacency/conductance matrix
    # But doing this for every gamma is expensive if we rebuild.
    # Actually, we can just subset P after building it.
    
    # Targets must be in SCC
    # We select targets from scc_indices
    # Use the same stratification logic but filter nodes first
    scc_nodes = nodes.iloc[scc_indices].copy()
    target_indices_scc_local = get_stratified_targets(scc_nodes, n_targets=50, seed=42)
    # The returned indices are relative to scc_nodes index (which matches original if we didn't reset index)
    # get_stratified_targets returns values from .index. Since we passed subset, check what it returns.
    # It returns nodes.index values.
    # My get_stratified_targets implementation uses nodes.index.
    # So target_indices_scc_local are original bodyIds or indices?
    # Wait, nodes['idx'] maps bodyId to 0..N-1.
    # If I pass scc_nodes, its index is the original DataFrame index (0..N-1).
    target_original_indices = target_indices_scc_local
    
    logging.info(f"Selected {len(target_original_indices)} targets (all in SCC).")

    for gamma in gammas:
        logging.info(f"Processing Gamma={gamma}...")
        C = build_conductance_matrix(edges, gamma, adj.shape)
        P_full = normalize_transition_matrix(C)
        
        # Extract SCC submatrix
        # P_scc = P_full[scc_mask, :][:, scc_mask]
        # This copies data.
        P_scc = P_full[scc_indices, :][:, scc_indices]
        
        # We need to map target_original_indices to scc-local indices
        # scc_indices is list of original indices included.
        # fast lookup:
        orig_to_local = {orig: local for local, orig in enumerate(scc_indices)}
        
        targets_local = [orig_to_local[t] for t in target_original_indices]
        
        fpt_values = []
        
        for t_idx in targets_local:
             t_fpts = solve_absorbing_fpt(P_scc, np.array([t_idx]))
             
             valid_fpts = t_fpts[~np.isnan(t_fpts) & (t_fpts > 0)]
             if len(valid_fpts) > 0:
                 if len(valid_fpts) > 100:
                     fpt_samples = np.random.choice(valid_fpts, 100, replace=False)
                 else:
                     fpt_samples = valid_fpts
                 fpt_values.extend(fpt_samples)
        
        if len(fpt_values) > 0:
            stats = {
                "mean": float(np.mean(fpt_values)),
                "median": float(np.median(fpt_values)),
                "p90": float(np.percentile(fpt_values, 90)),
                "p99": float(np.percentile(fpt_values, 99))
            }
            logging.info(f"FPT Stats (gamma={gamma}): {stats}")
            
            sweep_results.append({
                "metric": "latency",
                "eta": 1.0, # Latency independent of eta
                "gamma": gamma,
                "mean": stats['mean'],
                "median": stats['median'],
                "p90": stats['p90']
            })
            
            # Save distribution plot for base case
            if gamma == 0:
                 plt.figure()
                 plt.hist(fpt_values, bins=50, log=True)
                 plt.title(f"FPT Distribution (gamma={gamma})")
                 plt.savefig(plots_dir / "latency_hist.png")
                 plt.close()
                 
    # Save Results
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_parquet(sweeps_dir / "sweep_summary.parquet")
    
    # Save manifest
    write_manifest(out_dir / "metrics_manifest.yaml", vars(args))
    
    # Report
    with open(reports_dir / "metrics_report.md", "w") as f:
        f.write("# Metrics Report\n\n")
        f.write("## Energy\n")
        e_rows = df_sweep[df_sweep['metric'] == 'energy']
        f.write(e_rows.to_markdown(index=False))
        f.write("\n\n## Latency\n")
        l_rows = df_sweep[df_sweep['metric'] == 'latency']
        f.write(l_rows.to_markdown(index=False))
        
    logging.info("Metrics build complete.")
    close_logging()
    print(f"DONE: {args.version}")

if __name__ == "__main__":
    main()

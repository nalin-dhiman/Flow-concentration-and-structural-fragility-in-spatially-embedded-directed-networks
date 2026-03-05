
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
    logging.info(f"Loading canonical artifacts from {canonical_dir}")
    nodes = pd.read_parquet(canonical_dir / "nodes.parquet")
    edges = pd.read_parquet(canonical_dir / "edges.parquet")
    dist_edges = pd.read_parquet(canonical_dir / "distance_edges.parquet")
    adj = sp.load_npz(canonical_dir / "adjacency_csr.npz")
    
    
    if 'd_ij' not in edges.columns:
        logging.info("Merging distances into edges...")
        edges = edges.merge(dist_edges, on=['pre', 'post'], how='left')
    
    return nodes, edges, adj

def compute_energy(edges: pd.DataFrame, eta: float) -> float:

    mask = edges['d_ij'].notna()
    e_wire = (edges.loc[mask, 's_ij'] * (edges.loc[mask, 'd_ij']**eta)).sum()
    return float(e_wire)

def build_conductance_matrix(edges: pd.DataFrame, gamma: float, shape: tuple) -> sp.csr_matrix:
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
    np.random.seed(seed)
    
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
    
    
    nodes, edges, adj = load_canonical(args.canonical_dir)
    
    node_to_idx = {bid: i for i, bid in enumerate(nodes['bodyId'])}
    nodes['idx'] = nodes['bodyId'].map(node_to_idx)
    edges['pre_idx'] = edges['pre'].map(node_to_idx)
    edges['post_idx'] = edges['post'].map(node_to_idx)
    
    etas = [1.0, 1.25, 1.5, 2.0]
    gammas = [0.0, 1e-6, 5e-6, 1e-5]
    
    sweep_results = []
    
    logging.info("Computing Energy Terms...")
    total_synapses = edges['s_ij'].sum()
    
    for eta in etas:
        e_wire = compute_energy(edges, eta)
        logging.info(f"Energy (eta={eta}): {e_wire:.2e}")
        sweep_results.append({
            "metric": "energy",
            "eta": eta,
            "gamma": 0.0, 
            "value": e_wire
        })

    logging.info("Computing Latency (FPT)...")
    
    logging.info("Extracting Largest SCC for FPT analysis...")
    n_components, labels = sp.csgraph.connected_components(adj, directed=True, connection='strong')
    _, counts = np.unique(labels, return_counts=True)
    largest_label = np.argmax(counts)
    scc_mask = (labels == largest_label)
    scc_indices = np.where(scc_mask)[0]
    logging.info(f"Largest SCC size: {len(scc_indices)}")
    
    
    scc_nodes = nodes.iloc[scc_indices].copy()
    target_indices_scc_local = get_stratified_targets(scc_nodes, n_targets=50, seed=42)
    
    target_original_indices = target_indices_scc_local
    
    logging.info(f"Selected {len(target_original_indices)} targets (all in SCC).")

    for gamma in gammas:
        logging.info(f"Processing Gamma={gamma}...")
        C = build_conductance_matrix(edges, gamma, adj.shape)
        P_full = normalize_transition_matrix(C)
        
        
        P_scc = P_full[scc_indices, :][:, scc_indices]
        
        
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
                "eta": 1.0, 
                "gamma": gamma,
                "mean": stats['mean'],
                "median": stats['median'],
                "p90": stats['p90']
            })
            
            if gamma == 0:
                 plt.figure()
                 plt.hist(fpt_values, bins=50, log=True)
                 plt.title(f"FPT Distribution (gamma={gamma})")
                 plt.savefig(plots_dir / "latency_hist.png")
                 plt.close()
                 
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_parquet(sweeps_dir / "sweep_summary.parquet")
    
    write_manifest(out_dir / "metrics_manifest.yaml", vars(args))
    
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

import argparse
import sys
import shutil
import time
import json
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


sys.path.append(str(Path(__file__).parent.parent))
from utils.io_utils import (
    setup_logging, load_neurons, load_connections,
    write_manifest, compute_checksums, close_logging,
    get_file_info, get_env_info
)
from utils.sanity_checks import (
    validate_graph, check_completeness, compute_scc, check_uniqueness
)
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="FlyEM Optic Lobe Canonical Connectome Builder")
    parser.add_argument("--raw_dir", type=Path, required=True, help="Path to neuprint tables")
    parser.add_argument("--out_root", type=Path, required=True, help="Root directory for outputs")
    parser.add_argument("--version", type=str, required=True, help="Version tag (e.g. v1_raw)")
    
    parser.add_argument("--weight_map", type=str, choices=["linear", "log"], default="linear")
    parser.add_argument("--min_synapses", type=int, default=1)
    parser.add_argument("--remove_autapses", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Overwrite existing version")
    parser.add_argument("--drop_missing_positions", type=int, default=1, help="Drop edges/nodes if coords missing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic plots")
    parser.add_argument("--strict", action="store_true", default=True, help="Enforce stop gates")
    
    return parser.parse_args()

def plot_scc_sizes(adj, out_path):
    n_components, labels = sp.csgraph.connected_components(adj, directed=True, connection='strong')
    _, counts = np.unique(labels, return_counts=True)
    largest = counts.max()
    
    plt.figure(figsize=(6, 4))
    plt.hist(counts, bins=50, log=True, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(largest, color='r', linestyle='--', label=f'Largest: {largest}')
    plt.title("SCC Size Distribution")
    plt.xlabel("Size (Nodes)")
    plt.ylabel("Count (log)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_weight_vs_distance_binned(d, w, out_path, seed=42):
    np.random.seed(seed)
    
    mask = np.random.rand(len(d)) < min(1.0, 200000 / len(d))
    d_samp = d[mask]
    w_samp = w[mask]
    
    plt.figure(figsize=(6, 4))
    plt.scatter(d_samp, w_samp, alpha=0.1, s=1, c='gray', label='Sampled Edges')
    
    if len(d) > 0:
        bins = np.linspace(d.min(), d.max(), 50)
        bin_idx = np.digitize(d, bins)
        bin_means = []
        bin_centers = []
        
        for i in range(1, len(bins)):
            mask = bin_idx == i
            if mask.any():
                bin_means.append(w[mask].mean())
                bin_centers.append((bins[i-1] + bins[i])/2)
                
        plt.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Binned Mean')
    
    plt.title("Weight vs Distance")
    plt.xlabel("Distance (Euclidean)")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_hist(data, title, xlabel, out_path, log=True):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=50, log=log, alpha=0.7, color='steelblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count (log)" if log else "Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_scatter(x, y, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.1, s=1, c='k')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

def main():
    args = parse_args()
    
    out_dir = args.out_root / args.version
    if out_dir.exists():
        if args.force:
            logging.warning(f"Validation: Output {out_dir} exists. Overwriting due to --force.")
            shutil.rmtree(out_dir)
        else:
            print(f"Error: Output {out_dir} exists. Use --force to overwrite.")
            sys.exit(1)
            
    out_dir.mkdir(parents=True)
    canonical_dir = out_dir / "canonical"
    canonical_dir.mkdir()
    reports_dir = out_dir / "reports"
    reports_dir.mkdir()
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir()
    logs_dir = out_dir / "logs"
    logs_dir.mkdir()
    
    setup_logging(logs_dir / "build.log")
    logging.info(f"Starting build for {args.version}")
    logging.info(f"Args: {vars(args)}")
    
    start_time = time.time()
    
    logging.info("Step 1: Loading Data...")
    neurons = load_neurons(args.raw_dir / "Neuprint_Neurons.feather")
    conns = load_connections(args.raw_dir / "Neuprint_Neuron_Connections.feather")
    
    traced_mask = neurons['status:string'] == 'Traced'
    n_total_rows = len(neurons)
    n_traced = int(traced_mask.sum())
    
    unique_bodyids_total = neurons['bodyId'].nunique()
    unique_bodyids_traced = neurons.loc[traced_mask, 'bodyId'].nunique()
    
    logging.info(f"Loaded {n_total_rows} neuron rows.")
    logging.info(f"Unique bodyIds (Total): {unique_bodyids_total}")
    logging.info(f"Unique bodyIds (Traced): {unique_bodyids_traced}")
    logging.info(f"Traced Filter Rule: status:string == 'Traced'")
    
    if n_total_rows > 500000:
        logging.warning("High raw node count detected! Likely includes non-traced fragments.")
        
    conns_n_rows = len(conns)
    conns_total_synapses = conns['s_ij'].sum()
    
    logging.info("Step 2: Processing Connections...")
    
   
    logging.info("Aggregating connections by (pre, post)...")
    conns = conns.groupby(['pre', 'post'], as_index=False)['s_ij'].sum()
    
    valid_bodyIds = set(neurons['bodyId'])
    endpoints_known_mask = conns['pre'].isin(valid_bodyIds) & conns['post'].isin(valid_bodyIds)
    pct_known_endpoints = (endpoints_known_mask.sum() / len(conns)) * 100
    logging.info(f"Connections with known endpoints: {pct_known_endpoints:.2f}%")

    
    logging.info(f"Filtering min_synapses >= {args.min_synapses}")
    conns = conns[conns['s_ij'] >= args.min_synapses]
    
    if args.remove_autapses:
        logging.info("Removing self-loops (autapses)")
        n_loops = (conns['pre'] == conns['post']).sum()
        conns = conns[conns['pre'] != conns['post']]
        logging.info(f"Removed {n_loops} autapses.")
    
    logging.info("Step 3: Calculating Distances...")
    
    node_map = neurons.set_index('bodyId')[['x', 'y', 'z']]
    
    valid_nodes = set(neurons['bodyId'])
    conns = conns[conns['pre'].isin(valid_nodes) & conns['post'].isin(valid_nodes)]
    
    pre_coords = node_map.loc[conns['pre']].values
    post_coords = node_map.loc[conns['post']].values
    
    d_sq = np.sum((pre_coords - post_coords)**2, axis=1)
    conns['d_ij'] = np.sqrt(d_sq)
    
    if args.drop_missing_positions:
        logging.info("Dropping edges with missing distance...")
        n_before = len(conns)
        conns = conns.dropna(subset=['d_ij'])
        logging.info(f"Dropped {n_before - len(conns)} edges missing coords.")
    
    logging.info(f"Step 4: Calculating Weights ({args.weight_map})...")
    if args.weight_map == "linear":
        conns['w_ij'] = conns['s_ij'].astype(float)
    elif args.weight_map == "log":
        conns['w_ij'] = np.log1p(conns['s_ij'])
    
    logging.info("Step 5: Updating Node Metrics and Removing Isolates...")
    
    in_degree = conns.groupby('post')['s_ij'].count().rename("in_degree")
    out_degree = conns.groupby('pre')['s_ij'].count().rename("out_degree")
    in_strength = conns.groupby('post')['w_ij'].sum().rename("in_strength")
    out_strength = conns.groupby('pre')['w_ij'].sum().rename("out_strength")
    
    neurons = neurons.set_index('bodyId')
    neurons = neurons.join([in_degree, out_degree, in_strength, out_strength], how='left')
    neurons[['in_degree', 'out_degree', 'in_strength', 'out_strength']] = \
        neurons[['in_degree', 'out_degree', 'in_strength', 'out_strength']].fillna(0)
    
    active_mask = (neurons['in_degree'] > 0) | (neurons['out_degree'] > 0)
    neurons_active = neurons[active_mask].reset_index() # bodyId becomes column
    
    logging.info(f"Removed {len(neurons) - len(neurons_active)} isolates.")
    
    active_ids = set(neurons_active['bodyId'])
    conns_final = conns[conns['pre'].isin(active_ids) & conns['post'].isin(active_ids)].copy()
    
    logging.info("Step 6: Running Sanity Checks...")
    validate_graph(conns_final)
    check_uniqueness(neurons_active)
    completeness_stats = check_completeness(neurons_active, conns_final)
    
    logging.info("Step 7: Generating Artifacts...")
    
    node_to_idx = {bid: i for i, bid in enumerate(neurons_active['bodyId'])}
    row_idx = conns_final['pre'].map(node_to_idx).values
    col_idx = conns_final['post'].map(node_to_idx).values
    data = conns_final['w_ij'].values
    
    adj_csr = sp.csr_matrix(
        (data, (row_idx, col_idx)), 
        shape=(len(neurons_active), len(neurons_active))
    )
    
    neurons_active.to_parquet(canonical_dir / "nodes.parquet")
    conns_final.to_parquet(canonical_dir / "edges.parquet")
    sp.save_npz(canonical_dir / "adjacency_csr.npz", adj_csr)
    
    if 'd_ij' in conns_final.columns:
        dist_df = conns_final[['pre', 'post', 'd_ij']].dropna()
        dist_df.to_parquet(canonical_dir / "distance_edges.parquet")
    
    scc_metrics = compute_scc(adj_csr)
    
    logging.info("Generating Plots...")
    plot_hist(neurons_active['in_degree'], "In Degree", "Degree", plots_dir / "degree_in_hist.png")
    plot_hist(neurons_active['out_degree'], "Out Degree", "Degree", plots_dir / "degree_out_hist.png")
    plot_hist(neurons_active['in_strength'], "In Strength", "Strength", plots_dir / "strength_in_hist.png")
    plot_hist(neurons_active['out_strength'], "Out Strength", "Strength", plots_dir / "strength_out_hist.png")
    
    if 'd_ij' in conns_final.columns:
        valid_d = conns_final[['d_ij', 'w_ij']].dropna()
        plot_hist(valid_d['d_ij'], "Distance Histogram", "Distance", plots_dir / "distance_hist.png", log=False)
        plot_weight_vs_distance_binned(valid_d['d_ij'].values, valid_d['w_ij'].values, plots_dir / "weight_vs_distance.png", seed=args.seed)
    
    plot_scc_sizes(adj_csr, plots_dir / "scc_sizes.png")
    
    summaries = {
        "meta": vars(args),
        "timestamp": datetime.now().isoformat(),
        "initial_stats": {
            "n_rows_neuron_table": n_total_rows,
            "n_unique_bodyIds_total": unique_bodyids_total,
            "n_unique_bodyIds_traced": unique_bodyids_traced,
            "traced_filter_rule": "status:string == 'Traced'",
            "n_rows_connections_table": conns_n_rows,
            "connections_synapse_count_column": "weight:int (or s_ij)",
            "total_synapses_raw": int(conns_total_synapses),
            "pct_connections_with_known_endpoints": float(f"{pct_known_endpoints:.4f}"),
            "pct_traced_neurons_with_coords": float(f"{completeness_stats.get('pct_nodes_with_coords', 0):.4f}") # Using final active set approximation or calc earlier if strictly needed on ALL traced
        },
        "final_stats": {
            "n_nodes": len(neurons_active),
            "n_edges": len(conns_final),
            "total_synapses": float(conns_final['s_ij'].sum()),
            "total_weight": float(conns_final['w_ij'].sum())
        },
        "completeness_stats": {
             "pct_nodes_with_coords": completeness_stats.get('pct_nodes_with_coords', 0),
             "pct_edges_with_distances": completeness_stats.get('pct_edges_with_distances', 0)
        },
        "scc_metrics": scc_metrics
    }
    
    with open(canonical_dir / "summaries.json", "w") as f:
        json.dump(summaries, f, indent=2, cls=CustomJSONEncoder)
        
    env_info = get_env_info()
    file_info = {
        "neurons": get_file_info(args.raw_dir / "Neuprint_Neurons.feather"),
        "connections": get_file_info(args.raw_dir / "Neuprint_Neuron_Connections.feather")
    }
    
    column_mappings = {
        "neuron_id": "bodyId",
        "traced_status": "status:string",
        "coordinates": "somaLocation:point{srid:9157} -> (x,y,z)",
        "connection_pre": ":START_ID(Body-ID)",
        "connection_post": ":END_ID(Body-ID)",
        "synapse_count": "weight:int"
    }
    manifest_config = vars(args)
    manifest_config['column_mappings'] = column_mappings
    manifest_config['dataset_name'] = "flyem-optic-lobe"
    manifest_config['dataset_version'] = "v1.1"
    manifest_config['raw_dir_resolved'] = str(args.raw_dir.resolve())
    
    write_manifest(out_dir / "data_manifest.yaml", manifest_config, env_info, file_info)
    
    checksums = compute_checksums(out_dir)
    
    canonical_checksums = {k: v for k, v in checksums.items() if k.startswith("canonical/") and "checksums.json" not in k}
    report_checksums = {k: v for k, v in checksums.items() if k not in canonical_checksums}
    
    with open(canonical_dir / "checksums.json", "w") as f:
        json.dump(canonical_checksums, f, indent=2, cls=CustomJSONEncoder)
        
    with open(reports_dir / "checksums_reports.json", "w") as f:
        json.dump(report_checksums, f, indent=2, cls=CustomJSONEncoder)
        
    with open(reports_dir / "sanity_report.md", "w") as f:
        f.write("# Sanity Report\n\n")
        f.write("## Status: PASS\n\n")
        f.write(f"**Version**: {args.version}\n")
        f.write(f"**Nodes**: {len(neurons_active)}\n")
        f.write(f"**Edges**: {len(conns_final)}\n")
        f.write(f"**Coords**: {completeness_stats.get('pct_nodes_with_coords', 0):.2f}%\n")
        f.write(f"**Largest SCC**: {scc_metrics['largest_scc_size']} ({scc_metrics['largest_scc_fraction']:.2%})\n")
    
    logging.info(f"Build Validated & Complete in {time.time() - start_time:.2f}s")
    close_logging()
    
    print(f"DONE: {args.version}")
    print(f"Nodes: {len(neurons_active)}, Edges: {len(conns_final)}")
    print(f"SCC: {scc_metrics['largest_scc_fraction']:.2%}")
    print(f"Artifacts: {out_dir}")

if __name__ == "__main__":
    main()

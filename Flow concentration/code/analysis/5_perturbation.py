import pandas as pd
import numpy as np
import scipy.sparse as sp
import logging
from utils.ranking import to_networkx, rank_edges_by_efficiency
from utils.compute_metrics import build_conductance_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

def check_sccs(args):
    path = args.edge_rankings
    logger.info(f"Loading {path}...")
    edges = pd.read_parquet(path)
    
    
    if 'pre_idx' not in edges.columns:
        logger.warning("pre_idx not found, re-mapping...")
       
        print(edges.columns)
        return

    n_nodes = max(edges['pre_idx'].max(), edges['post_idx'].max()) + 1
    adj_shape = (n_nodes, n_nodes)
    
    fractions = [0.0, 0.01, 0.05, 0.10, 0.20]
    
    for f in fractions:
        cutoff = int(len(edges) * f)
        edges_ablated = edges.sort_values('efficiency', ascending=False).iloc[cutoff:]
        
        gamma = 1e-6
        C = build_conductance_matrix(edges_ablated, gamma, adj_shape)
        
        n_components, labels = sp.csgraph.connected_components(C, directed=True, connection='strong')
        _, counts = np.unique(labels, return_counts=True)
        largest_label = np.argmax(counts)
        largest_scc_size = np.max(counts)
        
        if f == 0.20:
             prev_cutoff = int(len(edges) * 0.10)
             curr_cutoff = int(len(edges) * 0.20)
             removed_subset = edges.sort_values('efficiency', ascending=False).iloc[prev_cutoff:curr_cutoff]
             
             logger.info(f"--- Analysis of Edges Removed (10% -> 20%) ---")
             logger.info(f"Mean s_ij: {removed_subset['s_ij'].mean():.2e}")
             logger.info(f"Mean d_ij: {removed_subset['d_ij'].mean():.2f}")
             logger.info(f"Mean Betweenness: {removed_subset['betweenness'].mean():.2e}")
             logger.info(f"Mean Efficiency: {removed_subset['efficiency'].mean():.2e}")
             
             remaining = edges.sort_values('efficiency', ascending=False).iloc[curr_cutoff:]
             logger.info(f"--- Remaining Edges ---")
             logger.info(f"Mean s_ij: {remaining['s_ij'].mean():.2e}")
             logger.info(f"Mean d_ij: {remaining['d_ij'].mean():.2f}")
             
        scc_mask = (labels == largest_label)
        scc_indices = np.where(scc_mask)[0]
        
        
        from utils.matrix_utils import normalize_transition_matrix, solve_absorbing_fpt
        P = normalize_transition_matrix(C)
        P_scc = P[scc_indices, :][:, scc_indices]
        
        np.random.seed(42)
        targets = np.random.choice(len(scc_indices), 50, replace=False)
        
        fpt_values = []
        
        sample_targets = targets[:5]
        mean_fpts = []
        reachability_pcts = []
        for t in sample_targets:
            fpt_vec = solve_absorbing_fpt(P_scc, np.array([t]))
            valid = fpt_vec[fpt_vec < 1e10] 
            reachability_pcts.append(len(valid) / len(scc_indices))
            if len(valid) > 0:
                mean_fpts.append(np.mean(valid))
        
        avg_fpt = np.mean(mean_fpts) if mean_fpts else np.nan
        avg_reach = np.mean(reachability_pcts) if reachability_pcts else 0.0
        
        logger.info(f"Fraction: {f:.0%}, SCC: {largest_scc_size}, Mean FPT: {avg_fpt:.2f}, Reachable: {avg_reach:.1%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_rankings", type=str, required=True, help="Path to edge rankings parquet")
    args = parser.parse_args()
    check_sccs(args)

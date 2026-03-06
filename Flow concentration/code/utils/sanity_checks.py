import numpy as np
import pandas as pd
import logging
from scipy.sparse import csgraph, csr_matrix

def validate_graph(edges: pd.DataFrame):
    logging.info("Running graph integrity checks...")
    
    if (edges['s_ij'] < 0).any():
        raise ValueError("Found negative synapse counts!")
    
    if (edges['w_ij'] < 0).any():
        raise ValueError("Found negative weights!")
        
    if edges['s_ij'].isna().any() or edges['w_ij'].isna().any():
        raise ValueError("Found NaN weights or counts!")

    if edges.duplicated(subset=['pre', 'post']).any():
        raise ValueError("Found duplicate (pre, post) edges!")
        
    logging.info("Integrity checks passed.")

def check_uniqueness(nodes: pd.DataFrame):
    if nodes['bodyId'].duplicated().any():
        raise ValueError("Found duplicate bodyIds in node table!")
    logging.info("Uniqueness check passed.")

def check_completeness(nodes: pd.DataFrame, edges: pd.DataFrame, 
                      drop_missing_positions: bool = False) -> dict:
    logging.info("Checking data completeness...")
    
    n_total = len(nodes)
    n_coords = nodes[['x', 'y', 'z']].notna().all(axis=1).sum()
    pct_coords = (n_coords / n_total) * 100
    logging.info(f"Nodes with coordinates: {n_coords}/{n_total} ({pct_coords:.2f}%)")
    
    if pct_coords < 60.0: 
        raise RuntimeError(f"FATAL: Only {pct_coords:.2f}% of nodes have coordinates. Threshold is 60%.")
        
    stats = {
        "pct_nodes_with_coords": pct_coords
    }
    
   
    if 'd_ij' in edges.columns:
        e_total = len(edges)
        e_dist = edges['d_ij'].notna().sum()
        pct_dist = (e_dist / e_total) * 100
        logging.info(f"Edges with distances: {e_dist}/{e_total} ({pct_dist:.2f}%)")
        
        if pct_dist < 70.0: 
             raise RuntimeError(f"FATAL: Only {pct_dist:.2f}% of edges have distances. Threshold is 70%.")
        
        stats["pct_edges_with_distances"] = pct_dist
        
    return stats

def compute_scc(adj: csr_matrix) -> dict:
    logging.info("Computing SCCs...")
    n_components, labels = csgraph.connected_components(
        adj, directed=True, connection='strong'
    )
    
    _, counts = np.unique(labels, return_counts=True)
    largest_scc = counts.max()
    
    metrics = {
        "scc_count": int(n_components),
        "largest_scc_size": int(largest_scc),
        "largest_scc_fraction": float(largest_scc / adj.shape[0])
    }
    
    logging.info(f"SCC Count: {metrics['scc_count']}, Largest: {metrics['largest_scc_size']}")
    return metrics

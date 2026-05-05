import pandas as pd
import networkx as nx
import logging

logger = logging.getLogger(__name__)

def ablate_edges_by_rank(edges: pd.DataFrame, rank_col: str, top_k_percent: float) -> pd.DataFrame:
    """
    Removes the top k% of edges based on a ranking column.
    Returns a new edges DataFrame with those rows removed.
    """
    if rank_col not in edges.columns:
        raise ValueError(f"Column {rank_col} not found in edges.")
        
    n = len(edges)
    k = int(n * top_k_percent)
    logger.info(f"Ablating top {top_k_percent:.1%} edges ({k} edges) based on {rank_col}.")
    
    # Sort descending (assuming higher is 'more important' to remove)
    sorted_edges = edges.sort_values(rank_col, ascending=False)
    
    # Keep the bottom (n - k)
    remaining = sorted_edges.iloc[k:].copy()
    
    return remaining

def ablate_specific_edges(edges: pd.DataFrame, edges_to_remove: pd.DataFrame) -> pd.DataFrame:
    """
    Removes specific edges found in edges_to_remove (must have pre, post).
    """
    # Create set of tuples for fast lookup
    remove_set = set(zip(edges_to_remove['pre'], edges_to_remove['post']))
    
    # Filter
    # iterate and keep if not in set
    # Vectorized approach: merge?
    # Or just use an index
    
    keys = list(zip(edges['pre'], edges['post']))
    mask = [k not in remove_set for k in keys]
    
    return edges[mask].copy()

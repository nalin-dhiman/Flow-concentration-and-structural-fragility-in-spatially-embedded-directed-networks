import networkx as nx
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def to_networkx(nodes: pd.DataFrame, edges: pd.DataFrame, weight_col: str = 's_ij') -> nx.DiGraph:
    """Converts nodes and edges to a NetworkX DiGraph."""
    G = nx.DiGraph()
    # Nodes
    for _, row in nodes.iterrows():
        G.add_node(row['bodyId'], **row.to_dict())
    
    # Edges
    # Iterate tuples is faster than iterrows
    for row in edges.itertuples(index=False):
        # row has pre, post, s_ij, d_ij etc.
        # getattr is safe
        pre = getattr(row, 'pre')
        post = getattr(row, 'post')
        w = getattr(row, weight_col)
        d = getattr(row, 'd_ij', 1.0)
        G.add_edge(pre, post, weight=w, distance=d)
        
    return G

def compute_edge_betweenness(G: nx.DiGraph, k: int = None, weight: str = None) -> pd.DataFrame:
    """
    Computes edge betweenness centrality.
    Args:
        G: NetworkX graph
        k: Number of samples for approximation (default None = all nodes)
        weight: Edge attribute to use as distance (e.g. 'distance'). 
                If None, unweighted BFS. 
                Note: 'weight' in nx usually means cost/distance. 
                Our 's_ij' is conductance/affinity. 
                If we want shortest paths, we might want 1/s_ij or 'd_ij'.
    """
    logger.info(f"Computing edge betweenness (k={k}, weight={weight})...")
    eb = nx.edge_betweenness_centrality(G, k=k, weight=weight, normalized=True)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {'pre': u, 'post': v, 'betweenness': score} 
        for (u, v), score in eb.items()
    ])
    return df

def rank_edges_by_efficiency(edges: pd.DataFrame, eta: float = 1.0) -> pd.DataFrame:
    """
    Ranks edges by a heuristic efficiency metric.
    Efficiency = Betweenness / Cost
    Cost = d_ij^eta
    """
    if 'betweenness' not in edges.columns:
        raise ValueError("Edges dataframe must contain 'betweenness' column.")
    if 'd_ij' not in edges.columns:
        raise ValueError("Edges dataframe must contain 'd_ij' column.")
        
    edges['cost'] = edges['d_ij'] ** eta
    # Avoid division by zero
    edges['efficiency'] = edges['betweenness'] / (edges['cost'] + 1e-9)
    
    return edges.sort_values('efficiency', ascending=False)

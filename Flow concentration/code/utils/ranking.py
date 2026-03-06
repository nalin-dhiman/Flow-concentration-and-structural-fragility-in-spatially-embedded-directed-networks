import networkx as nx
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def to_networkx(nodes: pd.DataFrame, edges: pd.DataFrame, weight_col: str = 's_ij') -> nx.DiGraph:
    G = nx.DiGraph()
    
    for _, row in nodes.iterrows():
        G.add_node(row['bodyId'], **row.to_dict())
    
    
    for row in edges.itertuples(index=False):
        
        pre = getattr(row, 'pre')
        post = getattr(row, 'post')
        w = getattr(row, weight_col)
        d = getattr(row, 'd_ij', 1.0)
        G.add_edge(pre, post, weight=w, distance=d)
        
    return G

def compute_edge_betweenness(G: nx.DiGraph, k: int = None, weight: str = None) -> pd.DataFrame:
    
    logger.info(f"Computing edge betweenness (k={k}, weight={weight})...")
    eb = nx.edge_betweenness_centrality(G, k=k, weight=weight, normalized=True)
    
    df = pd.DataFrame([
        {'pre': u, 'post': v, 'betweenness': score} 
        for (u, v), score in eb.items()
    ])
    return df

def rank_edges_by_efficiency(edges: pd.DataFrame, eta: float = 1.0) -> pd.DataFrame:
   
    if 'betweenness' not in edges.columns:
        raise ValueError("Edges dataframe must contain 'betweenness' column.")
    if 'd_ij' not in edges.columns:
        raise ValueError("Edges dataframe must contain 'd_ij' column.")
        
    edges['cost'] = edges['d_ij'] ** eta
    edges['efficiency'] = edges['betweenness'] / (edges['cost'] + 1e-9)
    
    return edges.sort_values('efficiency', ascending=False)

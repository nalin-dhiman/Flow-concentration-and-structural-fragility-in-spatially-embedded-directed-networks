import pandas as pd
import scipy.sparse as sp
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_canonical_v6a(canonical_dir: Path):
    """
    Loads nodes, edges, distances, and adjacency.
    Reused from src/metrics/compute_metrics.py logic.
    """
    canonical_dir = Path(canonical_dir)
    logging.info(f"Loading canonical artifacts from {canonical_dir}")
    
    nodes = pd.read_parquet(canonical_dir / "nodes.parquet")
    edges = pd.read_parquet(canonical_dir / "edges.parquet")
    dist_edges = pd.read_parquet(canonical_dir / "distance_edges.parquet")
    
    # Merge distance into edges if not already present
    if 'd_ij' not in edges.columns:
        logging.info("Merging distances into edges...")
        edges = edges.merge(dist_edges, on=['pre', 'post'], how='left')
    
    # Fill missing distances with NaN or default?
    # For now, drop edges without distance or handle metric computation carefully
    # edges = edges.dropna(subset=['d_ij']) 
    
    return nodes, edges

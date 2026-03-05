import numpy as np
import pandas as pd
import scipy.sparse as sp
import logging
from .base import NullModel

class N0WeightedNull(NullModel):
    """
    N0: Weighted Configuration Model (Soft).
    Preserves in-strength and out-strength sequences exactly (in expectation).
    Randomizes connections.
    """
    
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, adj: sp.csr_matrix):
        super().__init__("N0", nodes, edges, adj)
        
        # Create mappings
        self.node_map = {bid: i for i, bid in enumerate(nodes['bodyId'])}
        self.n_nodes = len(nodes)
        
        # Calculate strengths aligned to internal indices (0..N-1)
        # We need to aggregate edges by pre_idx/post_idx
        # edges might use bodyIds
        
        if 'pre_idx' not in edges.columns:
             edges['pre_idx'] = edges['pre'].map(self.node_map)
             edges['post_idx'] = edges['post'].map(self.node_map)
             
        # Group by index
        # This is fast enough (6M rows)
        s_out_series = edges.groupby('pre_idx')['s_ij'].sum()
        s_in_series = edges.groupby('post_idx')['s_ij'].sum()
        
        # Convert to full arrays (0..N-1)
        self.s_out = np.zeros(self.n_nodes)
        self.s_in = np.zeros(self.n_nodes)
        
        self.s_out[s_out_series.index] = s_out_series.values
        self.s_in[s_in_series.index] = s_in_series.values
        
        self.W_total = int(self.s_out.sum())
        
        # Probabilities
        self.p_out = self.s_out / self.W_total
        self.p_in = self.s_in / self.W_total
        
        # Indices to sample from
        self.indices = np.arange(self.n_nodes)
        
        # Cache coordinate array for details
        self.coords = nodes[['x', 'y', 'z']].values

    def generate(self, seed: int) -> pd.DataFrame:
        """
        Generates N0 using Soft Configuration Model (sampling).
        """
        rng = np.random.default_rng(seed)
        
        logging.info(f"N0: Generating {self.W_total} synapses via Soft Config Model...")
        
        # Sample source and target indices for each synapse
        # We sample W_total synapses
        
        # Optimization: chunking if W_total is too large?
        # 24M is fine for numpy (24M * 8 bytes = 200MB).
        
        src_idxs = rng.choice(self.indices, size=self.W_total, p=self.p_out)
        tgt_idxs = rng.choice(self.indices, size=self.W_total, p=self.p_in)
        
        # Aggregate to weighted edges
        # key = src * N + tgt
        # Use int64
        N = self.n_nodes
        keys = src_idxs.astype(np.int64) * N + tgt_idxs.astype(np.int64)
        
        logging.info("N0: Aggregating synapses to edges...")
        unique_keys, counts = np.unique(keys, return_counts=True)
        
        # Decode keys
        pre_idxs = unique_keys // N
        post_idxs = unique_keys % N
        
        # Build DataFrame
        df = pd.DataFrame({
            'pre_idx': pre_idxs,
            'post_idx': post_idxs,
            's_ij': counts
        })
        
        # Add distances
        logging.info("N0: Calculating distances...")
        p1 = self.coords[pre_idxs]
        p2 = self.coords[post_idxs]
        dists = np.linalg.norm(p1 - p2, axis=1)
        df['d_ij'] = dists
        
        # Add bodyIds
        # Reverse map? Or just map using series
        # self.nodes['bodyId'] is in order 0..N-1
        body_ids = self.nodes['bodyId'].values
        df['pre'] = body_ids[pre_idxs]
        df['post'] = body_ids[post_idxs]
        
        return df

import numpy as np
import pandas as pd
import scipy.sparse as sp
import logging
from .base import NullModel

class N0WeightedNull(NullModel):
   
    
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, adj: sp.csr_matrix):
        super().__init__("N0", nodes, edges, adj)
        
        self.node_map = {bid: i for i, bid in enumerate(nodes['bodyId'])}
        self.n_nodes = len(nodes)
        
        
        
        if 'pre_idx' not in edges.columns:
             edges['pre_idx'] = edges['pre'].map(self.node_map)
             edges['post_idx'] = edges['post'].map(self.node_map)
             
       
        s_out_series = edges.groupby('pre_idx')['s_ij'].sum()
        s_in_series = edges.groupby('post_idx')['s_ij'].sum()
        
        self.s_out = np.zeros(self.n_nodes)
        self.s_in = np.zeros(self.n_nodes)
        
        self.s_out[s_out_series.index] = s_out_series.values
        self.s_in[s_in_series.index] = s_in_series.values
        
        self.W_total = int(self.s_out.sum())
        
        self.p_out = self.s_out / self.W_total
        self.p_in = self.s_in / self.W_total
        
        self.indices = np.arange(self.n_nodes)
        
        self.coords = nodes[['x', 'y', 'z']].values

    def generate(self, seed: int) -> pd.DataFrame:
        
        rng = np.random.default_rng(seed)
        
        logging.info(f"N0: Generating {self.W_total} synapses via Soft Config Model...")
        
        
        src_idxs = rng.choice(self.indices, size=self.W_total, p=self.p_out)
        tgt_idxs = rng.choice(self.indices, size=self.W_total, p=self.p_in)
        
        
        N = self.n_nodes
        keys = src_idxs.astype(np.int64) * N + tgt_idxs.astype(np.int64)
        
        logging.info("N0: Aggregating synapses to edges...")
        unique_keys, counts = np.unique(keys, return_counts=True)
        
        pre_idxs = unique_keys // N
        post_idxs = unique_keys % N
        
        df = pd.DataFrame({
            'pre_idx': pre_idxs,
            'post_idx': post_idxs,
            's_ij': counts
        })
        
        logging.info("N0: Calculating distances...")
        p1 = self.coords[pre_idxs]
        p2 = self.coords[post_idxs]
        dists = np.linalg.norm(p1 - p2, axis=1)
        df['d_ij'] = dists
        
        body_ids = self.nodes['bodyId'].values
        df['pre'] = body_ids[pre_idxs]
        df['post'] = body_ids[post_idxs]
        
        return df

import numpy as np
import pandas as pd
import scipy.sparse as sp
import logging
from tqdm import tqdm
from .base import NullModel

class SpatialUtils:
    
    @staticmethod
    def get_distance_histogram(dists, bins=50, max_dist=None):
        if max_dist is None:
            max_dist = np.max(dists)
        hist, bin_edges = np.histogram(dists, bins=bins, range=(0, max_dist))
        return hist, bin_edges, max_dist

    @staticmethod
    def estimate_proposal_dist(n_samples, pre_nodes, post_nodes, pre_probs, post_probs, coords):
       
        s_idx = np.random.choice(len(pre_nodes), size=n_samples, p=pre_probs)
        t_idx = np.random.choice(len(post_nodes), size=n_samples, p=post_probs)
        
        sources = pre_nodes[s_idx]
        targets = post_nodes[t_idx]
        
        p1 = coords[sources]
        p2 = coords[targets]
        sample_dists = np.linalg.norm(p1 - p2, axis=1)
        
        return sample_dists

class N1SpatialNull(NullModel):
    
    
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, adj: sp.csr_matrix):
        super().__init__("N1", nodes, edges, adj)
        
        if 'pre_idx' not in edges.columns:
             node_map = {bid: i for i, bid in enumerate(nodes['bodyId'])}
             edges['pre_idx'] = edges['pre'].map(node_map)
             edges['post_idx'] = edges['post'].map(node_map)

        self.node_map = {bid: i for i, bid in enumerate(nodes['bodyId'])}
        self.coords = nodes[['x', 'y', 'z']].values
        

        s_out = np.array(adj.sum(axis=1)).flatten()
        s_in = np.array(adj.sum(axis=0)).flatten()
        
        self.total_out = s_out.sum()
        self.total_in = s_in.sum()
        
        self.out_probs = s_out / self.total_out
        self.in_probs = s_in / self.total_in
        
        self.indices = np.arange(len(nodes))
        
        logging.info("N1: Fitting distance acceptance curve...")
        
        
        if 'd_ij' not in edges.columns:
            logging.error("d_ij missing in edges for N1 support. Computing...")
            p1 = self.coords[edges['pre_idx'].values]
            p2 = self.coords[edges['post_idx'].values]
            edges['d_ij'] = np.linalg.norm(p1 - p2, axis=1)
            
        real_dists = edges['d_ij'].values
        self.max_dist = np.max(real_dists)
        n_bins = 100
        
        self.real_hist, self.bin_edges = np.histogram(real_dists, bins=n_bins, range=(0, self.max_dist))
        self.real_hist = self.real_hist / self.real_hist.sum() # Normalize
        
        
        sample_dists = SpatialUtils.estimate_proposal_dist(
            n_samples=1000000, 
            pre_nodes=self.indices, post_nodes=self.indices,
            pre_probs=self.out_probs, post_probs=self.in_probs,
            coords=self.coords
        )
        
        self.prop_hist, _ = np.histogram(sample_dists, bins=n_bins, range=(0, self.max_dist))
        
        self.prop_hist = self.prop_hist.astype(float)
        mask = self.prop_hist > 0
        
        self.acceptance_curve = np.zeros(n_bins)
        
        
        
        ratios = np.zeros(n_bins)
        ratios[mask] = self.real_hist[mask] / self.prop_hist[mask]
        
        max_ratio = ratios.max() if ratios.max() > 0 else 1.0
        self.acceptance_curve = ratios / max_ratio
        
        
        
        logging.info("N1: Distance curve fitted.")

    def generate(self, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        
        target_edges = len(self.edges)
        edges_list = []
        dists_list = []
        
        batch_size = 100000
        
        
        edges_collected = 0
        pbar = tqdm(total=target_edges, desc="N1 Generation", leave=False)
        
        while edges_collected < target_edges:
            needed = target_edges - edges_collected
            n_prop = int(needed * 5) 
            n_prop = max(n_prop, batch_size) 
            
            s_idx = rng.choice(self.indices, size=n_prop, p=self.out_probs)
            t_idx = rng.choice(self.indices, size=n_prop, p=self.in_probs)
            
            p1 = self.coords[s_idx]
            p2 = self.coords[t_idx]
            dists = np.linalg.norm(p1 - p2, axis=1)
            
            
            bin_idx = np.digitize(dists, self.bin_edges) - 1
            
            valid_mask = (bin_idx >= 0) & (bin_idx < len(self.acceptance_curve))
            
            
            probs = np.zeros(n_prop)
            probs[valid_mask] = self.acceptance_curve[bin_idx[valid_mask]]
            
            rand_vals = rng.random(n_prop)
            accepted = rand_vals < probs
            
            n_acc = accepted.sum()
            
            if n_acc > 0:
                
                batch_df = pd.DataFrame({
                    'pre_idx': s_idx[accepted],
                    'post_idx': t_idx[accepted],
                    'd_ij': dists[accepted]
                })
                
                edges_list.append(batch_df)
                edges_collected += n_acc
                pbar.update(n_acc)
                
        pbar.close()
        
        full_df = pd.concat(edges_list, ignore_index=True)
        if len(full_df) > target_edges:
            full_df = full_df.iloc[:target_edges]
            
        
        idx_to_body = {i: b for b, i in self.node_map.items()}
        full_df['pre'] = full_df['pre_idx'].map(idx_to_body)
        full_df['post'] = full_df['post_idx'].map(idx_to_body)
        
        
        weights = self.edges['s_ij'].values.copy()
        rng.shuffle(weights) 
        if len(weights) >= len(full_df):
            full_df['s_ij'] = weights[:len(full_df)]
        else:
            full_df['s_ij'] = np.resize(weights, len(full_df))
            
        return full_df


class N2BlockNull(NullModel):
   
    
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, adj: sp.csr_matrix):
        super().__init__("N2", nodes, edges, adj)
        
        if 'type' not in nodes.columns:
            raise ValueError("Nodes dataframe must have 'type' column for N2.")
            
        self.node_ids = nodes['bodyId'].values
        self.node_map = {bid: i for i, bid in enumerate(self.node_ids)}
        self.n_nodes = len(self.node_ids)
        
        self.node_types = nodes['type'].fillna('Unknown').values
        self.unique_types = np.unique(self.node_types)
        
        pre_in_map = edges['pre'].isin(self.node_map)
        post_in_map = edges['post'].isin(self.node_map)
        valid_mask = pre_in_map & post_in_map
        
        n_dropped = (~valid_mask).sum()
        if n_dropped > 0:
            pct_missing = n_dropped / len(edges)
            logging.warning(f"N2: Found {n_dropped} edges ({pct_missing:.2%}) with missing endpoints in node list.")
            
            if pct_missing > 0.01:
                raise ValueError(f"N2 Validation Failed: >1% edges missing endpoints ({pct_missing:.2%}). Check node/edge alignment.")
                
            logging.info(f"N2: Dropping {n_dropped} invalid edges.")
            self.edges = edges[valid_mask].copy()
           
        else:
            self.edges = edges
            
        self.type_indices = {}
        for t in self.unique_types:
            self.type_indices[t] = np.where(self.node_types == t)[0]
        
        
        logging.info("N2: Counting block edges...")
        if 'pre_idx' not in edges.columns:
             
             node_map = {bid: i for i, bid in enumerate(nodes['bodyId'])}
             edges['pre_idx'] = edges['pre'].map(node_map)
             edges['post_idx'] = edges['post'].map(node_map)
             
        pre_types = self.node_types[edges['pre_idx'].values]
        post_types = self.node_types[edges['post_idx'].values]
        
        block_edges = pd.DataFrame({'t_pre': pre_types, 't_post': post_types})
        self.block_counts = block_edges.groupby(['t_pre', 't_post']).size().to_dict()
        
        logging.info(f"N2: Found {len(self.block_counts)} non-empty block pairs.")
        
       
        
        self.coords = nodes[['x', 'y', 'z']].values
        
       
        
        real_dists = edges['d_ij'].values if 'd_ij' in edges.columns else self._calc_dists(edges)
        
        self.max_dist = np.max(real_dists)
        hist, bin_edges = np.histogram(real_dists, bins=100, range=(0, self.max_dist))
        real_prob = hist / hist.sum()
        
        
        
        idx = np.arange(len(nodes))
        
        s_out = np.array(adj.sum(axis=1)).flatten()
        s_in = np.array(adj.sum(axis=0)).flatten()
        self.out_probs = s_out / s_out.sum() 
        self.in_probs = s_in / s_in.sum()
        
        sample_dists = SpatialUtils.estimate_proposal_dist(
             1000000, idx, idx, self.out_probs, self.in_probs, self.coords
        )
        prop_hist, _ = np.histogram(sample_dists, bins=100, range=(0, self.max_dist))
        prop_hist = prop_hist.astype(float)
        
        
        ratios = np.zeros(100)
        mask = prop_hist > 0
        ratios[mask] = hist[mask] / prop_hist[mask]
        self.acceptance_curve = ratios / (ratios.max() if ratios.max()>0 else 1.0)
        self.bin_edges = bin_edges
        
        
        self.indices = np.arange(len(nodes))
        
    def _calc_dists(self, df):
        p1 = self.coords[df['pre_idx'].values]
        p2 = self.coords[df['post_idx'].values]
        return np.linalg.norm(p1 - p2, axis=1)

    def generate(self, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        all_edges = []
        
        
        pairs = sorted(self.block_counts.keys())
        
        pbar = tqdm(pairs, desc="N2 Generation", leave=False)
        
        for (t_pre, t_post) in pbar:
            n_target = self.block_counts[(t_pre, t_post)]
            
            src_indices = self.type_indices[t_pre]
            tgt_indices = self.type_indices[t_post]
            
            
            p_src = self.out_probs[src_indices]
            if p_src.sum() == 0: p_src = np.ones(len(p_src))/len(p_src)
            else: p_src = p_src / p_src.sum()
            
            p_tgt = self.in_probs[tgt_indices]
            if p_tgt.sum() == 0: p_tgt = np.ones(len(p_tgt))/len(p_tgt)
            else: p_tgt = p_tgt / p_tgt.sum()
            
            collected = 0
            block_edges_list = []
            
            while collected < n_target:
                needed = n_target - collected
                n_prop = max(int(needed * 20), 100) 
                n_prop = min(n_prop, 1000000)
                
                s_idx = rng.choice(src_indices, size=n_prop, p=p_src)
                t_idx = rng.choice(tgt_indices, size=n_prop, p=p_tgt)
                
                p1 = self.coords[s_idx]
                p2 = self.coords[t_idx]
                dists = np.linalg.norm(p1 - p2, axis=1)
                
                bin_idx = np.digitize(dists, self.bin_edges) - 1
                valid_mask = (bin_idx >= 0) & (bin_idx < len(self.acceptance_curve))
                
                probs = np.zeros(n_prop)
                probs[valid_mask] = self.acceptance_curve[bin_idx[valid_mask]]
                
                accepted = rng.random(n_prop) < probs
                
                n_acc = accepted.sum()
                if n_acc > 0:
                    batch_df = pd.DataFrame({
                        'pre_idx': s_idx[accepted],
                        'post_idx': t_idx[accepted],
                        'd_ij': dists[accepted]
                    })
                    block_edges_list.append(batch_df)
                    collected += n_acc
            
            block_df = pd.concat(block_edges_list, ignore_index=True)
            if len(block_df) > n_target:
                block_df = block_df.iloc[:n_target]
            all_edges.append(block_df)
            
        full_df = pd.concat(all_edges, ignore_index=True)
        
        idx_to_body = {i: b for b, i in self.node_map.items()}
        full_df['pre'] = full_df['pre_idx'].map(idx_to_body)
        full_df['post'] = full_df['post_idx'].map(idx_to_body)
        
        weights = self.edges['s_ij'].values.copy()
        rng.shuffle(weights)
        if len(weights) >= len(full_df):
            full_df['s_ij'] = weights[:len(full_df)]
        else:
            full_df['s_ij'] = np.resize(weights, len(full_df))
            
        return full_df

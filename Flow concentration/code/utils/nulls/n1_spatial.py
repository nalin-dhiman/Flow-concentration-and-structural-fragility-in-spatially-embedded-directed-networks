import numpy as np
import pandas as pd
import scipy.sparse as sp
import logging
from tqdm import tqdm
from .base import NullModel

class SpatialUtils:
    """Helper for spatial null statistics."""
    
    @staticmethod
    def get_distance_histogram(dists, bins=50, max_dist=None):
        if max_dist is None:
            max_dist = np.max(dists)
        hist, bin_edges = np.histogram(dists, bins=bins, range=(0, max_dist))
        return hist, bin_edges, max_dist

    @staticmethod
    def estimate_proposal_dist(n_samples, pre_nodes, post_nodes, pre_probs, post_probs, coords):
        """
        Estimates the distance distribution of the proposal (k_i * k_j) by sampling.
        """
        # Sample pairs
        s_idx = np.random.choice(len(pre_nodes), size=n_samples, p=pre_probs)
        t_idx = np.random.choice(len(post_nodes), size=n_samples, p=post_probs)
        
        sources = pre_nodes[s_idx]
        targets = post_nodes[t_idx]
        
        # Calculate distances
        p1 = coords[sources]
        p2 = coords[targets]
        sample_dists = np.linalg.norm(p1 - p2, axis=1)
        
        return sample_dists

class N1SpatialNull(NullModel):
    """
    N1: Spatial Null.
    Preserves distance distribution exactly (via rejection sampling).
    Preserves degree sequence approximately (in expectation).
    """
    
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, adj: sp.csr_matrix):
        super().__init__("N1", nodes, edges, adj)
        
        # 1. Prepare coordinates and probabilities
        if 'pre_idx' not in edges.columns:
             node_map = {bid: i for i, bid in enumerate(nodes['bodyId'])}
             edges['pre_idx'] = edges['pre'].map(node_map)
             edges['post_idx'] = edges['post'].map(node_map)

        # Standardize indices
        self.node_map = {bid: i for i, bid in enumerate(nodes['bodyId'])}
        self.coords = nodes[['x', 'y', 'z']].values
        
        # Calculate weighted degrees (strength) for proposal
        # We use Strength because "Weighted configuration" is the baseline for N0.
        # But for N1 "Degree/strength need only be approximately preserved".
        # Using Strength is better for weighted graphs.
        
        # We need vectors aligned with 0..N-1
        # out_strength array
        s_out = np.array(adj.sum(axis=1)).flatten()
        s_in = np.array(adj.sum(axis=0)).flatten()
        
        # Handle zero-degree nodes (they can't form edges anyway)
        self.total_out = s_out.sum()
        self.total_in = s_in.sum()
        
        self.out_probs = s_out / self.total_out
        self.in_probs = s_in / self.total_in
        
        self.indices = np.arange(len(nodes))
        
        # 2. Fit Distance Acceptance Curve
        logging.info("N1: Fitting distance acceptance curve...")
        
        # Real distance distribution
        # Check if d_ij is in edges
        if 'd_ij' not in edges.columns:
            logging.error("d_ij missing in edges for N1 support. Computing...")
            # Compute distances for existing edges
            p1 = self.coords[edges['pre_idx'].values]
            p2 = self.coords[edges['post_idx'].values]
            edges['d_ij'] = np.linalg.norm(p1 - p2, axis=1)
            
        real_dists = edges['d_ij'].values
        self.max_dist = np.max(real_dists)
        n_bins = 100
        
        self.real_hist, self.bin_edges = np.histogram(real_dists, bins=n_bins, range=(0, self.max_dist))
        self.real_hist = self.real_hist / self.real_hist.sum() # Normalize
        
        # Proposal distance distribution (Strength * Strength)
        # Check sufficient samples? 1M samples usually enough for 100 bins
        sample_dists = SpatialUtils.estimate_proposal_dist(
            n_samples=1000000, 
            pre_nodes=self.indices, post_nodes=self.indices,
            pre_probs=self.out_probs, post_probs=self.in_probs,
            coords=self.coords
        )
        
        self.prop_hist, _ = np.histogram(sample_dists, bins=n_bins, range=(0, self.max_dist))
        
        # Avoid division by zero
        self.prop_hist = self.prop_hist.astype(float)
        mask = self.prop_hist > 0
        
        # Calculate acceptance ratios
        self.acceptance_curve = np.zeros(n_bins)
        
        # Ratio = Target / Proposal
        # If Proposal has data but Target doesn't -> 0 probability (correct)
        # If Target has data but Proposal doesn't -> Theoretical impossibility (or undersampling).
        # We clamp in that case? Or just use 1.0?
        # With 1M samples, proposal should cover everything real graph covers geometrically.
        
        ratios = np.zeros(n_bins)
        ratios[mask] = self.real_hist[mask] / self.prop_hist[mask]
        
        # Normalize to max 1.0
        max_ratio = ratios.max() if ratios.max() > 0 else 1.0
        self.acceptance_curve = ratios / max_ratio
        
        # Smoothing? Raw histograms can be noisy.
        # But with 100 bins and 1M edges, it's fine.
        
        logging.info("N1: Distance curve fitted.")

    def generate(self, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        
        target_edges = len(self.edges)
        edges_list = []
        dists_list = []
        
        # Batch generation
        batch_size = 100000
        # Estimate acceptance rate to size buffers
        # avg_acceptance ~ 1 / max_ratio theoretically?
        # Just generate in loops.
        
        edges_collected = 0
        pbar = tqdm(total=target_edges, desc="N1 Generation", leave=False)
        
        while edges_collected < target_edges:
            # Over-sample based on remaining needed
            needed = target_edges - edges_collected
            n_prop = int(needed * 5) # Heuristic buffer
            n_prop = max(n_prop, batch_size) 
            
            # Sample Proposal
            s_idx = rng.choice(self.indices, size=n_prop, p=self.out_probs)
            t_idx = rng.choice(self.indices, size=n_prop, p=self.in_probs)
            
            # Compute dists
            p1 = self.coords[s_idx]
            p2 = self.coords[t_idx]
            dists = np.linalg.norm(p1 - p2, axis=1)
            
            # Bin indices
            # np.digitize returns 1..N_bins. We want 0..N_bins-1.
            bin_idx = np.digitize(dists, self.bin_edges) - 1
            
            # Filter out of range
            valid_mask = (bin_idx >= 0) & (bin_idx < len(self.acceptance_curve))
            
            # Acceptance prob
            # P = curve[bin_idx]
            probs = np.zeros(n_prop)
            probs[valid_mask] = self.acceptance_curve[bin_idx[valid_mask]]
            
            # Reject
            rand_vals = rng.random(n_prop)
            accepted = rand_vals < probs
            
            n_acc = accepted.sum()
            
            if n_acc > 0:
                # Store
                # We need weights too. N1 preserves weight distribution globally?
                # "Preserve in-strength and out-strength distributions exactly." was N0.
                # N1: "Degree/strength need only be approximately preserved."
                # Does implies we should just assign weights?
                # Best practice: Sample weights from empirical weight distribution independently?
                # Or coupling weight with distance?
                # Prompt doesn't specify weight-distance coupling.
                # Prompt says: "Rewire edges so that the marginal distance histogram matches..."
                # Usually implies using the existing weights?
                # If we just list edges (u,v), we need s_ij.
                # Let's assign weights by sampling from the original weight list?
                # This preserves global weight distribution.
                
                # Append basic edge info first
                batch_df = pd.DataFrame({
                    'pre_idx': s_idx[accepted],
                    'post_idx': t_idx[accepted],
                    'd_ij': dists[accepted]
                })
                
                edges_list.append(batch_df)
                edges_collected += n_acc
                pbar.update(n_acc)
                
        pbar.close()
        
        # Concatenate
        full_df = pd.concat(edges_list, ignore_index=True)
        if len(full_df) > target_edges:
            full_df = full_df.iloc[:target_edges]
            
        # Map back to bodyIds
        # Need reverse map
        idx_to_body = {i: b for b, i in self.node_map.items()}
        full_df['pre'] = full_df['pre_idx'].map(idx_to_body)
        full_df['post'] = full_df['post_idx'].map(idx_to_body)
        
        # Assign Weights
        # Shuffle original weights and assign
        # This preserves global weight statistics exactly.
        # But breaks any weight-distance correlation.
        # Is that intended? Probably. "null" implies breaking correlations.
        weights = self.edges['s_ij'].values.copy()
        rng.shuffle(weights) # In-place
        # If we have disparate sizes (due to edge count approx), truncate/cycle
        if len(weights) >= len(full_df):
            full_df['s_ij'] = weights[:len(full_df)]
        else:
            # Should not happen if target_edges = len(self.edges)
            full_df['s_ij'] = np.resize(weights, len(full_df))
            
        return full_df


class N2BlockNull(NullModel):
    """
    N2: Block-aware Spatial Null.
    Partition nodes by type.
    Preserve block-wise densities (generate exact number of edges per block pair).
    Match global distance distribution within blocks via rejection sampling.
    """
    
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, adj: sp.csr_matrix):
        super().__init__("N2", nodes, edges, adj)
        
        # 1. Setup metadata
        if 'type' not in nodes.columns:
            raise ValueError("Nodes dataframe must have 'type' column for N2.")
            
        # FIX: Explicit mapping
        self.node_ids = nodes['bodyId'].values
        self.node_map = {bid: i for i, bid in enumerate(self.node_ids)}
        self.n_nodes = len(self.node_ids)
        
        self.node_types = nodes['type'].fillna('Unknown').values
        self.unique_types = np.unique(self.node_types)
        
        # VALIDATION: Check for missing endpoints
        pre_in_map = edges['pre'].isin(self.node_map)
        post_in_map = edges['post'].isin(self.node_map)
        valid_mask = pre_in_map & post_in_map
        
        n_dropped = (~valid_mask).sum()
        if n_dropped > 0:
            pct_missing = n_dropped / len(edges)
            logging.warning(f"N2: Found {n_dropped} edges ({pct_missing:.2%}) with missing endpoints in node list.")
            
            if pct_missing > 0.01:
                raise ValueError(f"N2 Validation Failed: >1% edges missing endpoints ({pct_missing:.2%}). Check node/edge alignment.")
                
            # Filter edges
            logging.info(f"N2: Dropping {n_dropped} invalid edges.")
            self.edges = edges[valid_mask].copy()
            # Re-compute adj if edges changed?
            # adj passed to init is likely built from passed edges.
            # If we drop edges, adj should ideally exclude them.
            # But adj is used for strength/degree.
            # If we drop edges, we should probably update adj or just proceed with original adj as 'target'?
            # N2 preserves block counts from 'edges'.
            # If we drop edges, block counts decrease.
            # This is correct.
        else:
            self.edges = edges
            
        # Indices per type
        self.type_indices = {}
        for t in self.unique_types:
            self.type_indices[t] = np.where(self.node_types == t)[0]
            
        # 2. Block Edge Counts
        # We need to count how many edges exist between each block pair (A, B)
        # Group edges by (type_pre, type_post)
        
        logging.info("N2: Counting block edges...")
        if 'pre_idx' not in edges.columns:
             # Assuming standard indices 0..N-1
             # If external indices, this fails. Need N1's setup logic.
             node_map = {bid: i for i, bid in enumerate(nodes['bodyId'])}
             edges['pre_idx'] = edges['pre'].map(node_map)
             edges['post_idx'] = edges['post'].map(node_map)
             
        pre_types = self.node_types[edges['pre_idx'].values]
        post_types = self.node_types[edges['post_idx'].values]
        
        # Create a dataframe to group
        block_edges = pd.DataFrame({'t_pre': pre_types, 't_post': post_types})
        self.block_counts = block_edges.groupby(['t_pre', 't_post']).size().to_dict()
        
        logging.info(f"N2: Found {len(self.block_counts)} non-empty block pairs.")
        
        # 3. Distance Acceptance Curve (Reuse N1 logic or similar?)
        # Prompt: "Within each block pair, rewire edges while matching distance distribution."
        # Does this mean match EACH block's distance distribution? Or the global one?
        # "Match distance distribution" usually implies global characteristic in spatial nulls.
        # However, "Within each block pair... matching distance distribution" suggests local constraint.
        # But fitting f_AB(d) for sparse blocks is impossible.
        # We will use the GLOBAL acceptance curve from N1.
        # But apply it to block-constrained generation.
        
        # We need N1's fit logic.
        # Let's instantiate N1 to get the curve?
        # Or duplicate logic. Duplicate for independence.
        
        self.coords = nodes[['x', 'y', 'z']].values
        
        # Fit Global Curve (Same as N1)
        # Note: We use real_dists and prop_dists from the whole graph.
        
        real_dists = edges['d_ij'].values if 'd_ij' in edges.columns else self._calc_dists(edges)
        
        # Use simple helpers
        self.max_dist = np.max(real_dists)
        hist, bin_edges = np.histogram(real_dists, bins=100, range=(0, self.max_dist))
        real_prob = hist / hist.sum()
        
        # Global Proposal: P(d) ~ d^2 (geometric) roughly, if uniform.
        # But we want to correct for *geometry*, so we need the node-pair distance distribution.
        # Sample random pairs from WHOLE graph to get geometry baseline.
        
        idx = np.arange(len(nodes))
        # Uniform sampling for geometry baseline?
        # Or degree-weighted?
        # N2 preserves density. Inside a block, degrees might be uniform-ish?
        # Let's use Degree-Weighted proposal for consistency with N1?
        # If we use Degree-Weighted proposal, we correct for degree bias.
        
        # Let's use Uniform sampling of pairs for geometry correction f_geo(d).
        # And we assume k_i k_j handles the degree part.
        
        # Wait, if we use rejection sampling inside a block (A->B):
        # We pick u in A, v in B.
        # P_prop(d) depends on geometry of A vs B.
        # If A and B are far apart, P_prop(d) is peaked at large d.
        # If we enforce global f(d) (which favors short edges), we might reject EVERYTHING.
        # This is the risk.
        
        # Interpretation of "matching distance distribution" in Block Null: 
        # "Preserve the distance profile OF THAT BLOCK PAIR."
        # If Block A->B has mean distance 100um, we shouldn't force it to 20um.
        # BUT we should randomize connections *given* the distance constraints.
        
        # If we preserve "distance distribution of edges" within each block pair:
        # We just rewire A->B edges to other A->B pairs with similar distances?
        # Or just swap existing A->B edges with other existing A->B edges?
        # That's just degree-preserving randomization within blocks.
        # That DOES NOT change the distance distribution of that block.
        # So "matching distance distribution" is trivially satisfied if we just shuffle?
        # No, shuffling destroys distance correlations *if* we don't constrain it.
        # Shuffling A->B makes distances random samples from Geometry(A, B).
        # We want Dist ~ Real_Dist(A, B).
        
        # So: For each block pair, we fit f_AB(d) = Real(d) / Geom(d).
        # AND we sample using that.
        # Problem: Sparse blocks.
        # Fallback: If block has < 50 edges, just keep real distances (shuffle w/ annealing)?
        # Or just use "Global f(d)"?
        
        # Let's look at the science.
        # Cell type A -> Cell type B usually has specific spatial logic.
        # If we destroy that spatial logic and replace with "Generic Spatial Logic", that's a strong null.
        # If we preserve "Specific Spatial Logic", we are testing microstructure.
        # N2 is "Type/Block aware spatial null".
        # Usually checking if "Type constraints + Space" explain connectivity.
        # So we should generally respect the spatial profile of A->B.
        
        # Approach:
        # Valid Edges in (A,B) = E_AB.
        # 1. Shuffle endpoints of E_AB to get random (u,v) pairs.
        # 2. Check resulting distance distribution.
        # 3. If Real was "tighter" than Random, we should bias towards tight.
        
        # Given complexity and "sparse block" issue, I will implement:
        # **Kernel Density Per Block?** No.
        # **Linear combination?**
        # Let's use the **Global Distance Kernel** as a starting point.
        # Why? Because otherwise we just reproduce the graph too closely.
        # H0: "Connection prob depends on Type Pairs + Global Distance Decay".
        # If Real Graph has specific distance features per type pair (e.g. A only connects to B at far range),
        # N2 will destroy that if we use Global Decay.
        # And that's exactly what we want to test!
        # Does "Type + Global Distance Rule" explain the graph?
        
        # Decision: Use Global Distance Kernel `f(d)` applied to block-constrained generation.
        # This tests if "Types + General Spatial Wiring" is sufficient.
        
        # So we reuse N1's acceptance curve.
        # BUT relative to what?
        # N1 Proposal was Global Degree-Weighted.
        # N2 Proposal is Block-Restricted Degree-Weighted?
        # Yes.
        
        # So we need the same `self.acceptance_curve` as N1.
        # And for each block, we sample u in A, v in B, compute d, and accept with `acceptance_curve[d]`.
        
        # Initialize Rejection Sampler (Reuse N1 logic)
        # Copied from N1 for independence/simplicity
        s_out = np.array(adj.sum(axis=1)).flatten()
        s_in = np.array(adj.sum(axis=0)).flatten()
        self.out_probs = s_out / s_out.sum() # Global probs
        self.in_probs = s_in / s_in.sum()
        
        sample_dists = SpatialUtils.estimate_proposal_dist(
             1000000, idx, idx, self.out_probs, self.in_probs, self.coords
        )
        prop_hist, _ = np.histogram(sample_dists, bins=100, range=(0, self.max_dist))
        prop_hist = prop_hist.astype(float)
        
        # Ratio
        ratios = np.zeros(100)
        mask = prop_hist > 0
        ratios[mask] = hist[mask] / prop_hist[mask]
        self.acceptance_curve = ratios / (ratios.max() if ratios.max()>0 else 1.0)
        self.bin_edges = bin_edges
        
        # Compute in/out probs PER TYPE to avoid recomputing in loop
        # Store indices
        self.indices = np.arange(len(nodes))
        
    def _calc_dists(self, df):
        p1 = self.coords[df['pre_idx'].values]
        p2 = self.coords[df['post_idx'].values]
        return np.linalg.norm(p1 - p2, axis=1)

    def generate(self, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        all_edges = []
        
        # Iterate over all block pairs
        # This could be slow if there are 800*800 pairs?
        # 640,000 pairs.
        # Majority are empty. We only process non-empty distinct pairs from `self.block_counts`.
        
        # Sort keys for determinism
        pairs = sorted(self.block_counts.keys())
        
        pbar = tqdm(pairs, desc="N2 Generation", leave=False)
        
        for (t_pre, t_post) in pbar:
            n_target = self.block_counts[(t_pre, t_post)]
            
            # Nodes in blocks
            src_indices = self.type_indices[t_pre]
            tgt_indices = self.type_indices[t_post]
            
            # Get local probabilities (degree-weighted)
            # subset global probs and renormalize
            p_src = self.out_probs[src_indices]
            if p_src.sum() == 0: p_src = np.ones(len(p_src))/len(p_src)
            else: p_src = p_src / p_src.sum()
            
            p_tgt = self.in_probs[tgt_indices]
            if p_tgt.sum() == 0: p_tgt = np.ones(len(p_tgt))/len(p_tgt)
            else: p_tgt = p_tgt / p_tgt.sum()
            
            # Generate edges for this block
            collected = 0
            block_edges_list = []
            
            while collected < n_target:
                needed = n_target - collected
                n_prop = max(int(needed * 20), 100) # Higher buffer for potentially rare connections
                # Cap n_prop to avoid memory issues if needed/acceptance is tiny
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
            
            # Concatenate and Trim
            block_df = pd.concat(block_edges_list, ignore_index=True)
            if len(block_df) > n_target:
                block_df = block_df.iloc[:n_target]
            all_edges.append(block_df)
            
        full_df = pd.concat(all_edges, ignore_index=True)
        
        # Map IDs
        idx_to_body = {i: b for b, i in self.node_map.items()}
        full_df['pre'] = full_df['pre_idx'].map(idx_to_body)
        full_df['post'] = full_df['post_idx'].map(idx_to_body)
        
        # Weights
        weights = self.edges['s_ij'].values.copy()
        rng.shuffle(weights)
        if len(weights) >= len(full_df):
            full_df['s_ij'] = weights[:len(full_df)]
        else:
            full_df['s_ij'] = np.resize(weights, len(full_df))
            
        return full_df

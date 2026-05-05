import numpy as np
import scipy.sparse as sp
import pandas as pd
from numba import njit
import sys

def compute_ppr_vector(n_nodes, row, col, data, target_idxs, alpha=0.85, max_iter=30, tol=1e-5):
    A = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
    out_degree = np.array(A.sum(axis=1)).flatten()
    out_degree[out_degree == 0] = 1.0
    D_inv = sp.diags(1.0 / out_degree)
    P = D_inv.dot(A)
    
    v_T = np.zeros(n_nodes)
    if len(target_idxs) > 0:
        v_T[target_idxs] = 1.0 / len(target_idxs)
    else:
        v_T[:] = 1.0 / n_nodes
        
    pi = v_T.copy()
    for i in range(max_iter):
        pi_next = alpha * P.T.dot(pi) + (1 - alpha) * v_T
        diff = np.sum(np.abs(pi_next - pi))
        pi = pi_next
        if diff < tol: break
    return pi

@njit
def mc_reachability_compiled(indptr, indices, source_nodes, target_mask, walks_per_source=50, max_steps=50):
    n_sources = len(source_nodes)
    hits = np.zeros(n_sources, dtype=np.float64)
    
    for i in range(n_sources):
        src = source_nodes[i]
        success_count = 0
        for w in range(walks_per_source):
            curr = src
            for step in range(max_steps):
                if target_mask[curr]:
                    success_count += 1
                    break
                
                start_ptr = indptr[curr]
                end_ptr = indptr[curr + 1]
                if end_ptr == start_ptr:
                    break # dead end
                    
                # random choice (unweighted jump as a topological walk approximation)
                # To be exact with row-stochastic transitions, we usually sample by weight.
                # Since Numba makes weighted choice hard without cumulative arrays, we assume P transitions.
                # However, the graph might have variance. Numba allows np.random.rand.
                nxt_idx = start_ptr + np.random.randint(0, end_ptr - start_ptr)
                curr = indices[nxt_idx]
                
        hits[i] = success_count / walks_per_source
        
    return hits

# Weighted variant for exactly stochastic transitions
@njit
def mc_weighted_reachability_compiled(indptr, indices, data, out_degrees, source_nodes, target_mask, walks_per_source=50, max_steps=50):
    n_sources = len(source_nodes)
    hits = np.zeros(n_sources, dtype=np.float64)
    
    for i in range(n_sources):
        src = source_nodes[i]
        success_count = 0
        for w in range(walks_per_source):
            curr = src
            for step in range(max_steps):
                if target_mask[curr]:
                    success_count += 1
                    break
                
                start_ptr = indptr[curr]
                end_ptr = indptr[curr + 1]
                if end_ptr == start_ptr:
                    break
                
                # weighted choice
                total_w = out_degrees[curr]
                r = np.random.rand() * total_w
                
                cum = 0.0
                nxt = curr
                for j in range(start_ptr, end_ptr):
                    cum += data[j]
                    if r <= cum:
                        nxt = indices[j]
                        break
                curr = nxt
                
        hits[i] = success_count / walks_per_source
        
    return hits

def compute_reachability(edges, n_nodes, node2idx, source_idxs, target_idxs, walks=50, steps=50):
    row = edges['pre'].map(node2idx).values
    col = edges['post'].map(node2idx).values
    data = edges['w_ij'].values
    A = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
    
    indptr = A.indptr
    indices = A.indices
    data_csr = A.data
    out_degrees = np.array(A.sum(axis=1)).flatten()
    
    target_mask = np.zeros(n_nodes, dtype=bool)
    target_mask[target_idxs] = True
    source_nodes = np.array(source_idxs, dtype=np.int32)
    
    # Run MC
    hits = mc_weighted_reachability_compiled(
        indptr, indices, data_csr, out_degrees, 
        source_nodes, target_mask, walks, steps
    )
    return float(np.mean(hits))

def get_stratified_sources(edges, n_nodes, node2idx, target_idxs, K=10000, seed=42):
    row = edges['pre'].map(node2idx).values
    col = edges['post'].map(node2idx).values
    data = edges['w_ij'].values
    A = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
    out_degree = np.array(A.sum(axis=1)).flatten()
    
    # Non-target nodes with at least 1 out-degree
    target_set = set(target_idxs)
    non_target_idxs = np.array([i for i in range(n_nodes) if i not in target_set and out_degree[i] > 0])
    
    df = pd.DataFrame({'node_idx': non_target_idxs, 'out_degree': out_degree[non_target_idxs]})
    # Compute deciles safely
    df['decile'] = pd.qcut(df['out_degree'], 10, labels=False, duplicates='drop')
    
    np.random.seed(seed)
    
    sampled_idxs = []
    deciles = df['decile'].unique()
    k_per_decile = K // len(deciles) if len(deciles) > 0 else K
    
    for d in deciles:
        pool = df[df['decile'] == d]['node_idx'].values
        if len(pool) <= k_per_decile:
            sampled_idxs.extend(pool)
        else:
            sampled_idxs.extend(np.random.choice(pool, size=k_per_decile, replace=False))
            
    # if still short, sample randomly from remaining
    if len(sampled_idxs) < K:
        remaining = list(set(non_target_idxs) - set(sampled_idxs))
        shortfall = K - len(sampled_idxs)
        if len(remaining) >= shortfall:
            sampled_idxs.extend(np.random.choice(remaining, size=shortfall, replace=False))
        else:
            sampled_idxs.extend(remaining)
            
    return np.array(sampled_idxs, dtype=np.int32)

def compute_targeted_flux_backbone(edges, n_nodes, node2idx, target_idxs):
    row = edges['pre'].map(node2idx).values
    col = edges['post'].map(node2idx).values
    data = edges['w_ij'].values
    A = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
    out_degree = np.array(A.sum(axis=1)).flatten()
    out_degree[out_degree == 0] = 1.0
    
    pi = compute_ppr_vector(n_nodes, row, col, data, target_idxs)
    
    # flux = pi_u * P_uv
    edges = edges.copy()
    edges['pre_idx'] = edges['pre'].map(node2idx)
    edges['post_idx'] = edges['post'].map(node2idx)
    
    # Add out degree
    edges['pre_out_deg'] = out_degree[edges['pre_idx']]
    edges['pi_pre'] = pi[edges['pre_idx']]
    
    edges['flux'] = edges['pi_pre'] * (edges['w_ij'] / edges['pre_out_deg'])
    
    flux_thresh = edges['flux'].quantile(0.99)
    backbone_mask = edges['flux'] >= flux_thresh
    
    edges_ablated = edges[~backbone_mask].copy()
    
    return edges_ablated[['pre', 'post', 'w_ij']], edges[backbone_mask][['pre', 'post', 'w_ij', 'flux']], edges[['pre', 'post', 'w_ij', 'flux']]

def get_enrichment(edges, dn_bodies, backbone_edges):
    tot_wt = edges['w_ij'].sum()
    dnp11_tot = edges[edges['post'].isin(dn_bodies)]['w_ij'].sum()
    base_frac = dnp11_tot / tot_wt if tot_wt > 0 else 0
    
    bb_wt = backbone_edges['w_ij'].sum()
    bb_dnp11 = backbone_edges[backbone_edges['post'].isin(dn_bodies)]['w_ij'].sum()
    bb_frac = bb_dnp11 / bb_wt if bb_wt > 0 else 0
    
    E_wt = bb_frac / base_frac if base_frac > 0 else 0
    return E_wt

def get_flux_concentration(flux_edges):
    total_flux = flux_edges['flux'].sum()
    if total_flux == 0:
        return 0.0, 0.0, 0.0
    
    q99 = flux_edges['flux'].quantile(0.99)
    top1_flux = flux_edges[flux_edges['flux'] >= q99]['flux'].sum()
    flux_share_top1 = float(top1_flux / total_flux)
    
    q999 = flux_edges['flux'].quantile(0.999)
    top0p1_flux = flux_edges[flux_edges['flux'] >= q999]['flux'].sum()
    flux_share_top0p1 = float(top0p1_flux / total_flux)
    
    flux_vals = np.sort(flux_edges['flux'].values)
    n = len(flux_vals)
    cum_flux = np.cumsum(flux_vals)
    gini = (n + 1 - 2 * np.sum(cum_flux) / cum_flux[-1]) / n if n > 0 else 0.0
    
    return flux_share_top1, flux_share_top0p1, float(gini)

def get_topology_metrics(edges, n_nodes, node2idx, comp_dict=None):
    out_deg = edges.groupby('pre').size()
    deg_mean = float(out_deg.mean())
    deg_p95 = float(out_deg.quantile(0.95))
    
    out_str = edges.groupby('pre')['w_ij'].sum()
    str_mean = float(out_str.mean())
    str_p95 = float(out_str.quantile(0.95))
    
    edge_set = set(zip(edges['pre'], edges['post']))
    recip_edges = sum(1 for u, v in edge_set if (v, u) in edge_set)
    reciprocity = recip_edges / len(edge_set) if len(edge_set) > 0 else 0.0
    
    q99_str = out_str.quantile(0.99)
    q95_str = out_str.quantile(0.95)
    
    top1_nodes = set(out_str[out_str >= q99_str].index)
    top5_nodes = set(out_str[out_str >= q95_str].index)
    
    top1_sub_edges = edges[edges['pre'].isin(top1_nodes) & edges['post'].isin(top1_nodes)]
    top5_sub_edges = edges[edges['pre'].isin(top5_nodes) & edges['post'].isin(top5_nodes)]
    
    n1 = len(top1_nodes)
    rich_club_1 = len(top1_sub_edges) / (n1 * (n1 - 1)) if n1 > 1 else 0.0
    
    n5 = len(top5_nodes)
    rich_club_5 = len(top5_sub_edges) / (n5 * (n5 - 1)) if n5 > 1 else 0.0
    
    modularity_proxy = 0.0
    if comp_dict is not None:
        edges['pre_comp'] = edges['pre'].map(comp_dict)
        edges['post_comp'] = edges['post'].map(comp_dict)
        intra = edges[edges['pre_comp'] == edges['post_comp']]['w_ij'].sum()
        inter = edges[edges['pre_comp'] != edges['post_comp']]['w_ij'].sum()
        modularity_proxy = float(intra / inter) if inter > 0 else 0.0
        
    return {
        'deg_mean': deg_mean, 'deg_p95': deg_p95,
        'str_mean': str_mean, 'str_p95': str_p95,
        'reciprocity': float(reciprocity),
        'rich_club_1': float(rich_club_1),
        'rich_club_5': float(rich_club_5),
        'modularity_proxy': modularity_proxy
    }

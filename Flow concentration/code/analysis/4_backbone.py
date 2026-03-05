import numpy as np
import scipy.sparse as sp
import logging
from utils.matrix_utils import normalize_transition_matrix, solve_absorbing_fpt

logger = logging.getLogger(__name__)

def compute_reachability(adj: sp.csr_matrix, targets_indices: np.ndarray) -> np.ndarray:
    
    rev_adj = adj.transpose()
    
    n_nodes = adj.shape[0]
    visited = np.zeros(n_nodes, dtype=bool)
    queue = list(targets_indices)
    visited[targets_indices] = True
    
    
    dist_matrix = sp.csgraph.shortest_path(rev_adj, directed=True, indices=targets_indices)
    min_dist = np.min(dist_matrix, axis=0)
    reachable_mask = ~np.isinf(min_dist)
    
    return reachable_mask

def compute_global_latency_metrics(C: sp.csr_matrix, targets_indices: np.ndarray, penalty: float = 1e6):
    
    n_nodes = C.shape[0]
    
    reachable_mask = compute_reachability(C, targets_indices)
    reachable_count = np.sum(reachable_mask)
    reachability_fraction = reachable_count / n_nodes
    

    n_components, labels = sp.csgraph.connected_components(C, directed=True, connection='strong')
    _, counts = np.unique(labels, return_counts=True)
    largest_label = np.argmax(counts)
    scc_mask = (labels == largest_label)
    scc_indices = np.where(scc_mask)[0]
    scc_fraction = len(scc_indices) / n_nodes
    
    
    targets_in_scc = [t for t in targets_indices if scc_mask[t]]
    
    metric_a = np.nan
    if len(targets_in_scc) > 0:
        P = normalize_transition_matrix(C)
        P_scc = P[scc_indices, :][:, scc_indices]
        
        orig_to_local = {orig: local for local, orig in enumerate(scc_indices)}
        local_targets = [orig_to_local[t] for t in targets_in_scc]
        
        fpt_values = []
        for t in local_targets:
            fpt_vec = solve_absorbing_fpt(P_scc, np.array([t]))
            valid = fpt_vec[fpt_vec < 1e10]
            if len(valid) > 0:
                fpt_values.extend(valid)
        metric_a = np.mean(fpt_values) if fpt_values else np.nan
        
    
    
    metric_b = penalty 
    
    if reachable_count > 0:
        P = normalize_transition_matrix(C)
        P_sub = P[reachable_mask, :][:, reachable_mask]
        
        
        sub_indices = np.where(reachable_mask)[0]
        orig_to_sub = {orig: sub for sub, orig in enumerate(sub_indices)}
        sub_targets = [orig_to_sub[t] for t in targets_indices if t in orig_to_sub]
        
        
        target_sub_indices = set(sub_targets)
        non_target_mask = np.array([i not in target_sub_indices for i in range(len(sub_indices))])
        
        if np.sum(non_target_mask) > 0:
            Q = P_sub[non_target_mask, :][:, non_target_mask]
            
           
            n_vals = Q.shape[0]
            I = sp.eye(n_vals, format='csc')
            A = I - Q
            b = np.ones(n_vals)
            
            try:
                x, info = sp.linalg.bicgstab(A, b, rtol=1e-2, maxiter=200)
                
                if info != 0:
                    if info > 0:
                        logger.warning(f"bicgstab reached max iter ({info}). Using approximation.")
                    else:
                        logger.warning("bicgstab failed with breakdown. Approximating with penalty.")
                        
                        if np.any(np.isnan(x)):
                             x = np.full(n_vals, penalty)


                
                sum_fpt = np.sum(x) 
                mean_fpt_reachable = sum_fpt / reachable_count
                
                
                metric_b = (sum_fpt + (n_nodes - reachable_count) * penalty) / n_nodes
                
            except Exception as e:
                logger.warning(f"Linear solver failed for Metric B: {e}")
                metric_b = penalty
        else:
             metric_b = (n_nodes - reachable_count) * penalty / n_nodes 
             
    
    metric_c = np.nan
    
    return {
        "metric_a": metric_a,
        "metric_b": metric_b,
        "scc_fraction": scc_fraction,
        "reachability": reachability_fraction
    }

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def compute_pareto_frontier(points: np.ndarray) -> np.ndarray:
    """
    Identifies the Pareto frontier (non-dominated set) for minimization.
    Args:
        points: (N, 2) array of (energy, latency)
    Returns:
        mask: boolean array of length N, True if point is on frontier.
    """
    n_points = points.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if not is_pareto[i]:
            continue
        # Check if point i is dominated by any other point j
        # dominated if other.energy <= i.energy AND other.latency <= i.latency
        # AND (other.energy < i.energy OR other.latency < i.latency)
        
        # Vectorized check
        # We want to find if there exists any j such that j dominates i
        # j dominates i if:
        # all(p[j] <= p[i]) and any(p[j] < p[i])
        
        # Let's use a simpler approach for 2D:
        # Sort by one object (say energy). Then iterate and keep track of min latency seen so far.
        pass

    # Efficient 2D Pareto Sort
    # 1. Sort by objective 1 (energy) ascending.
    # 2. Iterate. A point is on frontier if its objective 2 (latency) is smaller than all previous points' objective 2.
    #    Actually for minimization:
    #    If we sort by Energy ascending:
    #       First point is definitely on frontier (lowest energy).
    #       For subsequent points, they have higher energy. They can only be on frontier if they have lower latency than
    #       ALL previously selected frontier points. But since we sorted by Energy, we just need to compare with the
    #       Latency of the *most recently added* frontier point (which has the best latency so far among processed)?
    #       Wait.
    #       Standard algo:
    #       Sort by X.
    #       Current Y_min = infinity.
    #       For point in sorted points:
    #           If point.Y < Y_min:
    #               Add to frontier
    #               Y_min = point.Y
    
    # Needs argsort to return mask in original order
    
    sort_idx = np.argsort(points[:, 0])
    sorted_points = points[sort_idx]
    
    # We need to handle duplicates carefully or strict strict inequality?
    # Pareto domination usually: x <= y in all, x < y in at least one.
    
    y_min = np.inf
    pareto_indices_sorted = []
    
    for k in range(n_points):
        # We process in increasing order of Energy.
        # If multiple points have same Energy, the one with lowest Latency comes first (stable sort or secondary sort).
        # We should probably do lexicographical sort.
        pass
        
    # Re-doing with lexical sort
    # sort by Energy (primary), Latency (secondary)
    # lexsort sorts by last key first, so:
    sort_idx = np.lexsort((points[:, 1], points[:, 0])) 
    sorted_points = points[sort_idx]
    
    is_pareto_sorted = np.zeros(n_points, dtype=bool)
    
    min_latency = np.inf
    
    for k in range(n_points):
        curr_latency = sorted_points[k, 1]
        
        # If strictly better latency than anything seen so far (which had <= energy)
        # Then it is non-dominated.
        if curr_latency < min_latency:
            is_pareto_sorted[k] = True
            min_latency = curr_latency
            
    # Map back to original indices
    original_mask = np.zeros(n_points, dtype=bool)
    original_mask[sort_idx] = is_pareto_sorted
    
    return original_mask
    

def compute_distance_to_frontier_z_score(real_point: np.ndarray, 
                                         null_points: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes:
    1. Distance of real point to the null frontier.
    2. Distances of all null points to the null frontier (excluding themselves - LOO?).
       Actually, standard practice is:
       d_real = dist(real, frontier(nulls))
       d_nulls = [dist(n, frontier(nulls_excluding_n)) for n in nulls] OR
       approx: d_nulls = [dist(n, frontier(nulls)) for n in nulls].
       
       Using "distance to THEIR OWN frontier" (as per prompt) suggests:
       "d_null is distance-to-frontier of null samples to their own frontier (leave-one-out or bootstrap)"
       LOO is computationally expensive if N is large (N=50 is fine).
       
    Args:
        real_point: (2,) array
        null_points: (K, 2) array
        
    Returns:
        z_score: (d_real - mean(d_nulls)) / std(d_nulls)
        d_real: absolute distance
        dominance_flag: True if d_real < 0 (i.e. beyond frontier? No, distance is usually positive).
        Wait.
        "real beats null?"
        Usually:
        - Frontier is the "best" obtainable.
        - Distance should be signed?
        - Or is real point *beyond* the null frontier (better)?
        
        If real point is dominated by null frontier -> Distance > 0 (worse)
        If real point dominates null frontier -> Distance < 0 (better) ??
        
        Definition of distance:
        "normalized Euclidean distance in z-scored coordinates"
        
        Let's standardize variables first using Null Ensemble Mean/Std.
        x' = (x - mu_x) / sigma_x
        y' = (y - mu_y) / sigma_y
        
        Then compute Euclidean distance to proper frontier segment.
        
        If real point is "better" than frontier (i.e. closer to ideal origin (0,0) than frontier), 
        we should probably treat it as "super-optimal".
        
        Prompt says:
        "distance of REAL point to that frontier"
        "d_null is distance-to-frontier of null samples to their own frontier"
    
    """
    # 1. Standardization stats
    mus = np.mean(null_points, axis=0)
    sigmas = np.std(null_points, axis=0) + 1e-9
    
    # 2. Transform
    null_z = (null_points - mus) / sigmas
    real_z = (real_point - mus) / sigmas
    
    # 3. Compute Null Interval Frontier (LOO)
    # For each null point i, compute frontier of nulls[-i], then dist(i, frontier[-i])
    
    d_nulls = []
    n = len(null_z)
    
    for i in range(n):
        subset = np.delete(null_z, i, axis=0)
        # Frontier of subset
        mask = compute_pareto_frontier(subset)
        frontier = subset[mask]
        
        # Dist i to frontier
        d = min_euclidean_dist(null_z[i], frontier)
        
        # Sign check: Is i "better" than frontier?
        # If i is dominated by frontier -> Positive distance
        # If i dominates frontier -> Negative distance?
        # Or just distance. Usually nulls are "at" the frontier or "behind" it.
        # But we want to test if real is *significantly better*.
        # Let's assume positive distance = worse (dominated), negative = better (dominating).
        
        # Check domination:
        # If i dominates ANY point on frontier? No, that's not right.
        # If i is non-dominated by ALL points in frontier, and strictly better than some?
        
        # Simple signed distance:
        # Vector to closest frontier point.
        # Dot product with normal? Hard for discrete frontier.
        
        # Alternative:
        # Just use raw distance. If Real is *better* than hull, it will have distance?
        # Actually, if Real is better than nulls, it will act as a new frontier point.
        # The prompt asks for "distance to that frontier".
        
        # Let's use unsigned distance for now, BUT we need to know if it's "better" or "worse".
        # If Real is dominated by Null Frontier -> Worse.
        # If Real dominates Null Frontier (or part of it) -> Better.
        
        # Let's implement function `signed_distance_to_frontier(point, frontier)`
        # Positive if point is dominated by frontier.
        # Negative if point dominates frontier (is better).
        d_signed = signed_dist(null_z[i], frontier)
        d_nulls.append(d_signed)
        
    d_nulls = np.array(d_nulls)
    
    # 4. Real Distance
    # Frontier of ALL nulls
    full_mask = compute_pareto_frontier(null_z)
    full_frontier = null_z[full_mask]
    d_real = signed_dist(real_z, full_frontier)
    
    # 5. Z-score
    # z = (d_real - mean(d_nulls)) / std(d_nulls)
    # If d_real is very negative (super optimal) and d_nulls are around 0, z will be negative.
    
    z_score = (d_real - np.mean(d_nulls)) / (np.std(d_nulls) + 1e-9)
    
    return z_score, d_real, np.mean(d_nulls), np.std(d_nulls)


def min_euclidean_dist(point, frontier_points):
    """Euclidean distance to closest point in frontier set."""
    dists = np.linalg.norm(frontier_points - point, axis=1)
    return np.min(dists)

def signed_dist(point, frontier_points):
    """
    Returns signed distance. 
    +dist if point is dominated by frontier (worse).
    -dist if point dominates frontier (better).
    0 if on frontier (approximately).
    
    Ambigous case: Neither dominated nor dominating (in the gaps).
    Treat as +dist (closer to "worse" region typically?).
    
     actually:
    If point is better (lower E, lower L) than closest frontier point -> negative.
    If point is worse -> positive.
    """
    # Find closest frontier point
    dists = np.linalg.norm(frontier_points - point, axis=1)
    idx = np.argmin(dists)
    closest = frontier_points[idx]
    dist = dists[idx]
    
    # Check dominance relationship with closest point
    # vector = point - closest
    # If both components positive -> point is worse (dominated) -> +dist
    # If both components negative -> point is better (dominating) -> -dist
    # Mixed?
    
    diff = point - closest
    if np.all(diff >= -1e-9): # dominated or equal
        return dist
    elif np.all(diff <= 1e-9): # dominating or equal
        return -dist
    else:
        # Mixed case. It's "incomparable" locally.
        # Usually implies it's in a concave region or gap.
        # Let's fallback to standard geometry:
        # If it's "above" the piecewise linear interpolation of frontier?
        # Too complex.
        # Fallback: Is it dominated by *any* frontier point?
        
        is_dominated = np.any(np.all(frontier_points <= point + 1e-9, axis=1))
        if is_dominated:
            return dist
            
        # Is it dominating *any* frontier point? (Unlikely if we picked closest, but possible)
        is_dominating = np.any(np.all(point <= frontier_points - 1e-9, axis=1))
        if is_dominating:
            return -dist
            
        # If neither, it's incomparable. Return metric distance but sign?
        # Treat as positive (not clearly better).
        return dist

def compute_scaling_stats(points: np.ndarray, method: str = 'z_score') -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns location (loc) and scale (scale) parameters.
    transformed = (x - loc) / scale
    """
    if method == 'z_score':
        loc = np.mean(points, axis=0)
        scale = np.std(points, axis=0) + 1e-9
    elif method == 'iqr':
        q75, q25 = np.percentile(points, [75 ,25], axis=0)
        iqr = q75 - q25
        scale = iqr
        scale[scale == 0] = 1.0 # Avoid div by zero
        loc = np.median(points, axis=0)
    else:
        raise ValueError(f"Unknown scaling method: {method}")
        
    return loc, scale

def compute_distance_robustness_check(real_point: np.ndarray, 
                                      null_points: np.ndarray) -> Dict[str, float]:
    """
    Computes Z-scores using multiple scaling methods to check robustness.
    Returns dictionary with z-scores for 'z_score' and 'iqr' methods.
    """
    results = {}
    
    for method in ['z_score', 'iqr']:
        loc, scale = compute_scaling_stats(null_points, method)
        
        # Transform
        null_scaled = (null_points - loc) / scale
        real_scaled = (real_point - loc) / scale
        
        # Compute Null distances (LOO)
        d_nulls = []
        n = len(null_scaled)
        
        for i in range(n):
            subset = np.delete(null_scaled, i, axis=0)
            mask = compute_pareto_frontier(subset)
            frontier = subset[mask]
            d = signed_dist(null_scaled[i], frontier)
            d_nulls.append(d)
            
        d_nulls = np.array(d_nulls)
        
        # Real Distance
        mask = compute_pareto_frontier(null_scaled)
        frontier = null_scaled[mask]
        d_real = signed_dist(real_scaled, frontier)
        
        # Z-score
        z = (d_real - np.mean(d_nulls)) / (np.std(d_nulls) + 1e-9)
        
        results[method] = z
        
    return results

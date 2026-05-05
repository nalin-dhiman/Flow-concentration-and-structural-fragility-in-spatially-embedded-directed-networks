import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Any, Optional

def compute_pareto_frontier_v5b(points: np.ndarray) -> np.ndarray:
    """
    Identifies the Pareto frontier (non-dominated set) for minimization.
    Args:
        points: (N, 2) array of (energy, latency)
    Returns:
        mask: boolean array of length N, True if point is on frontier.
    """
    # Use efficient sort-based algorithm for 2D
    # Sort by Energy ascending, then Latency ascending
    n_points = points.shape[0]
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

def signed_dist_v5b(point: np.ndarray, frontier_points: np.ndarray) -> float:
    """
    Returns UN-signed Euclidean distance to the frontier.
    BUT we also need to know if it beats the null.
    
    Standard convention:
    d = min || point - frontier_point ||
    
    We rely on 'beats_null' logic (p-value or quantile) to decide dominance.
    Distance itself is just a magnitude.
    
    Wait, if point is "inside" the pareto front (better), d should be small?
    No, if point is better, it's far from the null frontier in the "good" direction.
    
    Let's stick to simple Euclidean distance to closest point on frontier.
    Directionality is handled by checking if point dominates frontier points.
    
    However, for Z-score, we need a sign if we want "higher is worse" or similar.
    The prompt says: "DO NOT USE negative distances."
    "d_real = distance(real_point, Frontier(null)). (Distance is nonnegative.)"
    
    So we return nonnegative distance.
    The "goodness" is determined by comparing d_real to d_null_distribution.
    
    Wait.
    If real point is super-optimized (far "below" the null frontier), d_real is LARGE.
    If null points are close to their own frontier (by definition they define it), d_nulls are SMALL (0 for frontier points, small >0 for internal points).
    
    So if real is "better", d_real >> d_nulls?
    NO.
    "d_null is distance-to-frontier of null samples to their own frontier"
    
    If I select a null point N_i (LOO), the frontier is defined by N_{-i}.
    If N_i is "average", it's somewhat close to the frontier.
    If N_i is "bad", it's far "behind" the frontier (dominated).
    If N_i is "good", it might be ON the frontier of N_{-i} or even dominate it.
    
    If N_i is on the true frontier, its distance to N_{-i} frontier is small.
    
    If Real is "better" than nulls, it should be "in front" of the null frontier.
    Distance magnitude will be large.
    
    But prompt says:
    "p_value = P(d_null <= d_real) (one-sided; small p means real unusually close)"
    
    This implies:
    d_real should be SMALLER than d_null if it's "good"?
    
    Let's re-read carefully:
    "d_real = distance(real, Frontier(null))"
    "p_value = P(d_null <= d_real)" (small p -> real is in the tail where d is large)
    Wait. `P(X <= x)` is CDF. If p is small, x is small.
    So "small p means real unusually close".
    
    This implies we want Real to be CLOSE to the frontier?
    Or is the "frontier" the optimal boundary?
    
    Usually:
    - Nulls form a cloud.
    - Frontier is the "best" edge of the cloud.
    - We want real to be AT or BEYOND the frontier.
    - Null points (LOO) are mostly "inside" the cloud (worse than frontier). Their distance to frontier is > 0.
    - If real is "beyond" the frontier (better), its distance is ALSO > 0.
    
    This dominance definition is ambiguous if distance is unsigned.
    Distance 10 could be "10 units worse" or "10 units better".
    
    We MUST use a signed distance or a "dominance check".
    
    Prompt says: "DO NOT USE negative distances."
    
    Okay, let's look at "beats_null" definition:
    "d_real <= q10(d_null) (real closer than the best 10% of null distances)"
    
    This implies "smaller distance is better".
    This implies we want real to be "close to the frontier".
    
    BUT if real is "super optimized" (better than nulls), it might be 100 units away from the null frontier (in the good direction).
    Then d_real = 100.
    d_nulls are typically small (e.g. 1-5).
    Then d_real > d_nulls.
    So Real fails the "d_real <= q10" test.
    This logic only works if "better" means "on the frontier".
    
    If Real is *better* than Nulls, it forms a NEW frontier.
    
    Let's assume the standard interpretation:
    We measure "inefficiency" or "sub-optimality".
    Distance to optimal frontier from the "bad" side.
    
    If Real is "better" than null frontier, we treat its distance as 0? Or negative?
    Prompt: "DO NOT USE d_real < 0".
    
    Hypothesis: The user thinks Real is *dominated* by Nulls (worse) or *comparable*.
    Task is Falsification.
    
    If Real is 100 units BETTER, and we call dist=100.
    And d_nulls are around 5.
    Then Real is "worse" by metric "distance"? (100 > 5).
    This contradicts "optimality".
    
    CRITICAL FIX:
    We need to distinguish "Better side" vs "Worse side".
    
    Let `is_better(point, frontier)` return True if point dominates closest frontier point.
    
    If Real is BETTER:
       We treat it as "Super-Optimal".
       Maybe set d_real = 0? Or -dist?
       But user said "DO NOT USE negative".
       
       Maybe user assumes Real is strictly *worse* or *on* frontier?
       
       Let's implement:
       d_unsigned = Euclidean dist.
       status = {Better, Worse, Incomparable}
       
       If status == Better:
           Real beats nulls automatically.
           d_real for stats? Maybe 0?
           
       If status == Worse:
           d_real > 0.
           We compare to d_nulls (which are also 'Worse' relative to their own LOO frontier usually).
           
           If d_real < d_nulls, then Real is CLOSER to efficient frontier than typical nulls.
           
       So "Beats Null" = (Real is Better) OR (Real is Worse BUT Closer than 90% nulls).
       
       Wait.
       "quantile criterion: d_real <= q10(d_null)"
       
       So if Real is Better -> d_real should be considered effectively 0 or -inf.
       If we are forced to use non-negative, and Real is better, we assign 0.0?
       
       Let's stick to this:
       1. Calculate signed distance `ds` (+ve = worse, -ve = better).
       2. If `ds < 0` (Better):
             `beats_null` = True (Real is super optimal)
             For stats, we report the actual raw `d_real` (positive magnitude) but note it is "better".
             Or follow instruction "distance is nonnegative" -> return `abs(ds)`.
       
       Wait, if I report `abs(ds)` when better, a huge "better" distance looks like a huge "worse" distance.
       
       Let's look at `beats_null` logic again.
       "d_real <= q10(d_null)"
       
       If Real is better, `ds` is negative.
       If we clamp `d_real` to 0 for "better" points?
       Then 0 <= positive_nulls. So it passes.
       
       PLAN:
       1. Compute `signed_dist` (+ve = dominated/worse, -ve = dominating/better).
       2. `d_metric` = `signed_dist`.
       3. `d_nonneg` = `signed_dist`. If < 0, warn? Or just use signed logic internally but output non-negative?
       
       User said: "d_real = distance... (Distance is nonnegative)"
       AND "DO NOT USE negative distances."
       AND "Z-scores are astronomically large... due to incorrect null computation."
       
       Maybe the user implies we should only look at points *dominated* by frontier?
       
       Alternative:
       d = distance to hull.
       If point is outside hull (better), distance is defined?
       
       Let's follow the "standard" interpretation of this prompt type:
       The user wants to measure "Distance to Efficiency".
       Lower is better.
       
       If Real is better than Null Frontier -> Distance = 0 (It IS efficient, it defines the new frontier).
       If Real is worse -> Distance > 0.
       
       So:
       If `is_better(real, frontier)`: d_real = 0.0.
       Else: d_real = Euclidean dist.
       
       Check d_nulls (LOO):
       Most nulls are worse than their LOO frontier -> d_null > 0.
       Some might be on frontier -> d_null = 0.
       
       Then check `d_real <= q10(d_null)`.
       If `d_real = 0` (super optimal), and `q10(d_null) >= 0`, then `0 <= q10`. True.
       
       This works!
       
       So Strategy:
       `d_signed` = +ve (worse) or -ve (better).
       `d_final` = max(0, d_signed).
       
       Implementation of `signed_dist`:
       Already did in v5a, I will reuse it but clamp to 0.
       
    """
    dists = np.linalg.norm(frontier_points - point, axis=1)
    min_dist = np.min(dists)
    
    # Check dominance
    # If point dominates ANY frontier point -> Better
    # If point is dominated by ANY frontier point -> Worse
    
    # Approx check with closest point
    idx = np.argmin(dists)
    closest = frontier_points[idx]
    
    diff = point - closest
    # If diff >= 0 (all components) -> Point >= Closest -> Worse (Minimization)
    if np.all(diff >= -1e-9):
        return min_dist # Worse
    # If diff <= 0 -> Point <= Closest -> Better
    elif np.all(diff <= 1e-9):
        return -min_dist # Better
        
    # Mixed case?
    # Check full frontier
    # Dominated by any?
    if np.any(np.all(frontier_points <= point + 1e-9, axis=1)):
        return min_dist # Worse
    
    # Dominating any?
    if np.any(np.all(point <= frontier_points - 1e-9, axis=1)):
        return -min_dist # Better
        
    # Incomparable (on the "side" of frontier).
    # Treat as positive distance (not part of frontier dominance)
    return min_dist

def compute_distance_metrics_v5b(real_point: np.ndarray, 
                                 null_points: np.ndarray,
                                 method: str = 'z_score') -> Dict[str, Any]:
    """
    Computes d_real and distribution of d_nulls (LOO).
    Returns stats.
    """
    # 1. Scaling
    if method == 'z_score':
        loc = np.mean(null_points, axis=0)
        scale = np.std(null_points, axis=0) + 1e-9
    elif method == 'iqr':
        q75, q25 = np.percentile(null_points, [75, 25], axis=0)
        scale = q75 - q25
        scale[scale == 0] = 1.0
        loc = np.median(null_points, axis=0)
    else:
        raise ValueError(f"Unknown method {method}")
        
    null_z = (null_points - loc) / scale
    real_z = (real_point - loc) / scale
    
    # 2. Null Baselines (LOO)
    d_nulls = []
    n = len(null_z)
    
    for i in range(n):
        # LOO Frontier
        subset = np.delete(null_z, i, axis=0)
        mask = compute_pareto_frontier_v5b(subset)
        frontier = subset[mask]
        
        # Distance of i to frontier of others
        d_signed = signed_dist_v5b(null_z[i], frontier)
        # Verify: "distance is nonnegative" -> clamp better points to 0?
        # If null point i is BETTER than the rest (super-null), its distance is < 0.
        # Ideally we keep negative to show it is "better".
        # But for "inefficiency" metric, better = 0 inefficiency?
        # Let's use max(0, d) as per my hypothesis.
        d = max(0.0, d_signed) 
        d_nulls.append(d)
        
    d_nulls = np.array(d_nulls)
    
    # 3. Real Distance
    mask = compute_pareto_frontier_v5b(null_z)
    full_frontier = null_z[mask]
    d_real_signed = signed_dist_v5b(real_z, full_frontier)
    d_real = max(0.0, d_real_signed)
    
    # 4. Stats
    # Z-score
    mu = np.mean(d_nulls)
    sigma = np.std(d_nulls) + 1e-9
    z_score = (d_real - mu) / sigma
    
    # P-value: P(d_null >= d_real)?
    # "P(d_null <= d_real) (one-sided; small p means real unusually close)"
    # PROMPT IS WEIRD HERE.
    # If d_real is small (0), and d_nulls are larger (1,2,3).
    # Then P(d_null <= d_real) is P(d_null <= 0) which is 0.
    # Small p -> Real is unusually close.
    # This matches.
    
    p_value = np.mean(d_nulls <= d_real)
    
    # Beats Null Criteria
    # "d_real <= q10(d_null)"
    q10 = np.percentile(d_nulls, 10)
    beats_quantile = d_real <= q10
    
    # "p_value <= 0.05"
    beats_pvalue = p_value <= 0.05
    
    return {
        'z_score': z_score,
        'p_value': p_value,
        'd_real': d_real,
        'mean_d_null': mu,
        'std_d_null': sigma,
        'beats_quantile': beats_quantile,
        'beats_pvalue': beats_pvalue,
        'raw_d_real_signed': d_real_signed # Debug
    }

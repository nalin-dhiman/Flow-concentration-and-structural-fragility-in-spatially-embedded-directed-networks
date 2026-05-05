import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_matched_edges(edges: pd.DataFrame, edges_to_remove: pd.DataFrame, match_on: list = ['d_ij'], bins: int = 10) -> pd.DataFrame:
    """
    Selects random edges from 'edges' that match the distribution of 'edges_to_remove'
    for the specified columns (e.g. weight, distance).
    MATCHES HISTOGRAMS.
    """
    # Create bins for matching columns
    # We combine columns into a single bin ID
    
    temp_edges = edges.copy()
    temp_remove = edges_to_remove.copy()
    
    bin_labels = []
    
    for col in match_on:
        # Define global bins based on full edges range
        # Logspace might be better for weight/dist? 
        # Let's use qcut on full data for robust quantiles
        try:
             # Use rank/qcut
             # We need consistent bins.
             # Compute breaks on full set
             ret, breaks = pd.qcut(temp_edges[col], bins, retbins=True, duplicates='drop')
             
             # Apply to remove set
             temp_remove[f'{col}_bin'] = pd.cut(temp_remove[col], bins=breaks, include_lowest=True).astype(str)
             temp_edges[f'{col}_bin'] = pd.cut(temp_edges[col], bins=breaks, include_lowest=True).astype(str)
             
             bin_labels.append(f'{col}_bin')
        except Exception as e:
             logger.warning(f"Binning failed for {col}: {e}")
             
    # Combine bins
    if not bin_labels:
        # Fallback: Random
        logger.warning("No matching columns valid. Returning random.")
        return edges.sample(n=len(edges_to_remove))
        
    # Group and Sample
    # We want to select same COUNT from each composite bin
    
    # Create composite key
    temp_remove['key'] = temp_remove[bin_labels].agg('-'.join, axis=1)
    temp_edges['key'] = temp_edges[bin_labels].agg('-'.join, axis=1)
    
    # Count targets
    target_counts = temp_remove['key'].value_counts()
    
    sampled_indices = []
    
    grouped = temp_edges.groupby('key')
    
    for key, count in target_counts.items():
        if count == 0: continue
        
        try:
            group = grouped.get_group(key)
            # Sample without replacement
            # Need to exclude edges that ARE in edges_to_remove (conceptually)
            # But edges_to_remove is subset of edges.
            # We want to produce an ALTERNATIVE set.
            # So can we pick the same edges?
            # 'Matched Control' usually means 'Alternative' set.
            # Ideally we exclude the *actual* Top Efficient edges from the pool if possible?
            # Or just sample from the pool. Overlap is allowed but unlikely if pool is large.
            # User requirement: "Remove f random edges matched..."
            # Let's sample from full edges.
            
            n_sample = min(len(group), count)
            sample = group.sample(n=n_sample)
            sampled_indices.extend(sample.index.tolist())
            
        except KeyError:
            # key not found in main edges? Impossible if remove is subset.
            continue
            
    # If we are short (due to binning discretization issues?), fill random
    if len(sampled_indices) < len(edges_to_remove):
        shortfall = len(edges_to_remove) - len(sampled_indices)
        remaining = temp_edges.index.difference(sampled_indices)
        if len(remaining) >= shortfall:
             fill = np.random.choice(remaining, shortfall, replace=False)
             sampled_indices.extend(fill)
             
    return edges.loc[sampled_indices]

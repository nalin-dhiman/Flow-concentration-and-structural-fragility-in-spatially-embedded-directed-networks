import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_fixed_targets(nodes: pd.DataFrame, n_targets: int = 50, seed: int = 42, output_path: Path = None):
    """
    Selects n_targets nodes stratified by total degree (in + out).
    Ensures targets are fixed and reproducible.
    """
    np.random.seed(seed)
    
    # Calculate degree if not present
    # This might require edges, but assuming 'nodes' has some degree info or we compute it?
    # run_attribution_v6a loads nodes and edges.
    # But this function only takes nodes.
    # Check if nodes has degree columns.
    
    # If not, allow passing degree series or compute random if strictly necessary.
    # Better: require degree info.
    # For now, let's assume we can compute it outside or it's in nodes.
    # If not, simple random sampling with seed is better than failing.
    
    # Let's try to assume nodes has 'bodyId'.
    
    targets = []
    
    # Stratification logic
    # If we have degree info, use it.
    if 'degree' in nodes.columns:
        scoring_col = 'degree'
    elif 'in_degree' in nodes.columns:
        scoring_col = 'in_degree'
    else:
        logger.warning("No degree column found in nodes. using random sampling.")
        scoring_col = None
        
    if scoring_col:
        # Create deciles
        try:
             nodes['decile'] = pd.qcut(nodes[scoring_col].rank(method='first'), 10, labels=False)
             per_decile = n_targets // 10
             for d in range(10):
                 cands = nodes[nodes['decile'] == d]['bodyId'].values
                 if len(cands) >= per_decile:
                     targets.extend(np.random.choice(cands, per_decile, replace=False))
                 else:
                     targets.extend(cands)
        except Exception as e:
            logger.warning(f"Stratification failed: {e}. Fallback to random.")
            scoring_col = None
            
    if not scoring_col:
        all_ids = nodes['bodyId'].values
        targets = np.random.choice(all_ids, n_targets, replace=False).tolist()
        
    logger.info(f"Generated {len(targets)} fixed targets.")
    
    # Save if path provided
    if output_path:
        pd.DataFrame({'bodyId': targets}).to_parquet(output_path)
        
    return targets

import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_null_metrics(base_path: str, null_model: str) -> pd.DataFrame:
    """Loads metrics for a specific null model."""
    path = Path(base_path) / f"{null_model}_metrics.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Null metrics not found at {path}")
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows for {null_model} from {path}")
    
    # Pivot to get energy and latency as columns
    # content of 'metric' column: 'energy', 'latency'
    # value column: 'val'
    
    # Check if we have duplicates for the index
    # We expect one value per (sample_idx, seed, eta, gamma, metric)
    
    pivot_df = df.pivot_table(index=['null_model', 'sample_idx', 'seed', 'eta', 'gamma'], 
                              columns='metric', values='val').reset_index()
    
    # Check if 'energy' and 'latency' columns exist
    if 'energy' not in pivot_df.columns or 'latency' not in pivot_df.columns:
        raise ValueError(f"Pivot failed to produce 'energy' and 'latency' columns. Columns found: {pivot_df.columns}")
        
    return pivot_df

def load_real_metrics(path: str) -> pd.DataFrame:
    """Loads real graph metrics from sweep summary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Real metrics not found at {path}")
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows for real metrics from {path}")
    
    # Pivot or filter to get energy and latency per (eta, gamma)
    # The sweep summary has 'metric', 'eta', 'gamma', 'value', 'mean'
    # Energy is 'value', Latency is 'mean' (as per plan)
    
    energy_df = df[df['metric'] == 'energy'][['eta', 'gamma', 'value']].rename(columns={'value': 'energy'})
    latency_df = df[df['metric'] == 'latency'][['eta', 'gamma', 'mean']].rename(columns={'mean': 'latency'})
    
    # Round to avoid float issues
    energy_df['eta'] = energy_df['eta'].round(5)
    energy_df['gamma'] = energy_df['gamma'].round(5)
    latency_df['eta'] = latency_df['eta'].round(5)
    latency_df['gamma'] = latency_df['gamma'].round(5)
    
    real_df = pd.merge(energy_df, latency_df, on=['eta', 'gamma'], how='inner')
    
    if len(real_df) == 0:
        logger.warning("Merged real metrics dataframe is empty!")
        
    return real_df

def validate_coverage(null_df: pd.DataFrame, real_df: pd.DataFrame, null_name: str):
    """Checks if null metrics cover the same parameter grid as real metrics."""
    null_params = set(zip(null_df['eta'], null_df['gamma']))
    real_params = set(zip(real_df['eta'], real_df['gamma']))
    
    missing = real_params - null_params
    if missing:
        logger.warning(f"Null model {null_name} missing coverage for {len(missing)} parameter configs present in real data.")
        # We might want to abort or just log, but user said 'Real metrics missing or not aligned -> Abort'
        # But here it's nulls missing. 
        # "Ensure identical pairset... If not, WARN and abort."
        if len(missing) > 0:
             raise ValueError(f"CRITICAL: {null_name} missing coverage for {len(missing)} configs.")

def check_sample_sizes(null_df: pd.DataFrame, min_samples: int = 20):
    """Checks if each parameter config has enough samples."""
    counts = null_df.groupby(['eta', 'gamma']).size()
    under_sampled = counts[counts < min_samples]
    if not under_sampled.empty:
        logger.error(f"Found {len(under_sampled)} configs with < {min_samples} samples.")
        raise ValueError(f"Insufficient samples for some configs. Min found: {under_sampled.min()}")

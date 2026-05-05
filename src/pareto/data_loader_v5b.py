import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

def load_null_metrics_v5b(base_path: str, null_model: str) -> pd.DataFrame:
    """Loads metrics for a specific null model and pivots to wide format."""
    path = Path(base_path) / f"{null_model}_metrics.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Null metrics not found at {path}")
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows for {null_model} from {path}")
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows for {null_model} from {path}")
    
    # Check if we have 'metric', 'val' columns to process
    if 'metric' not in df.columns or 'val' not in df.columns:
         # Maybe already pivoted?
         if 'energy' in df.columns and 'latency' in df.columns:
             return df
         raise ValueError(f"Unknown format for {null_model} metrics. Columns: {df.columns}")
         
    # 1. Extract Energy (invariant to gamma in null generation)
    # In evaluate_nulls, energy computed with gamma=0.0
    energy_df = df[df['metric'] == 'energy'][['sample_idx', 'seed', 'eta', 'val']].rename(columns={'val': 'energy'})
    # Round eta
    energy_df['eta'] = energy_df['eta'].round(6)
    
    # 2. Extract Latency (invariant to eta in null generation)
    # In evaluate_nulls, latency computed with eta=1.0
    latency_df = df[df['metric'] == 'latency'][['sample_idx', 'seed', 'gamma', 'val']].rename(columns={'val': 'latency'})
    # Round gamma
    latency_df['gamma'] = latency_df['gamma'].round(6)
    
    # 3. Merge on sample/seed to form cross product
    # Each sample will get all combinations of eta (from energy) and gamma (from latency)
    merged_df = pd.merge(energy_df, latency_df, on=['sample_idx', 'seed'], how='inner')
    
    merged_df['null_model'] = null_model
    
    logger.info(f"Reconstructed {len(merged_df)} rows for {null_model} (cross-product).")
    
    return merged_df

def load_real_metrics_v5b(path: str) -> pd.DataFrame:
    """
    Loads real graph metrics from sweep summary.
    Returns DataFrame with columns: eta, gamma, energy, latency
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Real metrics not found at {path}")
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows for real metrics from {path}")
    
    # Process
    # We need eta, gamma, energy, latency
    # sweep_summary has 'metric', 'eta', 'gamma', 'value', 'mean'
    # Energy is deterministic -> 'value'
    # Latency is stochastic (FPT) -> 'mean'
    
    # Process
    # We need eta, gamma, energy, latency
    # sweep_summary has 'metric', 'eta', 'gamma', 'value', 'mean'
    # Energy is deterministic -> 'value'
    # Latency is stochastic (FPT) -> 'mean'
    
    # 1. Extract Energy per Eta (Invariant to Gamma)
    # Filter for metric=energy
    energy_subset = df[df['metric'] == 'energy'][['eta', 'value']].rename(columns={'value': 'energy'})
    # Drop duplicates just in case (e.g. if gamma was swept but ignored)
    energy_unique = energy_subset.drop_duplicates(subset=['eta'])
    energy_unique['eta'] = energy_unique['eta'].round(6)
    
    # 2. Extract Latency per Gamma (Invariant to Eta)
    # Filter for metric=latency
    latency_subset = df[df['metric'] == 'latency'][['gamma', 'mean']].rename(columns={'mean': 'latency'})
    # Drop duplicates
    latency_unique = latency_subset.drop_duplicates(subset=['gamma'])
    latency_unique['gamma'] = latency_unique['gamma'].round(6)
    
    logging.info(f"Unique Etas for Energy: {energy_unique['eta'].tolist()}")
    logging.info(f"Unique Gammas for Latency: {latency_unique['gamma'].tolist()}")
    
    # 3. Cross Product
    real_df = pd.merge(energy_unique.assign(key=1), latency_unique.assign(key=1), on='key').drop('key', axis=1)
    
    logging.info(f"Reconstructed Real Metrics Grid: {len(real_df)} configs")
    
    return real_df

def get_full_grid(real_df: pd.DataFrame) -> List[Tuple[float, float]]:
    """Extracts all unique (eta, gamma) pairs from real metrics."""
    grid = real_df[['eta', 'gamma']].drop_duplicates().values
    return [(row[0], row[1]) for row in grid]

def check_coverage(null_df: pd.DataFrame, grid: List[Tuple[float, float]], min_samples: int = 20) -> pd.DataFrame:
    """
    Checks coverage of null metrics against the expected grid.
    Returns a DataFrame with columns: eta, gamma, n_samples, covered (bool)
    """
    null_df['eta'] = null_df['eta'].round(6)
    null_df['gamma'] = null_df['gamma'].round(6)
    
    coverage_data = []
    
    for eta, gamma in grid:
        # Filter for this config
        # Use simple float comparison tolerance if needed, but rounding should suffice
        subset = null_df[(np.isclose(null_df['eta'], eta)) & (np.isclose(null_df['gamma'], gamma))]
        n = len(subset)
        covered = n >= min_samples
        
        coverage_data.append({
            'eta': eta,
            'gamma': gamma,
            'n_samples': n,
            'covered': covered
        })
        
    return pd.DataFrame(coverage_data)

import argparse
import sys
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def identify_pareto(scores):
    """
    Finds the Pareto frontier in 2D (Energy, Latency).
    Both are to be minimized.
    Returns boolean mask of Pareto-optimal points.
    """
    # scores: (N, 2) array.
    n_points = scores.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        # If any point j dominates i, i is not Pareto.
        # j dominates i if score_j <= score_i for all dims and score_j < score_i for at least one.
        # Only compare with other active Pareto candidates for speed?
        # Brute force O(N^2) is fine for N=50.
        for j in range(n_points):
            if i == j: continue
            if (scores[j] <= scores[i]).all() and (scores[j] < scores[i]).any():
                is_pareto[i] = False
                break
                
    return is_pareto

def compute_dominance_volume(real_point, null_points):
    """
    Computes fraction of null points dominated by the real point.
    Dominated means Real <= Null in both dims and < in at least one.
    """
    n_dominated = 0
    total = len(null_points)
    
    for idx, null in enumerate(null_points):
         if (real_point <= null).all() and (real_point < null).any():
             n_dominated += 1
             
    return n_dominated / total

def compute_distance_to_frontier_z(real_point, null_points):
    """
    Computes Z-score relative to null cloud center?
    Or distance to the frontier?
    Prompt: "For the real graph, compute distance to the null Pareto frontier. Normalize distances (e.g., z-score within null ensemble)."
    
    Distance to set P = min_{p in P} ||real - p||?
    But we care about *signed* distance (optimization).
    
    Standard metric: how many standard deviations is the real point away from the mean of the null cloud?
    Projected onto the "optimization vector" (-1, -1)?
    
    Let's compute Euclidean Z-score in the (E, L) space.
    Z = (Real - Mean(Null)) / Std(Null).
    Returns (Z_E, Z_L).
    """
    mean = np.mean(null_points, axis=0)
    std = np.std(null_points, axis=0)
    
    # Avoid zero division
    std[std == 0] = 1.0
    
    z_scores = (real_point - mean) / std
    return z_scores

def main():
    parser = argparse.ArgumentParser(description="Pareto Analysis")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    out_root = Path(config['paths']['output_root'])
    metrics_dir = out_root / "metrics"
    pareto_dir = out_root / "pareto"
    pareto_dir.mkdir(exist_ok=True)
    plots_dir = out_root / "reports" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(out_root / "logs" / "pareto.log")
    
    # Load Metrics
    logging.info("Loading metrics...")
    
    # Expect: N0_metrics.parquet, N1_metrics.parquet, etc.
    # And Real Metrics from v3_a? 
    # Or we construct a "real_metrics.parquet" using existing v3_a sweep?
    # Config points to v3_a/metrics.
    
    metrics_ref_dir = Path(config['paths']['metrics_ref'])
    real_metrics_path = metrics_ref_dir / "sweeps" / "sweep_summary.parquet"
    
    if not real_metrics_path.exists():
        logging.error(f"Real metrics not found at {real_metrics_path}")
        # sys.exit(1)
        # For now, continue if partial?
        # Try finding anywhere.
        pass
        
    df_real = pd.read_parquet(real_metrics_path)
    # real metrics has columns: metric, eta, gamma, value, mean, etc.
    # We need to pivot to (Energy, Latency) pairs for each (Eta, Gamma).
    
    null_dfs = {}
    null_types = config['null_models']['types']
    
    for nt in null_types:
        p = metrics_dir / f"{nt}_metrics.parquet"
        if p.exists():
            null_dfs[nt] = pd.read_parquet(p)
        else:
            logging.warning(f"Null metrics for {nt} not found.")
    
    # We need to compare Real vs Nulls for EACH (eta, gamma) combination.
    # Energy depends on eta. Latency depends on gamma.
    # So we iterate over grid (eta, gamma).
    
    etas = config['metrics']['etas']
    gammas = config['metrics']['gammas']
    
    # Store results
    pareto_results = []
    
    for eta in etas:
        for gamma in gammas:
            # 1. Get Real Point
            # Energy
            row_e = df_real[(df_real['metric'] == 'energy') & (df_real['eta'] == eta)]
            if row_e.empty: continue
            real_E = float(row_e['value'].iloc[0])
            
            # Latency
            row_l = df_real[(df_real['metric'] == 'latency') & (df_real['gamma'] == gamma)]
            if row_l.empty: continue
            real_L = float(row_l['mean'].iloc[0] if 'mean' in row_l.columns else row_l['value'].iloc[0]) # check col name
            
            real_point = np.array([real_E, real_L])
            
            # 2. Get Null Clouds
            for nt, df_null in null_dfs.items():
                if df_null.empty: continue
                
                # Get Energy values for this eta (all samples)
                # Null metrics df: [sample_idx, seed, metric, eta, gamma, val]
                # Filter energy
                null_Es = df_null[
                    (df_null['metric'] == 'energy') & 
                    (df_null['eta'] == eta)
                ][['sample_idx', 'val']].set_index('sample_idx')
                
                # Filter latency
                null_Ls = df_null[
                    (df_null['metric'] == 'latency') & 
                    (df_null['gamma'] == gamma)
                ][['sample_idx', 'val']].set_index('sample_idx')
                
                # Join on sample_idx
                # (Some samples might be missing if failed)
                joined = null_Es.join(null_Ls, lsuffix='_E', rsuffix='_L').dropna()
                
                if joined.empty:
                    continue
                
                null_points = joined[['val_E', 'val_L']].values
                
                # Metrics
                z_score = compute_distance_to_frontier_z(real_point, null_points)
                dominance = compute_dominance_volume(real_point, null_points)
                
                pareto_results.append({
                    "null_type": nt,
                    "eta": eta,
                    "gamma": gamma,
                    "real_E": real_E,
                    "real_L": real_L,
                    "z_E": z_score[0],
                    "z_L": z_score[1],
                    "dominance": dominance,
                    "n_nulls": len(null_points)
                })
                
                # Plot for specific interesting case?
                # Plot first eta/gamma or all?
                # Plot specific "canonical" params? eta=1.5, gamma=1e-5?
                if eta == 1.5 and gamma == 1e-5:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(null_points[:, 0], null_points[:, 1], c='gray', alpha=0.5, label=f"{nt} Ensemble")
                    plt.scatter([real_E], [real_L], c='red', marker='*', s=200, label='Real Connectome')
                    plt.xlabel(f"Wiring Energy (eta={eta})")
                    plt.ylabel(f"Latency (gamma={gamma})")
                    plt.title(f"Pareto Analysis: Real vs {nt}")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"pareto_scatter_{nt}_eta{eta}_gamma{gamma}.png")
                    plt.close()
                    
    # Save Results
    df_res = pd.DataFrame(pareto_results)
    df_res.to_parquet(pareto_dir / "dominance_volume.parquet")
    
    # Generate Heatmaps
    # Dominance Volume heatmap over (eta, gamma) for each null
    for nt in null_types:
        subset = df_res[df_res['null_type'] == nt]
        if subset.empty: continue
        
        pivot = subset.pivot(index='gamma', columns='eta', values='dominance')
        
        plt.figure(figsize=(8, 6))
        plt.imshow(pivot.values, aspect='auto', cmap='RdYlGn', origin='lower')
        plt.colorbar(label='Dominance Fraction')
        plt.title(f"{nt} Dominance Landscape")
        plt.xticks(range(len(etas)), etas)
        plt.yticks(range(len(gammas)), gammas)
        plt.xlabel("Eta (Distance definition)")
        plt.ylabel("Gamma (Conduction params)")
        plt.tight_layout()
        plt.savefig(plots_dir / f"dominance_heatmap_{nt}.png")
        plt.close()
        
    logging.info("Pareto analysis complete.")

if __name__ == "__main__":
    main()

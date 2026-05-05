import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

def plot_coverage_table_v5b(coverage_df: pd.DataFrame, output_path: Path):
    """
    Plots a table/heatmap of coverage: % covered per null model?
    Or just a grid visual.
    """
    # Group by null model
    # Wait, the coverage df passed from data loader might be per null?
    # We should aggregate.
    plt.figure(figsize=(8, 4))
    pivot = coverage_df.pivot_table(index='eta', columns='gamma', values='covered')
    sns.heatmap(pivot, annot=True, cbar=False, cmap="RdYlGn", linewidths=1, linecolor='gray')
    plt.title('Coverage Map (Green = Valid)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_pareto_scatter_v5b(null_points: np.ndarray, 
                            real_point: np.ndarray, 
                            frontier_mask: np.ndarray,
                            title: str,
                            output_path: Path):
    """Correlation Plot with Frontier."""
    plt.figure(figsize=(7, 6))
    
    # Nulls
    plt.scatter(null_points[~frontier_mask, 0], null_points[~frontier_mask, 1], 
                c='gray', alpha=0.4, label='Null Ensemble')
                
    # Frontier
    fp = null_points[frontier_mask]
    sort_idx = np.argsort(fp[:, 0]) # Energy sort
    plt.plot(fp[sort_idx, 0], fp[sort_idx, 1], 'r--', lw=2, label='Null Frontier')
    plt.scatter(fp[:, 0], fp[:, 1], c='red', s=20)
    
    # Real
    plt.scatter(real_point[0], real_point[1], c='blue', marker='*', s=150, label='Real Graph', edgecolor='white')
    
    plt.xlabel('Energy (Total)')
    plt.ylabel('Latency (FPT)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_dominance_heatmap_v5b(df: pd.DataFrame, value_col: str, title: str, output_path: Path):
    """Heatmap over eta/gamma."""
    pivot = df.pivot(index='gamma', columns='eta', values=value_col)
    pivot.sort_index(ascending=False, inplace=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0) # Red = High Z (Bad), Blue = Low Z (Good)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_failure_regions_heatmap(failures_df: pd.DataFrame, output_path: Path):
    """If failures exist, plot a heatmap of their Z-scores."""
    if failures_df.empty:
        return
        
    # Aggregate Max Z-Score per config (across null models)
    agg_df = failures_df.groupby(['eta', 'gamma'])['z_score'].max().reset_index()
    
    pivot = agg_df.pivot(index='gamma', columns='eta', values='z_score')
    pivot.sort_index(ascending=False, inplace=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds")
    plt.title("Failure Regions (Z-Score)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

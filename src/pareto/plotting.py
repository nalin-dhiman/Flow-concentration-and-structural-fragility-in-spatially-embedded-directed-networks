import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple

def plot_pareto_frontier(null_points: np.ndarray, 
                         real_point: np.ndarray, 
                         frontier_mask: np.ndarray,
                         null_name: str,
                         eta: float, 
                         gamma: float,
                         output_path: Path):
    """
    Plots the null ensemble, the computed frontier, and the real point.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot all null points
    plt.scatter(null_points[~frontier_mask, 0], null_points[~frontier_mask, 1], 
                alpha=0.5, color='gray', label=f'{null_name} Nulls')
    
    # Plot frontier points
    frontier_points = null_points[frontier_mask]
    # Sort for line plot
    sort_idx = np.argsort(frontier_points[:, 0])
    plt.plot(frontier_points[sort_idx, 0], frontier_points[sort_idx, 1], 
             color='red', linestyle='--', linewidth=2, label='Pareto Frontier')
    plt.scatter(frontier_points[:, 0], frontier_points[:, 1], color='red', s=40)
    
    # Plot Real Point
    plt.scatter(real_point[0], real_point[1], color='blue', s=100, marker='*', label='Real Drosophila')
    
    plt.xlabel('Energy (Total)')
    plt.ylabel('Latency (FPT)')
    plt.title(f'Pareto Analysis: {null_name} @ eta={eta}, gamma={gamma}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_dominance_heatmap(dominance_df: pd.DataFrame, 
                           metric_col: str, 
                           title: str, 
                           output_path: Path):
    """
    Plots a heatmap of the dominance metric over eta and gamma.
    """
    pivot_table = dominance_df.pivot(index='gamma', columns='eta', values=metric_col)
    pivot_table.sort_index(ascending=False, inplace=True) # Gamma Y-axis desc
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_distance_distribution(dist_df: pd.DataFrame, output_path: Path):
    """
    Boxplot of distance z-scores per null model.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dist_df, x='null_model', y='z_score')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Distribution of Real-to-Frontier Z-Scores')
    plt.ylabel('Z-Score (Negative = Better than Null Frontier)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

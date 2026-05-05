import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import glob

def main():
    print("TASK 4: Plotting Mechanism Panels v3", flush=True)
    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="notebook")
    
    try:
        real_df = pd.read_csv("tables/real_metrics_aligned.csv")
        base_df = pd.read_csv("tables/n2_reference.csv")
        gen_df = pd.read_csv("tables/generated_runs_all.csv")
        topo_df = pd.read_csv("tables/topology_summary_by_f.csv")
    except Exception as e:
        print(f"Skipping plot, missing data: {e}")
        return
        
    f_vals = sorted(gen_df['f_val'].unique())
    
    # 1. Phase Diagram
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Panel A: DNP11 enrichment vs f
    sns.lineplot(data=gen_df, x='f_val', y='DNP11_E_weight', marker='o', ci='sd', ax=axes[0], color='salmon', label='Generated')
    axes[0].axhline(real_df['DNP11_E_weight'].iloc[0], color='darkred', linestyle='--', label='REAL')
    axes[0].axhline(base_df['DNP11_E_weight'].iloc[0], color='gray', linestyle=':', label='Base (N2)')
    axes[0].set_title("A. DNP11 Enrichment vs Rewiring Fraction ($f$)")
    axes[0].set_xlabel("Rewiring Fraction $f$")
    axes[0].set_ylabel("Routing Enrichment ($E_{weight}$)")
    axes[0].legend()
    
    # Panel B: Fragility vs f
    # Fragility is Relative Drop 
    sns.lineplot(data=gen_df, x='f_val', y='Relative_Drop', marker='o', ci='sd', ax=axes[1], color='salmon', label='Generated')
    axes[1].axhline(real_df['Relative_Drop'].iloc[0], color='darkred', linestyle='--', label='REAL')
    axes[1].axhline(base_df['Relative_Drop'].iloc[0], color='gray', linestyle=':', label='Base (N2)')
    axes[1].set_title("B. Fragility ($\Delta R$ relative drop) vs $f$")
    axes[1].set_xlabel("Rewiring Fraction $f$")
    axes[1].set_ylabel("Relative Reachability Drop")
    axes[1].legend()
    
    # Panel C: Flux Concentration vs f
    sns.lineplot(data=gen_df, x='f_val', y='flux_share_top1', marker='o', ci='sd', ax=axes[2], color='salmon', label='Generated')
    axes[2].axhline(real_df['flux_share_top1'].iloc[0], color='darkred', linestyle='--', label='REAL')
    axes[2].axhline(base_df['flux_share_top1'].iloc[0], color='gray', linestyle=':', label='Base (N2)')
    axes[2].set_title("C. Flux Concentration (Top 1% share) vs $f$")
    axes[2].set_xlabel("Rewiring Fraction $f$")
    axes[2].set_ylabel("Fraction of Total Flux")
    axes[2].legend()
    
    # Panel D: Constraint Validity
    sns.lineplot(data=gen_df, x='f_val', y='Block_Mass_Err', marker='o', ci='sd', ax=axes[3], color='blue', label='Mass Err')
    sns.lineplot(data=gen_df, x='f_val', y='Dist_Divergence', marker='s', ci='sd', ax=axes[3], color='green', label='Dist Div')
    axes[3].axhline(0, color='k', linestyle='--')
    axes[3].set_title("D. Macroscopic Constraints vs $f$")
    axes[3].set_xlabel("Rewiring Fraction $f$")
    axes[3].set_ylabel("Relative Error vs REAL")
    axes[3].legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "phase_diagram.png", dpi=300)
    plt.close()
    
    # 2. Flux ECDF Compare
    # We need edge files to plot ECDF exactly, so we select one seed for f=0.01 and f=0.05
    try:
        repo_root = Path(__file__).resolve().parents[1]
        real_edges = pd.read_parquet(repo_root / "data" / "canonical" / "edges.parquet")
        from generative_v3_metrics import compute_targeted_flux_backbone
        nodes = pd.read_parquet(repo_root / "data" / "canonical" / "nodes.parquet")
        targets = pd.read_csv(repo_root / "data" / "targets" / "descending_neurons.csv")
        n_nodes = len(nodes['bodyId'].unique())
        node2idx = {node: i for i, node in enumerate(nodes['bodyId'].unique())}
        target_idxs = [node2idx[t] for t in targets['bodyId'].unique() if t in node2idx]
        
        _, _, e_flux_real = compute_targeted_flux_backbone(real_edges, n_nodes, node2idx, target_idxs)
        base_edges = pd.read_parquet("../generative_principle_v1/graphs/base_N2_like_edges.parquet")
        _, _, e_flux_base = compute_targeted_flux_backbone(base_edges, n_nodes, node2idx, target_idxs)
        
        # Load examples
        gen_0p01_edges = pd.read_parquet("runs/f_0p01/seed_000/generated_edges.parquet")
        _, _, e_flux_0p01 = compute_targeted_flux_backbone(gen_0p01_edges, n_nodes, node2idx, target_idxs)
        gen_0p05_edges = pd.read_parquet("runs/f_0p05/seed_000/generated_edges.parquet")
        _, _, e_flux_0p05 = compute_targeted_flux_backbone(gen_0p05_edges, n_nodes, node2idx, target_idxs)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.ecdfplot(e_flux_real['flux'], color='darkred', label='REAL', ax=ax)
        sns.ecdfplot(e_flux_base['flux'], color='gray', label='Base', ax=ax)
        sns.ecdfplot(e_flux_0p01['flux'], color='salmon', label='Gen (f=0.01)', ax=ax)
        sns.ecdfplot(e_flux_0p05['flux'], color='darkorange', label='Gen (f=0.05)', ax=ax)
        
        ax.set_xscale('log')
        ax.set_title("Edge Flux ECDF")
        ax.set_xlabel("Target-Conditioned Flux")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "flux_ecdf_compare.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping ECDF: {e}")
        
    # 3. Topology Grid
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes = axes.flatten()
    
    # We can plot bar charts from topo_df
    topo_df.set_index('Model', inplace=True)
    
    # Out-degree
    axes[0].bar(topo_df.index, topo_df['deg_mean'], color=['darkred', 'gray'] + ['salmon'] * (len(topo_df)-2))
    axes[0].set_title("Mean Out-Degree")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Modularity Proxy
    axes[1].bar(topo_df.index, topo_df['modularity_proxy'], color=['darkred', 'gray'] + ['salmon'] * (len(topo_df)-2))
    axes[1].set_title("Modularity (Intra/Inter)")
    axes[1].tick_params(axis='x', rotation=45)
    
    # Reciprocity
    axes[2].bar(topo_df.index, topo_df['reciprocity'], color=['darkred', 'gray'] + ['salmon'] * (len(topo_df)-2))
    axes[2].set_title("Reciprocity")
    axes[2].tick_params(axis='x', rotation=45)
    
    # Rich-club 1%
    axes[3].bar(topo_df.index, topo_df['rich_club_1'], color=['darkred', 'gray'] + ['salmon'] * (len(topo_df)-2))
    axes[3].set_title("Rich-Club (Top 1%)")
    axes[3].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(out_dir / "topology_grid.png", dpi=300)
    plt.close()
    
    print("Saved v3 plots successfully.", flush=True)

if __name__ == "__main__":
    main()

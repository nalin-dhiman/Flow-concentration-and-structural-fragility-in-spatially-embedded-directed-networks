import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import networkx as nx
from pathlib import Path

def setup_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.transparent': False
    })
    
import matplotlib.colors as mcolors

def get_rgba(color, alpha):
    return mcolors.to_rgba(color, alpha)

def plot_network_panel(ax, G, pos, plot_mode, dn_nodes, base_node_size, dn_node_size):
    # Colors
    comp_colors = {
        'medulla': '#1f77b4',
        'lobula': '#2ca02c',
        'lobula_plate': '#ff7f0e',
    }
    
    # Filter nodes based on plot_mode
    if plot_mode == 'all':
        plot_nodes = list(G.nodes(data=True))
    else:
        # e.g. plot_mode = 'medulla'
        plot_nodes = [(n, d) for n, d in G.nodes(data=True) if d['comp'] == plot_mode]
        
    filtered_node_ids = set([n for n, d in plot_nodes])
    
    # Draw regular nodes for this compartment selection
    for comp, color in comp_colors.items():
        if plot_mode != 'all' and comp != plot_mode:
            continue
            
        nodes = [n for n, d in plot_nodes if d['comp'] == comp and not d['is_dn']]
        if nodes:
            nodes_col = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, 
                                   node_size=1.5, alpha=0.65, 
                                   edgecolors=(0,0,0,0.4), linewidths=0.1, ax=ax)
            if nodes_col is not None:
                nodes_col.set_zorder(3)

    # Draw DNs
    dn_plot_nodes = [n for n, d in plot_nodes if d.get('is_dn', False)]
    if dn_plot_nodes:
        dn_col = nx.draw_networkx_nodes(G, pos, nodelist=dn_plot_nodes, node_color='#d62728', 
                               node_size=7.0, alpha=0.85, 
                               edgecolors=(0,0,0,0.6), linewidths=0.3, ax=ax)
        if dn_col is not None:
            dn_col.set_zorder(5)

    # Filter edges based on plot_mode. Edges must connect two nodes in the panel.
    valid_edges = [(u, v, d) for u, v, d in G.edges(data=True) if u in filtered_node_ids and v in filtered_node_ids]
    
    if plot_mode == 'all':
        reg_color = '#222222'
        reg_alpha = 0.10
        reg_lw = 0.15
    else:
        reg_color = comp_colors.get(plot_mode, '#222222')
        reg_alpha = 0.10
        reg_lw = 0.15
        
    reg_edges = [(u, v) for u, v, d in valid_edges if not d['is_dn_target']]
    dn_edges = [(u, v) for u, v, d in valid_edges if d['is_dn_target']]

    # Subsample background edges if too dense for visualization
    max_backbone_edges_display = 15000
    if len(reg_edges) > max_backbone_edges_display:
        import random
        # Ensure repeatable sampling
        random.seed(42)
        reg_edges = random.sample(reg_edges, max_backbone_edges_display)

    # Enable rasterization for zorder < 1
    ax.set_rasterization_zorder(1)
    
    def draw_line_collection(edges, color, lw, zorder, rasterized):
        if not edges: return
        segments = []
        for u, v in edges:
            segments.append([pos[u], pos[v]])
        lc = LineCollection(segments, colors=color, linewidths=lw, alpha=1.0, zorder=zorder, rasterized=rasterized)
        ax.add_collection(lc)

    # Draw white shadow layer behind background edges to boost visibility (Rasterized)
    draw_line_collection(reg_edges, color='#f5f5f5', lw=0.4, zorder=-3, rasterized=True)

    # Draw background edges (Rasterized solid hex to simulate alpha=0.18 without breaking PDF rasterization)
    draw_line_collection(reg_edges, color='#dddddd', lw=reg_lw, zorder=-2, rasterized=True)
    
    # Draw target-terminating edges (Vector)
    # Using #d62728 alpha 0.60 simulation -> #e67d7e 
    draw_line_collection(dn_edges, color='#e67d7e', lw=0.45, zorder=-1, rasterized=False)

    # Clean axes
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_dir", type=str, required=True, help="Base Phase1 Canonical Build path")
    parser.add_argument("--out_path", type=str, required=True, help="Output figure path")
    args = parser.parse_args()

    base_dir = Path(args.pipeline_dir)
    
    bb_edges_path = base_dir / "natcomm_upgrade_v3/tables/backbone_top_edges_top1pct.csv"
    if not bb_edges_path.exists():
        bb_edges_path = base_dir / "natcomm_defense_v2_4/tables/backbone_top_edges_top1pct.csv"
        
    dn_audit_path = base_dir / "natcomm_defense_v2_4/tables/DN_identification_audit.csv"

    setup_plot_style()
    
    # Load data
    dn_df = pd.read_csv(dn_audit_path)
    dn_nodes = set(dn_df['bodyId'].values)

    edges_df = pd.read_csv(bb_edges_path)
    G = nx.DiGraph()

    # Collect nodes and compartment properties
    node_props = {}
    for _, row in edges_df.iterrows():
        # Source node
        u = row['pre_idx']
        if u not in node_props:
            node_props[u] = {
                'pos': (row['xyz_pre_x'], row['xyz_pre_y']),
                'comp': str(row['pre_comp']).lower(),
                'is_dn': u in dn_nodes
            }
        # Target node
        v = row['post_idx']
        if v not in node_props:
            node_props[v] = {
                'pos': (row['xyz_post_x'], row['xyz_post_y']),
                'comp': str(row['post_comp']).lower(),
                'is_dn': v in dn_nodes
            }
        
        flux = row['flux']
        is_dn_target = v in dn_nodes
        G.add_edge(u, v, flux=flux, is_dn_target=is_dn_target)
        
    for u, props in node_props.items():
        G.add_node(u, **props)

    pos = nx.get_node_attributes(G, 'pos')
    
    # Determine global limits to include ALL nodes (no cropping)
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    
    x_range = xmax - xmin
    y_range = ymax - ymin
    
    x_pad = 0.05 * x_range
    y_pad = 0.05 * y_range
    
    x_lim = (xmin - x_pad, xmax + x_pad)
    y_lim = (ymin - y_pad, ymax + y_pad)

    fig = plt.figure(figsize=(7.0, 5.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.25, 1.0], hspace=0.10, wspace=0.18)
    
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[1, 2])
    
    panels = [
        (axA, 'all', '(a)', 3.0),
        (axB, 'medulla', '(b)', 3.0),
        (axC, 'lobula', '(c)', 3.0),
        (axD, 'lobula_plate', '(d)', 3.0)
    ]
    
    for ax, mode, label, size in panels:
        plot_network_panel(ax, G, pos, mode, dn_nodes, base_node_size=size, dn_node_size=10.0)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.text(0.01, 0.99, label, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')

    # Add legend to Panel A
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Medulla neurons', markerfacecolor='#1f77b4', markersize=6, markeredgecolor='black', markeredgewidth=0.3),
        Line2D([0], [0], marker='o', color='w', label='Lobula neurons', markerfacecolor='#2ca02c', markersize=6, markeredgecolor='black', markeredgewidth=0.3),
        Line2D([0], [0], marker='o', color='w', label='Lobula plate neurons', markerfacecolor='#ff7f0e', markersize=6, markeredgecolor='black', markeredgewidth=0.3),
        Line2D([0], [0], marker='o', color='w', label='Descending neurons (target set $T$)', markerfacecolor='#d62728', markersize=8, markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], color='#444444', lw=1.5, alpha=0.35, label='Backbone edges'),
        Line2D([0], [0], color='#d62728', lw=2.0, alpha=0.9, label='Backbone edges terminating in targets')
    ]
    axA.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7, markerscale=0.9, handlelength=1.2)

    fig.subplots_adjust(right=0.80)
    plt.savefig(args.out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"Figure 6 successfully generated at {args.out_path}")

if __name__ == "__main__":
    main()

import os
import yaml
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
from pathlib import Path
import json
import time
import sys

from v3_metrics import compute_ppr_vector

def main(f_val, seed_idx):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    print(f"--- RUNNING GENERATOR SEED {seed_idx} ---", flush=True)
    out_dir = Path(f"runs/f_{str(f_val).replace('.', 'p')}/seed_{seed_idx:03d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    random.seed(seed_idx)
    np.random.seed(seed_idx)
    
    # Load Real nodes & Base Edges
    print("Loading Base N2-like Graph and Metadata...", flush=True)
    nodes = pd.read_parquet(cfg['paths']['real_nodes'])
    edges_base = pd.read_parquet(cfg['paths']['base_graph_parquet'])
    targets = pd.read_parquet(cfg['paths']['targets'])
    
    node_list = nodes['bodyId'].unique()
    n_nodes = len(node_list)
    node2idx = {node: i for i, node in enumerate(node_list)}
    target_idxs = [node2idx[t] for t in targets['bodyId'].unique() if t in node2idx]
    
    # 1. Blocks & Compartments
    roi_df = pd.read_feather(cfg['paths']['roi_elements'])
    b_col_roi = 'body' if 'body' in roi_df.columns else ('bodyId' if 'bodyId' in roi_df.columns else ':ID(Body-ID)')
    if 'roi' in roi_df.columns:
        valid_rois = roi_df[roi_df['roi'] != '<unspecified>']
        primary_roi = valid_rois.drop_duplicates(b_col_roi)
        # Note: could use synweight but sticking to simple mapping
        comp_map_raw = dict(zip(primary_roi[b_col_roi], primary_roi['roi']))
    else: comp_map_raw = {}
    
    def get_compartment(roi_str):
        roi_str = str(roi_str).upper()
        if 'ME' in roi_str: return 'medulla'
        if 'LO' in roi_str:
            if 'LOP' in roi_str: return 'lobula_plate'
            return 'lobula'
        return 'other'
    nodes['comp'] = nodes['bodyId'].map(comp_map_raw).apply(get_compartment)
    comp_dict = nodes.set_index('bodyId')['comp'].to_dict()
    
    edges_base['pre_comp'] = edges_base['pre'].map(comp_dict)
    edges_base['post_comp'] = edges_base['post'].map(comp_dict)
    
    # 2. Coordinates
    coord_dict = nodes.set_index('bodyId')[['x', 'y', 'z']].to_dict('index')
    def get_d(u, v):
        try:
            c1, c2 = coord_dict[u], coord_dict[v]
            return np.sqrt((c1['x']-c2['x'])**2 + (c1['y']-c2['y'])**2 + (c1['z']-c2['z'])**2)
        except: return 1000.0
        
    eta = 1.0
    nodes_by_comp = {c: nodes[nodes['comp'] == c]['bodyId'].values for c in nodes['comp'].unique()}
    
    edges = edges_base.copy()
    existing_edges = set(zip(edges['pre'], edges['post']))
    edge_list = edges[['pre', 'post', 'w_ij', 'pre_comp', 'post_comp']].values.tolist()
    
    # Calculate initially
    row = edges['pre'].map(node2idx).values
    col = edges['post'].map(node2idx).values
    data = edges['w_ij'].values
    pi = compute_ppr_vector(n_nodes, row, col, data, target_idxs, 
                            alpha=cfg['parameters']['alpha_ppr'],
                            max_iter=cfg['parameters']['max_iter_ppr'])
    
    K_swaps = int(len(edges) * f_val)
    M_recompute = 500
    print(f"Generating {K_swaps} swaps for f={f_val}...", flush=True)
    
    t0 = time.time()
    for step in range(K_swaps):
        if step > 0 and step % M_recompute == 0:
            temp_df = pd.DataFrame(edge_list, columns=['pre', 'post', 'w_ij', 'pre_comp', 'post_comp'])
            row = temp_df['pre'].map(node2idx).values
            col = temp_df['post'].map(node2idx).values
            data = temp_df['w_ij'].values
            pi = compute_ppr_vector(n_nodes, row, col, data, target_idxs, 
                                    alpha=cfg['parameters']['alpha_ppr'],
                                    max_iter=cfg['parameters']['max_iter_ppr'])
            if step % (M_recompute * 10) == 0:
                print(f"Seed {seed_idx}: Step {step}/{K_swaps}", flush=True)
                
        # 1. Drop Edge
        idx_samples = random.sample(range(len(edge_list)), min(20, len(edge_list)))
        lowest_score = float('inf')
        drop_idx = -1
        
        # Optimize by caching pi lookups
        for i in idx_samples:
            u, v, w, pc, poc = edge_list[i]
            su = pi[node2idx[u]] if u in node2idx else 0
            sv = pi[node2idx[v]] if v in node2idx else 0
            score = su * sv
            if score < lowest_score:
                lowest_score = score
                drop_idx = i
                
        u_drop, v_drop, w_drop, cpre, cpost = edge_list[drop_idx]
        
        # 2. Pick an edge to add (same block)
        pool_u = nodes_by_comp[cpre]
        pool_v = nodes_by_comp[cpost]
        
        best_add_u, best_add_v = -1, -1
        highest_score = -1.0
        
        attempts = 0
        candidates_found = 0
        while candidates_found < 20 and attempts < 100:
            attempts += 1
            cu = random.choice(pool_u)
            cv = random.choice(pool_v)
            if cu != cv and (cu, cv) not in existing_edges:
                su = pi[node2idx[cu]] if cu in node2idx else 0
                sv = pi[node2idx[cv]] if cv in node2idx else 0
                d = get_d(cu, cv) + 1.0
                score = (su * sv) / (d ** eta)
                if score > highest_score:
                    highest_score = score
                    best_add_u, best_add_v = cu, cv
                candidates_found += 1
                
        if best_add_u != -1:
            existing_edges.remove((u_drop, v_drop))
            existing_edges.add((best_add_u, best_add_v))
            edge_list[drop_idx] = [best_add_u, best_add_v, w_drop, cpre, cpost]
            
    print(f"Seed {seed_idx} generation completed in {time.time()-t0:.1f}s.")
    generated_df = pd.DataFrame(edge_list, columns=['pre', 'post', 'w_ij', 'pre_comp', 'post_comp'])
    generated_df.to_parquet(out_dir / "generated_edges.parquet")

if __name__ == "__main__":
    f_val = float(sys.argv[1])
    seed_idx = int(sys.argv[2])
    main(f_val, seed_idx)

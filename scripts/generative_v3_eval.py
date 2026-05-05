import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import json
from v3_metrics import compute_targeted_flux_backbone, compute_reachability, get_enrichment, get_flux_concentration, get_topology_metrics

def get_block_mass(edges, nodes, comp_dict):
    edges = edges.copy()
    edges['pre_comp'] = edges['pre'].map(comp_dict)
    edges['post_comp'] = edges['post'].map(comp_dict)
    mass = edges.groupby(['pre_comp', 'post_comp'])['w_ij'].sum()
    return mass

def compute_spatial_dist(edges, nodes, coord_dict):
    def get_d(u, v):
        try:
            c1, c2 = coord_dict[u], coord_dict[v]
            return np.sqrt((c1['x']-c2['x'])**2 + (c1['y']-c2['y'])**2 + (c1['z']-c2['z'])**2)
        except: return 1000.0
    
    # Avoid applying on 6M edges, sample 1M
    samp = edges.sample(min(len(edges), 1000000), random_state=42)
    dists = [get_d(u, v) for u, v in zip(samp['pre'], samp['post'])]
    return float(np.median(dists))

def eval_graph(name, edges, nodes, node2idx, source_idxs, target_idxs, dn_bodies, cfg, real_mass=None, real_dist=None, comp_dict=None, coord_dict=None):
    n_nodes = len(node2idx)
    
    print(f"[{name}] Computing backbone...", flush=True)
    edges_ablated, backbone_edges, edges_flux = compute_targeted_flux_backbone(edges, n_nodes, node2idx, target_idxs)
    
    print(f"[{name}] Computing Enrichment...", flush=True)
    e_wt = get_enrichment(edges, dn_bodies, backbone_edges)
    
    print(f"[{name}] Computing Flux Concentration...", flush=True)
    flux_top1, flux_top0p1, gini = get_flux_concentration(edges_flux)
    
    print(f"[{name}] Computing Topology Summary...", flush=True)
    topo_res = get_topology_metrics(edges, n_nodes, node2idx, comp_dict)
    
    print(f"[{name}] Computing Baseline R(Tmax=50)...", flush=True)
    r_base = compute_reachability(edges, n_nodes, node2idx, source_idxs, target_idxs, walks=cfg['parameters']['mc_walks_per_source'], steps=cfg['parameters']['t_max'])
    
    print(f"[{name}] Computing Ablated R(Tmax=50)...", flush=True)
    r_abl = compute_reachability(edges_ablated, n_nodes, node2idx, source_idxs, target_idxs, walks=cfg['parameters']['mc_walks_per_source'], steps=cfg['parameters']['t_max'])
    
    mass_err = 0.0
    dist_div = 0.0
    median_d = 0.0
    if comp_dict and coord_dict:
        g_mass = get_block_mass(edges, nodes, comp_dict)
        if real_mass is not None:
            err = np.sum(np.abs(real_mass.reindex(g_mass.index, fill_value=0) - g_mass)) / real_mass.sum()
            mass_err = float(err)
        median_d = compute_spatial_dist(edges, nodes, coord_dict)
        if real_dist is not None:
            dist_div = abs(median_d - real_dist) / real_dist
            
    res = {
        'Model': name,
        'DNP11_E_weight': float(e_wt),
        'Base_R50': r_base,
        'Ablated_R50': r_abl,
        'Relative_Drop': (r_base - r_abl) / r_base if r_base > 0 else 0,
        'flux_share_top1': flux_top1,
        'flux_share_top0p1': flux_top0p1,
        'flux_gini': gini,
        'Block_Mass_Err': mass_err,
        'Median_Dist': median_d,
        'Dist_Divergence': float(dist_div)
    }
    res.update(topo_res)
    return res

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    print("Loading Base Metadata...", flush=True)
    nodes = pd.read_parquet(cfg['paths']['real_nodes'])
    edges_base = pd.read_parquet(cfg['paths']['base_graph_parquet'])
    edges_real = pd.read_parquet(cfg['paths']['real_edges'])
    targets = pd.read_parquet(cfg['paths']['targets'])
    dn_list = pd.read_csv(cfg['paths']['dn_list'])
    dn_bodies = set(dn_list[dn_list['type'] == 'DNp11']['bodyId'])
    
    node_list = nodes['bodyId'].unique()
    n_nodes = len(node_list)
    node2idx = {node: i for i, node in enumerate(node_list)}
    target_idxs = [node2idx[t] for t in targets['bodyId'].unique() if t in node2idx]
    
    # Compartments & Coords
    roi_df = pd.read_feather(cfg['paths']['roi_elements'])
    b_col_roi = 'body' if 'body' in roi_df.columns else ('bodyId' if 'bodyId' in roi_df.columns else ':ID(Body-ID)')
    if 'roi' in roi_df.columns:
        valid_rois = roi_df[roi_df['roi'] != '<unspecified>']
        primary_roi = valid_rois.drop_duplicates(b_col_roi)
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
    coord_dict = nodes.set_index('bodyId')[['x', 'y', 'z']].to_dict('index')
    
    # Stratified Source Set (Load or Generate)
    if os.path.exists("tables/stratified_sources.npy"):
        source_idxs = np.load("tables/stratified_sources.npy")
    else:
        from v3_metrics import get_stratified_sources
        source_idxs = get_stratified_sources(edges_real, n_nodes, node2idx, target_idxs, K=cfg['parameters']['source_sample_size'], seed=42)
        np.save("tables/stratified_sources.npy", source_idxs)
        
    print(f"Sources Sampled: {len(source_idxs)}", flush=True)

    # 1. Evaluate REAL
    real_mass = get_block_mass(edges_real, nodes, comp_dict)
    real_dist = compute_spatial_dist(edges_real, nodes, coord_dict)
    
    res_real = eval_graph('REAL', edges_real, nodes, node2idx, source_idxs, target_idxs, dn_bodies, cfg, real_mass, real_dist, comp_dict, coord_dict)
    pd.DataFrame([res_real]).to_csv("tables/real_metrics_aligned.csv", index=False)
    
    # 2. Evaluate BASE (N2)
    res_base = eval_graph('N2_Base', edges_base, nodes, node2idx, source_idxs, target_idxs, dn_bodies, cfg, real_mass, real_dist, comp_dict, coord_dict)
    pd.DataFrame([res_base]).to_csv("tables/n2_reference.csv", index=False)
    
    # 3. Evaluate SEEDS 0-9
    seed_records = []
    
    for f_val in cfg['parameters']['f_sweep']:
        f_str = str(f_val).replace('.', 'p')
        for s in range(cfg['parameters']['num_seeds']):
            seed_dir = Path(f"runs/f_{f_str}/seed_{s:03d}")
            edge_file = seed_dir / "generated_edges.parquet"
            if not edge_file.exists(): continue
            
            edges_gen = pd.read_parquet(edge_file)
            res_s = eval_graph(f'f_{f_str}_Seed_{s:03d}', edges_gen, nodes, node2idx, source_idxs, target_idxs, dn_bodies, cfg, real_mass, real_dist, comp_dict, coord_dict)
            res_s['f_val'] = f_val
            res_s['seed'] = s
            
            with open(seed_dir / "dn_enrichment.json", "w") as f_out:
                json.dump({'DNP11_E_weight': res_s['DNP11_E_weight']}, f_out, indent=2)
            with open(seed_dir / "fragility.json", "w") as f_out:
                json.dump({'Base_R50': res_s['Base_R50'], 'Ablated_R50': res_s['Ablated_R50'], 'Relative_Drop': res_s['Relative_Drop']}, f_out, indent=2)
            with open(seed_dir / "flux_concentration.json", "w") as f_out:
                json.dump({'flux_share_top1': res_s['flux_share_top1'], 'flux_gini': res_s['flux_gini']}, f_out, indent=2)
            with open(seed_dir / "constraints.json", "w") as f_out:
                json.dump({'Block_Mass_Err': res_s['Block_Mass_Err'], 'Median_Dist': res_s['Median_Dist'], 'Dist_Divergence': res_s['Dist_Divergence']}, f_out, indent=2)
                
            seed_records.append(res_s)
            
    if len(seed_records) > 0:
        df_all = pd.DataFrame(seed_records)
        df_all.to_csv("tables/generated_runs_all.csv", index=False)
        
        topo_cols = ['Model', 'deg_mean', 'deg_p95', 'str_mean', 'str_p95', 'reciprocity', 'rich_club_1', 'rich_club_5', 'modularity_proxy']
        
        topo_real = {c: res_real.get(c, None) for c in topo_cols}
        topo_base = {c: res_base.get(c, None) for c in topo_cols}
        
        topo_df_records = [topo_real, topo_base]
        
        for f_val in cfg['parameters']['f_sweep']:
            f_df = df_all[df_all['f_val'] == f_val]
            if len(f_df) == 0: continue
            rec = {'Model': f'Generated (f={f_val})'}
            for c in topo_cols[1:]:
                rec[c] = f_df[c].mean()
            topo_df_records.append(rec)
            
        pd.DataFrame(topo_df_records).to_csv("tables/topology_summary_by_f.csv", index=False)

if __name__ == "__main__":
    main()

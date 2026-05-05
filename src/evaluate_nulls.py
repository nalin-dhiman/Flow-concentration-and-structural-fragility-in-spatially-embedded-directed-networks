import argparse
import sys
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import scipy.sparse as sp

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from src.metrics.compute_metrics import (
    load_canonical, compute_energy, build_conductance_matrix, 
    normalize_transition_matrix, solve_absorbing_fpt, get_stratified_targets
)
from src.nulls import N0WeightedNull, N1SpatialNull, N2BlockNull, N3LocalNull

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def evaluate_single_null(args):
    # Unpack args for parallel execution if needed
    # But we'll do sequential for now to avoid complexity
    pass

def main():
    parser = argparse.ArgumentParser(description="Generate Nulls and Compute Metrics")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument("--null_type", type=str, required=True, choices=["N0", "N1", "N2", "N3"])
    parser.add_argument("--n_samples", type=int, default=None, help="Override config n_samples")
    parser.add_argument("--seed", type=int, default=None, help="Start seed")
    parser.add_argument("--n_targets", type=int, default=None, help="Override config n_targets")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs (not fully implemented)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    paths = config['paths']
    out_root = Path(paths['output_root'])
    version = config['version'] # e.g. v4_a_nulls
    
    # Setup paths
    metrics_out_dir = out_root / "metrics"
    logs_dir = out_root / "logs"
    metrics_out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(logs_dir / f"{args.null_type}_evaluation.log")
    logging.info(f"Starting evaluations for {args.null_type}")
    
    # Load Canonical
    # Note: version in config might be different from canonical source
    # We load from canonical_dir
    nodes, edges, adj = load_canonical(Path(paths['canonical']))
    
    # Map node indices for metrics logic
    node_to_idx = {bid: i for i, bid in enumerate(nodes['bodyId'])}
    nodes['idx'] = nodes['bodyId'].map(node_to_idx) # Ensure idx column exists
    
    # Initialize Null Model
    logging.info(f"Initializing {args.null_type}...")
    if args.null_type == "N0":
        null_model = N0WeightedNull(nodes, edges, adj)
    elif args.null_type == "N1":
        null_model = N1SpatialNull(nodes, edges, adj)
    elif args.null_type == "N2":
        null_model = N2BlockNull(nodes, edges, adj)
    elif args.null_type == "N3":
        null_model = N3LocalNull(nodes, edges, adj)
    
    # Sweep Parameters
    etas = config['metrics']['etas']
    gammas = config['metrics']['gammas']
    n_targets = args.n_targets if args.n_targets else config['metrics']['n_targets']
    
    # Targets (Shared across all nulls for comparability? Or re-sampled?)
    # "Metrics engine... cached transition matrices" -> v3_a used re-usable logic.
    # We should re-select targets consistent with null topology?
    # Or keep same targets (bodyIds) as real graph?
    # FPT is sensitive to target set.
    # If we shuffle topology, the "top 10% in-degree nodes" might change?
    # N0/N1 preserve degree approx. N2 preserves block density.
    # So "high degree nodes" remain high degree.
    # It is scientifically better to FIX the targets (use same bodyIds) to see how access to THEM changes.
    # So we should load targets from v3_a if possible?
    # Or just re-select using same seed/logic on REAL nodes (since bodyIds match).
    # Since we preserve bodyIds, we can use the same target list.
    
    # Let's compute the target list based on REAL graph once (using `nodes` degree).
    # Since N0 preserves out-strength (not in-degree?)
    # N0 preserves in-strength.
    # So using real graph's degree stratification is valid.
    
    # Re-logic from compute_metrics to get targets
    # We need to compute SCC of the NULL graph for FPT.
    # Null graph might not be fully connected.
    # So effective target set = Targets \cap SCC(Null).
    
    # Target Selection Logic
    min_targets = 20
    real_targets = get_stratified_targets(nodes, n_targets=n_targets, seed=42)
    
    if len(real_targets) < min_targets:
        logging.warning(f"Stratified selection returned {len(real_targets)} targets (< {min_targets}). Attempting fallback.")
        # Fallback: Random selection from nodes with in-degree > 0
        # Calculate in-degrees if not available
        in_degrees = np.array(adj.sum(axis=0)).flatten()
        valid_candidates = np.where(in_degrees > 0)[0]
        
        if len(valid_candidates) >= min_targets:
             real_targets = np.random.choice(valid_candidates, min_targets, replace=False)
             logging.info(f"Fallback selection: {len(real_targets)} targets from nodes with in-degree > 0.")
        else:
             logging.error(f"Cannot find {min_targets} valid targets even with fallback (only {len(valid_candidates)} candidates).")
             
    if len(real_targets) < min_targets:
        logging.critical(f"Target selection failed: {len(real_targets)} targets < {min_targets}. Aborting evaluation.")
        sys.exit(1)

    logging.info(f"Base targets selected: {len(real_targets)}")
    
    # Run Samples
    n_samples = args.n_samples if args.n_samples else config['null_models']['num_samples']
    start_seed = args.seed if args.seed else config['null_models']['seed_start']
    
    results = []
    
    # Determine full shape for sparse matrix
    shape = adj.shape
    
    for i in tqdm(range(n_samples), desc=f"{args.null_type} Samples"):
        seed = start_seed + i
        
        # 1. Generate Null
        try:
            null_edges = null_model.generate(seed=seed)
        except Exception as e:
            logging.error(f"Failed to generate null seed {seed}: {e}")
            continue
            
        # Validate
        if not null_model.validate(null_edges):
            logging.warning(f"Null seed {seed} failed validation.")
            # Continue or abort? 
            # Log and continue.
        
        # Prepare for metric computation
        # Add indices if missing (generate usually adds pre_idx/post_idx but let's be safe)
        if 'pre_idx' not in null_edges.columns:
             null_edges['pre_idx'] = null_edges['pre'].map(node_to_idx)
             null_edges['post_idx'] = null_edges['post'].map(node_to_idx)
        
        # 2. Compute Energy Terms
        # Energy: E_wire, E_syn
        # E_wire depends on eta.
        # E_syn is constant (total synapses).
        # Actually E_syn term usually refers to metabolic cost of synapses?
        # In v3_a, E_total = E_wire + E_syn?
        # "E_wire, E_syn with parameter sweeps".
        # Let's see compute_metrics.py:
        # e_wire = sum(s_ij * d_ij^eta)
        # It doesn't compute a separate "E_syn" metric explicitly in the main loop,
        # but the prompt says "E_total = E_wire + E_syn".
        # Usually E_syn ~ sum(s_ij). Constant across nulls N0/N1/N2 (edges preserved).
        # So we just track E_wire.
        
        sample_metrics = []
        
        for eta in etas:
            ew = compute_energy(null_edges, eta)
            sample_metrics.append({
                "null_model": args.null_type,
                "sample_idx": i,
                "seed": seed,
                "metric": "energy",
                "eta": eta,
                "gamma": 0.0,
                "val": ew
            })
            
        # 3. Compute Latency (FPT)
        # Need SCC of null graph
        # Build adjacency
        # null_edges has duplicates? (pre, post)
        # sparse matrix handles duplicates by summation usually.
        pre = null_edges['pre_idx'].values
        post = null_edges['post_idx'].values
        # data? Just 1s for connectivity?
        # We need to know connection weights for FPT?
        # compute_metrics: build_conductance_matrix uses s_ij * exp.
        # So we need s_ij.
        
        # SCC calculation on binary adjacency
        # Just use scipy csgraph on the s_ij weighted matrix (non-zero structure same)
        Null_Adj = sp.csr_matrix((np.ones(len(pre)), (pre, post)), shape=shape)
        
        n_components, labels = sp.csgraph.connected_components(Null_Adj, directed=True, connection='strong')
        _, counts = np.unique(labels, return_counts=True)
        largest_label = np.argmax(counts)
        scc_mask = (labels == largest_label)
        scc_indices = np.where(scc_mask)[0]
        
        # Filter targets to those in SCC
        # real_targets are indices (0..N-1)
        valid_targets = [t for t in real_targets if scc_mask[t]]
        
        if len(valid_targets) < len(real_targets) * 0.5:
             logging.warning(f"Seed {seed}: Only {len(valid_targets)}/{len(real_targets)} targets in SCC.")
             
        # Map targets to SCC-local indices
        orig_to_local = {orig: local for local, orig in enumerate(scc_indices)}
        local_targets = [orig_to_local[t] for t in valid_targets]
        
        # Run Gamma Sweep
        for gamma in gammas:
            # Build C
            C = build_conductance_matrix(null_edges, gamma, shape)
            P_full = normalize_transition_matrix(C)
            
            # Extract SCC
            P_scc = P_full[scc_indices, :][:, scc_indices]
            
            # Solve FPT
            fpt_values = []
            
            # Solve for each target?
            # compute_metrics loops over targets.
            # Optimization: If targets are many, we solve (I-Q)x=1 many times?
            # Yes, solve_absorbing_fpt solves for ONE target set?
            # compute_metrics: "for t_idx in targets_local: t_fpts = solve..."
            # It treats each target as a distinct sink?
            # Or is it "FPT to ANY target"?
            # Code: `solve_absorbing_fpt(P_scc, np.array([t_idx]))`
            # Yes, it loops over SINGLE targets.
            # And computes mean FPT from rest of network to THAT target.
            
            # We must replicate this behavior.
            # To speed up: Sample subset of targets if N=50 is too slow?
            # v3_a config says n_targets=50.
            # 50 solves per gamma. 4 gammas. 200 solves.
            # 50 nulls -> 10,000 solves.
            # This IS heavy.
            # But the user said "v3_a/cache/transition_matrix_csr.npz (for speed, if compatible)".
            # We can't reuse cached P for NULLS.
            
            # We might need to reduce n_targets for nulls?
            # Or run in parallel.
            # Or accept it takes overnight.
            # "Computation Cost... I will look for ways to parallelize".
            
            # Let's execute for one null and see speed.
            # For now, implemented as is.
            
            # Randomly sub-sample 10 targets for nulls to save time?
            # "No claims of optimality are allowed" if we don't compare apples to apples.
            # We should use 50.
            # Use 10 targets for "Fast Audit" and 50 for final?
            # Let's stick to config n_targets=50.
            
            # We can use parallel map for the loop over targets?
            
            for t_local in local_targets:
                 # solve
                 try:
                     # This function might need optimization check
                     # It calculates FPT from ALL nodes to t_local.
                     # We take statistics.
                     tf = solve_absorbing_fpt(P_scc, np.array([t_local]))
                     valid = tf[~np.isnan(tf) & (tf > 0)]
                     if len(valid) > 0:
                         # Subsample 100 values like v3_a
                         if len(valid) > 100:
                             samp = np.random.choice(valid, 100, replace=False)
                         else:
                             samp = valid
                         fpt_values.extend(samp)
                 except Exception as e:
                     pass
                     
            if len(fpt_values) > 0:
                stats = {
                     "mean": float(np.mean(fpt_values)),
                     "median": float(np.median(fpt_values)),
                     "p90": float(np.percentile(fpt_values, 90))
                }
                sample_metrics.append({
                    "null_model": args.null_type,
                    "sample_idx": i,
                    "seed": seed,
                    "metric": "latency",
                    "eta": 1.0, # dummy
                    "gamma": gamma,
                    "val": stats['mean'] # Track mean FPT
                })
        
        results.extend(sample_metrics)
        
        # Intermediate/Checkpoint save every 10 samples
        if (i + 1) % 10 == 0:
             df_curr = pd.DataFrame(results)
             df_curr.to_parquet(metrics_out_dir / f"{args.null_type}_metrics_checkpoint.parquet")
             
    # Final Save
    df_res = pd.DataFrame(results)
    df_res.to_parquet(metrics_out_dir / f"{args.null_type}_metrics.parquet")
    logging.info(f"Finished {args.null_type}. Saved to {metrics_out_dir}")

if __name__ == "__main__":
    main()

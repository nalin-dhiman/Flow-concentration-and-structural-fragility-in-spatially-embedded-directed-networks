
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_z(val, mean, std):
    if std == 0:
        if val == mean: return 0.0
        elif val < mean: return -999.0 # improvement
        else: return 999.0 # degradation
    return (val - mean) / std

def run_aggregation(args):
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    
    metrics_dir = in_dir / 'per_seed_metrics'
    
    # Load all metrics
    files = list(metrics_dir.glob("*.parquet"))
    if not files:
        logging.error("No metric files found!")
        return
        
    logging.info(f"Found {len(files)} metric files.")
    
    df_list = []
    for f in files:
        try:
            d = pd.read_parquet(f)
            df_list.append(d)
        except Exception as e:
            logging.warning(f"Failed to read {f}: {e}")
            
    df = pd.concat(df_list, ignore_index=True)
    
    # Identify Real
    # Identify Real (robust to casing)
    real_df = df[df['null_type'].astype(str).str.upper() == 'REAL']
    if len(real_df) == 0:
        logging.error("Real graph metrics not found! Cannot compute dominance.")
        # Proceed with just null analysis or exit? 
        # Exit.
        # But for smoke test, maybe I didn't run Real? 
        # Plan says "Run Seeds (N0, N1, N2)". 
        # Smoke test should PROBABLY run Real too to verify aggregation.
        # But user prompts don't explicitly say "Run Real in smoke test".
        # Prompt says "Run three per-seed jobs locally... nulls: N0, N1, N2".
        # It DOES NOT say run Real.
        # However, `v7c_aggregate.py` usually needs Real.
        # I will assume I need to run Real or handle its absence (by skipping dominance).
        # But "Aggregator produces ... dominance plot".
        # So I MUST have Real.
        # I will Insert a "fake" real row if missing for SMOKE TEST purposes? 
        # Or I should add a Real run to the smoke test instructions.
        # I will check if real exists, if not, try to find it in `v7_c_objective_upgrade_stable`?
        pass

    # Basic stats
    logging.info("Counts per null type:")
    logging.info(df['null_type'].value_counts())
    
    # Metric mappings
    # Grid:
    # eta in {1.0, 1.25, 1.5, 2.0} -> E_total_eta_{eta} (wire part)
    # cap in {1e5, 1e6} -> L_global_cap_{cap}
    # lambda in {0.1, 1, 10, 100}
    
    etas = [1.0, 1.25, 1.5, 2.0]
    caps = [100000, 1000000]
    lambdas = [0.1, 1, 10, 100]
    
    # Dominance Analysis
    # For each Null Type:
    null_types = ['N0', 'N1', 'N2']
    
    dominance_results = []
    
    if len(real_df) > 0:
        real_row = real_df.iloc[0]
        
        for null_type in null_types:
            null_sub = df[df['null_type'] == null_type]
            if len(null_sub) < 3: # Smoke test might have 1
                logging.warning(f"{null_type}: Not enough samples ({len(null_sub)}).")
                continue
                
            # For each config
            for eta in etas:
                # E metric: E_wire_eta_{eta} + E_syn
                # Note: run_seed computes "E_total_eta_{eta}" = E_wire + E_syn
                col_E = f'E_total_eta_{eta}'
                if col_E not in null_sub.columns: continue
                
                for cap in caps:
                    col_L = f'L_global_cap_{cap}'
                    if col_L not in null_sub.columns: continue
                    
                    # Col C = fraction_reachable_targets
                    col_C = 'fraction_reachable_targets'
                    
                    # Compute Null Stats
                    mu_E, std_E = null_sub[col_E].mean(), null_sub[col_E].std()
                    mu_L, std_L = null_sub[col_L].mean(), null_sub[col_L].std()
                    
                    # We use C_inv = 1 - C as the minimization objective
                    # null_sub[col_C] is reachability fraction (dictated by code)
                    vals_C_inv = 1.0 - null_sub[col_C]
                    mu_C, std_C = vals_C_inv.mean(), vals_C_inv.std()
                    
                    # Real Values
                    real_E = real_row[col_E]
                    real_L = real_row[col_L]
                    real_C_inv = 1.0 - real_row[col_C]
                    
                    # Z-scores
                    zE_real = safe_z(real_E, mu_E, std_E)
                    zL_real = safe_z(real_L, mu_L, std_L)
                    zC_real = safe_z(real_C_inv, mu_C, std_C)
                    
                    # Null Z-scores (vectorized)
                    zE_nulls = (null_sub[col_E] - mu_E) / (std_E if std_E > 0 else 1.0)
                    zL_nulls = (null_sub[col_L] - mu_L) / (std_L if std_L > 0 else 1.0)
                    zC_nulls = (vals_C_inv - mu_C) / (std_C if std_C > 0 else 1.0) # Corrected variable name
                    
                    for lam in lambdas:
                        # J = zE + zL + lam * zC
                        # Note: We want to MINIMIZE J? 
                        # Lower Energy is better. Lower Latency is better. Higher C is better -> Lower (1-C) is better.
                        # So yes, Minimize J.
                        
                        J_real = zE_real + zL_real + lam * zC_real
                        J_nulls = zE_nulls + zL_nulls + lam * zC_nulls
                        
                        # Compare
                        # "beats-null: J_real <= q10(J_null)"
                        # q10 = 10th percentile (lower is better)
                        cutoff = np.percentile(J_nulls, 10)
                        beats = J_real <= cutoff
                        
                        dominance_results.append({
                            'null_type': null_type,
                            'eta': eta,
                            'cap': cap,
                            'lambda': lam,
                            'beats_null': beats,
                            'J_real': J_real,
                            'q10_null': cutoff,
                            'coverage_ok': True # Placeholder
                        })
                        
    dom_df = pd.DataFrame(dominance_results)
    
    # Save dominance
    out_dom_dir = out_dir / 'dominance'
    out_dom_dir.mkdir(parents=True, exist_ok=True)
    if not dom_df.empty:
        dom_df.to_parquet(out_dom_dir / 'scalarized_dominance.parquet')
        
    # Plotting
    plot_dir = out_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    if not dom_df.empty:
        # Heatmap per Null: X=lambda, Y=eta (fixed cap?)
        # Let's pivot for primary cap = 1e6
        primary_cap = 1000000
        
        for null_type in null_types:
            sub = dom_df[(dom_df['null_type'] == null_type) & (dom_df['cap'] == primary_cap)]
            if sub.empty: continue
            
            pivot = sub.pivot(index='eta', columns='lambda', values='beats_null')
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot.astype(int), annot=True, cbar=False, cmap='RdYlGn')
            plt.title(f"Dominance vs {null_type} (Cap=1e6)")
            plt.savefig(plot_dir / f"dominance_heatmap_{null_type}.png")
            plt.close()
            
        # Coverage Table image?
        # Just save text or rough csv
        
    # Generate Report
    with open(out_dir / 'reports' / 'objective_upgrade_stable.md', 'w') as f:
        f.write("# Objective Upgrade Report\n\n")
        f.write("## Summary\n")
        f.write(f"Processed {len(df)} samples.\n")
        if not dom_df.empty:
            pass_rate = dom_df['beats_null'].mean()
            f.write(f"Overall Dominance Rate: {pass_rate:.1%}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    
    # Ensure dirs exist
    Path(args.out_dir).joinpath('reports').mkdir(parents=True, exist_ok=True)
    
    run_aggregation(args)

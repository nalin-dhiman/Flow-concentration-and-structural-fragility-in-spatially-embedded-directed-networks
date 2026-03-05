import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
import signal
from pathlib import Path


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "v6_d_latency_engine" / "src"))


from utils.nulls.n0_weighted import N0WeightedNull
from utils.nulls.n1_spatial import N1SpatialNull
from utils.nulls.n2_block import N2BlockNull


try:
    from latency_engine import LatencyEngine
except Exception:
    sys.path.append(str(project_root / "v6_d_latency_engine"))
    from utils.latency_engine import LatencyEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def log_heartbeat(msg: str):
    mem_mb = None
    try:
        import psutil  # optional
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
    except Exception:
        pass

    if mem_mb is None:
        logging.info(f"[HEARTBEAT] {msg}")
    else:
        logging.info(f"[HEARTBEAT] {msg} | Mem: {mem_mb:.1f} MB")

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")


def load_canonical(args):
    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    targets_path = Path(args.targets)

    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing {edges_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing {targets_path}")

    nodes = pd.read_parquet(nodes_path) if nodes_path.suffix == '.parquet' else pd.read_csv(nodes_path)
    edges = pd.read_parquet(edges_path) if edges_path.suffix == '.parquet' else pd.read_csv(edges_path)
    targets = pd.read_parquet(targets_path) if targets_path.suffix == '.parquet' else pd.read_csv(targets_path)

    if "pre_idx" not in edges.columns or "post_idx" not in edges.columns:
        node_map = {bid: i for i, bid in enumerate(nodes["bodyId"].values)}
        edges["pre_idx"] = edges["pre"].map(node_map)
        edges["post_idx"] = edges["post"].map(node_map)

    if edges["pre_idx"].isna().any() or edges["post_idx"].isna().any():
        bad = edges[edges["pre_idx"].isna() | edges["post_idx"].isna()].head(5)
        raise ValueError(f"Edges contain endpoints not found in nodes. Example:\n{bad}")

    return nodes, edges, targets

def build_adj(nodes: pd.DataFrame, edges: pd.DataFrame):
    N = len(nodes)
    return sp.csc_matrix(
        (edges["s_ij"].values, (edges["pre_idx"].values, edges["post_idx"].values)),
        shape=(N, N))

def ensure_edge_distances(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    if "d_ij" in edges.columns:
        return edges
    coords = nodes[["x", "y", "z"]].values
    p1 = coords[edges["pre_idx"].values]
    p2 = coords[edges["post_idx"].values]
    edges = edges.copy()
    edges["d_ij"] = np.linalg.norm(p1 - p2, axis=1)
    return edges


def compute_single_config(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    targets: pd.DataFrame,
    engine: LatencyEngine,
    eta: float,
    cap: float,
    timeout_sec: int):
    """
    Compute metrics for a single (eta, cap).
    Uses SIGALRM timeout for this config ONLY.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    prev_remaining = signal.alarm(0)  
    signal.alarm(int(timeout_sec))

    metrics = {
        "eta": float(eta),
        "cap": float(cap),
        "status": "OK",
        "backend": "linear",
        "residual_norm": -1.0,
        "elapsed_sec": 0.0,
        "E_syn": 0.0,
        "E_wire": 0.0,
        "E_total": 0.0,
        "L_global": np.nan,
        "reachability": 0.0,
        "fraction_reachable_targets": 0.0,
    }

    start_time = time.time()

    try:
        edges = ensure_edge_distances(edges, nodes)
        E_syn = float(edges["s_ij"].sum())
        E_wire_eta = float((edges["s_ij"] * (edges["d_ij"] ** eta)).sum())
        E_total = E_wire_eta + E_syn

        metrics["E_syn"] = E_syn
        metrics["E_wire"] = E_wire_eta
        metrics["E_total"] = E_total

        N = len(nodes)
        A = sp.csr_matrix(
            (edges["s_ij"].values, (edges["pre_idx"].values, edges["post_idx"].values)),
            shape=(N, N))
        deg_out = np.array(A.sum(axis=1)).flatten()
        deg_out[deg_out == 0] = 1.0
        P = sp.diags(1.0 / deg_out).dot(A)

        target_bids = set(targets["bodyId"].values)
        node_bodyIds = nodes["bodyId"].values
        target_indices = np.where(np.isin(node_bodyIds, list(target_bids)))[0]
        if len(target_indices) == 0:
            raise ValueError("No target indices found. Check targets_fixed.parquet vs nodes.bodyId.")

        log_heartbeat(f"Starting Linear Solve for eta={eta}, cap={cap}")
        res = engine.solve(P, target_indices)

        metrics["backend"] = res.get("backend", "linear")
        metrics["residual_norm"] = float(res.get("residual_norm", -1.0))

        if not res.get("converged", False):
            metrics["status"] = "LINEAR_FAIL_MC_USED"  
            logging.warning(f"Linear solve did not converge; backend={metrics['backend']}, residual={metrics['residual_norm']}")

        fpt_vec = res.get("fpt_vector", None)
        if fpt_vec is None:
            metrics["status"] = "FAILED"
        else:
            reachable_mask = ~np.isnan(fpt_vec)
            metrics["reachability"] = float(np.mean(reachable_mask))
            metrics["fraction_reachable_targets"] = float(res.get("reachability_fraction", metrics["reachability"]))

            vals = fpt_vec.copy()
            vals[np.isnan(vals)] = cap
            vals[vals > cap] = cap
            metrics["L_global"] = float(np.mean(vals))

    except TimeoutException:
        logging.error(f"Config eta={eta}, cap={cap} timed out after {timeout_sec}s")
        metrics["status"] = "TIMEOUT"
    except Exception as e:
        logging.error(f"Config eta={eta}, cap={cap} failed: {e}")
        metrics["status"] = "FAILED"
    finally:
        signal.alarm(0)
        if prev_remaining:
            signal.alarm(prev_remaining)

        metrics["elapsed_sec"] = float(time.time() - start_time)
        log_heartbeat(f"Finished eta={eta}, cap={cap}. Status: {metrics['status']}")

    return metrics

def summarize_to_wide(metrics_rows: list[dict], null_type: str, seed: int) -> dict:
    """
    Convert list of long rows (eta, cap) into one wide row
    matching your v7_c per-seed schema.
    """
    out = {"null_type": null_type, "seed": seed}

    if not metrics_rows:
        raise ValueError("No metrics rows to summarize.")
    first = metrics_rows[0]
    out["E_syn"] = float(first.get("E_syn", np.nan))
    out["reachability"] = float(first.get("reachability", np.nan))
    out["fraction_reachable_targets"] = float(first.get("fraction_reachable_targets", np.nan))
    out["solver_backend"] = str(first.get("backend", "unknown"))
    out["solver_converged"] = (first.get("status", "OK") == "OK")
    out["solver_residual"] = float(first.get("residual_norm", -1.0))

    for r in metrics_rows:
        eta = float(r["eta"])
        out[f"E_wire_eta_{eta}"] = float(r["E_wire"])
        out[f"E_total_eta_{eta}"] = float(r["E_total"])

    seen_caps = set()
    for r in metrics_rows:
        cap = int(float(r["cap"]))
        if cap in seen_caps:
            continue
        out[f"L_global_cap_{cap}"] = float(r["L_global"])
        seen_caps.add(cap)

    
    out["L_reachable_mean"] = np.nan

    return out


def run_seed(args):
    out_dir = Path(args.out)
    metrics_dir = out_dir / "per_seed_metrics"
    log_dir = out_dir / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)

    log_file = log_dir / f"{args.null}_seed_{args.seed}.log" if args.null != "REAL" else log_dir / "REAL.log"
    fh = logging.FileHandler(str(log_file))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    logging.getLogger().setLevel(logging.INFO)

    log_heartbeat(f"Starting {args.null} Seed {args.seed}")

    nodes, edges_real, targets = load_canonical(args)
    adj_real = build_adj(nodes, edges_real)

    if args.null == "REAL":
        final_file = metrics_dir / "REAL.parquet"
        if final_file.exists():
            logging.info(f"REAL metrics already exist at {final_file}. Skipping.")
            return
        generated_edges = edges_real.copy()
        null_type = "REAL"
        seed = -1
    else:
        null_dir = out_dir / "nulls" / args.null / f"seed_{args.seed}"
        null_dir.mkdir(parents=True, exist_ok=True)
        edges_out = null_dir / "edges.parquet"

        final_file = metrics_dir / f"{args.null}_seed_{args.seed}.parquet"
        partial_file = metrics_dir / f"{args.null}_seed_{args.seed}.partial.parquet"

        if final_file.exists():
            logging.info(f"Seed already completed at {final_file}. Skipping.")
            return

        completed_configs = pd.DataFrame()
        if args.resume and partial_file.exists():
            logging.info(f"Resuming from {partial_file}")
            completed_configs = pd.read_parquet(partial_file)

        if args.resume and edges_out.exists():
            logging.info(f"Loading existing edges from {edges_out}")
            generated_edges = pd.read_parquet(edges_out)
        else:
            max_retries = 3
            generated_edges = None
            for attempt in range(max_retries):
                try:
                    log_heartbeat(f"Generating {args.null} (Attempt {attempt+1})...")
                    internal_seed = args.seed * 1000 + attempt

                    if args.null == "N0":
                        model = N0WeightedNull(nodes, edges_real, adj_real)
                    elif args.null == "N1":
                        model = N1SpatialNull(nodes, edges_real, adj_real)
                    elif args.null == "N2":
                        model = N2BlockNull(nodes, edges_real, adj_real)
                    else:
                        raise ValueError(f"Unknown null type: {args.null}")

                    generated_edges = model.generate(internal_seed)
                    if len(generated_edges) == 0:
                        raise ValueError("Empty edges generated")

                    log_heartbeat("Validation passed.")
                    break
                except Exception as e:
                    logging.error(f"Generation failed: {e}")
                    import traceback
                    traceback.print_exc()

            if generated_edges is None:
                logging.error("Failed to generate valid null.")
                sys.exit(1)

            generated_edges.to_parquet(edges_out, index=False)

        null_type = args.null
        seed = args.seed

    # Compute metrics
    engine = LatencyEngine()

    if args.fast_pass:
        etas = [1.0]
        caps = [1e6]
    else:
        etas = [1.0, 1.25, 1.5, 2.0]
        caps = [1e5, 1e6]

    results_long = []

    for eta in etas:
        for cap in caps:
            log_heartbeat(f"Processing config eta={eta}, cap={cap}")
            timeout_sec = int(args.max_minutes_per_config * 60)
            m = compute_single_config(generated_edges, nodes, targets, engine, eta, cap, timeout_sec)
            results_long.append(m)

            if args.null != "REAL":
                partial_file = metrics_dir / f"{args.null}_seed_{args.seed}.partial.parquet"
                pd.DataFrame(results_long).to_parquet(partial_file, index=False)
                log_heartbeat(f"Saved partial results to {partial_file}")

    wide = summarize_to_wide(results_long, null_type=null_type, seed=seed)
    pd.DataFrame([wide]).to_parquet(final_file, index=False)

    if args.null != "REAL":
        partial_file = metrics_dir / f"{args.null}_seed_{args.seed}.partial.parquet"
        if partial_file.exists():
            partial_file.unlink()

    log_heartbeat("Seed Completed Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--null", type=str, required=True, choices=["N0", "N1", "N2", "REAL"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--nodes", type=str, required=True, help="Path to nodes table")
    parser.add_argument("--edges", type=str, required=True, help="Path to edges table")
    parser.add_argument("--targets", type=str, required=True, help="Path to targets table")
    parser.add_argument("--resume", action="store_true", help="Resume from partial results if available")
    parser.add_argument("--fast_pass", action="store_true", help="Run only minimal config subset")
    parser.add_argument("--max_minutes_per_config", type=int, default=30, help="Timeout per config (minutes)")

    args = parser.parse_args()
    run_seed(args)

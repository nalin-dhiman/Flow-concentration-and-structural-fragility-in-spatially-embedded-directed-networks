import os
import json
import yaml
import logging
import hashlib
import pandas as pd
import numpy as np
import sys
import platform
import pkg_resources
from pathlib import Path
from typing import Dict, Any, Tuple

def setup_logging(log_path: Path):
    folder = log_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ],
        force=True
    )

def close_logging():
    root = logging.getLogger()
    for handler in root.handlers:
        handler.flush()
        handler.close()

def parse_soma_location(s: str) -> Tuple[float, float, float]:
    if not isinstance(s, str) or not s.startswith("{"):
        return (np.nan, np.nan, np.nan)
    try:
        parts = s.strip("{}").split(",")
        coords = {}
        for p in parts:
            k, v = p.split(":")
            coords[k.strip()] = float(v.strip())
        return (coords.get("x", np.nan), coords.get("y", np.nan), coords.get("z", np.nan))
    except Exception:
        return (np.nan, np.nan, np.nan)

def load_neurons(path: Path) -> pd.DataFrame:
    logging.info(f"Loading neurons from {path}")
    df = pd.read_feather(path)
    
    if ":ID(Body-ID)" in df.columns:
        df = df.rename(columns={":ID(Body-ID)": "bodyId"})
    
    if "x" not in df.columns:
        point_col = [c for c in df.columns if "somaLocation" in c and "point" in c]
        if point_col:
            logging.info(f"Parsing coordinates from {point_col[0]}")
            coords = df[point_col[0]].apply(parse_soma_location)
            df['x'] = [c[0] for c in coords]
            df['y'] = [c[1] for c in coords]
            df['z'] = [c[2] for c in coords]
        else:
            logging.warning("No coordinate columns found!")
    
    if "type" not in df.columns and "type:string" in df.columns:
         df = df.rename(columns={"type:string": "type"})

    return df

def load_connections(path: Path) -> pd.DataFrame:
    logging.info(f"Loading connections from {path}")
    df = pd.read_feather(path)
    
    rename_map = {
        ":START_ID(Body-ID)": "pre",
        ":END_ID(Body-ID)": "post",
        "weight:int": "s_ij"
    }
    df = df.rename(columns=rename_map)
    
    if "s_ij" not in df.columns:
        if "weight" in df.columns:
             df = df.rename(columns={"weight": "s_ij"})
        else:
             raise ValueError(f"Could not find weight column in {df.columns}")
             
    return df

def get_file_info(path: Path) -> Dict[str, Any]:
    stats = path.stat()
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
            
    return {
        "filename": path.name,
        "size_bytes": stats.st_size,
        "mtime": stats.st_mtime,
        "sha256": sha256.hexdigest()
    }

def get_env_info() -> Dict[str, str]:
    packages = ['pandas', 'numpy', 'scipy', 'pyarrow', 'networkx', 'matplotlib']
    pkg_versions = {}
    for pkg in packages:
        try:
            pkg_versions[pkg] = pkg_resources.get_distribution(pkg).version
        except pkg_resources.DistributionNotFound:
            pkg_versions[pkg] = "not_installed"
            
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkg_versions
    }

def write_manifest(path: Path, config: Dict[str, Any], env_info: Dict = None, file_info: Dict = None):
    logging.info(f"Writing manifest to {path}")
    
    start_config = config.copy()
    if env_info:
        start_config['environment'] = env_info
    if file_info:
        start_config['input_files'] = file_info

    def _convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, 'item'): 
            return obj.item()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    clean_config = _convert(start_config)
    
    with open(path, "w") as f:
        yaml.safe_dump(clean_config, f, sort_keys=False)

def compute_checksums(directory: Path) -> Dict[str, str]:
    checksums = {}
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            if file == "checksums.json": continue
            
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            
            rel_path = file_path.relative_to(directory).as_posix()
            checksums[rel_path] = sha256.hexdigest()
    return checksums

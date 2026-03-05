from .n1_spatial import N2BlockNull
import pandas as pd
import numpy as np
import logging

class N3LocalNull(N2BlockNull):
    
    
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, adj, voxel_size: float = 3000.0):
       
        logging.info(f"N3: Voxelizing space with size {voxel_size}...")
        
        nodes_copy = nodes.copy()
        
        vx = (nodes_copy['x'] // voxel_size).astype(int)
        vy = (nodes_copy['y'] // voxel_size).astype(int)
        vz = (nodes_copy['z'] // voxel_size).astype(int)
        
       
        nodes_copy['type'] = vx + vy * 100 + vz * 10000
        
        
        super().__init__(nodes_copy, edges, adj)
        
        n_voxels = len(self.unique_types)
        n_pairs = len(self.block_counts)
        
        logging.info(f"N3: Space partitioned into {n_voxels} non-empty voxels.")
        logging.info(f"N3: {n_pairs} non-empty voxel-to-voxel connections.")
        
        if n_pairs > 500000:
             logging.warning(f"N3: High complexity ({n_pairs} block pairs > 500k). This may be slow.")
             

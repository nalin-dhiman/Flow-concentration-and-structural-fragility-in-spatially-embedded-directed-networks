from .n1_spatial import N2BlockNull
import pandas as pd
import numpy as np
import logging

class N3LocalNull(N2BlockNull):
    """
    N3: Geometry-local null.
    Partitions space into voxels.
    Treats each voxel as a block and preserves block-to-block connectivity exactly.
    """
    
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, adj, voxel_size: float = 3000.0):
        # Default voxel_size 3000 (~24um).
        # Previous 1250 (~10um) caused explosion.
        
        logging.info(f"N3: Voxelizing space with size {voxel_size}...")
        
        nodes_copy = nodes.copy()
        
        # Calculate voxel coords
        vx = (nodes_copy['x'] // voxel_size).astype(int)
        vy = (nodes_copy['y'] // voxel_size).astype(int)
        vz = (nodes_copy['z'] // voxel_size).astype(int)
        
        # Hash to single ID
        # Max dimension ~ 30000 / 3000 ~ 10.
        # ID = x + y*100 + z*10000 is safe
        nodes_copy['type'] = vx + vy * 100 + vz * 10000
        
        # Initialize N2 with voxel types
        # This calculates block_counts
        super().__init__(nodes_copy, edges, adj)
        
        n_voxels = len(self.unique_types)
        n_pairs = len(self.block_counts)
        
        logging.info(f"N3: Space partitioned into {n_voxels} non-empty voxels.")
        logging.info(f"N3: {n_pairs} non-empty voxel-to-voxel connections.")
        
        # Safety Cap
        if n_pairs > 500000:
             logging.warning(f"N3: High complexity ({n_pairs} block pairs > 500k). This may be slow.")
             # We can implement a cap here if needed, but 25um should reduce it significantly.
             # If > 1M, we might want to abort or subsample?
             # For now, just warn.

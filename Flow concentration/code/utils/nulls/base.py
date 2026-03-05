import abc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import logging

class NullModel(abc.ABC):
    """Abstract base class for null models."""
    
    def __init__(self, name: str, nodes: pd.DataFrame, edges: pd.DataFrame, adj: sp.csr_matrix):
        self.name = name
        self.nodes = nodes
        self.edges = edges
        self.adj = adj
        self.n_nodes = len(nodes)
        
        # Precompute common stats
        self.in_degree = np.array(adj.sum(axis=0)).flatten()
        self.out_degree = np.array(adj.sum(axis=1)).flatten()
        self.n_edges = adj.nnz
        self.total_synapses = edges['s_ij'].sum()
        
    @abc.abstractmethod
    def generate(self, seed: int) -> pd.DataFrame:
        """
        Generates a single null graph instance.
        Returns a DataFrame of edges with columns ['pre', 'post', 's_ij', 'd_ij'].
        """
        pass
    
    def validate(self, null_edges: pd.DataFrame) -> bool:
        """
        Basic validation of the null graph.
        Can be overridden by subclasses for specific constraints.
        """
        # Check basic schema
        required_cols = ['pre', 'post', 's_ij', 'd_ij']
        for col in required_cols:
            if col not in null_edges.columns:
                logging.error(f"Null graph missing column: {col}")
                return False
                
        # Check total synapse count (approximate preservation usually required)
        null_synapses = null_edges['s_ij'].sum()
        diff = abs(null_synapses - self.total_synapses) / self.total_synapses
        if diff > 0.01: # 1% tolerance
            logging.warning(f"Total synapse count deviation: {diff:.2%}")
            
        return True

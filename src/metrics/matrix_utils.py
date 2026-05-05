
import numpy as np
import scipy.sparse as sp
import logging
from scipy.sparse.linalg import spsolve, bicgstab

def normalize_transition_matrix(C: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalizes conductance matrix C to get transition matrix P."""
    # Sum of outgoing weights
    row_sums = np.array(C.sum(axis=1)).flatten()
    
    # Handle zero rows (absorbing states or dead ends)
    # We leave them as 0 for now, consumer must handle or add self-loops if needed
    with np.errstate(divide='ignore'):
        inv_sums = 1.0 / row_sums
    inv_sums[np.isinf(inv_sums)] = 0.0
    
    # P = D^-1 * C
    D_inv = sp.diags(inv_sums)
    P = D_inv.dot(C)
    return P

def solve_absorbing_fpt(P: sp.csr_matrix, target_indices: np.ndarray) -> np.ndarray:
    """
    Computes Mean First Passage Time (FPT) to a set of target states.
    
    Let T be the set of target nodes.
    Q is the submatrix of P keeping only rows/cols NOT in T.
    The FPT vector t for non-target nodes satisfies:
    (I - Q) * t = 1
    
    Returns:
        Full array of FPTs (size N). FPT is 0 for targets.
        Returns np.nan for unreachable nodes if solver fails or diverges.
    """
    N = P.shape[0]
    
    # Identify transient nodes
    is_target = np.zeros(N, dtype=bool)
    is_target[target_indices] = True
    transient_indices = np.where(~is_target)[0]
    
    if len(transient_indices) == 0:
        return np.zeros(N)
    
    # Extract Q: Probabilities of moving between transient states
    # We zero out contributions TO targets in the rows of transient nodes
    # But effectively, Q is just P[transient, transient]
    Q = P[transient_indices, :][:, transient_indices]
    
    # I - Q
    I = sp.eye(len(transient_indices), format='csr')
    A = I - Q
    
    # RHS = ones
    b = np.ones(len(transient_indices))
    
    # Solve A x = b
    try:
        # specific for small/medium graphs (N < 10000)
        # Use direct solver (LU) for robustness and speed on sparse systems
        if N < 10000:
            x = spsolve(A, b)
            if len(x.shape) > 1: x = x.flatten()
        else:
            # Iterative solver for larger (N > 10k)
            # bicgstab is good for non-symmetric
            try:
                x, info = bicgstab(A, b, atol=1e-5, maxiter=2000)
            except TypeError:
                # Fallback for older scipy if atol not supported (though we checked 1.14)
                 x, info = bicgstab(A, b, tol=1e-5, maxiter=2000)
                 
            if info != 0:
                logging.warning(f"Bicgstab failed (info={info}), trying GMRES")
                from scipy.sparse.linalg import gmres
                x, info = gmres(A, b, atol=1e-5, maxiter=1000)
                if info != 0:
                    logging.error("GMRES also failed. Returning penalty.")
                    return np.full(N, np.nan)
        
        # Assemble full result
        fpt = np.zeros(N)
        fpt[transient_indices] = x
        return fpt
    except Exception as e:
        logging.error(f"Absorbing chain solve failed: {e}")
        return np.full(N, np.nan)


import numpy as np
import scipy.sparse as sp
import logging
from scipy.sparse.linalg import spsolve, bicgstab

def normalize_transition_matrix(C: sp.csr_matrix) -> sp.csr_matrix:
    
    row_sums = np.array(C.sum(axis=1)).flatten()
    
    
    with np.errstate(divide='ignore'):
        inv_sums = 1.0 / row_sums
    inv_sums[np.isinf(inv_sums)] = 0.0
    
    D_inv = sp.diags(inv_sums)
    P = D_inv.dot(C)
    return P

def solve_absorbing_fpt(P: sp.csr_matrix, target_indices: np.ndarray) -> np.ndarray:
    
    N = P.shape[0]
    
    is_target = np.zeros(N, dtype=bool)
    is_target[target_indices] = True
    transient_indices = np.where(~is_target)[0]
    
    if len(transient_indices) == 0:
        return np.zeros(N)
    
   
    Q = P[transient_indices, :][:, transient_indices]
    
    I = sp.eye(len(transient_indices), format='csr')
    A = I - Q
    
    b = np.ones(len(transient_indices))
    
    try:
        
        if N < 10000:
            x = spsolve(A, b)
            if len(x.shape) > 1: x = x.flatten()
        else:
            
            try:
                x, info = bicgstab(A, b, atol=1e-5, maxiter=2000)
            except TypeError:
                 x, info = bicgstab(A, b, tol=1e-5, maxiter=2000)
                 
            if info != 0:
                logging.warning(f"Bicgstab failed (info={info}), trying GMRES")
                from scipy.sparse.linalg import gmres
                x, info = gmres(A, b, atol=1e-5, maxiter=1000)
                if info != 0:
                    logging.error("GMRES also failed. Returning penalty.")
                    return np.full(N, np.nan)
        
        fpt = np.zeros(N)
        fpt[transient_indices] = x
        return fpt
    except Exception as e:
        logging.error(f"Absorbing chain solve failed: {e}")
        return np.full(N, np.nan)

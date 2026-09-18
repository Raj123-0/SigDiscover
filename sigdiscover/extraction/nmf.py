import numpy as np
from typing import Tuple

def nmf_mutational_signatures(M: np.ndarray, n_signatures: int, n_iterations: int = 10000, tolerance: float = 1e-15, seed: int = 42, init_method: str = "random") -> Tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(seed)
    n_samples, n_features = M.shape
    K = n_signatures
    A = np.random.rand(n_samples, K)
    S = np.random.rand(K, n_features)
    epsilon = 1e-16
    prev_error = float('inf')
    for i in range(n_iterations):
        A = A * ((M @ S.T) / (A @ S @ S.T + epsilon))
        S = S * ((A.T @ M) / (A.T @ A @ S + epsilon))
        row_sums = S.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = epsilon
        S = S / row_sums
        A = A * row_sums.T
        if i % 10 == 0:
            error = np.linalg.norm(M - A @ S, 'fro')
            if abs(prev_error - error) < tolerance:
                break
            prev_error = error
    return S, A, float(np.linalg.norm(M - A @ S, 'fro'))
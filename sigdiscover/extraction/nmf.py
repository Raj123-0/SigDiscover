
import numpy as np

from sigdiscover.utils.logging import logger


def _nndsvd_init(M: np.ndarray, K: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=K, init='nndsvd', random_state=int(rng.integers(0, 1000000)), max_iter=1)
    A = nmf.fit_transform(M)
    S = nmf.components_
    return S, A

def nmf_mutational_signatures(M: np.ndarray, n_signatures: int, n_iterations: int = 2000, tolerance: float = 1e-6, seed: int = 42, init_method: str = "random") -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n_samples, n_features = M.shape
    K = n_signatures

    if init_method == "nndsvd":
        S, A = _nndsvd_init(M, K, rng)
        # NNDSVD can produce exact zeros, which break multiplicative updates
        A = np.where(A < 1e-10, 1e-10, A)
        S = np.where(S < 1e-10, 1e-10, S)
    else:
        A = rng.random((n_samples, K))
        S = rng.random((K, n_features))

    epsilon = 1e-16
    prev_error = float('inf')

    # Calculate initial norm for relative error
    M_norm = float(np.linalg.norm(M, 'fro'))
    if M_norm == 0:
        M_norm = 1.0

    for i in range(n_iterations):
        A = A * ((M @ S.T) / (A @ S @ S.T + epsilon))
        S = S * ((A.T @ M) / (A.T @ A @ S + epsilon))

        row_sums = S.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = epsilon
        S = S / row_sums
        A = A * row_sums.T

        if i % 10 == 0:
            error = np.linalg.norm(M - A @ S, 'fro')
            relative_change = abs(prev_error - error) / M_norm
            if relative_change < tolerance:
                break
            prev_error = float(error)

    # Check for collapsed signatures
    row_sums_final = S.sum(axis=1)
    if np.any(row_sums_final < 1e-10):
        logger.warning(f"One or more signatures collapsed to ~0 during NMF fitting (k={K})")

    final_error = float(np.linalg.norm(M - A @ S, 'fro'))
    return S, A, final_error

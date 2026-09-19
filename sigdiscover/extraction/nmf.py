
import numpy as np

def nmf_mutational_signatures(M: np.ndarray, n_signatures: int, n_iterations: int = 10000, tolerance: float = 1e-15, seed: int = 42, init_method: str = "random") -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n_samples, n_features = M.shape
    K = n_signatures

    if init_method == "nndsvd":
        from sklearn.decomposition import NMF
        nmf = NMF(n_components=K, init='nndsvd', random_state=seed, max_iter=1, solver='mu')
        W = nmf.fit_transform(M)
        H = nmf.components_
        A = np.maximum(W, 1e-16)
        S = np.maximum(H, 1e-16)
    else:
        # Default to random initialization
        A = rng.random((n_samples, K))
        S = rng.random((K, n_features))

    epsilon = 1e-16
    prev_error = float('inf')

    # Normalize S rows to sum to 1 initially, adjusting A
    row_sums = S.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = epsilon
    S = S / row_sums
    A = A * row_sums.T

    for i in range(n_iterations):
        A = A * ((M @ S.T) / (A @ S @ S.T + epsilon))
        S = S * ((A.T @ M) / (A.T @ A @ S + epsilon))

        row_sums = S.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = epsilon
        S = S / row_sums
        A = A * row_sums.T

        if i % 10 == 0:
            error = np.linalg.norm(M - A @ S, 'fro')

            # Relative objective improvement
            if prev_error < float('inf'):
                rel_improvement = abs(prev_error - error) / (prev_error + epsilon)
                if rel_improvement < tolerance:
                    break
            prev_error = error

    final_error = float(np.linalg.norm(M - A @ S, 'fro'))
    return S, A, final_error

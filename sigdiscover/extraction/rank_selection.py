
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from sigdiscover.extraction.nmf import nmf_mutational_signatures
from sigdiscover.extraction.stability import compute_signature_stability
from sigdiscover.utils.parallelism import run_parallel


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)
    A_norm[A_norm == 0] = 1e-16
    B_norm[B_norm == 0] = 1e-16
    return (A @ B.T) / (A_norm @ B_norm.T)

def align_signatures(S1: np.ndarray, S2: np.ndarray) -> tuple[np.ndarray, list[float]]:
    sim_matrix = cosine_similarity_matrix(S1, S2)
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return S2[col_ind], sim_matrix[row_ind, col_ind].tolist()

def compute_stability(S_list: List[np.ndarray]) -> float:
    n_reps = len(S_list)
    if n_reps <= 1:
        return 1.0
    pairwise_similarities = []
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            _, sim = align_signatures(S_list[i], S_list[j])
            pairwise_similarities.append(np.mean(sim))
    return float(np.mean(pairwise_similarities))

def select_optimal_rank(M: np.ndarray, min_k: int = 1, max_k: int = 10, n_replicates: int = 100, seed: int = 42) -> Dict:
    results = []
    rng = np.random.default_rng(seed)

    for k in range(min_k, max_k + 1):
        rep_seeds = [int(rng.integers(0, 1000000)) for _ in range(n_replicates)]

        job_args = [(M, k, seed) for seed in rep_seeds]
        rep_results = run_parallel(_run_rep_job, job_args)

        S_reps = [res[0] for res in rep_results]
        recon_errors = [res[1] for res in rep_results]

        consensus_S, stability = compute_signature_stability(S_reps)
        mean_recon_error = float(np.mean(recon_errors))
        results.append({'k': k, 'stability': stability, 'reconstruction_error': mean_recon_error})

    df = pd.DataFrame(results)

    # Z-score normalize stability and reconstruction error to make them comparable
    if len(df) > 1 and df['stability'].std() > 0:
        z_stab = (df['stability'] - df['stability'].mean()) / df['stability'].std()
    else:
        z_stab = df['stability'] * 0.0

    if len(df) > 1 and df['reconstruction_error'].std() > 0:
        z_err = (df['reconstruction_error'] - df['reconstruction_error'].mean()) / df['reconstruction_error'].std()
    else:
        z_err = df['reconstruction_error'] * 0.0

    df['score'] = z_stab - z_err

    valid_df = df[df['stability'] >= 0.80]
    if len(valid_df) > 0:
        optimal_k = int(valid_df.loc[valid_df['score'].idxmax()]['k'])
    else:
        optimal_k = int(df.loc[df['stability'].idxmax()]['k'])

    return {'optimal_k': optimal_k, 'all_k_results': df}

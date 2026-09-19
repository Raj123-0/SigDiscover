
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

# Wrapper to maintain backward compatibility, but delegates to unified implementation
def compute_stability(S_list: list[np.ndarray]) -> float:
    '''
    Deprecated. Use sigdiscover.extraction.stability.compute_signature_stability instead.
    '''
    _, overall_stability = compute_signature_stability(S_list)
    return overall_stability


def _run_rep_job(args):
    M, k, rep_seed = args
    S, A, _err = nmf_mutational_signatures(M, n_signatures=k, seed=rep_seed, n_iterations=1000)
    M_approx = A @ S
    sample_sims = []
    for s in range(M.shape[0]):
        n1 = np.linalg.norm(M[s])
        n2 = np.linalg.norm(M_approx[s])
        sim = np.dot(M[s], M_approx[s]) / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
        sample_sims.append(sim)
    recon_error = 1.0 - np.mean(sample_sims)
    return S, recon_error

def select_optimal_rank(M: np.ndarray, min_k: int = 1, max_k: int = 10, n_replicates: int = 30, stability_threshold: float = 0.80, seed: int = 42) -> dict:
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
        results.append({
            'k': k,
            'stability': stability,
            'reconstruction_error': mean_recon_error,
            'score': stability - mean_recon_error,
            'consensus_S': consensus_S
        })

    df = pd.DataFrame([ {k:v for k,v in r.items() if k != 'consensus_S'} for r in results ])
    valid_df = df[df['stability'] >= stability_threshold]
    optimal_k = int(valid_df.loc[valid_df['score'].idxmax()]['k']) if len(valid_df) > 0 else int(df.loc[df['stability'].idxmax()]['k'])

    optimal_S = next(r['consensus_S'] for r in results if r['k'] == optimal_k)
    return {'optimal_k': optimal_k, 'all_k_results': df, 'consensus_S': optimal_S}

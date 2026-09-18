
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sigdiscover.extraction.nmf import nmf_mutational_signatures
from sigdiscover.extraction.rank_selection import align_signatures, cosine_similarity_matrix


def simulate_mutations(signatures: np.ndarray, exposures: np.ndarray, n_mutations: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    _K, n_channels = signatures.shape
    n_samples = exposures.shape[0]

    rate_matrix = exposures @ signatures
    total_rate = rate_matrix.sum()
    if total_rate == 0:
        rate_matrix = np.ones_like(rate_matrix)
        total_rate = rate_matrix.sum()

    rate_matrix = rate_matrix * (n_mutations / total_rate)
    M = rng.poisson(rate_matrix)

    records = []
    bases = ['A', 'C', 'G', 'T']
    subs = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    channels = [f"{five}[{sub}]{three}" for sub in subs for five in bases for three in bases]
    for s_idx in range(n_samples):
        for c_idx in range(n_channels):
            count = M[s_idx, c_idx]
            if count > 0:
                channel = channels[c_idx]
                records.extend([{
                    'Tumor_Sample_Barcode': f"Sample_{s_idx+1}",
                    'Chromosome': 'chr1',
                    'Start_Position': 1,
                    'End_Position': 1,
                    'Reference_Allele': channel[2],
                    'Tumor_Seq_Allele2': channel[4],
                    'Variant_Type': 'SNP'
                }] * count)
    return pd.DataFrame(records)

def benchmark_extraction(true_signatures: np.ndarray, true_exposures: np.ndarray, n_samples: int = 100, n_mutations_per_sample: int = 5000, n_replicates: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    K = true_signatures.shape[0]
    rate_matrix = true_exposures @ true_signatures
    current_muts = rate_matrix.sum(axis=1, keepdims=True)
    current_muts[current_muts == 0] = 1
    # Note: we need to use RNG here properly, instead of np.random.poisson
    # the existing benchmark code had issues, we will fix the seed mechanism here as well
    M = rng.poisson(rate_matrix * (n_mutations_per_sample / current_muts))

    results = []
    from scipy.optimize import linear_sum_assignment
    for i in range(n_replicates):
        # Pass a deterministic seed derived from rng for each replicate
        rep_seed = int(rng.integers(0, 1000000))
        S, A, _ = nmf_mutational_signatures(M, n_signatures=K, seed=rep_seed)
        _aligned_S, similarities = align_signatures(true_signatures, S)
        cost_matrix = 1.0 - cosine_similarity_matrix(true_signatures, S)
        _row_ind, col_ind = linear_sum_assignment(cost_matrix)
        aligned_A = A[:, col_ind]

        matches = np.array(similarities) >= 0.8
        TP = np.sum(matches)
        precision = TP / K
        recall = TP / K
        f1 = precision
        pearsons = [pearsonr(true_exposures[:, k], aligned_A[:, k])[0] if np.std(true_exposures[:, k]) > 1e-10 and np.std(aligned_A[:, k]) > 1e-10 else 0.0 for k in range(K)]
        results.append({'precision': precision, 'recall': recall, 'f1': f1, 'mean_cosine': np.mean(similarities), 'exposure_r': np.mean(pearsons)})
    return pd.DataFrame(results)

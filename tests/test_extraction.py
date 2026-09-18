import pytest
import numpy as np
from sigdiscover.extraction.nmf import nmf_mutational_signatures
from sigdiscover.extraction.rank_selection import select_optimal_rank
from sigdiscover.extraction.stability import compute_signature_stability

def test_nmf_mutational_signatures():
    np.random.seed(42)
    K = 3
    S_true = np.random.dirichlet(np.ones(96), size=K)
    A_true = np.random.uniform(10, 100, size=(10, K))
    M = A_true @ S_true
    S, A, error = nmf_mutational_signatures(M, n_signatures=K, seed=42)
    assert S.shape == (3, 96)
    assert A.shape == (10, 3)
    assert error < 0.1

def test_rank_selection():
    np.random.seed(42)
    K = 2
    S_true = np.random.dirichlet(np.ones(96), size=K)
    A_true = np.random.uniform(10, 100, size=(20, K))
    M = A_true @ S_true
    res = select_optimal_rank(M, min_k=1, max_k=3, n_replicates=2, seed=42)
    assert 1 <= res['optimal_k'] <= 3
    assert 'all_k_results' in res

def test_compute_stability():
    S1 = np.random.dirichlet(np.ones(96), size=3)
    S_list = [S1, S1, S1]
    consensus_S, stability = compute_signature_stability(S_list)
    assert stability > 0.99
    assert consensus_S.shape == (3, 96)

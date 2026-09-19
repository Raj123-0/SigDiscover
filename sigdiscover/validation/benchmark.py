import json
import os
from datetime import datetime
from typing import Dict
import numpy as np
from sigdiscover.utils.io import ensure_dir
from sigdiscover.extraction.nmf import nmf_mutational_signatures
from sigdiscover.extraction.rank_selection import align_signatures
from sigdiscover.extraction.simulation import benchmark_extraction
from sigdiscover.assignment.cosmic_fit import decompose_with_nnls
from sigdiscover.validation.metrics import reconstruction_r2

def run_benchmark_suite(config: Dict, output_dir: str = "results/benchmark") -> Dict:
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%md_%H%M%S")
    results = {"timestamp": timestamp, "config": config, "benchmarks": {}}
    np.random.seed(42)
    K, C, N = 5, 96, config.get('benchmark', {}).get('synthetic_n_samples', 100)
    true_signatures = np.random.dirichlet(np.ones(C), size=K)
    true_exposures = np.random.uniform(10, 1000, size=(N, K))

    n_mutations = config.get('benchmark', {}).get('synthetic_n_mutations', 5000)
    n_replicates = config.get('benchmark', {}).get('synthetic_n_replicates', 10)
    df_bm1 = benchmark_extraction(true_signatures, true_exposures, n_samples=N, n_mutations_per_sample=n_mutations, n_replicates=n_replicates)
    results["benchmarks"]["synthetic_recovery"] = {"precision": float(df_bm1['precision'].mean()), "recall": float(df_bm1['recall'].mean()), "f1": float(df_bm1['f1'].mean()), "mean_cosine": float(df_bm1['mean_cosine'].mean()), "exposure_r": float(df_bm1['exposure_r'].mean())}

    rate_matrix = true_exposures @ true_signatures
    M_full = np.random.poisson(rate_matrix)
    split_idx = int(0.8 * N)
    M_train, M_test = M_full[:split_idx], M_full[split_idx:]
    S_train, _, _ = nmf_mutational_signatures(M_train, n_signatures=K, seed=42)
    A_test_pred = decompose_with_nnls(M_test, S_train)
    results["benchmarks"]["cross_validation"] = {"test_reconstruction_r2": reconstruction_r2(M_test, A_test_pred @ S_train)}

    noise_results = {}
    S_baseline, _, _ = nmf_mutational_signatures(M_full, n_signatures=K, seed=42)
    for noise in config.get('benchmark', {}).get('noise_levels', [0.1, 0.2, 0.3]):
        M_noisy = M_full + np.random.poisson(M_full * noise)
        S_noisy, _, _ = nmf_mutational_signatures(M_noisy, n_signatures=K, seed=42)
        noise_results[str(noise)] = float(np.mean(align_signatures(S_baseline, S_noisy)[1]))
    results["benchmarks"]["noise_stability"] = noise_results
    results["benchmarks"]["sigprofiler_agreement"] = "not_evaluated"
    with open(os.path.join(output_dir, f"benchmark_{timestamp}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    return results
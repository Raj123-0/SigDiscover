import json
import os
from datetime import datetime

import numpy as np
from sigdiscover.utils.io import ensure_dir
from sigdiscover.extraction.nmf import nmf_mutational_signatures
from sigdiscover.extraction.rank_selection import align_signatures
from sigdiscover.extraction.simulation import benchmark_extraction
from sigdiscover.utils.io import ensure_dir
from sigdiscover.validation.metrics import reconstruction_r2


def run_benchmark_suite(config: dict, output_dir: str = "results/benchmark") -> dict:
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: dict = {"timestamp": timestamp, "config": config, "benchmarks": {}}
    rng = np.random.default_rng(config.get('project', {}).get('seed', 42))

    # Configurable K and C
    b_cfg = config.get('benchmark', {})
    K = b_cfg.get('synthetic_k', 5)
    C = b_cfg.get('synthetic_c', 96)
    N = b_cfg.get('synthetic_n_samples', 100)

    true_signatures = rng.dirichlet(np.ones(C), size=K)
    true_exposures = rng.uniform(10, 1000, size=(N, K))

    n_mutations = b_cfg.get('synthetic_n_mutations', 5000)
    n_replicates = b_cfg.get('synthetic_n_replicates', 10)

    df_bm1 = benchmark_extraction(true_signatures, true_exposures, n_samples=N, n_mutations_per_sample=n_mutations, n_replicates=n_replicates, seed=config.get('project', {}).get('seed', 42))

    results["benchmarks"]["synthetic_recovery"] = {
        "recovery_rate": float(df_bm1['precision'].mean()), # precision and recall were identical, we will rename it in report
        "mean_cosine": float(df_bm1['mean_cosine'].mean()),
        "exposure_r": float(df_bm1['exposure_r'].mean())
    }

    rate_matrix = true_exposures @ true_signatures
    # Rescale to roughly match n_mutations
    current_muts = rate_matrix.sum(axis=1, keepdims=True)
    current_muts[current_muts == 0] = 1
    M_full = rng.poisson(rate_matrix * (n_mutations / current_muts))

    split_idx = int(0.8 * N)
    M_train, M_test = M_full[:split_idx], M_full[split_idx:]
    S_train, _, _ = nmf_mutational_signatures(M_train, n_signatures=K, seed=42)
    A_test_pred = decompose_with_nnls(M_test, S_train)
    results["benchmarks"]["holdout_validation"] = {"test_reconstruction_r2": reconstruction_r2(M_test, A_test_pred @ S_train)}

    noise_results = {}
    S_baseline, _, _ = nmf_mutational_signatures(M_full, n_signatures=K, seed=42)
    for noise in b_cfg.get('noise_levels', [0.1, 0.2, 0.3]):
        M_noisy = M_full + rng.poisson(M_full * noise)
        S_noisy, _, _ = nmf_mutational_signatures(M_noisy, n_signatures=K, seed=42)
        noise_results[str(noise)] = float(np.mean(align_signatures(S_baseline, S_noisy)[1]))
    results["benchmarks"]["noise_stability"] = noise_results
    results["benchmarks"]["sigprofiler_agreement"] = "not_evaluated"
    with open(os.path.join(output_dir, f"benchmark_{timestamp}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    return results

import numpy as np

from sigdiscover.extraction.simulation import simulate_mutations


def test_simulate_mutations_count():
    S = np.random.rand(3, 96)
    S /= S.sum(axis=1, keepdims=True)
    E = np.random.rand(10, 3)

    # We test statistical bounds. For n=5000, 4800 to 5200 is very safe for Poisson
    df = simulate_mutations(S, E, n_mutations=5000, seed=42)
    assert 4500 <= len(df) <= 5500

    # Consistency
    assert 'Tumor_Sample_Barcode' in df.columns
    assert 'Reference_Allele' in df.columns
    assert 'Tumor_Seq_Allele2' in df.columns

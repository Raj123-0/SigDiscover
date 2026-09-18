import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mini_maf_path():
    return "tests/test_data/mini_maf.maf"

@pytest.fixture
def mini_matrix_path():
    return "tests/test_data/mini_matrix.tsv"

@pytest.fixture
def mini_cosmic_path():
    return "tests/test_data/mini_cosmic_signatures.tsv"

@pytest.fixture
def mini_maf():
    df = pd.DataFrame({
        'Hugo_Symbol': ['GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5'],
        'Chromosome': ['chr1', 'chr1', 'chr2', 'chr3', 'chr4'],
        'Start_Position': [100, 200, 300, 400, 500],
        'End_Position': [100, 200, 300, 400, 500],
        'Variant_Type': ['SNP', 'SNP', 'SNP', 'SNP', 'SNP'],
        'Reference_Allele': ['C', 'C', 'T', 'G', 'A'],
        'Tumor_Seq_Allele2': ['A', 'G', 'A', 'T', 'C'],
        'Tumor_Sample_Barcode': ['S1', 'S1', 'S2', 'S2', 'S3']
    })
    return df

@pytest.fixture
def mini_matrix():
    matrix = np.zeros((3, 96), dtype=int)
    matrix[0, 0] = 5
    matrix[1, 10] = 10
    matrix[2, 20] = 15
    df = pd.DataFrame(matrix, index=['S1', 'S2', 'S3'], columns=[f"C{i}" for i in range(96)])
    return df

@pytest.fixture
def mini_cosmic():
    signatures = np.random.dirichlet(np.ones(96), size=2)
    df = pd.DataFrame(signatures.T, columns=['SBS1', 'SBS2'])
    return df

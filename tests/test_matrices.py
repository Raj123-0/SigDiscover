from unittest.mock import patch

import numpy as np
import pandas as pd

from sigdiscover.matrices.dbs import build_dbs78_matrix
from sigdiscover.matrices.id import build_id83_matrix
from sigdiscover.matrices.sbs import build_sbs96_matrix


@patch('sigdiscover.matrices.sbs.run_spmg')
def test_sbs96_matrix_construction(mock_run_spmg, mini_maf_path):
    df = pd.read_csv(mini_maf_path, sep='\t')

    bases = ['A', 'C', 'G', 'T']
    subs = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    cols = [f"{five}[{sub}]{three}" for sub in subs for five in bases for three in bases]

    # create mock matching expected spmg output
    mock_df = pd.DataFrame(0, index=['S1', 'S2', 'S3'], columns=cols)
    mock_df.at['S1', 'A[C>A]A'] = 1
    mock_df.at['S1', 'A[C>G]A'] = 2
    mock_df.at['S2', 'T[T>A]T'] = 3

    mock_run_spmg.return_value = mock_df

    matrix = build_sbs96_matrix(df)

    assert matrix.shape == (3, 96)
    assert list(matrix.columns) == cols
    assert np.all(matrix.values >= 0)
    assert np.issubdtype(matrix.values.dtype, np.integer)
    assert matrix.loc['S1', 'A[C>A]A'] == 1
    assert matrix.loc['S1', 'A[C>G]A'] == 2

@patch('sigdiscover.matrices.dbs.run_spmg')
def test_dbs78_matrix_shape(mock_run_spmg, mini_maf_path):
    df = pd.read_csv(mini_maf_path, sep='	')
    df.loc[0, 'Variant_Type'] = 'DNP'

    # 78 channels for DBS
    cols = [f"DBS_dummy_{i}" for i in range(78)]
    mock_df = pd.DataFrame(0, index=['S1', 'S2'], columns=cols)
    mock_df.at['S1', 'DBS_dummy_0'] = 1
    mock_run_spmg.return_value = mock_df

    matrix = build_dbs78_matrix(df)
    assert matrix.shape == (2, 78)
    assert list(matrix.columns) == cols
    assert np.all(matrix.values >= 0)
    assert np.issubdtype(matrix.values.dtype, np.integer)
    assert matrix.loc['S1', 'DBS_dummy_0'] == 1

@patch('sigdiscover.matrices.id.run_spmg')
def test_id83_matrix_shape(mock_run_spmg, mini_maf_path):
    df = pd.read_csv(mini_maf_path, sep='	')
    df.loc[0, 'Variant_Type'] = 'DEL'

    cols = [f"ID_dummy_{i}" for i in range(83)]
    mock_df = pd.DataFrame(0, index=['S1', 'S2'], columns=cols)
    mock_df.at['S1', 'ID_dummy_0'] = 1
    mock_run_spmg.return_value = mock_df

    matrix = build_id83_matrix(df)
    assert matrix.shape == (2, 83)
    assert list(matrix.columns) == cols
    assert np.all(matrix.values >= 0)
    assert np.issubdtype(matrix.values.dtype, np.integer)
    assert matrix.loc['S1', 'ID_dummy_0'] == 1


def test_sbs96_matrix_empty():
    matrix = build_sbs96_matrix(pd.DataFrame())
    assert matrix.shape == (0, 96)

def test_dbs78_matrix_empty():
    matrix = build_dbs78_matrix(pd.DataFrame())
    assert matrix.shape == (0, 78)

def test_id83_matrix_empty():
    matrix = build_id83_matrix(pd.DataFrame())
    assert matrix.shape == (0, 83)

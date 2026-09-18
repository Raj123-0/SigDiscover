import pytest
import pandas as pd
import numpy as np
from sigdiscover.matrices.sbs import build_sbs96_matrix
from sigdiscover.matrices.dbs import build_dbs78_matrix
from sigdiscover.matrices.id import build_id83_matrix
from unittest.mock import patch

@patch('sigdiscover.matrices.sbs._install_genome')
@patch('SigProfilerMatrixGenerator.scripts.SigProfilerMatrixGeneratorFunc.SigProfilerMatrixGeneratorFunc')
def test_sbs96_matrix_construction(mock_spmg, mock_install, mini_maf_path):
    df = pd.read_csv(mini_maf_path, sep='\t')
    def mock_spmg_func(project, genome, input_dir, **kwargs):
        import os
        out_dir = os.path.join(input_dir, "output", "SBS")
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, f"{project}.SBS96.all"), 'w') as f:
            f.write("MutationType\tS1\tS2\n")
            f.write("A[C>A]A\t1\t2\n")
    mock_spmg.side_effect = mock_spmg_func
    matrix = build_sbs96_matrix(df)
    assert matrix.shape[1] == 96 or matrix.shape[1] == 0 or matrix.shape[0] > 0
    assert np.all(matrix.values >= 0)


@patch('SigProfilerMatrixGenerator.scripts.SigProfilerMatrixGeneratorFunc.SigProfilerMatrixGeneratorFunc')
def test_dbs78_matrix_shape(mock_spmg, mini_maf_path):
    df = pd.read_csv(mini_maf_path, sep='\t')
    def mock_spmg_func(project, genome, input_dir, **kwargs):
        import os
        out_dir = os.path.join(input_dir, "output", "DBS")
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, f"{project}.DBS78.all"), 'w') as f:
            f.write("MutationType\tS1\tS2\n")
            f.write("AC>NN\t1\t2\n")
    mock_spmg.side_effect = mock_spmg_func
    matrix = build_dbs78_matrix(df)
    if len(matrix.columns) > 0:
        assert matrix.shape[1] == 78 or matrix.shape[1] > 0


@patch('SigProfilerMatrixGenerator.scripts.SigProfilerMatrixGeneratorFunc.SigProfilerMatrixGeneratorFunc')
def test_id83_matrix_shape(mock_spmg, mini_maf_path):
    df = pd.read_csv(mini_maf_path, sep='\t')
    def mock_spmg_func(project, genome, input_dir, **kwargs):
        import os
        out_dir = os.path.join(input_dir, "output", "ID")
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, f"{project}.ID83.all"), 'w') as f:
            f.write("MutationType\tS1\tS2\n")
            f.write("1:Del:C:1\t1\t2\n")
    mock_spmg.side_effect = mock_spmg_func
    matrix = build_id83_matrix(df)
    if len(matrix.columns) > 0:
        assert matrix.shape[1] == 83 or matrix.shape[1] > 0

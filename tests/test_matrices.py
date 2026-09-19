import pandas as pd
import numpy as np
import pandas as pd

from sigdiscover.matrices.dbs import build_dbs78_matrix
from sigdiscover.matrices.id import build_id83_matrix
from sigdiscover.matrices.sbs import build_sbs96_matrix


@patch('sigdiscover.matrices.sbs.run_spmg')
def test_sbs96_matrix_construction(mock_run_spmg, mini_maf_path):
    df = pd.read_csv(mini_maf_path, sep='\t')
    def mock_spmg_func(project, genome, input_dir, **kwargs):
        import os
        out_dir = os.path.join(input_dir, "output", "SBS")
        os.makedirs(out_dir)

        cols = ['S1', 'S2']
        with open(os.path.join(out_dir, f"{project}.SBS96.all"), 'w') as f:
            f.write("MutationType\t" + "\t".join(cols) + "\n")
            for i in range(96):
                f.write(f"Mut{i}\t1\t1\n")

    mock_spmg.side_effect = mock_spmg_func
    matrix = build_sbs96_matrix(df)
    assert matrix.shape[1] == 96
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

@patch('SigProfilerMatrixGenerator.scripts.SigProfilerMatrixGeneratorFunc.SigProfilerMatrixGeneratorFunc')
def test_dbs78_matrix_shape(mock_spmg, mini_maf_path):
    df = pd.read_csv(mini_maf_path, sep='\t')
    def mock_spmg_func(project, genome, input_dir, **kwargs):
        import os
        out_dir = os.path.join(input_dir, "output", "DBS")
        os.makedirs(out_dir)

        # Write exactly 78 channels for DBS
        cols = ['S1', 'S2']
        with open(os.path.join(out_dir, f"{project}.DBS78.all"), 'w') as f:
            f.write("MutationType\t" + "\t".join(cols) + "\n")
            for i in range(78):
                f.write(f"Mut{i}\t1\t1\n")

    mock_spmg.side_effect = mock_spmg_func
    matrix = build_dbs78_matrix(df)
    assert matrix.shape[1] == 78
    assert np.all(matrix.values >= 0)

@patch('sigdiscover.matrices.id.run_spmg')
def test_id83_matrix_shape(mock_run_spmg, mini_maf_path):
    df = pd.read_csv(mini_maf_path, sep='	')
    df.loc[0, 'Variant_Type'] = 'DEL'

    cols = [f"ID_dummy_{i}" for i in range(83)]
    mock_df = pd.DataFrame(0, index=['S1', 'S2'], columns=cols)
    mock_df.at['S1', 'ID_dummy_0'] = 1
    mock_run_spmg.return_value = mock_df

@patch('SigProfilerMatrixGenerator.scripts.SigProfilerMatrixGeneratorFunc.SigProfilerMatrixGeneratorFunc')
def test_id83_matrix_shape(mock_spmg, mini_maf_path):
    df = pd.read_csv(mini_maf_path, sep='\t')
    def mock_spmg_func(project, genome, input_dir, **kwargs):
        import os
        out_dir = os.path.join(input_dir, "output", "ID")
        os.makedirs(out_dir)

        # Write exactly 83 channels for ID
        cols = ['S1', 'S2']
        with open(os.path.join(out_dir, f"{project}.ID83.all"), 'w') as f:
            f.write("MutationType\t" + "\t".join(cols) + "\n")
            for i in range(83):
                f.write(f"Mut{i}\t1\t1\n")

    mock_spmg.side_effect = mock_spmg_func
    matrix = build_id83_matrix(df)
    assert matrix.shape[1] == 83
    assert np.all(matrix.values >= 0)

def test_sbs96_known_biological_fixture():
    df = pd.DataFrame({
        'Tumor_Sample_Barcode': ['S1', 'S1'],
        'Chromosome': ['chr1', 'chr1'],
        'Start_Position': [100, 200],
        'End_Position': [100, 200],
        'Reference_Allele': ['C', 'T'],
        'Tumor_Seq_Allele2': ['A', 'G']
    })

    with patch('sigdiscover.matrices.sbs.os.path.exists', return_value=True):
        with patch('pysam.FastaFile') as mock_fasta:
            mock_fasta_instance = mock_fasta.return_value
            # Fetch for first mut (C>A), need context 'A' before and 'T' after -> "ACT"
            # Fetch for second mut (T>G), need context 'G' before and 'C' after -> "GTC"
            mock_fasta_instance.fetch.side_effect = ["ACT", "GTC"]

            matrix = build_sbs96_matrix(df, genome="mock")

            assert matrix.shape[1] == 96
            assert matrix.loc['S1', 'A[C>A]T'] == 1
            assert matrix.loc['S1', 'G[T>G]C'] == 1

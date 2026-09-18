import os
import tempfile
import pandas as pd
from sigdiscover.matrices.context import _parse_maf_to_spmg_format

def build_dbs78_matrix(mutations: pd.DataFrame, genome: str = "GRCh37") -> pd.DataFrame:
    if len(mutations) == 0:
        return pd.DataFrame()
    try:
        from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as spmg
    except ImportError:
        return pd.DataFrame()
    spmg_df = _parse_maf_to_spmg_format(mutations)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        input_file = os.path.join(input_dir, "input.txt")
        spmg_df.to_csv(input_file, sep='\t', index=False)
        try:
            spmg.SigProfilerMatrixGeneratorFunc("test", genome, input_dir, plot=False, exome=False, bed_file=None, chrom_based=False, tsb_stat=False, seqInfo=False, cushion=100)
        except Exception as e:
            pass
        matrix_file = os.path.join(input_dir, "output", "DBS", "test.DBS78.all")
        if os.path.exists(matrix_file):
            df = pd.read_csv(matrix_file, sep='\t', index_col=0)
            return df.T
        return pd.DataFrame()
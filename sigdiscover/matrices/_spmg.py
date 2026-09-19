import os
import tempfile
import uuid

import pandas as pd


def run_spmg(mutations: pd.DataFrame, context: str, genome: str) -> pd.DataFrame:
    """Run SigProfilerMatrixGenerator and extract the output matrix for a given context.

    Args:
        mutations: DataFrame of mutations formatted for SPMG
        context: One of 'SBS', 'DBS', 'ID'
        genome: The reference genome (e.g. 'GRCh37')

    Returns:
        DataFrame representing the mutational matrix.
    """
    try:
        from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as spmg
    except ImportError as e:
        raise ImportError(f"SigProfilerMatrixGenerator is required for {context} matrix construction. Install with: pip install 'sigdiscover[sigprofiler]'") from e

    project_name = uuid.uuid4().hex[:8]

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        input_file = os.path.join(input_dir, "input.txt")
        mutations.to_csv(input_file, sep='\t', index=False)
        try:
            spmg.SigProfilerMatrixGeneratorFunc(project_name, genome, input_dir, plot=False, exome=False, bed_file=None, chrom_based=False, tsb_stat=False, seqInfo=False, cushion=100)
        except Exception as e:
            raise RuntimeError(f"SigProfilerMatrixGenerator failed: {e}") from e

        if context == 'SBS':
            suffix = 'SBS96.all'
        elif context == 'DBS':
            suffix = 'DBS78.all'
        elif context == 'ID':
            suffix = 'ID83.all'
        else:
            raise ValueError(f"Unknown context {context}")

        matrix_file = os.path.join(input_dir, "output", context, f"{project_name}.{suffix}")
        if os.path.exists(matrix_file):
            df = pd.read_csv(matrix_file, sep='\t', index_col=0)
            return df.T
        else:
            return pd.DataFrame()

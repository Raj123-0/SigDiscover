import pandas as pd

from sigdiscover.matrices._spmg import run_spmg
from sigdiscover.matrices.context import to_spmg_format


def build_dbs78_matrix(mutations: pd.DataFrame, genome: str = "GRCh37") -> pd.DataFrame:
    '''Build a 78-channel DBS mutation count matrix.

    Args:
        mutations: DataFrame containing mutations in MAF format
        genome: Reference genome version

    Returns:
        DataFrame with samples as rows and 78 channels as columns
    '''
    # Simplified empty matrix for DBS
    if len(mutations) == 0:
        return pd.DataFrame()
    try:
        from SigProfilerMatrixGenerator.scripts import SigProfilerMatrixGeneratorFunc as spmg
    except ImportError:
        return pd.DataFrame()
    spmg_df = _parse_maf_to_spmg_format(mutations, genome)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        input_file = os.path.join(input_dir, "input.txt")
        spmg_df.to_csv(input_file, sep='\t', index=False)
        try:
            spmg.SigProfilerMatrixGeneratorFunc("test", genome, input_dir, plot=False, exome=False, bed_file=None, chrom_based=False, tsb_stat=False, seqInfo=False, cushion=100)
        except Exception as e:
            from sigdiscover.utils.logging import logger
            logger.error(f"SigProfilerMatrixGenerator failed for DBS78: {e}")
            raise RuntimeError(f"Failed to generate DBS78 matrix: {e}")
        matrix_file = os.path.join(input_dir, "output", "DBS", "test.DBS78.all")
        if os.path.exists(matrix_file):
            df = pd.read_csv(matrix_file, sep='\t', index_col=0)
            return df.T

        from sigdiscover.utils.logging import logger
        logger.error(f"DBS78 output file not found at {matrix_file}")
        raise RuntimeError("SigProfilerMatrixGenerator completed but no DBS78 output file was found.")

import pandas as pd

from sigdiscover.matrices._spmg import run_spmg
from sigdiscover.matrices.context import to_spmg_format


def build_id83_matrix(mutations: pd.DataFrame, genome: str = "GRCh37") -> pd.DataFrame:
    '''Build a 83-channel ID mutation count matrix.

    Args:
        mutations: DataFrame containing mutations in MAF format
        genome: Reference genome version

    Returns:
        DataFrame with samples as rows and 83 channels as columns
    '''
    if len(mutations) == 0:
        return pd.DataFrame(columns=[f"ID_dummy_{i}" for i in range(83)])

    spmg_df = to_spmg_format(mutations, genome=genome, context='ID')
    if len(spmg_df) == 0:
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
            logger.error(f"SigProfilerMatrixGenerator failed for ID83: {e}")
            raise RuntimeError(f"Failed to generate ID83 matrix: {e}")
        matrix_file = os.path.join(input_dir, "output", "ID", "test.ID83.all")
        if os.path.exists(matrix_file):
            df = pd.read_csv(matrix_file, sep='\t', index_col=0)
            return df.T

        from sigdiscover.utils.logging import logger
        logger.error(f"ID83 output file not found at {matrix_file}")
        raise RuntimeError("SigProfilerMatrixGenerator completed but no ID83 output file was found.")

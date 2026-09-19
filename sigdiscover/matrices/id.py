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

    df = run_spmg(spmg_df, context="ID", genome=genome)
    return df

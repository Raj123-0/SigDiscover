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
        return pd.DataFrame(columns=[f"DBS_dummy_{i}" for i in range(78)]) # We will fix these labels later if required

    spmg_df = to_spmg_format(mutations, genome=genome, context='DBS')
    if len(spmg_df) == 0:
        return pd.DataFrame() # returning empty

    df = run_spmg(spmg_df, context="DBS", genome=genome)
    return df

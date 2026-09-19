import pandas as pd

from sigdiscover.matrices._spmg import run_spmg
from sigdiscover.matrices.context import to_spmg_format


def build_sbs96_matrix(mutations: pd.DataFrame, genome: str = "GRCh37") -> pd.DataFrame:
    '''Build a 96-channel SBS mutation count matrix.

    Args:
        mutations: DataFrame containing mutations in MAF format
        genome: Reference genome version

    Returns:
        DataFrame with samples as rows and 96 channels as columns
    '''
    bases = ['A', 'C', 'G', 'T']
    subs = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    cols = [f"{five}[{sub}]{three}" for sub in subs for five in bases for three in bases]

    if len(mutations) == 0:
        return pd.DataFrame(columns=cols)

    spmg_df = to_spmg_format(mutations, genome=genome, context='SBS')
    spmg_df = spmg_df[spmg_df['ref'].str.len() == 1]
    spmg_df = spmg_df[spmg_df['alt'].str.len() == 1]
    if len(spmg_df) == 0:
        return pd.DataFrame(columns=cols)

    df = run_spmg(spmg_df, context="SBS", genome=genome)
    if len(df.columns) == 0:
        return pd.DataFrame(columns=cols)
    return df

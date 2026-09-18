import os
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.optimize import nnls
from sigdiscover.utils.logging import logger
from sigdiscover.data.loaders import load_cosmic_signatures

def cosmic_fit_with_sigprofiler(matrix_path: str, output_dir: str, cosmic_version: float = 3.4, genome_build: str = "GRCh37") -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    '''Fit samples to COSMIC signatures using SigProfilerAssignment.

    Args:
        matrix_path: Path to the input mutation matrix TSV
        output_dir: Directory to save SPA results
        cosmic_version: COSMIC version
        genome_build: Genome build

    Returns:
        Tuple containing activities df, signatures array, and assignment info df

    Raises:
        FileNotFoundError: If COSMIC signatures cannot be loaded during fallback.
    '''
    try:
        from SigProfilerAssignment import Analyzer as Analyze
        Analyze.cosmic_fit(matrix_path, output_dir, input_type="matrix", context_type="96", cosmic_version=cosmic_version, genome_build=genome_build, collapse_to_SBS96=True, make_plots=False)
        activity_file = os.path.join(output_dir, "Assignment_Solution", "Activities", "Assignment_Solution_Activities.txt")
        activities_df = pd.read_csv(activity_file, sep='	', index_col=0)
        signatures_file = os.path.join(output_dir, "Assignment_Solution", "Signatures", "Assignment_Solution_Signatures.txt")
        if os.path.exists(signatures_file):
            signatures_df = pd.read_csv(signatures_file, sep='	', index_col=0)
            return activities_df, signatures_df.values.T, pd.DataFrame({'signature': signatures_df.columns.tolist()})
        return activities_df, np.array([]), pd.DataFrame({'signature': []})
    except ImportError:
        logger.warning("SigProfilerAssignment not installed. Falling back to NNLS decomposition.")
        M_df = pd.read_csv(matrix_path, sep='	', index_col=0)
        S_cosmic, names = load_cosmic_signatures("SBS96", "data/cosmic")
        A = decompose_with_nnls(M_df.values, S_cosmic)
        activities_df = pd.DataFrame(A, index=M_df.index, columns=names)
        return activities_df, S_cosmic, pd.DataFrame({'signature': names})

def decompose_with_nnls(M: np.ndarray, S_ref: np.ndarray) -> np.ndarray:
    '''Decompose samples into reference signatures using Non-Negative Least Squares.

    Args:
        M: Mutation matrix (N samples x C channels)
        S_ref: Reference signatures

    Returns:
        Exposure matrix A
    '''
    n_samples, n_signatures = M.shape[0], S_ref.shape[0]
    A = np.zeros((n_samples, n_signatures))
    for i in range(n_samples):
        A[i], _ = nnls(S_ref.T, M[i])
    return A

import os
import numpy as np
import pandas as pd
from typing import Tuple, Union
from pathlib import Path

def load_cosmic_signatures(sig_type: str = "SBS96", data_dir: str = "data/cosmic") -> Tuple[np.ndarray, list]:
    npz_path = os.path.join(data_dir, f"{sig_type}.npz")
    data = np.load(npz_path)
    return data['signatures'], data['names'].tolist()

def load_maf(filepath: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep='\t', comment='#', low_memory=False)
    return df
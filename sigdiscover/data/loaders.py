from pathlib import Path

import numpy as np
import pandas as pd

from sigdiscover.data.download import _parse_cosmic_tsv


def load_maf(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='	', comment='#', low_memory=False)

def load_cosmic_signatures(mutation_type: str, path: str) -> tuple[np.ndarray, list]:
    p = Path(path)
    if p.is_dir():
        npz_file = p / f"{mutation_type}.npz"
        if npz_file.exists():
            data = np.load(npz_file)
            return data['signatures'], data['names'].tolist()
        tsv_file = p / f"{mutation_type}.tsv"
        if not tsv_file.exists():
            # cosmic download saves them like COSMIC_v3.4_SBS_GRCh37.txt, fallback if needed
            txt_files = list(p.glob(f"*_{mutation_type}_*.txt"))
            if txt_files:
                tsv_file = txt_files[0]
            else:
                raise FileNotFoundError(f"Could not find {mutation_type} references in {path}")
        return _parse_cosmic_tsv(str(tsv_file))
    elif p.is_file():
        if p.suffix == '.npz':
            data = np.load(p)
            return data['signatures'], data['names'].tolist()
        else:
            return _parse_cosmic_tsv(str(p))
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

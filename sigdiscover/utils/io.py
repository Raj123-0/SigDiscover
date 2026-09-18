from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def save_matrix(df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, sep='\t', index=True)

def load_matrix(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', index_col=0)

def save_signatures(S: pd.DataFrame, path: str | Path, channels: list[str] | None = None) -> None:
    ensure_dir(Path(path).parent)
    df = S.copy()
    if channels is not None and len(channels) == df.shape[1]:
        df.columns = channels
    # Rename rows to Sig_1, Sig_2...
    df.index = [f"Sig_{i+1}" for i in range(df.shape[0])]
    df.to_csv(path, sep='\t', index=True, index_label="Signature")

def save_activities(A: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    df = A.copy()
    df.columns = [f"Sig_{i+1}" for i in range(df.shape[1])]
    if df.index.name is None:
        df.index.name = "Sample"
    df.to_csv(path, sep='\t', index=True)

import os
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import yaml

def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def save_matrix(df: pd.DataFrame, path: Union[str, Path]) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, sep='\t', index=True)

def load_matrix(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', index_col=0)
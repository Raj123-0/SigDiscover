import numpy as np
import pandas as pd

from sigdiscover.data.validators import validate_matrix


def test_validate_matrix_valid():
    df = pd.DataFrame(np.random.randint(1, 10, (5, 96)))
    report = validate_matrix(df, mutation_type="SBS96")
    assert report["is_valid"] is True

def test_validate_matrix_invalid():
    df = pd.DataFrame(np.random.randint(1, 10, (5, 95)))
    report = validate_matrix(df, mutation_type="SBS96")
    assert report["is_valid"] is False
    assert "Expected 96 channels" in report["errors"][0]

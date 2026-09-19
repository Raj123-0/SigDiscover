import numpy as np
import pandas as pd

from sigdiscover.utils.io import load_matrix, save_matrix


def test_io_matrix(tmp_path):
    df = pd.DataFrame(np.random.randint(0, 10, (5, 5)))
    df.columns = [str(c) for c in df.columns] # String columns to survive roundtrip
    out_file = tmp_path / "matrix.tsv"

    save_matrix(df, out_file)
    assert out_file.exists()

    df_loaded = load_matrix(out_file)
    pd.testing.assert_frame_equal(df, df_loaded)

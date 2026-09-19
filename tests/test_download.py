from unittest.mock import patch

import pytest

from sigdiscover.data.download import download_cosmic_signatures


@patch('sigdiscover.data.download.download_url')
@patch('sigdiscover.data.download._parse_cosmic_tsv')
def test_download_cosmic(mock_parse, mock_download, tmp_path):
    import numpy as np
    mock_parse.return_value = (np.array([[1]]), ["sig1"])

    def side_effect(url, output_path, timeout=30):
        with open(output_path, "w") as f:
            f.write("dummy content")

    mock_download.side_effect = side_effect

    download_cosmic_signatures(version="3.4", output_dir=str(tmp_path))

    assert mock_download.call_count >= 3 # called for 3 signatures
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()

@patch('sigdiscover.data.download.download_url')
def test_download_cosmic_failure(mock_download, tmp_path):
    # Simulate failure for all sources
    mock_download.side_effect = Exception("Download failed")

    with pytest.raises(RuntimeError):
        download_cosmic_signatures(version="3.4", output_dir=str(tmp_path))

    tmp_path / "manifest.json"
    # Although it raised, it might have partially written or not depending on the loop.
    # It raises on the first failed download.

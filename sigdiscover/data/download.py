import gzip
import json
import os
import shutil
import tarfile
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sigdiscover.utils.logging import logger
from sigdiscover.utils.io import ensure_dir

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url: str, output_path: str) -> None:
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def calculate_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def _parse_cosmic_tsv(filepath: str) -> Tuple[np.ndarray, list]:
    df = pd.read_csv(filepath, sep='\t')
    if 'Type' in df.columns and 'SubType' in df.columns:
        sig_cols = [c for c in df.columns if c.startswith(('SBS', 'DBS', 'ID'))]
        if not sig_cols:
            sig_cols = [c for c in df.columns if c not in ('Type', 'SubType')]
    elif 'Mutation Type' in df.columns:
        sig_cols = [c for c in df.columns if c != 'Mutation Type']
    else:
        sig_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    signatures = df[sig_cols].values.T
    return signatures, sig_cols

def download_cosmic_signatures(version: str = "3.4", output_dir: str = "data/cosmic") -> None:
    ensure_dir(output_dir)
    logger.info(f"Downloading COSMIC signatures version {version} to {output_dir}")
    base_url_sanger = "https://cancer.sanger.ac.uk/signatures/downloads/"
    if str(version) == "3.4":
        files = {
            "SBS96": "COSMIC_v3.4_SBS_GRCh37.txt",
            "DBS78": "COSMIC_v3.4_DBS_GRCh37.txt",
            "ID83": "COSMIC_v3.4_ID_GRCh37.txt"
        }
        fallback_files = {
            "SBS96": "https://raw.githubusercontent.com/Rozen-Lab/cosmicsig/master/data-raw/COSMIC_v3.4_SBS_GRCh37.txt",
            "DBS78": "https://raw.githubusercontent.com/Rozen-Lab/cosmicsig/master/data-raw/COSMIC_v3.4_DBS_GRCh37.txt",
            "ID83": "https://raw.githubusercontent.com/Rozen-Lab/cosmicsig/master/data-raw/COSMIC_v3.4_ID_GRCh37.txt"
        }
    else:
        files = {
            "SBS96": f"COSMIC_v{version}_SBS_GRCh37.txt",
            "DBS78": f"COSMIC_v{version}_DBS_GRCh37.txt",
            "ID83": f"COSMIC_v{version}_ID_GRCh37.txt"
        }
        fallback_files = {
            "SBS96": f"https://raw.githubusercontent.com/Rozen-Lab/cosmicsig/master/data-raw/COSMIC_v{version}_SBS_GRCh37.txt",
            "DBS78": f"https://raw.githubusercontent.com/Rozen-Lab/cosmicsig/master/data-raw/COSMIC_v{version}_DBS_GRCh37.txt",
            "ID83": f"https://raw.githubusercontent.com/Rozen-Lab/cosmicsig/master/data-raw/COSMIC_v{version}_ID_GRCh37.txt"
        }
    manifest = {"version": version, "download_date": datetime.now().isoformat(), "files": {}}
    for sig_type, filename in files.items():
        url = base_url_sanger + filename
        output_file = os.path.join(output_dir, filename)
        try:
            download_url(url, output_file)
        except Exception as e:
            logger.warning(f"Primary download failed for {sig_type}: {e}. Trying fallback.")
            fallback_url = fallback_files[sig_type]
            try:
                download_url(fallback_url, output_file)
            except Exception as e2:
                logger.error(f"Fallback download failed for {sig_type}: {e2}")
                raise RuntimeError(f"Failed to download COSMIC signature {sig_type} from both primary and fallback URLs.")
        try:
            signatures, sig_names = _parse_cosmic_tsv(output_file)
            npz_file = os.path.join(output_dir, f"{sig_type}.npz")
            np.savez(npz_file, signatures=signatures, names=sig_names)
            manifest["files"][sig_type] = {
                "source_file": filename, "npz_file": f"{sig_type}.npz", "sha256": calculate_sha256(output_file), "shape": signatures.shape
            }
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}")
            raise RuntimeError(f"Failed to parse downloaded signature file {filename}: {e}")
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

def download_tcga_mutations(project: str = "TCGA-BRCA", output_dir: str = "data/tcga") -> pd.DataFrame:
    ensure_dir(output_dir)
    files_endpt = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "data_category", "value": ["Simple Nucleotide Variation"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Masked Somatic Mutation"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["MAF"]}}
        ]
    }
    params = {"filters": json.dumps(filters), "fields": "file_id,file_name", "format": "JSON", "size": "1"}
    url = files_endpt + "?" + urllib.parse.urlencode(params)
    response = urllib.request.urlopen(url)
    data = json.loads(response.read().decode("utf-8"))
    file_info = data["data"]["hits"][0]
    file_id = file_info["file_id"]
    file_name = file_info["file_name"]
    download_url = f"https://api.gdc.cancer.gov/data/{file_id}"
    gz_output_path = os.path.join(output_dir, file_name)
    maf_output_path = os.path.join(output_dir, f"{project}.maf")
    urllib.request.urlretrieve(download_url, gz_output_path)
    with gzip.open(gz_output_path, 'rb') as f_in:
        with open(maf_output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    df = pd.read_csv(maf_output_path, sep='\t', comment='#', low_memory=False)
    return df
# SigDiscover

SigDiscover is a complete cancer mutational signature discovery pipeline. It provides an end-to-end framework to take raw somatic mutation data (MAFs, VCFs) and identify the mutational signatures active in those samples using Non-Negative Matrix Factorization (NMF).

## Features

- **Data Acquisition**: Download TCGA data and COSMIC signatures.
- **Matrix Generation**: Create SBS96, DBS78, and ID83 matrices using SigProfilerMatrixGenerator.
- **De Novo Extraction**: Extract mutational signatures using an NMF algorithm from first principles.
- **Assignment**: Compare and assign discovered signatures to COSMIC reference signatures.
- **Validation & Benchmarking**: Quantify the stability and accuracy of the signatures.

## Installation

```bash
conda env create -f environment.yml
conda activate sigdiscover
```
Or via pip:
```bash
pip install -r requirements.txt
```

## Quickstart

This example runs the full pipeline on synthetically generated data:

```bash
# You must install the sigprofiler extra to extract contexts
pip install -e ".[sigprofiler]"

# Generate synthetic mutations, build matrix, extract signatures, and assign to COSMIC
python -m sigdiscover run --synthetic --output results/synthetic_run/

# Run benchmarking
python -m sigdiscover benchmark --output results/benchmark/

# Generate an HTML report
python -m sigdiscover report --results results/benchmark/ --output results/report.html
```

## CLI Reference

- `sigdiscover download-cosmic`: Download COSMIC reference signatures.
- `sigdiscover download-tcga`: Download TCGA somatic mutation data.
- `sigdiscover build-matrix`: Convert a MAF file into mutation count matrices (SBS96, etc).
- `sigdiscover extract`: Extract signatures using NMF.
- `sigdiscover assign`: Assign extracted signatures to COSMIC.
- `sigdiscover run`: Run the full end-to-end pipeline.
- `sigdiscover benchmark`: Benchmark the extraction algorithms.
- `sigdiscover report`: Generate an HTML summary report.

## Citation and License

Licensed under the MIT License.

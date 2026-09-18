# SigDiscover

SigDiscover is a complete cancer mutational signature discovery pipeline. It provides an end-to-end framework to take raw somatic mutation data (MAFs, VCFs) and identify the mutational signatures active in those samples using Non-Negative Matrix Factorization (NMF).

## Features

- **Data Acquisition**: Download TCGA data and COSMIC signatures.
- **Matrix Generation**: Create SBS96, DBS78, and ID83 matrices using SigProfilerMatrixGenerator.
- **De Novo Extraction**: Extract mutational signatures using an NMF algorithm from first principles.
- **Assignment**: Compare and assign discovered signatures to COSMIC reference signatures.
- **Validation & Benchmarking**: Quantify the stability and accuracy of the signatures.

## Installation

See [Installation](installation.md) for details on setting up the environment.

## Quickstart

See [Quickstart](quickstart.md) for a worked example.

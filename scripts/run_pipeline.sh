#!/bin/bash
set -e

# Run the full pipeline on synthetic data
python -m sigdiscover run --synthetic --output results/synthetic_run

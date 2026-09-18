# Quickstart

This example runs the full pipeline on synthetically generated data. No internet connection is required after the initial setup.

```bash
# Activate the environment
conda activate sigdiscover

# Generate synthetic mutations, build matrix, extract signatures, and assign to COSMIC
python -m sigdiscover run --synthetic --output results/synthetic_run/

# Run benchmarking to validate the algorithm
python -m sigdiscover benchmark --output results/benchmark/

# Generate an HTML report from the benchmark results
python -m sigdiscover report --results results/benchmark/ --output results/report.html
```

You can view the extracted signatures and activities in `results/synthetic_run/`.

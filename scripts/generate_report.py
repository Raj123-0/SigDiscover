#!/usr/bin/env python
import argparse
from sigdiscover.cli import report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML validation report.")
    parser.add_argument("--results", type=str, required=True, help="Results directory containing benchmark JSON")
    parser.add_argument("--output", type=str, required=True, help="Output HTML file")
    args = parser.parse_args()

    # We call the CLI logic directly or run via subprocess
    import subprocess
    subprocess.run(["python", "-m", "sigdiscover", "report", "--results", args.results, "--output", args.output], check=True)

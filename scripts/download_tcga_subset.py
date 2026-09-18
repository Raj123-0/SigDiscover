#!/usr/bin/env python
import argparse
from sigdiscover.data.download import download_tcga_mutations

def main():
    parser = argparse.ArgumentParser(description="Download TCGA somatic mutations.")
    parser.add_argument("--project", type=str, default="TCGA-BRCA", help="TCGA project code (e.g., TCGA-BRCA)")
    parser.add_argument("--output", type=str, default="data/tcga", help="Output directory")

    args = parser.parse_args()
    download_tcga_mutations(project=args.project, output_dir=args.output)

if __name__ == "__main__":
    main()

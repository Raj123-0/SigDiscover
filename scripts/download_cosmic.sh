#!/bin/bash
set -e

# Defaults
VERSION="3.4"
OUTPUT_DIR="data/cosmic"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --version) VERSION="$2"; shift ;;
        --output) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python -m sigdiscover download-cosmic --version "$VERSION" --output "$OUTPUT_DIR"

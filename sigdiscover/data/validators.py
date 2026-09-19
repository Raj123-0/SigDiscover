import numpy as np
import pandas as pd
from typing import Dict

def validate_matrix(matrix: pd.DataFrame, mutation_type: str = "SBS96") -> Dict:
    expected_channels = {"SBS96": 96, "DBS78": 78, "ID83": 83}
    report = {
        "is_valid": True, "n_samples": matrix.shape[0], "n_channels": matrix.shape[1],
        "total_mutations": 0, "min_per_sample": 0, "max_per_sample": 0, "errors": [], "warnings": []
    }

    if matrix.empty:
        report["is_valid"] = False
        report["errors"].append("Matrix is empty")
        return report

    # Check for NaNs and Infinities
    if matrix.isnull().values.any():
        report["is_valid"] = False
        report["errors"].append("Matrix contains NaN values")

    if np.isinf(matrix.values).any():
        report["is_valid"] = False
        report["errors"].append("Matrix contains infinite values")

    # Check for non-negative values
    if (matrix.values < 0).any():
        report["is_valid"] = False
        report["errors"].append("Matrix contains negative values")

    # Check sample uniqueness
    if not matrix.index.is_unique:
        report["is_valid"] = False
        report["errors"].append("Matrix index (sample names) contains duplicates")

    # Check channel uniqueness
    if not matrix.columns.is_unique:
        report["is_valid"] = False
        report["errors"].append("Matrix columns (channels) contains duplicates")

    if mutation_type in expected_channels:
        expected = expected_channels[mutation_type]
        if matrix.shape[1] != expected:
            report["is_valid"] = False
            report["errors"].append(f"Expected {expected} channels but got {matrix.shape[1]}")

    if report["is_valid"]:
        report["total_mutations"] = int(matrix.sum().sum())
        report["min_per_sample"] = int(matrix.sum(axis=1).min())
        report["max_per_sample"] = int(matrix.sum(axis=1).max())

    return report

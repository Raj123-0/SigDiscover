
import pandas as pd


def validate_matrix(matrix: pd.DataFrame, mutation_type: str = "SBS96") -> dict:
    expected_channels = {"SBS96": 96, "DBS78": 78, "ID83": 83}
    report = {
        "is_valid": True, "n_samples": matrix.shape[0], "n_channels": matrix.shape[1],
        "total_mutations": int(matrix.sum().sum()), "min_per_sample": int(matrix.sum(axis=1).min()),
        "max_per_sample": int(matrix.sum(axis=1).max()), "errors": [], "warnings": []
    }
    if mutation_type in expected_channels:
        expected = expected_channels[mutation_type]
        if matrix.shape[1] != expected:
            report["is_valid"] = False
            report["errors"].append(f"Expected {expected} channels")
    return report
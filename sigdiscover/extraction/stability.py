import numpy as np
from typing import List, Tuple
from sigdiscover.extraction.rank_selection import align_signatures, cosine_similarity_matrix

def compute_signature_stability(S_list: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    if not S_list:
        raise ValueError("S_list cannot be empty")
    if len(S_list) == 1:
        return S_list[0], 1.0
    ref_S = S_list[0]
    aligned_signatures = [ref_S]
    for S in S_list[1:]:
        aligned_S, _ = align_signatures(ref_S, S)
        aligned_signatures.append(aligned_S)
    consensus_S = np.mean(aligned_signatures, axis=0)
    row_sums = consensus_S.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-16
    consensus_S = consensus_S / row_sums
    per_run_similarities = [np.diag(cosine_similarity_matrix(consensus_S, S)) for S in aligned_signatures]
    overall_stability = float(np.mean(np.mean(per_run_similarities, axis=0)))
    return consensus_S, overall_stability
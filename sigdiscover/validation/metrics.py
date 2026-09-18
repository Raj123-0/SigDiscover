import numpy as np
from typing import List, Tuple
from scipy.stats import pearsonr
from sigdiscover.extraction.rank_selection import align_signatures

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0

def frobenius_error(M: np.ndarray, M_hat: np.ndarray) -> float:
    return float(np.linalg.norm(M - M_hat, 'fro'))

def reconstruction_r2(M: np.ndarray, M_hat: np.ndarray) -> float:
    ss_res = np.sum((M - M_hat) ** 2)
    ss_tot = np.sum((M - np.mean(M)) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)

def signature_stability(S_list: List[np.ndarray]) -> float:
    from sigdiscover.extraction.stability import compute_signature_stability
    return compute_signature_stability(S_list)[1]

def exposure_correlation(A_true: np.ndarray, A_pred: np.ndarray) -> float:
    pearsons = [pearsonr(A_true[:, k], A_pred[:, k])[0] if np.std(A_true[:, k]) > 1e-10 and np.std(A_pred[:, k]) > 1e-10 else 0.0 for k in range(A_true.shape[1])]
    return float(np.mean(pearsons))

def signature_precision_recall(S_true: np.ndarray, S_pred: np.ndarray, threshold: float = 0.8) -> Tuple[float, float, float]:
    _, similarities = align_signatures(S_true, S_pred)
    TP = np.sum(np.array(similarities) >= threshold)
    precision = TP / S_pred.shape[0] if S_pred.shape[0] > 0 else 0.0
    recall = TP / S_true.shape[0] if S_true.shape[0] > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)
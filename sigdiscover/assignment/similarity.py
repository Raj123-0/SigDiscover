
import numpy as np
import pandas as pd


def assign_to_cosmic(discovered_signatures: np.ndarray, cosmic_signatures: np.ndarray, cosmic_names: list[str], threshold: float = 0.8) -> pd.DataFrame:
    disc_norm = np.linalg.norm(discovered_signatures, axis=1, keepdims=True)
    cosm_norm = np.linalg.norm(cosmic_signatures, axis=1, keepdims=True)
    disc_norm[disc_norm == 0] = 1e-16
    cosm_norm[cosm_norm == 0] = 1e-16
    sim_matrix = (discovered_signatures @ cosmic_signatures.T) / (disc_norm @ cosm_norm.T)
    results = []
    for i in range(discovered_signatures.shape[0]):
        best_idx = np.argmax(sim_matrix[i])
        max_sim = sim_matrix[i, best_idx]
        if max_sim >= threshold:
            results.append({'discovered_index': i, 'best_cosmic_match': cosmic_names[best_idx], 'cosine_similarity': max_sim, 'is_novel': False})
        else:
            results.append({'discovered_index': i, 'best_cosmic_match': f"Novel_{i+1}", 'cosine_similarity': max_sim, 'is_novel': True})
    return pd.DataFrame(results)
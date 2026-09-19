import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sigdiscover.utils.io import ensure_dir


def plot_cosine_heatmap(discovered: np.ndarray, cosmic: np.ndarray, cosmic_names: list[str], save_path: str | None = None) -> plt.Figure:
    disc_norm = np.linalg.norm(discovered, axis=1, keepdims=True)
    cosm_norm = np.linalg.norm(cosmic, axis=1, keepdims=True)
    disc_norm[disc_norm == 0], cosm_norm[cosm_norm == 0] = 1e-16, 1e-16
    sim_matrix = (discovered @ cosmic.T) / (disc_norm @ cosm_norm.T)
    fig, ax = plt.subplots(figsize=(12, min(8, len(cosmic_names) * 0.3)))
    keep_idx = np.max(sim_matrix, axis=0) > 0.5
    filtered_sim = sim_matrix[:, keep_idx] if np.any(keep_idx) else sim_matrix
    filtered_names = [name for i, name in enumerate(cosmic_names) if keep_idx[i]] if np.any(keep_idx) else cosmic_names
    sns.heatmap(filtered_sim, cmap='Blues', ax=ax, xticklabels=filtered_names, yticklabels=[f"DeNovo_{i+1}" for i in range(discovered.shape[0])], annot=True, fmt=".2f", vmin=0, vmax=1)
    ax.set_title("Cosine Similarity: De Novo vs COSMIC")
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save_path:
       ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig

def plot_rank_selection(rank_results: pd.DataFrame, save_path: str | None = None) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    k_vals, stability, recon_err = rank_results['k'].values, rank_results['stability'].values, rank_results['reconstruction_error'].values
    ax1.set_xlabel('Rank (k)')
    ax1.set_ylabel('Stability', color='tab:blue')
    ax1.plot(k_vals, stability, marker='o', color='tab:blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Reconstruction Error', color='tab:red')
    ax2.plot(k_vals, recon_err, marker='s', color='tab:red', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    opt_k = int(rank_results.loc[rank_results['score'].idxmax()]['k'])
    ax1.axvline(x=opt_k, color='green', linestyle=':', linewidth=2, label=f"Optimal k={opt_k}")
    fig.suptitle("Rank Selection Criteria vs. Number of Signatures (k)")
    ax1.set_xticks(k_vals)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    if save_path:
       ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig
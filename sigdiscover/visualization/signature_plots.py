import os

import matplotlib.pyplot as plt
import numpy as np

from sigdiscover.utils.io import ensure_dir


def plot_signature_profile(signature: np.ndarray, title: str = "Mutational Signature", save_path: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 4))
    n_channels = len(signature)

    if n_channels == 96:
        colors = ['#03BDFF', '#000000', '#E41A1C', '#A6CEE3', '#1F78B4', '#FB9A99']
        mutation_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
        bases = ['A', 'C', 'G', 'T']
        x = np.arange(96)
        for i, color in enumerate(colors):
            ax.bar(x[i*16:(i+1)*16], signature[i*16:(i+1)*16], color=color, width=0.8)
        ax.set_xticks(x)
        labels = [f"{five}[{mt[0]}>{mt[2]}]{three}" for mt in mutation_types for five in bases for three in bases]
        ax.set_xticklabels(labels, rotation=90, fontsize=6, fontfamily='monospace')
        ax.set_xlim(-1, 96); ax.set_title(title); ax.set_ylabel("Probability")
        for i, (color, mt) in enumerate(zip(colors, mutation_types)):
            ax.axvspan(i*16 - 0.5, (i+1)*16 - 0.5, facecolor=color, alpha=0.2, zorder=0)
            ax.text(i*16 + 7.5, ax.get_ylim()[1] * 1.05, mt, ha='center', fontweight='bold', fontsize=12)

    elif n_channels == 78:
        # Generic DBS plotting
        x = np.arange(78)
        ax.bar(x, signature, color='#1F78B4', width=0.8)
        ax.set_xticks(x)
        labels = [f"DBS_{i}" for i in range(78)] # Dummy labels if not provided
        ax.set_xticklabels(labels, rotation=90, fontsize=6, fontfamily='monospace')
        ax.set_xlim(-1, 78); ax.set_title(f"{title} (DBS78)"); ax.set_ylabel("Probability")

    elif n_channels == 83:
        # Generic ID plotting
        x = np.arange(83)
        ax.bar(x, signature, color='#E41A1C', width=0.8)
        ax.set_xticks(x)
        labels = [f"ID_{i}" for i in range(83)] # Dummy labels if not provided
        ax.set_xticklabels(labels, rotation=90, fontsize=6, fontfamily='monospace')
        ax.set_xlim(-1, 83); ax.set_title(f"{title} (ID83)"); ax.set_ylabel("Probability")

    else:
        plt.close(fig)
        raise ValueError(f"Unsupported signature dimension: {n_channels}. Expected 96, 78, or 83.")

    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig

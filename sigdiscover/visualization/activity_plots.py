import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sigdiscover.utils.io import ensure_dir


def plot_activity_heatmap(activities: pd.DataFrame, save_path: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(activities, cmap='YlOrRd', ax=ax, xticklabels=True, yticklabels=False)
    ax.set_title("Signature Activities Across Samples"); ax.set_ylabel("Samples"); ax.set_xlabel("Signatures")
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path)); fig.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    return fig

def plot_activity_barplot(activities: pd.DataFrame, top_n: int = 10, save_path: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    df = activities.copy()
    df['total'] = df.sum(axis=1)
    df = df.sort_values('total', ascending=False).drop('total', axis=1)
    if df.shape[1] > top_n:
        sig_sums = df.sum().sort_values(ascending=False)
        top_sigs, other_sigs = sig_sums.index[:top_n], sig_sums.index[top_n:]
        df_top = df[top_sigs].copy()
        df_top['Other'] = df[other_sigs].sum(axis=1)
        df = df_top
    df.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', width=0.8)
    if len(df) > 50: ax.set_xticks([])
    else: ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_title("Signature Contributions per Sample"); ax.set_ylabel("Mutations"); ax.set_xlabel("Samples")
    ax.legend(title="Signatures", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path)); fig.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    return fig
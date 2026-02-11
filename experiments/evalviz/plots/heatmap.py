from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_improvement_heatmap(
    imp_pivot: pd.DataFrame,
    title: str = "",
    clip_pp: float = 12.0,
):
    """
    Pure matplotlib diverging heatmap, centered at 0.
    imp_pivot values are in percentage points.
    """
    data = imp_pivot.copy().fillna(0.0)
    arr = np.clip(data.values, -clip_pp, clip_pp)

    fig, ax = plt.subplots(
        figsize=(max(10, 0.45 * data.shape[1]), max(3.5, 0.45 * data.shape[0]))
    )
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", vmin=-clip_pp, vmax=clip_pp)

    ax.set_title(title)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Improvement (pp)")

    plt.tight_layout()
    plt.show()

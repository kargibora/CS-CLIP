from __future__ import annotations
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt


def plot_grouped_bars(
    pivot: pd.DataFrame,
    methods: Optional[List[str]] = None,
    title: str = "",
    ylabel: str = "Accuracy",
):
    """
    pivot: rows=run_label, cols=things, values=score
    Produces grouped bars (transpose so groups are on x-axis).
    """
    if methods is None:
        methods = list(pivot.index)

    data = pivot.loc[methods].copy()
    data = data.dropna(axis=1, how="all")

    ax = data.T.plot(kind="bar", figsize=(14, 6))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    plt.tight_layout()
    plt.show()

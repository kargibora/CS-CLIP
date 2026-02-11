from __future__ import annotations
from typing import Callable, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def select_radar_axes(
    pivot: pd.DataFrame,
    max_axes: int = 12,
    strategy: str = "variance",
    baseline_label: str | None = None,
) -> List[str]:
    cols = list(pivot.columns)
    if len(cols) <= max_axes:
        return cols

    if strategy == "variance":
        scores = pivot.var(axis=0, skipna=True).sort_values(ascending=False)
        return list(scores.head(max_axes).index)

    if strategy == "baseline_gap":
        if baseline_label is None or baseline_label not in pivot.index:
            scores = pivot.var(axis=0, skipna=True).sort_values(ascending=False)
            return list(scores.head(max_axes).index)
        gap = (pivot.sub(pivot.loc[baseline_label], axis=1)).abs().mean(axis=0).sort_values(ascending=False)
        return list(gap.head(max_axes).index)

    raise ValueError(f"Unknown strategy: {strategy}")


def plot_radar_profile(
    pivot: pd.DataFrame,
    axes: List[str],
    methods: Optional[List[str]] = None,
    title: str = "",
    baseline_label: str | None = None,
    ours_predicate: Callable[[str], bool] | None = None,
    fill_alpha: float = 0.10,
):
    """
    pivot: rows=run_label, cols=axes, values in [0,1]
    """
    if methods is None:
        methods = list(pivot.index)

    data = pivot.loc[methods, axes].copy()
    data = data.dropna(axis=1, how="all")
    data = data.fillna(data.mean(axis=0))

    labels = list(data.columns)
    n = len(labels)
    if n < 3:
        raise ValueError("Radar needs >= 3 axes.")

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 7))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=9)

    for m in methods:
        vals = data.loc[m].values.tolist()
        vals += vals[:1]

        is_baseline = (baseline_label is not None and m == baseline_label)
        is_ours = (ours_predicate(m) if ours_predicate else False)

        lw = 3.0 if is_baseline else (2.5 if is_ours else 1.8)
        ls = "--" if is_baseline else "-"

        ax.plot(angles, vals, linewidth=lw, linestyle=ls, label=m)
        ax.fill(angles, vals, alpha=fill_alpha)

    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=False)
    plt.tight_layout()
    plt.show()

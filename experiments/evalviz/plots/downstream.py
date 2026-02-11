# evalviz/plots/downstream.py
"""
Plotting functions for downstream benchmarks (zero-shot classification, retrieval).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..downstream import (
    build_downstream_table,
    build_downstream_summary,
    DOWNSTREAM_GROUPINGS,
)
from ..tables import _run_info_from_df, order_rows_by_groups


def plot_downstream_bars(
    df: pd.DataFrame,
    task: str = "zero_shot_classification",
    metric: Optional[str] = None,
    title: str = "",
    figsize: Tuple[int, int] = (14, 6),
    as_percentage: bool = True,
    save_path: Optional[str | Path] = None,
    ylim: Optional[Tuple[float, float]] = None,
    legend_ncol: int = 4,
    bar_width: float = 0.8,
    rotation: int = 45,
):
    """
    Plot grouped bar chart for downstream benchmarks.
    X-axis = datasets, bars = models
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    task : str
        One of: "zero_shot_classification", "retrieval_t2i", "retrieval_i2t"
    metric : str, optional
        Override the default metric
    title : str
        Plot title
    figsize : tuple
        Figure size
    as_percentage : bool
        Multiply values by 100
    save_path : str | Path, optional
        Save plot to file
    ylim : tuple, optional
        Y-axis limits (default: 0-100 for percentage)
    legend_ncol : int
        Number of columns in legend
    bar_width : float
        Width of bars
    rotation : int
        X-label rotation
    """
    pivot, run_info = build_downstream_table(df, task=task, metric=metric)
    
    if pivot.empty:
        print("No data for downstream bar plot.")
        return
    
    # Convert to percentage
    data = pivot.copy()
    if as_percentage:
        data = data * 100.0
    
    # Order by groups
    methods = list(data.index)
    ordered_methods, _ = order_rows_by_groups(methods, run_info)
    data = data.loc[ordered_methods]
    
    # Clean column names
    data.columns = [c.replace("wds_", "").replace("_", "-") for c in data.columns]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    data.T.plot(kind="bar", ax=ax, width=bar_width)
    
    if not title:
        task_name = task.replace("_", " ").title()
        title = f"Downstream Results: {task_name}"
    
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Accuracy (%)" if as_percentage else "Accuracy")
    ax.set_xlabel("")
    
    if ylim:
        ax.set_ylim(ylim)
    elif as_percentage:
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(0, 1.0)
    
    # Rotate x labels
    plt.xticks(rotation=rotation, ha="right")
    
    # Legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=min(legend_ncol, len(data.index)),
        frameon=False,
    )
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def plot_downstream_comparison(
    df: pd.DataFrame,
    baseline_label: Optional[str] = None,
    title: str = "",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str | Path] = None,
    legend_ncol: int = 4,
):
    """
    Plot improvement over baseline for all downstream tasks as a horizontal bar chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    baseline_label : str, optional
        Label of the baseline model (auto-detected if None)
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str | Path, optional
        Save plot to file
    legend_ncol : int
        Number of columns in legend
    """
    pivot, run_info = build_downstream_summary(df)
    
    if pivot.empty:
        print("No data for downstream comparison.")
        return
    
    # Find baseline
    if baseline_label is None:
        for m, info in run_info.items():
            if info.get("is_baseline", False):
                baseline_label = m
                break
    
    if baseline_label is None or baseline_label not in pivot.index:
        print("No baseline found for comparison.")
        return
    
    # Compute improvements (in percentage points)
    baseline_vals = pivot.loc[baseline_label]
    improvements = (pivot - baseline_vals) * 100.0
    improvements = improvements.drop(index=baseline_label, errors="ignore")
    
    # Order rows
    methods = list(improvements.index)
    ordered_methods, _ = order_rows_by_groups(methods, run_info)
    improvements = improvements.loc[ordered_methods]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    improvements.plot(kind="barh", ax=ax)
    
    ax.axvline(x=0, color="black", linewidth=0.8)
    
    if not title:
        title = f"Improvement over {baseline_label} (percentage points)"
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Improvement (pp)")
    ax.set_ylabel("")
    
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=min(legend_ncol, len(improvements.columns)),
        frameon=False,
    )
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def plot_downstream_heatmap(
    df: pd.DataFrame,
    task: str = "zero_shot_classification",
    metric: Optional[str] = None,
    title: str = "",
    figsize: Optional[Tuple[int, int]] = None,
    as_percentage: bool = True,
    cmap: str = "RdYlGn",
    save_path: Optional[str | Path] = None,
    annotate: bool = True,
    annotation_fontsize: int = 8,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Plot heatmap of downstream benchmark results.
    Rows = models, Columns = datasets
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    task : str
        One of: "zero_shot_classification", "retrieval_t2i", "retrieval_i2t"
    metric : str, optional
        Override the default metric
    title : str
        Plot title
    figsize : tuple, optional
        Figure size (auto-calculated if None)
    as_percentage : bool
        Multiply values by 100
    cmap : str
        Colormap name
    save_path : str | Path, optional
        Save plot to file
    annotate : bool
        Add value annotations to cells
    annotation_fontsize : int
        Font size for annotations
    vmin, vmax : float, optional
        Colormap value limits
    """
    pivot, run_info = build_downstream_table(df, task=task, metric=metric)
    
    if pivot.empty:
        print("No data for heatmap.")
        return
    
    # Order rows
    methods = list(pivot.index)
    ordered_methods, _ = order_rows_by_groups(methods, run_info)
    pivot = pivot.loc[ordered_methods]
    
    # Convert to percentage
    data = pivot.copy()
    if as_percentage:
        data = data * 100.0
    
    # Clean column names
    data.columns = [c.replace("wds_", "").replace("_", "-") for c in data.columns]
    
    # Figure size
    if figsize is None:
        figsize = (max(10, 0.8 * len(data.columns)), max(4, 0.5 * len(data.index)))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set vmin/vmax
    if vmin is None:
        vmin = data.values.min() - 5
    if vmax is None:
        vmax = data.values.max() + 5
    
    im = ax.imshow(data.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Axis labels
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    
    # Annotations
    if annotate:
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                val = data.iloc[i, j]
                if not pd.isna(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", 
                            fontsize=annotation_fontsize)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)" if as_percentage else "Accuracy")
    
    if not title:
        task_name = task.replace("_", " ").title()
        title = f"Downstream Results: {task_name}"
    
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def plot_downstream_radar(
    df: pd.DataFrame,
    models: Optional[list] = None,
    title: str = "Downstream Benchmark Comparison",
    figsize: Tuple[int, int] = (8, 8),
    as_percentage: bool = True,
    save_path: Optional[str | Path] = None,
    fill_alpha: float = 0.25,
):
    """
    Plot radar/spider chart comparing models across downstream task categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    models : list, optional
        Subset of models to plot (default: all)
    title : str
        Plot title
    figsize : tuple
        Figure size
    as_percentage : bool
        Multiply values by 100
    save_path : str | Path, optional
        Save plot to file
    fill_alpha : float
        Alpha for filled area
    """
    pivot, run_info = build_downstream_summary(df)
    
    if pivot.empty:
        print("No data for radar plot.")
        return
    
    # Subset models
    if models is not None:
        pivot = pivot.loc[[m for m in models if m in pivot.index]]
    
    if pivot.empty:
        print("No matching models found.")
        return
    
    # Convert to percentage
    data = pivot.copy()
    if as_percentage:
        data = data * 100.0
    
    # Setup radar
    categories = list(data.columns)
    n_cats = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data.index)))
    
    for idx, (model, row) in enumerate(data.iterrows()):
        values = row.tolist()
        values += values[:1]  # Close the polygon
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
        ax.fill(angles, values, alpha=fill_alpha, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, fontsize=14, pad=20)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    
    plt.show()

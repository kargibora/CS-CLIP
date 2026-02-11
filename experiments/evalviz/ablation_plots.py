#!/usr/bin/env python3
"""
Ablation Visualization Module for CS-CLIP

Specialized plots for each type of ablation study.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors
CMAP_DIVERGING = "RdYlGn"
CMAP_SEQUENTIAL = "YlGnBu"

PAL = {
    "best": "#2ca02c",
    "good": "#98df8a",
    "neutral": "#1f77b4",
    "bad": "#ff9896",
    "worst": "#d62728",
    "highlight": "#ff7f0e",
}


# =============================================================================
# Lambda Sweep Plots (1D)
# =============================================================================

def plot_lambda_sweep(
    values: list[float],
    scores: list[float],
    xlabel: str = "λ",
    ylabel: str = "Accuracy (%)",
    title: str = "Lambda Sweep",
    highlight_best: bool = True,
    figsize: Tuple[float, float] = (6, 4),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Clean line plot for lambda/hyperparameter sweep.
    Shows trend with highlighted optimal point.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by x-value
    order = np.argsort(values)
    x = np.array(values)[order]
    y = np.array(scores)[order]
    
    # Main line
    ax.plot(x, y, marker="o", linewidth=2, markersize=8, color=PAL["neutral"], zorder=3)
    
    # Fill area under curve
    ax.fill_between(x, y.min() - 1, y, alpha=0.1, color=PAL["neutral"])
    
    # Highlight best
    if highlight_best:
        best_idx = np.argmax(y)
        ax.scatter([x[best_idx]], [y[best_idx]], s=200, color=PAL["best"], 
                   marker="*", zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(f"Best: {y[best_idx]:.1f}%\n({xlabel}={x[best_idx]})",
                    xy=(x[best_idx], y[best_idx]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=PAL["best"],
                    arrowprops=dict(arrowstyle="->", color=PAL["best"], lw=1.5))
    
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    y_margin = (y.max() - y.min()) * 0.15
    ax.set_ylim(y.min() - y_margin, y.max() + y_margin)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_multi_metric_sweep(
    values: list[float],
    scores_dict: dict[str, list[float]],
    xlabel: str = "λ",
    title: str = "Lambda Sweep (Multiple Metrics)",
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Line plot comparing multiple metrics across a parameter sweep.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(scores_dict)))
    
    for (metric_name, scores), color in zip(scores_dict.items(), colors):
        order = np.argsort(values)
        x = np.array(values)[order]
        y = np.array(scores)[order]
        
        ax.plot(x, y, marker="o", linewidth=2, markersize=6, 
                label=metric_name, color=color)
        
        # Mark best
        best_idx = np.argmax(y)
        ax.scatter([x[best_idx]], [y[best_idx]], s=100, color=color, 
                   marker="*", zorder=5, edgecolors="black", linewidth=0.5)
    
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# =============================================================================
# Grid/Heatmap Plots (2D)
# =============================================================================

def plot_2d_heatmap(
    data: pd.DataFrame,
    xlabel: str = "Column",
    ylabel: str = "Row",
    title: str = "2D Ablation Grid",
    cmap: str = CMAP_DIVERGING,
    annotate: bool = True,
    highlight_best: bool = True,
    figsize: Tuple[float, float] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Heatmap for 2D grid ablation (e.g., swap × inplace).
    """
    if figsize is None:
        figsize = (max(5, len(data.columns) * 1.0), max(4, len(data) * 0.7))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(data, annot=annotate, fmt=".1f", cmap=cmap,
                center=data.values.mean(), ax=ax,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Accuracy (%)", "shrink": 0.8},
                annot_kws={"size": 9})
    
    # Highlight best cell
    if highlight_best:
        best_val = data.max().max()
        best_loc = np.where(data.values == best_val)
        if len(best_loc[0]) > 0:
            row, col = best_loc[0][0], best_loc[1][0]
            ax.add_patch(patches.Rectangle((col, row), 1, 1, fill=False,
                                           edgecolor=PAL["best"], linewidth=3))
    
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_contour(
    data: pd.DataFrame,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Parameter Surface",
    figsize: Tuple[float, float] = (7, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Contour plot for smooth visualization of 2D parameter space.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    X = data.columns.astype(float).values
    Y = data.index.astype(float).values
    Z = data.values
    
    # Create meshgrid
    XX, YY = np.meshgrid(X, Y)
    
    # Contour plot
    levels = np.linspace(Z.min(), Z.max(), 15)
    cs = ax.contourf(XX, YY, Z, levels=levels, cmap=CMAP_DIVERGING)
    ax.contour(XX, YY, Z, levels=levels, colors="white", linewidths=0.5, alpha=0.5)
    
    # Colorbar
    cbar = plt.colorbar(cs, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)", fontsize=10)
    
    # Mark best point
    best_idx = np.unravel_index(np.argmax(Z), Z.shape)
    ax.scatter([X[best_idx[1]]], [Y[best_idx[0]]], s=200, color="white",
               marker="*", edgecolors="black", linewidth=1, zorder=5)
    ax.annotate(f"Best: {Z.max():.1f}%", xy=(X[best_idx[1]], Y[best_idx[0]]),
                xytext=(5, 5), textcoords="offset points", fontsize=9,
                color="white", fontweight="bold",
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="black")])
    
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# =============================================================================
# Comparison Plots
# =============================================================================

def plot_ablation_comparison_bar(
    labels: list[str],
    scores: list[float],
    baseline_score: float = None,
    baseline_label: str = "Baseline",
    title: str = "Ablation Comparison",
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Horizontal bar chart comparing ablation variants with optional baseline.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by score
    order = np.argsort(scores)[::-1]
    labels_sorted = [labels[i] for i in order]
    scores_sorted = [scores[i] for i in order]
    
    y = np.arange(len(labels_sorted))
    
    # Color by rank
    best_score = max(scores_sorted)
    colors = [PAL["best"] if s == best_score else PAL["neutral"] for s in scores_sorted]
    
    bars = ax.barh(y, scores_sorted, color=colors, height=0.6, edgecolor="white")
    
    # Baseline line
    if baseline_score is not None:
        ax.axvline(x=baseline_score, color=PAL["worst"], linestyle="--", 
                   linewidth=2, label=baseline_label)
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels_sorted, fontsize=9)
    ax.set_xlabel("Accuracy (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    
    # Value labels
    for bar, score in zip(bars, scores_sorted):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{score:.1f}", va="center", fontsize=8)
    
    if baseline_score is not None:
        ax.legend(loc="lower right", fontsize=9)
    
    ax.invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_delta_bar(
    labels: list[str],
    deltas: list[float],
    title: str = "Improvement over Baseline",
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Diverging bar chart showing delta from baseline.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by delta
    order = np.argsort(deltas)[::-1]
    labels_sorted = [labels[i] for i in order]
    deltas_sorted = [deltas[i] for i in order]
    
    y = np.arange(len(labels_sorted))
    colors = [PAL["best"] if d >= 0 else PAL["worst"] for d in deltas_sorted]
    
    bars = ax.barh(y, deltas_sorted, color=colors, height=0.6)
    ax.axvline(x=0, color="black", linewidth=1)
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels_sorted, fontsize=9)
    ax.set_xlabel("Δ Accuracy (pp)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    
    # Value labels
    for bar, delta in zip(bars, deltas_sorted):
        offset = 0.2 if delta >= 0 else -0.2
        ha = "left" if delta >= 0 else "right"
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                f"{delta:+.1f}", va="center", ha=ha, fontsize=8)
    
    ax.invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# =============================================================================
# Per-Dataset/Capability Analysis
# =============================================================================

def plot_capability_breakdown(
    data: pd.DataFrame,
    variants: list[str] = None,
    title: str = "Per-Capability Performance",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Grouped bar chart showing performance across capabilities for each variant.
    """
    if variants is None:
        variants = data.index.tolist()
    
    data = data.loc[variants]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_variants = len(variants)
    n_caps = len(data.columns)
    
    x = np.arange(n_caps)
    width = 0.8 / n_variants
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_variants))
    
    for i, (variant, color) in enumerate(zip(variants, colors)):
        offset = (i - n_variants/2 + 0.5) * width
        ax.bar(x + offset, data.loc[variant], width, label=variant, color=color)
    
    ax.set_xticks(x)
    ax.set_xticklabels(data.columns, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_dataset_heatmap(
    data: pd.DataFrame,
    title: str = "Per-Dataset Performance",
    figsize: Tuple[float, float] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Heatmap showing all variants × all datasets.
    """
    if figsize is None:
        figsize = (max(8, len(data.columns) * 0.6), max(4, len(data) * 0.4))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(data, annot=True, fmt=".1f", cmap=CMAP_DIVERGING,
                center=data.values.mean(), ax=ax,
                linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Accuracy (%)", "shrink": 0.8},
                annot_kws={"size": 7})
    
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_ylabel("Variant", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# =============================================================================
# Summary Dashboard
# =============================================================================

def plot_ablation_dashboard(
    ablation_name: str,
    summary_1d: pd.DataFrame = None,
    summary_2d: pd.DataFrame = None,
    per_dataset: pd.DataFrame = None,
    title: str = None,
    figsize: Tuple[float, float] = (14, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Multi-panel dashboard summarizing an ablation study.
    """
    if title is None:
        title = f"Ablation Dashboard: {ablation_name}"
    
    # Determine layout based on available data
    n_panels = sum([summary_1d is not None, summary_2d is not None, per_dataset is not None])
    
    if n_panels == 0:
        raise ValueError("Need at least one data source")
    
    fig = plt.figure(figsize=figsize)
    
    if n_panels == 1:
        gs = fig.add_gridspec(1, 1)
        axes = [fig.add_subplot(gs[0, 0])]
    elif n_panels == 2:
        gs = fig.add_gridspec(1, 2)
        axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    else:
        gs = fig.add_gridspec(2, 2)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])
        ]
    
    panel_idx = 0
    
    # 1D summary (bar or line)
    if summary_1d is not None:
        ax = axes[panel_idx]
        if "ablation_value" in summary_1d.columns:
            x = summary_1d["ablation_value"].values
            y = summary_1d["avg_score_pct"].values if "avg_score_pct" in summary_1d.columns else summary_1d.iloc[:, -1].values
            ax.bar(range(len(x)), y, color=PAL["neutral"])
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels(x, rotation=30, ha="right")
        else:
            summary_1d.plot(kind="bar", ax=ax, color=PAL["neutral"])
        ax.set_title("(a) Performance by Variant", fontsize=10, fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, axis="y", alpha=0.3)
        panel_idx += 1
    
    # 2D summary (heatmap)
    if summary_2d is not None:
        ax = axes[panel_idx]
        sns.heatmap(summary_2d, annot=True, fmt=".1f", cmap=CMAP_DIVERGING,
                    center=summary_2d.values.mean(), ax=ax,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title("(b) Grid Search", fontsize=10, fontweight="bold")
        panel_idx += 1
    
    # Per-dataset breakdown
    if per_dataset is not None:
        ax = axes[panel_idx]
        sns.heatmap(per_dataset, annot=True, fmt=".1f", cmap=CMAP_SEQUENTIAL,
                    ax=ax, linewidths=0.3, cbar_kws={"shrink": 0.6},
                    annot_kws={"size": 7})
        ax.set_title("(c) Per-Dataset Breakdown", fontsize=10, fontweight="bold")
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

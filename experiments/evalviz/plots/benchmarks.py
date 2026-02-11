# evalviz/plots/benchmarks_modern.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evalviz.benchmarks import explode_list_col, add_capabilities_from_primitives


def _autopct_counts(values: List[int]):
    total = float(sum(values)) if sum(values) > 0 else 1.0
    def _fmt(pct):
        count = int(round(pct * total / 100.0))
        return f"{pct:.0f}%\n(n={count})" if pct >= 5 else ""
    return _fmt


def plot_donut_task_breakdown(df_bench: pd.DataFrame, title: str = "Task types across used benchmarks"):
    d = df_bench.copy()
    counts = d["task"].fillna("Unknown").value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index.astype(str),
        autopct=_autopct_counts(counts.values.tolist()),
        startangle=90,
        wedgeprops=dict(width=0.45)  # donut
    )
    ax.set_title(title)
    ax.axis("equal")
    plt.tight_layout()
    plt.show()
    return counts


def plot_donut_coco_usage(df_bench: pd.DataFrame, coco_token: str = "COCO", title: str = "COCO usage among used benchmarks"):
    def uses_coco(srcs):
        return coco_token in (srcs or [])

    d = df_bench.copy()
    d["uses_coco"] = d["data_sources"].apply(uses_coco)
    counts = d["uses_coco"].value_counts().rename(index={True: "Uses COCO", False: "No COCO"})

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(
        counts.values,
        labels=counts.index.astype(str),
        autopct=_autopct_counts(counts.values.tolist()),
        startangle=90,
        wedgeprops=dict(width=0.45)
    )
    ax.set_title(title)
    ax.axis("equal")
    plt.tight_layout()
    plt.show()
    return counts


def plot_sources_barh(df_bench: pd.DataFrame, top_k: int = 15, title: str = "Most common data sources (used benchmarks)"):
    d = explode_list_col(df_bench, "data_sources").dropna(subset=["data_sources"]).copy()
    counts = d.groupby("data_sources")["dataset"].nunique().sort_values(ascending=True)
    counts = counts.tail(top_k)

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(counts))))
    ax.barh(counts.index.astype(str), counts.values)
    ax.set_xlabel("#Benchmarks")
    ax.set_title(title)

    # annotate counts
    for y, v in enumerate(counts.values):
        ax.text(v, y, f" {int(v)}", va="center")

    plt.tight_layout()
    plt.show()
    return counts.sort_values(ascending=False)


def plot_benchmark_sizes_barh(df_bench: pd.DataFrame, logx: bool = True, title: str = "Benchmark sizes (#samples)"):
    d = df_bench.dropna(subset=["num_samples"]).copy()
    d = d.sort_values("num_samples", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(d))))
    ax.barh(d["dataset"].astype(str), d["num_samples"].astype(float))
    ax.set_xlabel("#Samples" + (" (log scale)" if logx else ""))
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")

    # annotate (keep short)
    for y, v in enumerate(d["num_samples"].astype(float).values):
        ax.text(v, y, f" {v:,.0f}", va="center")

    plt.tight_layout()
    plt.show()
    return d[["dataset", "num_samples"]].sort_values("num_samples", ascending=False)


def plot_primitives_presence_heatmap(
    df_bench: pd.DataFrame,
    primitives_order: Optional[List[str]] = None,
    title: str = "Benchmark coverage by primitive"
):
    d = df_bench[["dataset", "primitives"]].copy()
    all_prims = sorted({p for ps in d["primitives"] for p in (ps or [])})

    if primitives_order:
        prims = [p for p in primitives_order if p in all_prims] + [p for p in all_prims if p not in primitives_order]
    else:
        prims = all_prims

    datasets = d["dataset"].astype(str).tolist()
    mat = np.zeros((len(datasets), len(prims)), dtype=float)

    for i, ps in enumerate(d["primitives"]):
        ps = set(ps or [])
        for j, p in enumerate(prims):
            mat[i, j] = 1.0 if p in ps else 0.0

    fig, ax = plt.subplots(figsize=(0.75 * len(prims) + 5, 0.35 * len(datasets) + 2.5))
    ax.imshow(mat, aspect="auto")
    ax.set_title(title)

    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)

    ax.set_xticks(range(len(prims)))
    ax.set_xticklabels(prims, rotation=45, ha="right")

    # gridlines to look cleaner
    ax.set_xticks(np.arange(-.5, len(prims), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(datasets), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.show()
    return pd.DataFrame(mat, index=datasets, columns=prims)


def plot_capability_presence_heatmap(
    df_bench: pd.DataFrame,
    primitive_to_capability: Dict[str, List[str]],
    title: str = "Benchmark coverage by capability category"
):
    d = add_capabilities_from_primitives(df_bench, primitive_to_capability)
    all_caps = sorted({c for cs in d["capabilities"] for c in (cs or [])})

    datasets = d["dataset"].astype(str).tolist()
    mat = np.zeros((len(datasets), len(all_caps)), dtype=float)

    for i, cs in enumerate(d["capabilities"]):
        cs = set(cs or [])
        for j, c in enumerate(all_caps):
            mat[i, j] = 1.0 if c in cs else 0.0

    fig, ax = plt.subplots(figsize=(0.75 * len(all_caps) + 5, 0.35 * len(datasets) + 2.5))
    ax.imshow(mat, aspect="auto")
    ax.set_title(title)

    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)

    ax.set_xticks(range(len(all_caps)))
    ax.set_xticklabels(all_caps, rotation=45, ha="right")

    ax.set_xticks(np.arange(-.5, len(all_caps), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(datasets), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.show()
    return pd.DataFrame(mat, index=datasets, columns=all_caps)


def benchmark_overview_table(df_bench: pd.DataFrame) -> pd.DataFrame:
    d = df_bench.copy()
    for col in ["data_sources", "primitives", "image_types", "text_types"]:
        if col in d.columns:
            d[col] = d[col].apply(lambda xs: ", ".join(xs) if isinstance(xs, list) else ("" if xs is None else str(xs)))
    d = d.sort_values(["task", "num_samples"], ascending=[True, False])
    cols = [c for c in ["dataset", "num_samples", "task", "data_sources", "primitives", "license", "image_types", "text_types"] if c in d.columns]
    return d[cols]

from __future__ import annotations
import numpy as np
import pandas as pd

from .preprocess import filter_benchmark


def build_dataset_avg_pivot(
    df: pd.DataFrame,
    metric: str = "text_contrastive_accuracy",
    benchmark_type: str | None = "compositional",
) -> pd.DataFrame:
    d = filter_benchmark(df, benchmark_type)
    d = d[d["metric"] == metric].copy()
    if d.empty:
        raise ValueError("No rows after filtering benchmark_type/metric.")
    dataset_avg = d.groupby(["dataset", "run_label"], as_index=False)["value"].mean()
    pivot = dataset_avg.pivot_table(index="run_label", columns="dataset", values="value")
    pivot = pivot[sorted(pivot.columns)]
    return pivot


def build_capability_pivot(
    df: pd.DataFrame,
    metric: str = "text_contrastive_accuracy",
    benchmark_type: str | None = "compositional",
    exclude_uncat: bool = True,
) -> pd.DataFrame:
    if "capability_category" not in df.columns:
        raise ValueError("df missing capability_category. Run annotate_capabilities first.")
    d = filter_benchmark(df, benchmark_type)
    d = d[d["metric"] == metric].copy()
    if exclude_uncat:
        d = d[~d["capability_category"].astype(str).str.startswith("Multi:")]
        d = d[d["capability_category"] != "Uncategorized"]

    cap_avg = d.groupby(["capability_category", "run_label"], as_index=False)["value"].mean()
    pivot = cap_avg.pivot_table(index="run_label", columns="capability_category", values="value")
    pivot = pivot[sorted(pivot.columns)]
    return pivot


def compute_improvement_vs_baseline(
    df: pd.DataFrame,
    metric: str = "text_contrastive_accuracy",
    benchmark_type: str | None = "compositional",
    baseline_label: str | None = None,
    group: str = "dataset",  # "dataset" or "capability"
    clip_pp: float | None = None,
) -> pd.DataFrame:
    """
    Returns pivot: rows=run_label (non-baseline), cols=group keys, values=improvement_pp
      improvement_pp = 100*(value - baseline_value)
    """
    d = filter_benchmark(df, benchmark_type)
    d = d[d["metric"] == metric].copy()

    if baseline_label is None:
        bl = d.loc[d.get("is_baseline", False) == True, "run_label"].dropna().unique()
        baseline_label = bl[0] if len(bl) else None
    if baseline_label is None:
        raise ValueError("No baseline label found/provided.")

    if group == "dataset":
        key = "dataset"
        grouped = d.groupby(["run_label", key], as_index=False)["value"].mean()
    elif group == "capability":
        if "capability_category" not in d.columns:
            raise ValueError("Missing capability_category. Run annotate_capabilities first.")
        key = "capability_category"
        dd = d[(d[key] != "Uncategorized") & (~d[key].astype(str).str.startswith("Multi:"))]
        grouped = dd.groupby(["run_label", key], as_index=False)["value"].mean()
    else:
        raise ValueError("group must be 'dataset' or 'capability'")

    base = grouped[grouped["run_label"] == baseline_label].set_index(key)["value"]
    other = grouped[grouped["run_label"] != baseline_label].copy()

    other["baseline_value"] = other[key].map(base)
    other["improvement_pp"] = 100.0 * (other["value"] - other["baseline_value"])
    other = other.dropna(subset=["improvement_pp"])

    pivot = other.pivot_table(index="run_label", columns=key, values="improvement_pp")
    pivot = pivot[sorted(pivot.columns)]

    if clip_pp is not None:
        pivot = pivot.clip(lower=-clip_pp, upper=clip_pp)

    return pivot

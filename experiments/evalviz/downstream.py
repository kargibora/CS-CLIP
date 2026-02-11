# evalviz/downstream.py
"""
Functions for building tables and generating LaTeX for downstream benchmarks
(zero-shot classification, retrieval).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .tables import (
    latex_escape,
    _ensure_method_flags,
    _run_info_from_df,
    order_rows_by_groups,
    DEFAULT_ROW_GROUPING,
    RowGrouping,
)


# -----------------------------------------------------------------------------
# DOWNSTREAM BENCHMARK GROUPINGS
# -----------------------------------------------------------------------------

DOWNSTREAM_GROUPINGS = {
    "zero_shot_classification": {
        "ImageNet": ["wds_imagenet1k", "wds_imagenetv2", "wds_imagenet_sketch", "wds_imagenet-o"],
        "Fine-Grained": ["wds_vtab-caltech101", "wds_vtab-cifar10", "wds_sun397"],
    },
    "retrieval": {
        "COCO": ["wds_mscoco_captions"],
        "Flickr": ["wds_flickr8k"],
    },
}

DOWNSTREAM_METRICS = {
    "zero_shot_classification": "acc1",
    "retrieval_t2i": "text_retrieval_recall@1",
    "retrieval_i2t": "image_retrieval_recall@1",
}


# -----------------------------------------------------------------------------
# BUILD DOWNSTREAM TABLES
# -----------------------------------------------------------------------------

def build_downstream_table(
    df: pd.DataFrame,
    task: str = "zero_shot_classification",
    metric: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, bool]]]:
    """
    Build a downstream benchmark table.
    
    Parameters
    ----------
    df : pd.DataFrame
        The full results DataFrame (should have benchmark_type or dataset columns)
    task : str
        One of: "zero_shot_classification", "retrieval_t2i", "retrieval_i2t"
    metric : str, optional
        Override the default metric for the task
    
    Returns
    -------
    pivot : pd.DataFrame
        Rows = run_label (models), columns = subsets (datasets)
    run_info : dict
        run_label -> {is_baseline, is_ours, is_pretrained}
    """
    df = _ensure_method_flags(df)
    
    # Filter to CLIPBench data
    if "benchmark_type" in df.columns:
        d = df[df["benchmark_type"] == "downstream_clipbench"].copy()
    elif "dataset" in df.columns:
        d = df[df["dataset"] == "CLIPBench"].copy()
    else:
        d = df.copy()
    
    if d.empty:
        return pd.DataFrame(), {}
    
    # Select metric
    if metric is None:
        metric = DOWNSTREAM_METRICS.get(task, "acc1")
    
    d = d[d["metric"] == metric].copy()
    
    if d.empty:
        return pd.DataFrame(), {}
    
    run_info = _run_info_from_df(d)
    
    # Average over any duplicates
    dataset_avg = d.groupby(["subset", "run_label"], as_index=False)["value"].mean()
    
    # Pivot: rows = run_label, columns = subset
    pivot = dataset_avg.pivot_table(
        index="run_label",
        columns="subset",
        values="value"
    )
    
    # Sort columns alphabetically
    pivot = pivot[sorted(pivot.columns)]
    
    return pivot, run_info


def build_downstream_summary(
    df: pd.DataFrame,
    include_zs: bool = True,
    include_retrieval: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, bool]]]:
    """
    Build a summary table with averages for each downstream task type.
    
    Returns a pivot with columns like:
    - ZS-ImageNet (avg of ImageNet variants)
    - ZS-FineGrained (avg of fine-grained datasets)
    - Retrieval-T2I (avg text-to-image retrieval)
    - Retrieval-I2T (avg image-to-text retrieval)
    """
    df = _ensure_method_flags(df)
    
    summary_data = []
    
    if include_zs:
        zs_pivot, _ = build_downstream_table(df, task="zero_shot_classification", metric="acc1")
        if not zs_pivot.empty:
            grouping = DOWNSTREAM_GROUPINGS.get("zero_shot_classification", {})
            for group_name, subsets in grouping.items():
                valid_cols = [c for c in subsets if c in zs_pivot.columns]
                if valid_cols:
                    group_avg = zs_pivot[valid_cols].mean(axis=1)
                    for run_label, val in group_avg.items():
                        summary_data.append({
                            "run_label": run_label,
                            "category": f"ZS-{group_name}",
                            "value": val,
                        })
            
            # Overall ZS average
            zs_avg = zs_pivot.mean(axis=1)
            for run_label, val in zs_avg.items():
                summary_data.append({
                    "run_label": run_label,
                    "category": "ZS-Overall",
                    "value": val,
                })
    
    if include_retrieval:
        # T2I retrieval
        t2i_pivot, _ = build_downstream_table(df, task="retrieval_t2i")
        if not t2i_pivot.empty:
            t2i_avg = t2i_pivot.mean(axis=1)
            for run_label, val in t2i_avg.items():
                summary_data.append({
                    "run_label": run_label,
                    "category": "Retrieval-T2I",
                    "value": val,
                })
        
        # I2T retrieval
        i2t_pivot, _ = build_downstream_table(df, task="retrieval_i2t")
        if not i2t_pivot.empty:
            i2t_avg = i2t_pivot.mean(axis=1)
            for run_label, val in i2t_avg.items():
                summary_data.append({
                    "run_label": run_label,
                    "category": "Retrieval-I2T",
                    "value": val,
                })
    
    if not summary_data:
        return pd.DataFrame(), {}
    
    summary_df = pd.DataFrame(summary_data)
    pivot = summary_df.pivot_table(
        index="run_label",
        columns="category",
        values="value"
    )
    
    run_info = _run_info_from_df(df)
    
    return pivot, run_info


# -----------------------------------------------------------------------------
# LATEX TABLE GENERATORS
# -----------------------------------------------------------------------------

def make_latex_downstream_table(
    df: pd.DataFrame,
    task: str = "zero_shot_classification",
    metric: Optional[str] = None,
    caption: str = "",
    label: str = "tab:downstream",
    decimals: int = 1,
    as_percentage: bool = True,
    save_path: Optional[str | Path] = None,
    fit_to_page: bool = True,
    font_size: str = "scriptsize",
    tabcolsep: str = "3pt",
    include_average: bool = True,
    use_grouping: bool = True,
    row_groups: Optional[List[RowGrouping]] = None,
) -> str:
    """
    Generate LaTeX table for downstream benchmarks.
    
    Rows = models, Columns = datasets/subsets
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    task : str
        One of: "zero_shot_classification", "retrieval_t2i", "retrieval_i2t"
    metric : str, optional
        Override the default metric
    caption : str
        LaTeX caption
    label : str
        LaTeX label
    decimals : int
        Decimal places
    as_percentage : bool
        Multiply values by 100
    save_path : str | Path, optional
        Save to file
    fit_to_page : bool
        Use adjustbox
    font_size : str
        LaTeX font size command
    tabcolsep : str
        Column separation
    include_average : bool
        Include Avg column
    use_grouping : bool
        Use multicolumn headers for dataset groups
    row_groups : list, optional
        Custom row grouping
    
    Returns
    -------
    str
        LaTeX table code
    """
    pivot, run_info = build_downstream_table(df, task=task, metric=metric)
    
    if pivot.empty:
        return "% Empty downstream table.\n"
    
    # Order rows by groups
    methods = list(pivot.index)
    ordered_methods, split_idxs = order_rows_by_groups(methods, run_info, groups=row_groups)
    pivot = pivot.loc[ordered_methods]
    
    # Get grouping for multicolumn headers
    grouping = DOWNSTREAM_GROUPINGS.get(task, {}) if use_grouping else {}
    
    # Compute average
    avg = pivot.mean(axis=1)
    
    # Best per column
    best_per_col = {c: pivot[c].idxmax() for c in pivot.columns}
    best_avg = avg.idxmax()
    
    def fmt(v: float) -> str:
        if pd.isna(v):
            return "--"
        v = float(v) * (100.0 if as_percentage else 1.0)
        return f"{v:.{decimals}f}"
    
    # Build LaTeX
    lines = []
    
    if not caption:
        task_name = task.replace("_", " ").title()
        caption = f"Downstream benchmark results ({task_name}). Best per column in \\textbf{{bold}}."
    
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(rf"  \{font_size}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    
    if fit_to_page:
        lines.append(r"  \begin{adjustbox}{max width=\textwidth}")
    
    lines.append(rf"  \setlength{{\tabcolsep}}{{{tabcolsep}}}")
    
    # Build column order based on grouping
    col_order = list(pivot.columns)
    if grouping:
        col_order = []
        for group_name, subsets in grouping.items():
            valid_cols = [c for c in subsets if c in pivot.columns]
            col_order.extend(valid_cols)
        # Add ungrouped columns
        ungrouped = [c for c in pivot.columns if c not in col_order]
        col_order.extend(ungrouped)
        pivot = pivot[[c for c in col_order if c in pivot.columns]]
    
    # Column spec
    n_cols = len(pivot.columns) + (1 if include_average else 0)
    col_spec = "l" + "c" * n_cols
    lines.append(r"  \begin{tabular}{" + col_spec + "}")
    lines.append(r"    \toprule")
    
    # Multicolumn header if grouping exists
    if grouping:
        header1 = [""]
        cmidrule_parts = []
        col_idx = 2  # Start after Model column
        
        for group_name, subsets in grouping.items():
            valid_cols = [c for c in subsets if c in pivot.columns]
            if valid_cols:
                header1.append(f"\\multicolumn{{{len(valid_cols)}}}{{c}}{{{latex_escape(group_name)}}}")
                cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + len(valid_cols) - 1}}}")
                col_idx += len(valid_cols)
        
        # Ungrouped columns
        ungrouped = [c for c in pivot.columns if c not in sum(grouping.values(), [])]
        if ungrouped:
            header1.append(f"\\multicolumn{{{len(ungrouped)}}}{{c}}{{Other}}")
            cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + len(ungrouped) - 1}}}")
        
        if include_average:
            header1.append("")
        
        lines.append("    " + " & ".join(header1) + r" \\")
        lines.append("    " + " ".join(cmidrule_parts))
    
    # Header row (subset names)
    header = ["Model"]
    for c in pivot.columns:
        # Clean up column names (remove wds_ prefix)
        clean_name = c.replace("wds_", "").replace("_", "-")
        header.append(latex_escape(clean_name))
    if include_average:
        header.append("Avg")
    
    lines.append("    " + " & ".join(header) + r" \\")
    lines.append(r"    \midrule")
    
    # Body rows
    for i, m in enumerate(pivot.index.tolist()):
        if i in split_idxs:
            lines.append(r"    \midrule")
        
        info = run_info.get(m, {})
        is_baseline = info.get("is_baseline", False)
        is_ours = info.get("is_ours", False)
        
        row = [latex_escape(m)]
        
        for c in pivot.columns:
            s = fmt(pivot.loc[m, c])
            if m == best_per_col.get(c):
                s = r"\textbf{" + s + "}"
            if is_baseline:
                s = r"\underline{" + s + "}"
            if is_ours:
                s = r"\emph{" + s + "}"
            row.append(s)
        
        if include_average:
            s = fmt(avg.loc[m])
            if m == best_avg:
                s = r"\textbf{" + s + "}"
            if is_baseline:
                s = r"\underline{" + s + "}"
            if is_ours:
                s = r"\emph{" + s + "}"
            row.append(s)
        
        lines.append("    " + " & ".join(row) + r" \\")
    
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    
    if fit_to_page:
        lines.append(r"  \end{adjustbox}")
    
    lines.append(r"\end{table}")
    
    full = "\n".join(lines)
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(full, encoding="utf-8")
        print(f"Saved to: {save_path}")
    
    return full


def build_retrieval_combined_table(
    df: pd.DataFrame,
    datasets: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, bool]]]:
    """
    Build a combined retrieval table with both I2T and T2I metrics.
    
    Returns
    -------
    pivot : pd.DataFrame
        Rows = run_label (models)
        Columns = MultiIndex: (dataset, metric) e.g. (wds_mscoco_captions, I2T), (wds_mscoco_captions, T2I)
    run_info : dict
        run_label -> {is_baseline, is_ours, is_pretrained}
    """
    df = _ensure_method_flags(df)
    
    # Filter to CLIPBench data
    if "benchmark_type" in df.columns:
        d = df[df["benchmark_type"] == "downstream_clipbench"].copy()
    elif "dataset" in df.columns:
        d = df[df["dataset"] == "CLIPBench"].copy()
    else:
        d = df.copy()
    
    if d.empty:
        return pd.DataFrame(), {}
    
    # Filter to retrieval metrics
    retrieval_metrics = ["image_retrieval_recall@1", "text_retrieval_recall@1"]
    d = d[d["metric"].isin(retrieval_metrics)].copy()
    
    if d.empty:
        return pd.DataFrame(), {}
    
    # Filter to specific datasets if provided
    if datasets is not None:
        d = d[d["subset"].isin(datasets)].copy()
    
    run_info = _run_info_from_df(d)
    
    # Map metric to short name
    metric_short = {
        "image_retrieval_recall@1": "I2T",
        "text_retrieval_recall@1": "T2I",
    }
    d["metric_short"] = d["metric"].map(metric_short)
    
    # Average over any duplicates
    dataset_avg = d.groupby(["subset", "metric_short", "run_label"], as_index=False)["value"].mean()
    
    # Pivot with MultiIndex columns
    pivot = dataset_avg.pivot_table(
        index="run_label",
        columns=["subset", "metric_short"],
        values="value"
    )
    
    # Sort columns: by dataset, then I2T before T2I
    if isinstance(pivot.columns, pd.MultiIndex):
        sorted_cols = sorted(pivot.columns, key=lambda x: (x[0], 0 if x[1] == "I2T" else 1))
        pivot = pivot[sorted_cols]
    
    return pivot, run_info


def make_latex_retrieval_table(
    df: pd.DataFrame,
    datasets: Optional[List[str]] = None,
    caption: str = "",
    label: str = "tab:retrieval",
    decimals: int = 1,
    as_percentage: bool = True,
    save_path: Optional[str | Path] = None,
    fit_to_page: bool = True,
    font_size: str = "scriptsize",
    tabcolsep: str = "3pt",
    include_average: bool = True,
    row_groups: Optional[List[RowGrouping]] = None,
) -> str:
    """
    Generate LaTeX table for retrieval benchmarks with BOTH I2T and T2I metrics.
    
    Rows = models
    Columns = datasets with sub-columns for I2T and T2I
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    datasets : list, optional
        Subset of datasets to include (e.g., ["wds_mscoco_captions", "wds_flickr8k"])
    caption : str
        LaTeX caption
    label : str
        LaTeX label
    decimals : int
        Decimal places
    as_percentage : bool
        Multiply values by 100
    save_path : str | Path, optional
        Save to file
    fit_to_page : bool
        Use adjustbox
    font_size : str
        LaTeX font size command
    tabcolsep : str
        Column separation
    include_average : bool
        Include Avg columns for I2T and T2I
    row_groups : list, optional
        Custom row grouping
    
    Returns
    -------
    str
        LaTeX table code
    """
    pivot, run_info = build_retrieval_combined_table(df, datasets=datasets)
    
    if pivot.empty:
        return "% Empty retrieval table.\n"
    
    # Order rows by groups
    methods = list(pivot.index)
    ordered_methods, split_idxs = order_rows_by_groups(methods, run_info, groups=row_groups)
    pivot = pivot.loc[ordered_methods]
    
    # Get unique datasets and metrics
    if isinstance(pivot.columns, pd.MultiIndex):
        unique_datasets = list(dict.fromkeys([c[0] for c in pivot.columns]))
        unique_metrics = ["I2T", "T2I"]
    else:
        return "% Expected MultiIndex columns.\n"
    
    # Compute averages per metric
    avg_i2t = pivot.xs("I2T", level=1, axis=1).mean(axis=1) if ("I2T" in [c[1] for c in pivot.columns]) else None
    avg_t2i = pivot.xs("T2I", level=1, axis=1).mean(axis=1) if ("T2I" in [c[1] for c in pivot.columns]) else None
    
    # Best per column
    best_per_col = {}
    for col in pivot.columns:
        best_per_col[col] = pivot[col].idxmax()
    
    best_avg_i2t = avg_i2t.idxmax() if avg_i2t is not None else None
    best_avg_t2i = avg_t2i.idxmax() if avg_t2i is not None else None
    
    def fmt(v: float) -> str:
        if pd.isna(v):
            return "--"
        v = float(v) * (100.0 if as_percentage else 1.0)
        return f"{v:.{decimals}f}"
    
    # Build LaTeX
    lines = []
    
    if not caption:
        caption = (
            "Retrieval results (Recall@1). I2T = Image-to-Text, T2I = Text-to-Image. "
            "Best per column in \\textbf{bold}."
        )
    
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(rf"  \{font_size}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    
    if fit_to_page:
        lines.append(r"  \begin{adjustbox}{max width=\textwidth}")
    
    lines.append(rf"  \setlength{{\tabcolsep}}{{{tabcolsep}}}")
    
    # Column spec: Model + 2 cols per dataset + 2 avg cols
    n_data_cols = len(unique_datasets) * 2
    n_avg_cols = 2 if include_average else 0
    col_spec = "l" + "c" * (n_data_cols + n_avg_cols)
    lines.append(r"  \begin{tabular}{" + col_spec + "}")
    lines.append(r"    \toprule")
    
    # First header row: dataset names with multicolumn
    header1 = [""]
    cmidrule_parts = []
    col_idx = 2
    
    for ds in unique_datasets:
        clean_ds = ds.replace("wds_", "").replace("_", " ")
        header1.append(f"\\multicolumn{{2}}{{c}}{{{latex_escape(clean_ds)}}}")
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + 1}}}")
        col_idx += 2
    
    if include_average:
        header1.append(r"\multicolumn{2}{c}{Average}")
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + 1}}}")
    
    lines.append("    " + " & ".join(header1) + r" \\")
    lines.append("    " + " ".join(cmidrule_parts))
    
    # Second header row: I2T / T2I for each dataset
    header2 = ["Model"]
    for ds in unique_datasets:
        header2.append("I2T")
        header2.append("T2I")
    if include_average:
        header2.append("I2T")
        header2.append("T2I")
    
    lines.append("    " + " & ".join(header2) + r" \\")
    lines.append(r"    \midrule")
    
    # Body rows
    for i, m in enumerate(pivot.index.tolist()):
        if i in split_idxs:
            lines.append(r"    \midrule")
        
        info = run_info.get(m, {})
        is_baseline = info.get("is_baseline", False)
        is_ours = info.get("is_ours", False)
        
        row = [latex_escape(m)]
        
        for ds in unique_datasets:
            for metric in ["I2T", "T2I"]:
                col = (ds, metric)
                if col in pivot.columns:
                    val = pivot.loc[m, col]
                    s = fmt(val)
                    if m == best_per_col.get(col):
                        s = r"\textbf{" + s + "}"
                else:
                    s = "--"
                
                if is_baseline:
                    s = r"\underline{" + s + "}"
                if is_ours:
                    s = r"\emph{" + s + "}"
                row.append(s)
        
        if include_average:
            # I2T average
            if avg_i2t is not None:
                s = fmt(avg_i2t.loc[m])
                if m == best_avg_i2t:
                    s = r"\textbf{" + s + "}"
                if is_baseline:
                    s = r"\underline{" + s + "}"
                if is_ours:
                    s = r"\emph{" + s + "}"
                row.append(s)
            else:
                row.append("--")
            
            # T2I average
            if avg_t2i is not None:
                s = fmt(avg_t2i.loc[m])
                if m == best_avg_t2i:
                    s = r"\textbf{" + s + "}"
                if is_baseline:
                    s = r"\underline{" + s + "}"
                if is_ours:
                    s = r"\emph{" + s + "}"
                row.append(s)
            else:
                row.append("--")
        
        lines.append("    " + " & ".join(row) + r" \\")
    
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    
    if fit_to_page:
        lines.append(r"  \end{adjustbox}")
    
    lines.append(r"\end{table}")
    
    full = "\n".join(lines)
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(full, encoding="utf-8")
        print(f"Saved to: {save_path}")
    
    return full


def make_latex_downstream_summary_table(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "tab:downstream_summary",
    decimals: int = 1,
    as_percentage: bool = True,
    save_path: Optional[str | Path] = None,
    font_size: str = "small",
    tabcolsep: str = "4pt",
    row_groups: Optional[List[RowGrouping]] = None,
) -> str:
    """
    Generate LaTeX summary table for all downstream tasks.
    """
    pivot, run_info = build_downstream_summary(df)
    
    if pivot.empty:
        return "% Empty downstream summary table.\n"
    
    # Order rows
    methods = list(pivot.index)
    ordered_methods, split_idxs = order_rows_by_groups(methods, run_info, groups=row_groups)
    pivot = pivot.loc[ordered_methods]
    
    # Best per column
    best_per_col = {c: pivot[c].idxmax() for c in pivot.columns}
    
    # Overall average
    avg = pivot.mean(axis=1)
    best_avg = avg.idxmax()
    
    def fmt(v: float) -> str:
        if pd.isna(v):
            return "--"
        v = float(v) * (100.0 if as_percentage else 1.0)
        return f"{v:.{decimals}f}"
    
    lines = []
    
    if not caption:
        caption = "Summary of downstream benchmark results. Best per column in \\textbf{bold}."
    
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(rf"  \{font_size}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(rf"  \setlength{{\tabcolsep}}{{{tabcolsep}}}")
    
    col_spec = "l" + "c" * (len(pivot.columns) + 1)
    lines.append(r"  \begin{tabular}{" + col_spec + "}")
    lines.append(r"    \toprule")
    
    # Header
    header = ["Model"] + [latex_escape(c) for c in pivot.columns] + ["Overall"]
    lines.append("    " + " & ".join(header) + r" \\")
    lines.append(r"    \midrule")
    
    # Body
    for i, m in enumerate(pivot.index.tolist()):
        if i in split_idxs:
            lines.append(r"    \midrule")
        
        info = run_info.get(m, {})
        is_baseline = info.get("is_baseline", False)
        is_ours = info.get("is_ours", False)
        
        row = [latex_escape(m)]
        
        for c in pivot.columns:
            s = fmt(pivot.loc[m, c])
            if m == best_per_col.get(c):
                s = r"\textbf{" + s + "}"
            if is_baseline:
                s = r"\underline{" + s + "}"
            if is_ours:
                s = r"\emph{" + s + "}"
            row.append(s)
        
        # Overall
        s = fmt(avg.loc[m])
        if m == best_avg:
            s = r"\textbf{" + s + "}"
        if is_baseline:
            s = r"\underline{" + s + "}"
        if is_ours:
            s = r"\emph{" + s + "}"
        row.append(s)
        
        lines.append("    " + " & ".join(row) + r" \\")
    
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    
    full = "\n".join(lines)
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(full, encoding="utf-8")
        print(f"Saved to: {save_path}")
    
    return full

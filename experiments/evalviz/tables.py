from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# 1) Benchmark subset groupings (your old dict)
# ---------------------------

BENCHMARK_GROUPINGS = {
    "SugarCrepe": {
        "add":     ["add_att", "add_obj"],
        "replace": ["replace_att", "replace_obj", "replace_rel"],
        "swap":    ["swap_att", "swap_obj"],
    },
    "SugarCrepe++": {
        "replace": ["replace_attribute", "replace_object", "replace_relation"],
        "swap":    ["swap_object", "swap_atribute"],
    },
    "ControlledImages": {
        "WhatsUp": ["A", "B"],
        "COCO":    ["COCO-One", "COCO-Two"],
        "VG":      ["VG-One", "VG-Two"],
    },
    "VL_CheckList": {
        "attr": ["attr_color", "attr_material", "attr_size", "attr_state", "attr_action"],
        "obj":  ["obj_location", "obj_size"],
        "rel":  ["rel_action", "rel_spatial"],
    },
}


# ---------------------------
# 2) Utilities
# ---------------------------

def latex_escape(s: str) -> str:
    s = str(s)
    s = s.replace("\\", r"\textbackslash ")
    s = s.replace("_", r"\_")
    s = s.replace("%", r"\%")
    s = s.replace("&", r"\&")
    s = s.replace("#", r"\#")
    return s


def _ensure_method_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce presence of the new-architecture flags.
    Assumes attach_method_metadata() has already run.
    """
    df = df.copy()
    for col in ["is_baseline", "is_ours", "is_pretrained"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    return df


def _run_info_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, bool]]:
    """
    run_label -> flags
    """
    cols = ["run_label", "is_baseline", "is_ours", "is_pretrained"]
    keep = [c for c in cols if c in df.columns]
    info = (
        df[keep]
        .drop_duplicates(subset=["run_label"])
        .set_index("run_label")
        .to_dict(orient="index")
    )
    # fill missing keys
    for rl in df["run_label"].dropna().unique():
        info.setdefault(rl, {})
        for k in ["is_baseline", "is_ours", "is_pretrained"]:
            info[rl].setdefault(k, False)
    return info


# ---------------------------
# 3) Subgroup ordering (new!)
# ---------------------------

@dataclass(frozen=True)
class RowGrouping:
    """
    Controls how rows are ordered and where midrules appear.
    """
    name: str
    predicate: callable  # predicate(run_label, run_info_dict) -> bool


DEFAULT_ROW_GROUPING: List[RowGrouping] = [
    RowGrouping("Baseline",   lambda rl, info: info.get(rl, {}).get("is_baseline", False)),
    RowGrouping("Pretrained", lambda rl, info: (not info.get(rl, {}).get("is_baseline", False)) and info.get(rl, {}).get("is_pretrained", False)),
    RowGrouping("External",   lambda rl, info: (not info.get(rl, {}).get("is_baseline", False)) and (not info.get(rl, {}).get("is_pretrained", False)) and (not info.get(rl, {}).get("is_ours", False))),
    RowGrouping("Ours",       lambda rl, info: info.get(rl, {}).get("is_ours", False)),
]


def order_rows_by_groups(
    row_labels: List[str],
    run_info: Dict[str, Dict[str, bool]],
    groups: Optional[List[RowGrouping]] = None,
) -> Tuple[List[str], List[int]]:
    """
    Returns:
      ordered_rows
      split_indices: indices in ordered_rows where you should insert a \midrule before that row.
    """
    groups = groups or DEFAULT_ROW_GROUPING

    assigned = set()
    buckets: List[List[str]] = []

    for g in groups:
        members = [rl for rl in row_labels if (rl not in assigned) and g.predicate(rl, run_info)]
        for m in members:
            assigned.add(m)
        buckets.append(members)

    # anything unassigned goes last (rare)
    leftovers = [rl for rl in row_labels if rl not in assigned]
    if leftovers:
        buckets.append(leftovers)

    ordered = [rl for b in buckets for rl in b]

    # split indices: add midrule between non-empty buckets
    splits = []
    idx = 0
    for b in buckets:
        if not b:
            continue
        if idx != 0:
            splits.append(idx)
        idx += len(b)

    return ordered, splits


# ---------------------------
# 4) Build compositional tables (detailed + compact)
# ---------------------------

def build_compositional_tables_for_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    metric: str = "text_contrastive_accuracy",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    detailed: rows=run_label, cols=subset
    compact:  rows=run_label, cols=combined categories (if grouping exists)
    """
    df = _ensure_method_flags(df)

    if "benchmark_type" in df.columns:
        d = df[(df["benchmark_type"] == "compositional") &
               (df["dataset"] == dataset_name) &
               (df["metric"] == metric)].copy()
    else:
        d = df[(df["dataset"] == dataset_name) & (df["metric"] == metric)].copy()

    if d.empty:
        return pd.DataFrame(), None

    # average duplicates (seeds)
    d_grouped = (
        d.groupby(["run_label", "subset"], as_index=False)["value"]
        .mean()
    )

    detailed = d_grouped.pivot_table(
        index="run_label",
        columns="subset",
        values="value"
    ).sort_index(axis=1)

    grouping = BENCHMARK_GROUPINGS.get(dataset_name)
    compact = None
    if grouping is not None:
        compact_rows = {}
        for run_label, row in detailed.iterrows():
            row_vals = {}
            for cat_name, subsets in grouping.items():
                valid = [s for s in subsets if s in detailed.columns]
                row_vals[cat_name] = float("nan") if not valid else row[valid].mean()
            compact_rows[run_label] = row_vals
        compact = pd.DataFrame.from_dict(compact_rows, orient="index").sort_index(axis=1)

    return detailed, compact


def build_all_compositional_tables(
    df: pd.DataFrame,
    metric: str = "text_contrastive_accuracy",
) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
    if "benchmark_type" in df.columns:
        comp = df[(df["benchmark_type"] == "compositional") & (df["metric"] == metric)].copy()
    else:
        comp = df[df["metric"] == metric].copy()

    results = {}
    for ds in sorted(comp["dataset"].dropna().unique()):
        det, compa = build_compositional_tables_for_dataset(comp, ds, metric=metric)
        results[ds] = {"detailed": det, "compact": compa}
    return results


# ---------------------------
# 5) Summary table (dataset averages)
# ---------------------------

def build_summary_table(
    df: pd.DataFrame,
    metric: str = "text_contrastive_accuracy",
    benchmark_type: str = "compositional",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, bool]]]:
    """
    Build summary table with hierarchical averaging:
    1. First average within each (dataset, subset, run_label) - handles duplicate rows
    2. Then average across subsets within each (dataset, run_label) - equal weight per subset
    """
    df = _ensure_method_flags(df)

    d = df[(df.get("benchmark_type", benchmark_type) == benchmark_type) & (df["metric"] == metric)].copy() \
        if "benchmark_type" in df.columns else df[df["metric"] == metric].copy()

    if d.empty:
        return pd.DataFrame(), {}

    run_info = _run_info_from_df(d)

    # Hierarchical averaging: first by subset (handles duplicates), then by dataset (equal weight per subset)
    subset_avg = d.groupby(["dataset", "subset", "run_label"], as_index=False)["value"].mean()
    dataset_avg = subset_avg.groupby(["dataset", "run_label"], as_index=False)["value"].mean()
    pivot = dataset_avg.pivot_table(index="run_label", columns="dataset", values="value")
    pivot = pivot[sorted(pivot.columns)]

    return pivot, run_info


# ---------------------------
# 6) Capability tables (absolute + improvement)
# ---------------------------

def build_capability_table(
    df: pd.DataFrame,
    metric: str = "text_contrastive_accuracy",
    level: str = "top",  # "top" for main paper, "sub" for appendix
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build capability table at specified hierarchy level.
    
    Args:
        df: DataFrame with capability_top and capability_sub columns
        metric: Which metric to use
        level: "top" for top-level capabilities, "sub" for sub-level
        
    Returns:
        pivot: rows=capability, cols=run_label
        meta: run_label -> flags
    """
    df = _ensure_method_flags(df)
    
    # Select capability column based on level
    if level == "top":
        cap_col = "capability_top"
    elif level == "sub":
        cap_col = "capability_sub"
    else:
        # Fallback for legacy format
        cap_col = "capability_category"
    
    if cap_col not in df.columns:
        # Fallback to old column name for backwards compatibility
        if "capability_category" in df.columns:
            cap_col = "capability_category"
        else:
            raise ValueError(f"df missing {cap_col}. Run annotate_capabilities_hierarchical() first.")

    d = df[(df["benchmark_type"] == "compositional") & (df["metric"] == metric)].copy()

    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = d[~d[cap_col].astype(str).str.startswith("Multi:")]
    d = d[d[cap_col] != "Uncategorized"]

    aggregated = d.groupby([cap_col, "run_label"], as_index=False)["value"].mean()

    pivot = aggregated.pivot_table(
        index=cap_col,
        columns="run_label",
        values="value"
    ).sort_index()

    meta = (
        d.drop_duplicates(subset=["run_label"])
        .set_index("run_label")[["is_baseline", "is_ours", "is_pretrained"]]
    )

    return pivot, meta


def compute_dataset_improvements_vs_baseline(
    df: pd.DataFrame,
    metric: str = "text_contrastive_accuracy",
    baseline_label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns improvement pivot: rows=run_label, cols=dataset, values=improvement_pp
    """
    df = _ensure_method_flags(df)
    d = df[(df["benchmark_type"] == "compositional") & (df["metric"] == metric)].copy()

    if baseline_label is None:
        bl = d.loc[d["is_baseline"] == True, "run_label"].dropna().unique()
        if len(bl) == 0:
            raise ValueError("No baseline found (is_baseline=True).")
        baseline_label = bl[0]

    dataset_avg = d.groupby(["dataset", "run_label"], as_index=False)["value"].mean()
    piv = dataset_avg.pivot_table(index="run_label", columns="dataset", values="value")

    base = piv.loc[baseline_label]
    imp = (piv.subtract(base, axis=1) * 100.0)
    imp = imp.drop(index=baseline_label, errors="ignore")
    return imp


def compute_capability_improvements_vs_baseline(
    df: pd.DataFrame,
    metric: str = "text_contrastive_accuracy",
    baseline_label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns improvement pivot: rows=run_label, cols=capability, values=improvement_pp
    """
    pivot, meta = build_capability_table(df, metric=metric)
    if pivot.empty:
        return pd.DataFrame()

    if baseline_label is None:
        # baseline label from meta
        candidates = meta.index[meta["is_baseline"] == True].tolist()
        if not candidates:
            raise ValueError("No baseline found (is_baseline=True).")
        baseline_label = candidates[0]

    base = pivot[baseline_label]
    imp = (pivot.subtract(base, axis=0) * 100.0)
    imp = imp.drop(columns=[baseline_label], errors="ignore").T  # rows=run_label
    return imp


# ---------------------------
# 7) LaTeX generators with subgroup midrules (NEW!)
# ---------------------------

def make_latex_summary_table(
    df: pd.DataFrame,
    metric: str = "text_contrastive_accuracy",
    caption: str = "",
    label: str = "tab:summary_compositional",
    decimals: int = 1,
    as_percentage: bool = True,
    save_path: str | Path | None = None,
    fit_to_page: bool = True,
    font_size: str = "scriptsize",
    tabcolsep: str = "2pt",
    include_overall: bool = True,
    max_cols_per_table: int = 12,
    row_groups: Optional[List[RowGrouping]] = None,
) -> str:
    pivot, run_info = build_summary_table(df, metric=metric)

    if pivot.empty:
        return "% Empty summary table.\n"

    methods = list(pivot.index)
    ordered_methods, split_idxs = order_rows_by_groups(methods, run_info, groups=row_groups)
    pivot = pivot.loc[ordered_methods]

    datasets = list(pivot.columns)
    chunks = [datasets[i:i + max_cols_per_table] for i in range(0, len(datasets), max_cols_per_table)]
    out_tables = []

    def fmt(v: float) -> str:
        if pd.isna(v):
            return "--"
        v = float(v) * (100.0 if as_percentage else 1.0)
        return f"{v:.{decimals}f}"

    for part_i, ds_chunk in enumerate(chunks, start=1):
        chunk = pivot[ds_chunk]
        avg = chunk.mean(axis=1)

        best_per_col = {c: chunk[c].idxmax() for c in ds_chunk}
        best_avg = avg.idxmax()

        part = f" (Part {part_i}/{len(chunks)})" if len(chunks) > 1 else ""
        cap = caption or (
            f"Summary of compositional benchmark results ({metric}){part}. "
            "Each dataset averaged over all subsets."
        )
        lab = f"{label}_part{part_i}" if len(chunks) > 1 else label

        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"  \centering")
        lines.append(rf"  \{font_size}")
        lines.append(f"  \\caption{{{cap}}}")
        lines.append(f"  \\label{{{lab}}}")
        if fit_to_page:
            lines.append(r"  \begin{adjustbox}{max width=\textwidth}")
        lines.append(rf"  \setlength{{\tabcolsep}}{{{tabcolsep}}}")

        n_cols = len(ds_chunk) + (1 if include_overall else 0)
        lines.append(r"  \begin{tabular}{l" + "c" * n_cols + "}")
        lines.append(r"    \toprule")

        header = ["Model"] + [latex_escape(c) for c in ds_chunk] + (["Avg"] if include_overall else [])
        lines.append("    " + " & ".join(header) + r" \\")
        lines.append(r"    \midrule")

        for i, m in enumerate(chunk.index.tolist()):
            if i in split_idxs:
                lines.append(r"    \midrule")

            info = run_info.get(m, {})
            is_baseline = info.get("is_baseline", False)
            is_ours = info.get("is_ours", False)
            is_pretrained = info.get("is_pretrained", False)

            row = [latex_escape(m)]
            for c in ds_chunk:
                s = fmt(chunk.loc[m, c])
                if m == best_per_col[c]:
                    s = r"\textbf{" + s + "}"
                if is_baseline:
                    s = r"\underline{" + s + "}"
                if is_ours:
                    s = r"\emph{" + s + "}"
                # optional: mark pretrained with \dagger
                if is_pretrained and (not is_baseline) and (not is_ours):
                    s = s  # keep style, or wrap in \textit{} if you want
                row.append(s)

            if include_overall:
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

        out_tables.append("\n".join(lines))

    full = "\n\n".join(out_tables)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(full, encoding="utf-8")
    return full


def make_latex_from_compositional_table(
    table: pd.DataFrame,
    df_src: pd.DataFrame,
    dataset_name: str,
    metric: str = "text_contrastive_accuracy",
    caption: str = "",
    label: str = "tab:comp",
    decimals: int = 1,
    as_percentage: bool = True,
    use_multicolumn_header: bool = True,
    fit_to_page: bool = True,
    font_size: str = "small",
    tabcolsep: str = "3pt",
    save_path: str | Path | None = None,
    row_groups: Optional[List[RowGrouping]] = None,
) -> str:
    """
    Works for both "detailed" and "compact" tables (same format).
    Adds subgroup midrules using new metadata.
    """
    if table.empty:
        return "% Empty table.\n"

    df_src = _ensure_method_flags(df_src)
    run_info = _run_info_from_df(df_src[(df_src["dataset"] == dataset_name) & (df_src["metric"] == metric)])

    tab = table.copy()
    methods = list(tab.index)
    ordered_methods, split_idxs = order_rows_by_groups(methods, run_info, groups=row_groups)
    tab = tab.loc[ordered_methods]

    grouping = BENCHMARK_GROUPINGS.get(dataset_name) if use_multicolumn_header else None

    # --- column segments (same idea as your old code) ---
    if grouping is not None:
        col_to_cat: Dict[str, Optional[str]] = {}
        for cat, subs in grouping.items():
            for s in subs:
                if s in tab.columns and s not in col_to_cat:
                    col_to_cat[s] = cat

        ordered_cols: List[str] = []
        for cat in grouping.keys():
            cols_for_cat = [c for c in tab.columns if col_to_cat.get(c) == cat]
            ordered_cols.extend(cols_for_cat)
        for c in tab.columns:
            if c not in ordered_cols:
                ordered_cols.append(c)

        tab = tab[ordered_cols]

        segments: List[Tuple[Optional[str], List[str]]] = []
        cur_cat = None
        cur_cols: List[str] = []
        for c in ordered_cols:
            cat = col_to_cat.get(c)
            if cat != cur_cat:
                if cur_cols:
                    segments.append((cur_cat, cur_cols))
                cur_cat, cur_cols = cat, [c]
            else:
                cur_cols.append(c)
        if cur_cols:
            segments.append((cur_cat, cur_cols))
    else:
        ordered_cols = list(tab.columns)
        segments = [(None, [c]) for c in ordered_cols]

    # category averages only if a segment has >1 column and has a real category name
    category_avgs = {}
    for cat, cols in segments:
        if cat is not None and len(cols) > 1:
            category_avgs[cat] = tab[cols].mean(axis=1)
    overall_avg = tab[ordered_cols].mean(axis=1)

    # best per column + best per avg
    best_per_col = {c: tab[c].idxmax() for c in ordered_cols}
    best_per_avg = {cat: s.idxmax() for cat, s in category_avgs.items()}
    best_overall = overall_avg.idxmax()

    def fmt(v: float) -> str:
        if pd.isna(v):
            return "--"
        v = float(v) * (100.0 if as_percentage else 1.0)
        return f"{v:.{decimals}f}"

    if not caption:
        caption = (
            f"{latex_escape(dataset_name)} compositional results ({metric}). "
            "Best per column in \\textbf{bold}."
        )

    # column spec with visual separators between segments + overall avg
    col_spec_parts = ["p{5cm}"]
    for idx, (cat, cols) in enumerate(segments):
        if idx > 0:
            col_spec_parts.append("|")
        col_spec_parts.append("c" * len(cols))
        if cat in category_avgs:
            col_spec_parts.append("c")
    col_spec_parts.append("|c")
    col_spec = "".join(col_spec_parts)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(rf"  \{font_size}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    if fit_to_page:
        lines.append(r"  \begin{adjustbox}{max width=\textwidth}")
    lines.append(rf"  \setlength{{\tabcolsep}}{{{tabcolsep}}}")
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    # header: either grouped multicolumn or simple
    has_multicol = any(cat is not None and len(cols) > 1 for cat, cols in segments)
    if has_multicol:
        row1 = ["Method"]
        row2 = [""]

        for cat, cols in segments:
            if cat is not None and len(cols) > 1:
                n = len(cols) + (1 if cat in category_avgs else 0)
                row1.append(rf"\multicolumn{{{n}}}{{c}}{{{latex_escape(cat)}}}")
                row2.extend([latex_escape(c) for c in cols])
                if cat in category_avgs:
                    row2.append("Avg")
            else:
                row1.append(latex_escape(cols[0]))
                row2.append("")

        row1.append("Overall")
        row2.append("Avg")

        lines.append("    " + " & ".join(row1) + r" \\")
        if any(x != "" for x in row2[1:]):
            lines.append("    " + " & ".join(row2) + r" \\")
    else:
        header = ["Method"] + [latex_escape(c) for c in ordered_cols] + ["Overall Avg"]
        lines.append("    " + " & ".join(header) + r" \\")

    lines.append(r"    \midrule")

    # rows with subgroup midrules
    for i, rl in enumerate(tab.index.tolist()):
        if i in split_idxs:
            lines.append(r"    \midrule")

        info = run_info.get(rl, {})
        is_baseline = info.get("is_baseline", False)
        is_ours = info.get("is_ours", False)

        row = [latex_escape(rl)]

        for cat, cols in segments:
            for c in cols:
                s = fmt(tab.loc[rl, c])
                if rl == best_per_col[c]:
                    s = r"\textbf{" + s + "}"
                if is_baseline:
                    s = r"\underline{" + s + "}"
                if is_ours:
                    s = r"\emph{" + s + "}"
                row.append(s)

            if cat in category_avgs:
                s = fmt(category_avgs[cat].loc[rl])
                if rl == best_per_avg[cat]:
                    s = r"\textbf{" + s + "}"
                if is_baseline:
                    s = r"\underline{" + s + "}"
                if is_ours:
                    s = r"\emph{" + s + "}"
                row.append(s)

        s = fmt(overall_avg.loc[rl])
        if rl == best_overall:
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

    out = "\n".join(lines)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(out, encoding="utf-8")
    return out


def make_latex_capability_table_models_rows(
    df: pd.DataFrame,
    metric: str = "text_contrastive_accuracy",
    level: str = "top",  # "top" for main paper, "sub" for appendix
    caption: str = "",
    label: str = "tab:capability_models_rows",
    decimals: int = 1,
    as_percentage: bool = True,
    save_path: str | Path | None = None,
    fit_to_page: bool = True,
    font_size: str = "scriptsize",
    tabcolsep: str = "2pt",
    max_cols_per_table: int = 10,         # split capabilities into parts if needed
    row_groups: Optional[List[RowGrouping]] = None,
    capability_order: Optional[List[str]] = None,  # optional ordering of capabilities
) -> str:
    """
    Capability-wise table with:
      rows = models (run_label)
      cols = capabilities (at specified level: top or sub)
      
    Args:
        level: "top" for top-level capabilities (main paper), 
               "sub" for sub-level (appendix)
        capability_order: Optional list to order capabilities
    """

    # build_capability_table returns:
    #   pivot: rows=capability, cols=run_label
    #   meta:  run_label -> flags
    pivot_cap, meta = build_capability_table(df, metric=metric, level=level)
    if pivot_cap.empty:
        return "% Empty capability table.\n"

    # transpose -> rows=models, cols=capabilities
    tab = pivot_cap.T  # index=run_label, columns=capability
    
    # Order capabilities if specified
    if capability_order:
        ordered_caps = [c for c in capability_order if c in tab.columns]
        other_caps = [c for c in tab.columns if c not in ordered_caps]
        tab = tab[ordered_caps + other_caps]

    # run_label -> flags
    run_info: Dict[str, Dict[str, bool]] = meta.fillna(False).astype(bool).to_dict(orient="index")

    # order rows by groups and compute where to insert midrules
    methods = list(tab.index)
    ordered_methods, split_idxs = order_rows_by_groups(methods, run_info, groups=row_groups)
    tab = tab.loc[ordered_methods]

    # split columns into chunks (capabilities)
    caps = list(tab.columns)
    chunks = [caps[i:i + max_cols_per_table] for i in range(0, len(caps), max_cols_per_table)]

    def fmt(v: float) -> str:
        if pd.isna(v):
            return "--"
        v = float(v) * (100.0 if as_percentage else 1.0)
        return f"{v:.{decimals}f}"

    def style_cell(s: str, method: str, is_best: bool) -> str:
        info = run_info.get(method, {})
        if info.get("is_baseline", False):
            s = r"\underline{" + s + "}"
        if info.get("is_ours", False):
            s = r"\emph{" + s + "}"
        if is_best:
            s = r"\textbf{" + s + "}"
        return s

    out_tables: List[str] = []

    for part_i, cap_chunk in enumerate(chunks, start=1):
        chunk = tab[cap_chunk]
        avg = chunk.mean(axis=1)

        # best per capability column
        best_per_col = {}
        for c in cap_chunk:
            col = chunk[c].dropna()
            best_per_col[c] = col.idxmax() if len(col) else None
        best_avg = avg.dropna().idxmax() if avg.dropna().shape[0] else None

        part = f" (Part {part_i}/{len(chunks)})" if len(chunks) > 1 else ""
        cap = caption or (
            f"Capability-wise performance ({metric}){part}. "
            r"Best per column in \textbf{bold}, baseline \underline{underlined}, our models \emph{italic}."
        )
        lab = f"{label}_part{part_i}" if len(chunks) > 1 else label

        lines: List[str] = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"  \centering")
        lines.append(rf"  \{font_size}")
        lines.append(f"  \\caption{{{cap}}}")
        lines.append(f"  \\label{{{lab}}}")
        if fit_to_page:
            lines.append(r"  \begin{adjustbox}{max width=\textwidth}")
        lines.append(rf"  \setlength{{\tabcolsep}}{{{tabcolsep}}}")

        # column spec: method + capabilities + Avg
        col_spec = "p{5.0cm}" + "c" * (len(cap_chunk) + 1)
        lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
        lines.append(r"    \toprule")

        header = ["Model"] + [latex_escape(c) for c in cap_chunk] + ["Avg"]
        lines.append("    " + " & ".join(header) + r" \\")
        lines.append(r"    \midrule")

        # body rows with midrules between method groups
        for i, m in enumerate(chunk.index.tolist()):
            if i in split_idxs:
                lines.append(r"    \midrule")

            row = [latex_escape(m)]
            for c in cap_chunk:
                s = fmt(chunk.loc[m, c])
                is_best = (best_per_col[c] is not None and m == best_per_col[c])
                s = style_cell(s, m, is_best)
                row.append(s)

            s_avg = fmt(avg.loc[m])
            is_best_avg = (best_avg is not None and m == best_avg)
            s_avg = style_cell(s_avg, m, is_best_avg)
            row.append(s_avg)

            lines.append("    " + " & ".join(row) + r" \\")

        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        if fit_to_page:
            lines.append(r"  \end{adjustbox}")
        lines.append(r"\end{table}")

        out_tables.append("\n".join(lines))

    out = "\n\n".join(out_tables)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(out, encoding="utf-8")
    return out


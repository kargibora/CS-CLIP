from __future__ import annotations
import pandas as pd
from typing import Any, Dict, List
import re


def apply_mappings(df: pd.DataFrame, benchmarks_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies dataset/subset/metric alias mappings from benchmarks.json.
    When aliasing creates duplicates, keeps the aliased row (drops the original).
    """
    df = df.copy()

    ds_alias = benchmarks_cfg.get("dataset_aliases", {})
    subset_alias = benchmarks_cfg.get("subset_aliases", {})
    metric_alias = benchmarks_cfg.get("metric_aliases", {})

    if "dataset" in df.columns and isinstance(ds_alias, dict):
        df["dataset"] = df["dataset"].replace(ds_alias)

    if "subset" in df.columns and isinstance(subset_alias, dict):
        df["subset"] = df["subset"].replace(subset_alias)

    # metric aliases (including scoped keys like "macro_contrastive_accuracy@ARO")
    if "metric" in df.columns:
        # Mark rows that WILL be aliased (these take priority over originals)
        df["_was_aliased"] = False
        
        # Collect all source metric names that will be renamed
        global_map = {k: v for k, v in metric_alias.items() if "@" not in k}
        
        # Mark rows that match global aliases
        if global_map:
            alias_mask = df["metric"].isin(global_map.keys())
            df.loc[alias_mask, "_was_aliased"] = True
            df["metric"] = df["metric"].replace(global_map)

        # Mark and apply scoped aliases
        for k, v in metric_alias.items():
            if "@" in k:
                m, ds = k.split("@", 1)
                mask = (df["metric"] == m) & (df["dataset"] == ds)
                df.loc[mask, "_was_aliased"] = True
                df.loc[mask, "metric"] = v

        # Drop duplicates: prefer ALIASED rows (drop originals)
        # Include step/run_label to avoid deduplicating across different checkpoints
        key_cols = ["run_label", "step", "dataset", "subset", "metric"]
        key_cols = [c for c in key_cols if c in df.columns]
        
        if key_cols:
            # Sort so aliased rows come FIRST (True > False when ascending=False)
            df = df.sort_values("_was_aliased", ascending=False)
            df = df.drop_duplicates(subset=key_cols, keep="first")
        
        df = df.drop(columns=["_was_aliased"])

    return df

_CLIPBENCH_PATTERNS = [
    r"^CLIPBench$",
]

def is_clipbench_dataset(dataset_name: str) -> bool:
    """
    Returns True if dataset belongs to downstream CLIPBench-style evaluations.
    Customize patterns above for your logging conventions.
    """
    if dataset_name is None:
        return False
    ds = str(dataset_name).strip()
    for pat in _CLIPBENCH_PATTERNS:
        if re.search(pat, ds, flags=re.IGNORECASE):
            return True
    return False


def add_dataset_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - is_clipbench: bool
      - benchmark_type: {'downstream_clipbench','compositional'}
    """
    df = df.copy()
    df["is_clipbench"] = df["dataset"].apply(is_clipbench_dataset)
    df["benchmark_type"] = df["is_clipbench"].map(
        {True: "downstream_clipbench", False: "compositional"}
    )
    return df



def apply_dataset_merge_rules(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    dataset_col: str = "dataset",
    subset_col: str = "subset",
) -> pd.DataFrame:
    """
    Applies cfg["merge_datasets_into_subsets"] rules.
    Each rule merges multiple dataset names into one dataset, moving the original dataset into subset.

    Idempotent:
      - If dataset already equals target and subset already equals one of the source dataset names, nothing breaks.
    """
    out = df.copy()

    rules: List[Dict[str, Any]] = cfg.get("merge_datasets_into_subsets", [])
    if not rules:
        return out

    if dataset_col not in out.columns:
        raise ValueError(f"apply_dataset_merge_rules: missing '{dataset_col}' column")

    if subset_col not in out.columns:
        out[subset_col] = "all"

    for rule in rules:
        target = rule["target_dataset"]
        sources = set(rule["source_datasets"])

        subset_policy = rule.get("subset_policy", {})
        mode = subset_policy.get("mode", "dataset_name")  # currently supports dataset_name
        only_if_subset_in = subset_policy.get("only_if_subset_in", ["all", "", None])

        mask = out[dataset_col].isin(sources)
        if not mask.any():
            continue

        # Determine which rows should have subset overwritten
        subset_vals = out.loc[mask, subset_col]

        def subset_is_overwritable(x):
            # normalize empties safely
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None in only_if_subset_in
            sx = str(x)
            if sx in only_if_subset_in:
                return True
            if sx.strip() == "" and "" in only_if_subset_in:
                return True
            if sx.lower() == "all" and ("all" in only_if_subset_in):
                return True
            return False

        overwrite_mask = subset_vals.apply(subset_is_overwritable).fillna(False).to_numpy()
        rows_to_overwrite = out.loc[mask].index[overwrite_mask]

        # subset <- old dataset
        if mode == "dataset_name":
            out.loc[rows_to_overwrite, subset_col] = out.loc[rows_to_overwrite, dataset_col].astype(str)
        else:
            raise ValueError(f"Unknown subset_policy.mode: {mode}")

        # dataset <- target
        out.loc[mask, dataset_col] = target

    return out


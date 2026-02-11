# evalviz/benchmarks_filter.py
from __future__ import annotations
from typing import Dict, Optional, Set
import pandas as pd


def get_used_datasets_from_eval(df_eval: pd.DataFrame) -> Set[str]:
    """
    df_eval: your loaded evaluation DF (already normalized via apply_mappings()).
    Returns set of dataset names present in df_eval.
    """
    if "dataset" not in df_eval.columns:
        raise ValueError("df_eval missing column 'dataset'. Did you load results correctly?")
    return set(df_eval["dataset"].dropna().astype(str).unique())


def filter_benchmarks_to_used(
    df_bench: pd.DataFrame,
    df_eval: pd.DataFrame,
    dataset_aliases: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Filters benchmark metadata to only benchmarks that exist in df_eval['dataset'].
    dataset_aliases: optional mapping for extra safety (old_name -> canonical_name).
    """
    dataset_aliases = dataset_aliases or {}

    used = get_used_datasets_from_eval(df_eval)

    d = df_bench.copy()
    if "dataset" not in d.columns:
        raise ValueError("df_bench missing column 'dataset'. Check your benchmarks loader.")

    # Apply aliases to benchmark side (so names match your eval normalization)
    d["dataset_norm"] = d["dataset"].astype(str).map(lambda x: dataset_aliases.get(x, x))

    # Filter
    d = d[d["dataset_norm"].isin(used)].copy()

    # Replace dataset with normalized name for downstream plots
    d["dataset"] = d["dataset_norm"]
    d = d.drop(columns=["dataset_norm"])

    return d

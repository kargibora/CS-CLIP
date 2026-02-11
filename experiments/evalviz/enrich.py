from __future__ import annotations
import pandas as pd
from typing import Any, Dict


def attach_method_metadata(df: pd.DataFrame, methods: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Adds columns like is_ours, is_baseline, is_pretrained, used_datasets, data_quality, etc.
    based on df['method_key'].
    """
    df = df.copy()

    def lookup(method_key: str) -> Dict[str, Any]:
        return methods.get(method_key, {})

    meta = df["method_key"].map(lookup)

    # expand dict into columns
    meta_df = pd.json_normalize(meta)
    meta_df.index = df.index

    out = pd.concat([df, meta_df], axis=1)

    # ensure flags exist
    for col in ["is_ours", "is_baseline", "is_pretrained"]:
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].fillna(False).astype(bool)

    return out


def attach_benchmark_metadata(df: pd.DataFrame, benchmarks_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Joins benchmark metadata from benchmarks.json 'benchmarks' keyed by dataset name.
    Adds fields like size, data_source, license, primitives, etc.
    """
    df = df.copy()
    bench = benchmarks_cfg.get("benchmarks", {})
    if not isinstance(bench, dict) or "dataset" not in df.columns:
        return df

    bench_df = pd.DataFrame.from_dict(bench, orient="index").reset_index().rename(columns={"index": "dataset"})
    return df.merge(bench_df, on="dataset", how="left")

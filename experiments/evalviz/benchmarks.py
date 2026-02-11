# evalviz/benchmarks.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import pandas as pd


def _as_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def load_benchmarks_cfg(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        cfg = json.load(f)
    if "benchmarks" not in cfg:
        raise ValueError(f"benchmarks.json missing top-level key 'benchmarks': {path}")
    cfg.setdefault("source_aliases", {})
    cfg.setdefault("primitive_to_capability", {})
    return cfg


# ---------------------------------------------------------------------------
# Capability schema helpers
# ---------------------------------------------------------------------------

def get_capability_schema(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract capability_schema from config, with defaults."""
    return cfg.get("capability_schema", {
        "top_level_buckets": ["EntityContent", "RelationalStructure", "Binding", "Linguistic"],
        "sub_buckets": {},
        "tag_convention": {"format": "Top/Sub"}
    })


def get_top_level_buckets(cfg: Dict[str, Any]) -> List[str]:
    """Get ordered list of top-level capability buckets."""
    schema = get_capability_schema(cfg)
    return schema.get("top_level_buckets", [])


def get_sub_buckets(cfg: Dict[str, Any], top_level: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Get sub-bucket mapping.
    If top_level is specified, return only that bucket's sub-buckets as a list.
    """
    schema = get_capability_schema(cfg)
    sub_buckets = schema.get("sub_buckets", {})
    if top_level:
        return sub_buckets.get(top_level, [])
    return sub_buckets


def build_subset_capability_lookup(cfg: Dict[str, Any]) -> Dict[Tuple[str, str], List[str]]:
    """
    Returns mapping (dataset, subset) -> list[str capabilities]
    Also supports wildcard (dataset, "all") from dataset-level capability_tags.
    Priority:
      1) subset_capability_map[subset] if present
      2) dataset capability_tags as (dataset, "all")
    """
    lookup: Dict[Tuple[str, str], List[str]] = {}

    for b in cfg.get("benchmarks", []):
        ds = b["dataset"]
        ds_caps = list(b.get("capability_tags", []) or [])
        if ds_caps:
            lookup[(ds, "all")] = ds_caps

        sub_map = b.get("subset_capability_map", {}) or {}
        for sub, caps in sub_map.items():
            lookup[(ds, sub)] = list(caps or [])

    return lookup


def build_top_level_lookup(cfg: Dict[str, Any]) -> Dict[Tuple[str, str], List[str]]:
    """
    Like build_subset_capability_lookup but returns only top-level buckets.
    Useful for main paper tables.
    """
    from evalviz.capabilities import get_top_level  # avoid circular import
    
    full_lookup = build_subset_capability_lookup(cfg)
    top_lookup: Dict[Tuple[str, str], List[str]] = {}
    
    for key, caps in full_lookup.items():
        top_caps = list(dict.fromkeys(get_top_level(c) for c in caps))  # preserve order, dedupe
        top_lookup[key] = top_caps
    
    return top_lookup

def benchmarks_to_df(cfg: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for b in cfg["benchmarks"]:
        # accept both keys
        num = b.get("num_samples", None)
        if num is None:
            num = b.get("n_samples", None)

        rows.append({
            "dataset": b.get("dataset"),
            "num_samples": num,                 # <-- store consistently as num_samples
            "task": b.get("task"),
            "data_sources": _as_list(b.get("data_sources")),
            "primitives": _as_list(b.get("primitives")),
            "image_types": _as_list(b.get("image_types")),
            "text_types": _as_list(b.get("text_types")),
            "license": b.get("license"),
        })
    df = pd.DataFrame(rows)

    # numeric conversion
    df["num_samples"] = pd.to_numeric(df["num_samples"], errors="coerce")

    # normalize sources
    aliases = cfg.get("source_aliases", {})
    if aliases and "data_sources" in df.columns:
        def norm_sources(srcs: List[str]) -> List[str]:
            out = []
            for s in srcs:
                s2 = aliases.get(s, s)
                out.append(s2)
            # de-dup, keep order
            seen = set()
            out2 = []
            for s in out:
                if s not in seen:
                    out2.append(s)
                    seen.add(s)
            return out2
        df["data_sources"] = df["data_sources"].apply(norm_sources)

    return df


def load_benchmarks_df(path: str | Path) -> pd.DataFrame:
    cfg = load_benchmarks_cfg(path)
    return benchmarks_to_df(cfg)


def explode_list_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df.copy()
    out = df.copy()
    out[col] = out[col].apply(lambda x: x if isinstance(x, list) else ([] if x is None else [x]))
    return out.explode(col)


def add_capabilities_from_primitives(df: pd.DataFrame, primitive_to_capability: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Adds:
      - capabilities: list[str]
    based on primitives in each row.
    """
    out = df.copy()

    def prims_to_caps(prims: List[str]) -> List[str]:
        caps = []
        for p in prims or []:
            for c in primitive_to_capability.get(p, []):
                if c not in caps:
                    caps.append(c)
        return caps

    out["capabilities"] = out["primitives"].apply(prims_to_caps)
    return out

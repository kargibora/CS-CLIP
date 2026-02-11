# evalviz/capabilities.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Hierarchical capability parsing utilities
# ---------------------------------------------------------------------------

def parse_capability_tag(tag: str) -> Tuple[str, str]:
    """
    Parse a capability tag in "Top/Sub" format.
    Returns (top_level, sub_level). Falls back gracefully for legacy tags.
    
    Examples:
        "EntityContent/ObjectRecognition" -> ("EntityContent", "ObjectRecognition")
        "Binding/AttributeBinding" -> ("Binding", "AttributeBinding")
        "Uncategorized" -> ("Uncategorized", "Uncategorized")
    """
    if "/" in tag:
        parts = tag.split("/", 1)
        return (parts[0].strip(), parts[1].strip())
    return (tag, tag)  # legacy fallback


def get_top_level(tag: str) -> str:
    """Extract top-level capability from a hierarchical tag."""
    return parse_capability_tag(tag)[0]


def get_sub_level(tag: str) -> str:
    """Extract sub-level capability from a hierarchical tag."""
    return parse_capability_tag(tag)[1]


# ---------------------------------------------------------------------------
# Core annotation function (updated for hierarchical schema)
# ---------------------------------------------------------------------------

def annotate_capabilities_from_cfg(
    df: pd.DataFrame,
    subset_cap_lookup: Dict[Tuple[str, str], List[str]],
    dataset_col: str = "dataset",
    subset_col: str = "subset",
    out_col: str = "capability_category",
    explode: bool = True,
    add_hierarchy_cols: bool = True,
) -> pd.DataFrame:
    """
    Adds capability_category using JSON-driven lookup.
    Uses exact (dataset, subset), else wildcard (dataset, "all"), else Uncategorized.
    
    Args:
        df: Input DataFrame with dataset/subset columns
        subset_cap_lookup: Mapping (dataset, subset) -> List[capability_tags]
        dataset_col: Column name for dataset
        subset_col: Column name for subset
        out_col: Output column name for full capability tag
        explode: If True, explode rows so each capability gets its own row
        add_hierarchy_cols: If True, add capability_top and capability_sub columns
    
    Returns:
        DataFrame with capability annotations
    """
    d = df.copy()
    if subset_col not in d.columns:
        d[subset_col] = "all"
    d[subset_col] = d[subset_col].fillna("all").astype(str)

    def get_caps(row):
        ds = str(row[dataset_col])
        sub = str(row[subset_col])
        caps = subset_cap_lookup.get((ds, sub))
        if caps is None:
            caps = subset_cap_lookup.get((ds, "all"))
        return caps if caps else ["Uncategorized"]

    d[out_col] = d.apply(get_caps, axis=1)

    if explode:
        d = d.explode(out_col)

    # Add hierarchical columns for top/sub level aggregation
    if add_hierarchy_cols:
        d["capability_top"] = d[out_col].apply(get_top_level)
        d["capability_sub"] = d[out_col].apply(get_sub_level)

    return d


# ---------------------------------------------------------------------------
# Aggregation helpers for tables/plots
# ---------------------------------------------------------------------------

def aggregate_by_capability_level(
    df: pd.DataFrame,
    level: str = "top",
    value_col: str = "value",
    group_cols: Optional[List[str]] = None,
    agg_func: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate metrics by capability level (top or sub).
    
    Args:
        df: DataFrame with capability_top and capability_sub columns
        level: "top" for main paper aggregation, "sub" for appendix
        value_col: Column containing metric values
        group_cols: Additional columns to group by (e.g., ["run_label", "metric"])
        agg_func: Aggregation function ("mean", "median", etc.)
    
    Returns:
        Aggregated DataFrame with one row per (group_cols..., capability_level)
    """
    if group_cols is None:
        group_cols = ["run_label"]
    
    cap_col = f"capability_{level}"
    if cap_col not in df.columns:
        raise ValueError(f"Column '{cap_col}' not found. Run annotate_capabilities_from_cfg first.")
    
    grouping = group_cols + [cap_col]
    
    if agg_func == "mean":
        result = df.groupby(grouping, as_index=False)[value_col].mean()
    elif agg_func == "median":
        result = df.groupby(grouping, as_index=False)[value_col].median()
    elif agg_func == "std":
        result = df.groupby(grouping, as_index=False)[value_col].std()
    else:
        result = df.groupby(grouping, as_index=False)[value_col].agg(agg_func)
    
    return result


def pivot_capability_table(
    df: pd.DataFrame,
    level: str = "top",
    value_col: str = "value",
    index_col: str = "run_label",
) -> pd.DataFrame:
    """
    Create a pivot table with runs as rows and capabilities as columns.
    
    Args:
        df: DataFrame with capability annotations and values
        level: "top" or "sub"
        value_col: Column containing metric values  
        index_col: Column to use as row index
    
    Returns:
        Pivot table suitable for paper tables
    """
    cap_col = f"capability_{level}"
    if cap_col not in df.columns:
        raise ValueError(f"Column '{cap_col}' not found.")
    
    # First aggregate if there are multiple entries per (index, capability)
    agg_df = df.groupby([index_col, cap_col], as_index=False)[value_col].mean()
    
    pivot = agg_df.pivot(index=index_col, columns=cap_col, values=value_col)
    return pivot


def get_capability_order(level: str = "top") -> List[str]:
    """
    Return canonical ordering of capability buckets for consistent table/plot layout.
    """
    if level == "top":
        return ["EntityContent", "RelationalStructure", "Binding", "Linguistic"]
    else:
        # Sub-level order grouped by parent
        return [
            # EntityContent
            "ObjectRecognition", "AttributeRecognition", "CountingQuantity", "ExistencePresence",
            # RelationalStructure
            "PredicateSensitivity", "RoleSensitivity",
            # Binding
            "AttributeBinding",
            # Linguistic
            "WordOrderSyntax", "Coreference", "Negation",
        ]


# ---------------------------------------------------------------------------
# Convenience aliases for notebook usage
# ---------------------------------------------------------------------------

def annotate_capabilities_hierarchical(
    df: pd.DataFrame,
    subset_cap_lookup: Dict[Tuple[str, str], List[str]],
    bench_cfg: Optional[Dict] = None,
    dataset_col: str = "dataset",
    subset_col: str = "subset",
    explode: bool = True,
) -> pd.DataFrame:
    """
    Annotate DataFrame with hierarchical capabilities (convenience wrapper).
    
    This is an alias for annotate_capabilities_from_cfg with add_hierarchy_cols=True.
    Adds capability_category, capability_top, and capability_sub columns.
    
    Args:
        df: Input DataFrame
        subset_cap_lookup: Mapping from (dataset, subset) to capability tags
        bench_cfg: Benchmark config (optional, for future extensions)
        dataset_col: Column name for dataset
        subset_col: Column name for subset
        explode: Whether to explode multi-capability rows
    
    Returns:
        DataFrame with capability_category, capability_top, capability_sub columns
    """
    return annotate_capabilities_from_cfg(
        df=df,
        subset_cap_lookup=subset_cap_lookup,
        dataset_col=dataset_col,
        subset_col=subset_col,
        out_col="capability_category",
        explode=explode,
        add_hierarchy_cols=True,
    )


def get_capability_ordering(bench_cfg: Optional[Dict] = None) -> Tuple[List[str], List[str]]:
    """
    Get canonical ordering for top-level and sub-level capabilities.
    
    Args:
        bench_cfg: Optional benchmark config dict. If provided and contains
                   'capability_schema', uses that for ordering. Otherwise
                   uses built-in defaults.
    
    Returns:
        Tuple of (top_level_order, sub_level_order)
    """
    # Try to get from config first
    if bench_cfg and "capability_schema" in bench_cfg:
        schema = bench_cfg["capability_schema"]
        top_order = schema.get("top_level_buckets", get_capability_order("top"))
        
        # Build sub order from schema
        sub_buckets = schema.get("sub_buckets", {})
        sub_order = []
        for top in top_order:
            sub_order.extend(sub_buckets.get(top, []))
        
        if not sub_order:
            sub_order = get_capability_order("sub")
        
        return (top_order, sub_order)
    
    # Fallback to built-in defaults
    return (get_capability_order("top"), get_capability_order("sub"))

# evalviz/io.py
from __future__ import annotations

from datetime import datetime
from typing import List
import pandas as pd

from .config import RunSpec
from .vismin import parse_vismin_value


def _normalize_external_csv_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize external CSV formats (e.g., DeGLA) to the standard format.
    
    Handles CSVs with format:
        checkpoint,dataset,metric,value
        DeGLA ViT-B/32,eval,BLA/ap/ap_group_contrastive_accuracy,0.529
    
    Converts to standard format:
        timestamp,step,epoch,dataset,subset,metric,value,total_samples
        2025-12-15T00:00:00,0,,BLA,ap,ap_group_contrastive_accuracy,0.529,0
    
    The 'metric' column in external format is actually: dataset/subset/metric
    """
    # Check if this is the external format (has 'checkpoint' column, no 'subset' column)
    if "checkpoint" in df.columns and "subset" not in df.columns:
        # Parse the metric column which contains: dataset/subset/metric
        parsed_rows = []
        
        for _, row in df.iterrows():
            metric_parts = str(row["metric"]).split("/")
            
            if len(metric_parts) >= 3:
                # Format: dataset/subset/metric (e.g., BLA/ap/ap_group_contrastive_accuracy)
                dataset = metric_parts[0]
                subset = metric_parts[1]
                metric = "/".join(metric_parts[2:])  # In case metric has slashes
            elif len(metric_parts) == 2:
                # Format: dataset/metric (e.g., SugarCrepe/add_att -> dataset=SugarCrepe, subset=add_att?)
                # Actually looking at the data, it seems like 2-part is dataset/subset_metric
                # Let's check: SugarCrepe/add_att/text_contrastive_accuracy -> 3 parts
                # So 2 parts shouldn't happen, but handle gracefully
                dataset = metric_parts[0]
                subset = "all"
                metric = metric_parts[1]
            else:
                # Single part - use as metric, dataset from 'dataset' column if different
                dataset = row.get("dataset", "unknown")
                if dataset == "eval":
                    dataset = metric_parts[0] if metric_parts else "unknown"
                subset = "all"
                metric = row["metric"]
            
            parsed_rows.append({
                "timestamp": datetime.now().isoformat(),
                "step": 0,
                "epoch": None,
                "dataset": dataset,
                "subset": subset,
                "metric": metric,
                "value": row["value"],
                "total_samples": 0,
            })
        
        return pd.DataFrame(parsed_rows)
    
    # Not external format, return as-is but ensure required columns exist
    return df


def _ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has all standard columns with default values.
    """
    # Set defaults for missing columns
    if "step" not in df.columns:
        df["step"] = 0
    if "epoch" not in df.columns:
        df["epoch"] = None
    if "timestamp" not in df.columns:
        df["timestamp"] = datetime.now().isoformat()
    if "total_samples" not in df.columns:
        df["total_samples"] = 0
    if "subset" not in df.columns:
        df["subset"] = "all"
    
    return df


def _expand_vismin_nested_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands VisMin rows where metric=='group_contrastive_accuracy' holds nested dict in 'value'.
    """
    if not {"dataset", "metric", "value"}.issubset(df.columns):
        return df

    mask = (df["dataset"] == "VisMin") & (df["metric"] == "group_contrastive_accuracy")
    if not mask.any():
        return df

    expanded_rows = []
    for _, row in df.loc[mask].iterrows():
        parsed = parse_vismin_value(row["value"])
        if not parsed:
            continue
        for subset, metric, value in parsed:
            new_row = row.copy()
            new_row["subset"] = subset
            new_row["metric"] = metric
            new_row["value"] = value
            expanded_rows.append(new_row)

    if not expanded_rows:
        return df

    df2 = df.loc[~mask].copy()
    return pd.concat([df2, pd.DataFrame(expanded_rows)], ignore_index=True)


def load_results_from_runs(
    runs: List[RunSpec],
    metric_for_step_selection: str = "text_contrastive_accuracy",
    dedup_latest_timestamp: bool = True,
) -> pd.DataFrame:
    """
    Loads each CSV and returns one long-form DataFrame.
    Attaches: run_id, run_label, method_key
    (No is_baseline/is_ours here; those come from methods.json via enrich.attach_method_metadata)
    
    Handles multiple CSV formats:
    1. Standard format: timestamp,step,epoch,dataset,subset,metric,value,total_samples
    2. External format: checkpoint,dataset,metric,value (where metric = dataset/subset/metric)
    """
    dfs: list[pd.DataFrame] = []

    for run in runs:
        df = pd.read_csv(run.csv_path)

        if "value" not in df.columns:
            raise ValueError(f"'value' column not found in {run.csv_path}. Columns={df.columns.tolist()}")

        # Normalize external CSV formats (e.g., DeGLA format)
        df = _normalize_external_csv_format(df)
        
        # Ensure all standard columns exist with defaults
        df = _ensure_standard_columns(df)

        # VisMin parsing
        df = _expand_vismin_nested_rows(df)

        # numeric value
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])

        # select best step if multiple
        if "step" in df.columns:
            steps = df["step"].dropna().unique()
            if len(steps) > 1:
                non_zero = [s for s in steps if s != 0]
                df_sel = df[df["step"].isin(non_zero)] if non_zero else df

                metric_mask = (df_sel["metric"] == metric_for_step_selection)
                if metric_mask.any():
                    scores = df_sel.loc[metric_mask].groupby("step")["value"].mean()
                else:
                    scores = df_sel.groupby("step")["value"].mean()

                best_step = scores.idxmax()
                df = df[df["step"] == best_step].copy()

        # attach run metadata (new scheme)
        df["run_id"] = run.run_id
        df["run_label"] = run.label
        df["method_key"] = run.method_key

        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    if dedup_latest_timestamp and "timestamp" in full.columns:
        full["timestamp"] = pd.to_datetime(full["timestamp"], errors="coerce")
        full = full.sort_values("timestamp", ascending=False)
        full = full.drop_duplicates(subset=["dataset", "subset", "metric", "run_id"], keep="first")

    return full

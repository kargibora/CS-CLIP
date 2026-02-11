"""
Shared utilities for ablation study notebooks.

This module provides common functions for:
- Loading and parsing ablation CSV files
- Checkpoint selection
- Metric extraction
- Dataset merging (ARO, etc.)
- LaTeX table generation
- Visualization helpers

Usage:
    from ablation_utils import (
        load_ablation_csv, select_checkpoint, extract_dataset_scores,
        load_all_ablation_models, load_all_models_all_metrics,
        make_latex_ablation_table, setup_plotting_style,
        load_benchmark_config, apply_dataset_merge_rules
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys

# Add parent directory to path for evalviz imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from evalviz.preprocess import apply_mappings as evalviz_apply_mappings

# =============================================================================
# CONSTANTS
# =============================================================================

METRICS = ["text_contrastive_accuracy", "image_contrastive_accuracy", "group_contrastive_accuracy"]
METRIC_DISPLAY = {
    "text_contrastive_accuracy": "I2T",
    "image_contrastive_accuracy": "T2I", 
    "group_contrastive_accuracy": "Group"
}

# Colorblind-friendly palette (Paul Tol)
METRIC_COLORS = {
    'I2T': '#4477AA',      # Blue
    'T2I': '#EE6677',      # Red/Pink
    'Group': '#228833',    # Green
    'Average': '#CCBB44',  # Yellow
}

DEFAULT_PRIMARY_METRIC = "text_contrastive_accuracy"

# Default benchmark config path (relative to ablations directory)
DEFAULT_BENCHMARK_CONFIG = "../configs/benchmarks.json"


# =============================================================================
# DATASET MERGING (ARO, etc.)
# =============================================================================

def load_benchmark_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the benchmark configuration file.
    
    Args:
        config_path: Path to benchmarks.json. If None, uses default path.
        
    Returns:
        Dictionary with benchmark configuration
    """
    if config_path is None:
        # Resolve relative to this file's location
        this_dir = Path(__file__).parent
        config_path = this_dir / DEFAULT_BENCHMARK_CONFIG
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"WARNING: Benchmark config not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)


def apply_dataset_merge_rules(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    dataset_col: str = "dataset",
    subset_col: str = "subset",
) -> pd.DataFrame:
    """
    Apply dataset merge rules from config (e.g., merge VG_Relation, VG_Attribution into ARO).
    
    This merges multiple dataset names into one dataset, moving the original dataset
    name into the subset column.
    
    Args:
        df: DataFrame with evaluation results
        cfg: Benchmark configuration (from load_benchmark_config)
        dataset_col: Name of the dataset column
        subset_col: Name of the subset column
        
    Returns:
        DataFrame with merge rules applied
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
        mode = subset_policy.get("mode", "dataset_name")
        only_if_subset_in = subset_policy.get("only_if_subset_in", ["all", "", None])
        
        mask = out[dataset_col].isin(sources)
        if not mask.any():
            continue
        
        # Determine which rows should have subset overwritten
        subset_vals = out.loc[mask, subset_col]
        
        def subset_is_overwritable(x):
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
        
        # subset <- old dataset name
        if mode == "dataset_name":
            out.loc[rows_to_overwrite, subset_col] = out.loc[rows_to_overwrite, dataset_col].astype(str)
        else:
            raise ValueError(f"Unknown subset_policy.mode: {mode}")
        
        # dataset <- target
        out.loc[mask, dataset_col] = target
    
    return out


# =============================================================================
# PLOTTING STYLE
# =============================================================================

def setup_plotting_style():
    """Set up matplotlib style for publication-ready figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
    })


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_ablation_csv(csv_path: str, apply_merge: bool = False,
                      benchmark_config: Optional[Dict[str, Any]] = None,
                      apply_metric_mappings: bool = True) -> pd.DataFrame:
    """Load and parse an ablation CSV file.
    
    Args:
        csv_path: Path to the CSV file
        apply_merge: Whether to apply dataset merge rules (e.g., ARO)
        benchmark_config: Optional pre-loaded benchmark config. If None and 
                         apply_merge=True or apply_metric_mappings=True, will load default config.
        apply_metric_mappings: Whether to apply metric aliases (e.g., contrastive_accuracy -> text_contrastive_accuracy)
        
    Returns:
        DataFrame with the loaded data, or empty DataFrame if file not found
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"WARNING: File not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    
    # Load benchmark config if needed
    if (apply_merge or apply_metric_mappings) and benchmark_config is None:
        benchmark_config = load_benchmark_config()
    
    # Apply metric mappings (e.g., contrastive_accuracy -> text_contrastive_accuracy)
    if apply_metric_mappings and benchmark_config:
        df = evalviz_apply_mappings(df, benchmark_config)
    
    # Apply dataset merge rules (e.g., ARO)
    if apply_merge and benchmark_config:
        df = apply_dataset_merge_rules(df, benchmark_config)
    
    return df


def select_checkpoint(df: pd.DataFrame, metric: str = DEFAULT_PRIMARY_METRIC, 
                      step: Optional[int] = None) -> pd.DataFrame:
    """Select best checkpoint or specific step.
    
    If step is None, selects the checkpoint with highest average metric across datasets,
    preferring steps with more datasets.
    
    Args:
        df: DataFrame with evaluation results
        metric: Metric to use for checkpoint selection
        step: Specific step to select, or None for best
        
    Returns:
        DataFrame filtered to the selected checkpoint
    """
    if df.empty:
        return df
    
    if step is not None:
        return df[df['step'] == step]
    
    # Find best checkpoint based on average metric
    metric_df = df[df['metric'] == metric].copy()
    if metric_df.empty:
        # Fallback to first available step
        return df[df['step'] == df['step'].iloc[0]]
    
    # Convert to numeric, coercing errors (handles JSON dict values)
    metric_df['value'] = pd.to_numeric(metric_df['value'], errors='coerce')
    metric_df = metric_df.dropna(subset=['value'])
    
    if metric_df.empty:
        return df[df['step'] == df['step'].iloc[0]]
    
    # Count datasets per step - prefer steps with more datasets
    datasets_per_step = metric_df.groupby('step')['dataset'].nunique()
    max_datasets = datasets_per_step.max()
    
    # Only consider steps that have the most datasets
    valid_steps = datasets_per_step[datasets_per_step == max_datasets].index
    metric_df_filtered = metric_df[metric_df['step'].isin(valid_steps)]
    
    # Among steps with max datasets, pick the one with highest average score
    avg_by_step = metric_df_filtered.groupby('step')['value'].mean()
    best_step = avg_by_step.idxmax()
    
    return df[df['step'] == best_step]


def extract_dataset_scores(df: pd.DataFrame, metric: str = DEFAULT_PRIMARY_METRIC) -> dict:
    """Extract per-dataset scores for a given metric.
    
    Args:
        df: DataFrame with evaluation results (typically filtered to one checkpoint)
        metric: Metric to extract
        
    Returns:
        Dictionary mapping dataset names to scores
    """
    metric_df = df[df['metric'] == metric].copy()
    
    # Convert to numeric, coercing errors (handles JSON dict values)
    metric_df['value'] = pd.to_numeric(metric_df['value'], errors='coerce')
    metric_df = metric_df.dropna(subset=['value'])
    
    scores = {}
    for _, row in metric_df.iterrows():
        dataset = row['dataset']
        subset = row.get('subset', 'all')
        key = f"{dataset}" if subset in ['all', '', None] or pd.isna(subset) else f"{dataset}/{subset}"
        scores[key] = row['value']
    
    return scores


def load_all_ablation_models(models_config: dict, metric: str = DEFAULT_PRIMARY_METRIC,
                             checkpoint_step: Optional[int] = None,
                             apply_merge: bool = False,
                             benchmark_config: Optional[Dict[str, Any]] = None,
                             apply_metric_mappings: bool = True) -> pd.DataFrame:
    """Load all ablation models and create a comparison table.
    
    Only includes datasets that are present in ALL models (intersection).
    
    Args:
        models_config: Dictionary of model configurations with 'csv_path' key
        metric: Metric to use for comparison
        checkpoint_step: Specific step to use, or None for best
        apply_merge: Whether to apply dataset merge rules (e.g., ARO)
        benchmark_config: Optional pre-loaded benchmark config
        apply_metric_mappings: Whether to apply metric aliases (default True)
        
    Returns:
        DataFrame with models as rows and datasets as columns
    """
    all_scores = {}
    
    # Load benchmark config once if needed for merge or metric mappings
    if (apply_merge or apply_metric_mappings) and benchmark_config is None:
        benchmark_config = load_benchmark_config()
    
    for model_name, cfg in models_config.items():
        print(f"Loading {model_name}...")
        df = load_ablation_csv(cfg['csv_path'], apply_merge=apply_merge, 
                               benchmark_config=benchmark_config,
                               apply_metric_mappings=apply_metric_mappings)
        
        if df.empty:
            print(f"  WARNING: No data for {model_name}")
            continue
        
        df_ckpt = select_checkpoint(df, metric=metric, step=checkpoint_step)
        scores = extract_dataset_scores(df_ckpt, metric=metric)
        
        step_used = df_ckpt['step'].iloc[0] if not df_ckpt.empty else 'N/A'
        print(f"  Loaded {len(scores)} datasets (step={step_used})")
        
        all_scores[model_name] = scores
    
    # Create DataFrame
    result = pd.DataFrame(all_scores).T
    result.index.name = 'Model'
    
    # Filter to common datasets (columns present in ALL models)
    common_cols = result.dropna(axis=1, how='any').columns.tolist()
    print(f"\nCommon datasets ({len(common_cols)}): {common_cols}")
    result = result[common_cols]
    
    return result


def load_all_metrics_for_model(csv_path: str, metrics: list = None, 
                                checkpoint_step: Optional[int] = None,
                                primary_metric: str = DEFAULT_PRIMARY_METRIC,
                                apply_merge: bool = False,
                                benchmark_config: Optional[Dict[str, Any]] = None,
                                apply_metric_mappings: bool = True) -> dict:
    """Load all metrics for a model, returning per-dataset and overall averages.
    
    Args:
        csv_path: Path to the CSV file
        metrics: List of metrics to load (default: METRICS)
        checkpoint_step: Specific step to use, or None for best
        primary_metric: Metric to use for checkpoint selection
        apply_merge: Whether to apply dataset merge rules (e.g., ARO)
        benchmark_config: Optional pre-loaded benchmark config
        apply_metric_mappings: Whether to apply metric aliases (default True)
        
    Returns:
        Dictionary with metric names as keys, containing 'per_dataset' and 'average'
    """
    if metrics is None:
        metrics = METRICS
        
    df = load_ablation_csv(csv_path, apply_merge=apply_merge, benchmark_config=benchmark_config,
                           apply_metric_mappings=apply_metric_mappings)
    if df.empty:
        return {}
    
    # Select checkpoint based on primary metric
    df_ckpt = select_checkpoint(df, metric=primary_metric, step=checkpoint_step)
    
    result = {}
    for metric in metrics:
        metric_df = df_ckpt[df_ckpt['metric'] == metric].copy()
        metric_df['value'] = pd.to_numeric(metric_df['value'], errors='coerce')
        metric_df = metric_df.dropna(subset=['value'])
        
        if metric_df.empty:
            continue
            
        # Per-dataset scores (average over subsets)
        dataset_scores = metric_df.groupby('dataset')['value'].mean().to_dict()
        
        # Overall average
        overall_avg = metric_df['value'].mean()
        
        result[metric] = {
            'per_dataset': dataset_scores,
            'average': overall_avg
        }
    
    return result


def load_all_models_all_metrics(models_config: dict, metrics: list = None, 
                                 checkpoint_step: Optional[int] = None,
                                 primary_metric: str = DEFAULT_PRIMARY_METRIC,
                                 apply_merge: bool = False,
                                 benchmark_config: Optional[Dict[str, Any]] = None,
                                 apply_metric_mappings: bool = True) -> pd.DataFrame:
    """Load all models with all metrics, create a summary table.
    
    Only includes datasets that are present in ALL models (intersection).
    
    Args:
        models_config: Dictionary of model configurations
        metrics: List of metrics to load (default: METRICS)
        checkpoint_step: Specific step to use, or None for best
        primary_metric: Metric to use for checkpoint selection
        apply_merge: Whether to apply dataset merge rules (e.g., ARO)
        benchmark_config: Optional pre-loaded benchmark config
        apply_metric_mappings: Whether to apply metric aliases (default True)
        
    Returns:
        DataFrame with models as index and metrics + per-dataset scores as columns
    """
    if metrics is None:
        metrics = METRICS
    
    # Load benchmark config once if needed
    if (apply_merge or apply_metric_mappings) and benchmark_config is None:
        benchmark_config = load_benchmark_config()
    
    # First pass: collect all data and find common datasets
    all_model_data = {}
    all_datasets_per_model = {}
    
    for model_name, cfg in models_config.items():
        print(f"Loading {model_name}...")
        model_data = load_all_metrics_for_model(
            cfg['csv_path'], metrics, checkpoint_step, primary_metric,
            apply_merge=apply_merge, benchmark_config=benchmark_config,
            apply_metric_mappings=apply_metric_mappings
        )
        
        if not model_data:
            print("  WARNING: No data")
            continue
        
        all_model_data[model_name] = model_data
        
        # Collect datasets for this model (from primary metric)
        pm = metrics[0]
        if pm in model_data:
            all_datasets_per_model[model_name] = set(model_data[pm]['per_dataset'].keys())
        
        print(f"  Loaded metrics: {[METRIC_DISPLAY.get(m, m) for m in model_data.keys()]}")
    
    # Find common datasets across ALL models
    if all_datasets_per_model:
        common_datasets = set.intersection(*all_datasets_per_model.values())
        print(f"\nCommon datasets across all models ({len(common_datasets)}): {sorted(common_datasets)}")
    else:
        common_datasets = set()
    
    # Second pass: build records using only common datasets
    records = []
    for model_name, cfg in models_config.items():
        if model_name not in all_model_data:
            continue
        
        model_data = all_model_data[model_name]
        record = {'Model': model_name, 'is_baseline': cfg.get('is_baseline', False)}
        
        for metric in metrics:
            if metric not in model_data:
                continue
            
            # Filter to common datasets only
            filtered_scores = {ds: val for ds, val in model_data[metric]['per_dataset'].items() 
                               if ds in common_datasets}
            
            # Compute average over common datasets only
            if filtered_scores:
                record[METRIC_DISPLAY.get(metric, metric)] = np.mean(list(filtered_scores.values()))
                
                # Also store per-dataset for later
                for ds, val in filtered_scores.items():
                    record[f"{ds}_{METRIC_DISPLAY.get(metric, metric)}"] = val
        
        records.append(record)
    
    return pd.DataFrame(records).set_index('Model')


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def make_latex_ablation_table(
    df: pd.DataFrame,
    models_config: dict,
    caption: str = "Ablation study results.",
    label: str = "tab:ablation",
    decimals: int = 1,
    font_size: str = "small",
) -> str:
    """Generate a LaTeX table for ablation study.
    
    Args:
        df: DataFrame with Model as index, metrics as columns (values in [0,1])
        models_config: Dict with model configs (for is_baseline flag)
        caption: Table caption
        label: Table label
        decimals: Decimal places for formatting
        font_size: LaTeX font size command
        
    Returns:
        LaTeX table string
    """
    
    def fmt(v):
        if pd.isna(v):
            return "--"
        return f"{v * 100:.{decimals}f}"
    
    def latex_escape(s):
        return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    
    # Find baseline and best per column
    baseline_model = [name for name, cfg in models_config.items() if cfg.get('is_baseline', False)]
    baseline_model = baseline_model[0] if baseline_model else None
    
    best_per_col = {col: df[col].idxmax() for col in df.columns}
    
    # Build table
    cols = list(df.columns)
    col_spec = "l" + "c" * len(cols)
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(rf"  \{font_size}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")
    
    # Header
    header = ["Model"] + [latex_escape(c) for c in cols]
    lines.append("    " + " & ".join(header) + r" \\")
    lines.append(r"    \midrule")
    
    # Body
    for model in df.index:
        row = [latex_escape(model)]
        
        for col in cols:
            val = fmt(df.loc[model, col])
            
            # Style: bold for best, underline for baseline
            is_best = (model == best_per_col[col])
            is_baseline = (model == baseline_model)
            
            if is_baseline:
                val = r"\underline{" + val + "}"
            if is_best:
                val = r"\textbf{" + val + "}"
            
            row.append(val)
        
        lines.append("    " + " & ".join(row) + r" \\")
    
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_ablation_line(
    summary_df: pd.DataFrame,
    models_config: dict,
    param_key: str,
    param_label: str,
    title: str,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
):
    """Create a line plot showing metric vs hyperparameter value.
    
    Args:
        summary_df: DataFrame with metrics (I2T, T2I, Group, Average)
        models_config: Dict with model configs including param_key
        param_key: Key in models_config to use for x-axis values
        param_label: Label for x-axis
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        Figure and axes objects
    """
    # Extract parameter values
    param_values = [models_config[model][param_key] for model in summary_df.index]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for metric in ['I2T', 'T2I', 'Group', 'Average']:
        if metric in summary_df.columns:
            values = summary_df[metric].values * 100
            ax.plot(param_values, values, 'o-', label=metric, 
                    color=METRIC_COLORS.get(metric, 'gray'), linewidth=2, markersize=8)
    
    # Mark baseline
    baseline_models = [name for name, cfg in models_config.items() if cfg.get('is_baseline', False)]
    if baseline_models:
        baseline_param = models_config[baseline_models[0]][param_key]
        ax.axvline(x=baseline_param, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(baseline_param + 0.02, ax.get_ylim()[1] * 0.98, 'Default', 
                ha='left', va='top', fontsize=9, color='gray')
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(param_values)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_ablation_bars(
    summary_df: pd.DataFrame,
    models_config: dict,
    title: str,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """Create a grouped bar chart comparing metrics across models.
    
    Args:
        summary_df: DataFrame with metrics (I2T, T2I, Group)
        models_config: Dict with model configs (for baseline)
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        Figure and axes objects
    """
    metric_cols = [col for col in ['I2T', 'T2I', 'Group'] if col in summary_df.columns]
    plot_data = summary_df[metric_cols] * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_models = len(plot_data)
    n_metrics = len(metric_cols)
    x = np.arange(n_models)
    width = 0.25
    
    for i, metric in enumerate(metric_cols):
        offset = (i - n_metrics/2 + 0.5) * width
        bars = ax.bar(x + offset, plot_data[metric], width, 
                      label=metric, color=METRIC_COLORS.get(metric, f'C{i}'),
                      edgecolor='white', linewidth=0.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.index, fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_xlabel('')
    ax.set_ylim(0, 100)
    
    # Add baseline marker
    baseline_models = [name for name, cfg in models_config.items() if cfg.get('is_baseline', False)]
    if baseline_models and baseline_models[0] in plot_data.index:
        baseline_idx = list(plot_data.index).index(baseline_models[0])
        ax.axvline(x=baseline_idx, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(baseline_idx, ax.get_ylim()[1] * 0.98, 'Default', 
                ha='center', va='top', fontsize=8, color='gray')
    
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    
    return fig, ax


def compute_deltas(summary_df: pd.DataFrame, models_config: dict) -> pd.DataFrame:
    """Compute deltas from baseline model.
    
    Args:
        summary_df: DataFrame with metrics
        models_config: Dict with model configs (for baseline)
        
    Returns:
        DataFrame with deltas in percentage points
    """
    baseline_models = [name for name, cfg in models_config.items() if cfg.get('is_baseline', False)]
    if not baseline_models:
        print("WARNING: No baseline model found")
        return summary_df * 0
    
    baseline_model = baseline_models[0]
    baseline_scores = summary_df.loc[baseline_model]
    deltas_df = (summary_df.sub(baseline_scores, axis=1)) * 100  # Convert to percentage points
    
    return deltas_df


# =============================================================================
# DATASET-WISE AND SUBCATEGORY-WISE TABLES
# =============================================================================

def load_all_models_per_dataset(models_config: dict, metric: str = DEFAULT_PRIMARY_METRIC,
                                 checkpoint_step: Optional[int] = None,
                                 apply_merge: bool = False,
                                 benchmark_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load all models and create a per-dataset comparison table.
    
    Args:
        models_config: Dictionary of model configurations
        metric: Metric to extract
        checkpoint_step: Specific step to use, or None for best
        apply_merge: Whether to apply dataset merge rules (e.g., ARO)
        benchmark_config: Optional pre-loaded benchmark config
        
    Returns:
        DataFrame with models as rows and datasets as columns (values in [0,1])
    """
    all_scores = {}
    
    # Load benchmark config once if needed
    if apply_merge and benchmark_config is None:
        benchmark_config = load_benchmark_config()
    
    for model_name, cfg in models_config.items():
        df = load_ablation_csv(cfg['csv_path'], apply_merge=apply_merge,
                               benchmark_config=benchmark_config)
        if df.empty:
            continue
        
        df_ckpt = select_checkpoint(df, metric=metric, step=checkpoint_step)
        
        # Get per-dataset scores (average over subsets within each dataset)
        metric_df = df_ckpt[df_ckpt['metric'] == metric].copy()
        metric_df['value'] = pd.to_numeric(metric_df['value'], errors='coerce')
        metric_df = metric_df.dropna(subset=['value'])
        
        dataset_scores = metric_df.groupby('dataset')['value'].mean().to_dict()
        all_scores[model_name] = dataset_scores
    
    result = pd.DataFrame(all_scores).T
    result.index.name = 'Model'
    
    # Filter to common datasets
    common_cols = result.dropna(axis=1, how='any').columns.tolist()
    result = result[common_cols]
    
    return result


def load_all_models_per_subset(models_config: dict, metric: str = DEFAULT_PRIMARY_METRIC,
                                checkpoint_step: Optional[int] = None,
                                apply_merge: bool = False,
                                benchmark_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load all models and create a per-subset (dataset/subset) comparison table.
    
    Args:
        models_config: Dictionary of model configurations
        metric: Metric to extract
        checkpoint_step: Specific step to use, or None for best
        apply_merge: Whether to apply dataset merge rules (e.g., ARO)
        benchmark_config: Optional pre-loaded benchmark config
        
    Returns:
        DataFrame with models as rows and dataset/subset as columns (values in [0,1])
    """
    all_scores = {}
    
    # Load benchmark config once if needed
    if apply_merge and benchmark_config is None:
        benchmark_config = load_benchmark_config()
    
    for model_name, cfg in models_config.items():
        df = load_ablation_csv(cfg['csv_path'], apply_merge=apply_merge,
                               benchmark_config=benchmark_config)
        if df.empty:
            continue
        
        df_ckpt = select_checkpoint(df, metric=metric, step=checkpoint_step)
        scores = extract_dataset_scores(df_ckpt, metric=metric)
        all_scores[model_name] = scores
    
    result = pd.DataFrame(all_scores).T
    result.index.name = 'Model'
    
    # Filter to common columns
    common_cols = result.dropna(axis=1, how='any').columns.tolist()
    result = result[common_cols]
    
    return result

def make_latex_dataset_table(
    df: pd.DataFrame,
    models_config: dict,
    caption: str = "Per-dataset ablation results.",
    label: str = "tab:ablation_datasets",
    decimals: int = 1,
    font_size: str = "scriptsize",
    include_average: bool = True,
    rotate_headers: bool = True,
) -> str:
    """Generate a LaTeX table showing per-dataset results.
    
    Args:
        df: DataFrame with models as rows and datasets as columns (values in [0,1])
        models_config: Dict with model configs (for is_baseline flag)
        caption: Table caption
        label: Table label
        decimals: Decimal places for formatting
        font_size: LaTeX font size command
        include_average: Whether to include an Average column
        rotate_headers: Whether to rotate column headers for space
        
    Returns:
        LaTeX table string
    """
    def fmt(v):
        if pd.isna(v):
            return "--"
        return f"{v * 100:.{decimals}f}"
    
    def latex_escape(s):
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    
    # Find baseline
    baseline_model = [name for name, cfg in models_config.items() if cfg.get('is_baseline', False)]
    baseline_model = baseline_model[0] if baseline_model else None
    
    # Add average column if requested
    df_display = df.copy()
    if include_average:
        df_display['Avg'] = df_display.mean(axis=1)
    
    # Find best per column
    best_per_col = {col: df_display[col].idxmax() for col in df_display.columns}
    
    # Build table
    cols = list(df_display.columns)
    col_spec = "l" + "c" * len(cols)
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(rf"  \{font_size}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"  \begin{adjustbox}{max width=\textwidth}")
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")
    
    # Header
    if rotate_headers:
        header = ["Model"] + [r"\rotatebox{60}{" + latex_escape(c) + "}" for c in cols]
    else:
        header = ["Model"] + [latex_escape(c) for c in cols]
    lines.append("    " + " & ".join(header) + r" \\")
    lines.append(r"    \midrule")
    
    # Body
    for model in df_display.index:
        row = [latex_escape(model)]
        
        for col in cols:
            val = fmt(df_display.loc[model, col])
            
            is_best = (model == best_per_col[col])
            is_baseline = (model == baseline_model)
            
            if is_baseline:
                val = r"\underline{" + val + "}"
            if is_best:
                val = r"\textbf{" + val + "}"
            
            row.append(val)
        
        lines.append("    " + " & ".join(row) + r" \\")
    
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \end{adjustbox}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def make_latex_subset_table(
    df: pd.DataFrame,
    models_config: dict,
    dataset_filter: Optional[str] = None,
    caption: str = "Per-subset ablation results.",
    label: str = "tab:ablation_subsets",
    decimals: int = 1,
    font_size: str = "scriptsize",
    include_average: bool = True,
) -> str:
    """Generate a LaTeX table showing per-subset results for a specific dataset.
    
    Args:
        df: DataFrame with models as rows and dataset/subset as columns (values in [0,1])
        models_config: Dict with model configs (for is_baseline flag)
        dataset_filter: If provided, only show subsets from this dataset
        caption: Table caption
        label: Table label
        decimals: Decimal places for formatting
        font_size: LaTeX font size command
        include_average: Whether to include an Average column
        
    Returns:
        LaTeX table string
    """
    # Filter columns if dataset specified
    df_filtered = df.copy()
    if dataset_filter:
        cols_to_keep = [c for c in df.columns if c.startswith(dataset_filter)]
        if cols_to_keep:
            df_filtered = df[cols_to_keep]
            # Simplify column names (remove dataset prefix)
            df_filtered.columns = [c.replace(f"{dataset_filter}/", "") for c in df_filtered.columns]
    
    return make_latex_dataset_table(
        df_filtered, models_config, caption, label, decimals, font_size, include_average
    )


def get_datasets_and_subsets(models_config: dict, metric: str = DEFAULT_PRIMARY_METRIC,
                              checkpoint_step: Optional[int] = None,
                              apply_merge: bool = False,
                              benchmark_config: Optional[Dict[str, Any]] = None) -> dict:
    """Get a mapping of datasets to their subsets.
    
    Args:
        models_config: Dictionary of model configurations
        metric: Metric to use
        checkpoint_step: Specific step to use, or None for best
        apply_merge: Whether to apply dataset merge rules (e.g., ARO)
        benchmark_config: Optional pre-loaded benchmark config
        
    Returns:
        Dictionary mapping dataset names to list of subset names
    """
    # Use first model to get structure
    first_model = list(models_config.keys())[0]
    df = load_ablation_csv(models_config[first_model]['csv_path'],
                           apply_merge=apply_merge, benchmark_config=benchmark_config)
    if df.empty:
        return {}
    
    df_ckpt = select_checkpoint(df, metric=metric, step=checkpoint_step)
    metric_df = df_ckpt[df_ckpt['metric'] == metric]
    
    datasets_subsets = {}
    for _, row in metric_df.iterrows():
        dataset = row['dataset']
        subset = row.get('subset', 'all')
        if pd.isna(subset) or subset == '':
            subset = 'all'
        
        if dataset not in datasets_subsets:
            datasets_subsets[dataset] = set()
        datasets_subsets[dataset].add(subset)
    
    # Convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in datasets_subsets.items()}


def display_all_tables(models_config: dict, metric: str = DEFAULT_PRIMARY_METRIC,
                       checkpoint_step: Optional[int] = None,
                       show_latex: bool = True,
                       apply_merge: bool = False,
                       benchmark_config: Optional[Dict[str, Any]] = None):
    """Display all dataset and subset tables with optional LaTeX output.
    
    Args:
        models_config: Dictionary of model configurations
        metric: Metric to use
        checkpoint_step: Specific step to use, or None for best
        show_latex: Whether to print LaTeX tables
        apply_merge: Whether to apply dataset merge rules (e.g., ARO)
        benchmark_config: Optional pre-loaded benchmark config
    """
    metric_name = METRIC_DISPLAY.get(metric, metric)
    
    # Load benchmark config once if needed
    if apply_merge and benchmark_config is None:
        benchmark_config = load_benchmark_config()
    
    # 1. Per-dataset table
    print("\n" + "="*60)
    print(f"PER-DATASET RESULTS ({metric_name})")
    print("="*60)
    
    dataset_df = load_all_models_per_dataset(models_config, metric, checkpoint_step,
                                              apply_merge=apply_merge, benchmark_config=benchmark_config)
    dataset_pct = dataset_df * 100
    dataset_pct['Average'] = dataset_pct.mean(axis=1)
    
    # Display
    from IPython.display import display
    display(dataset_pct.round(1).style.highlight_max(axis=0, color='lightgreen'))
    
    if show_latex:
        print("\nLaTeX:")
        print(make_latex_dataset_table(
            dataset_df, models_config,
            caption=f"Per-dataset {metric_name} accuracy.",
            label=f"tab:ablation_datasets_{metric_name.lower()}"
        ))
    
    # 2. Per-subset table (full)
    print("\n" + "="*60)
    print(f"PER-SUBSET RESULTS ({metric_name})")
    print("="*60)
    
    subset_df = load_all_models_per_subset(models_config, metric, checkpoint_step,
                                            apply_merge=apply_merge, benchmark_config=benchmark_config)
    subset_pct = subset_df * 100
    subset_pct['Average'] = subset_pct.mean(axis=1)
    
    display(subset_pct.round(1).style.highlight_max(axis=0, color='lightgreen'))
    
    if show_latex:
        print("\nLaTeX:")
        print(make_latex_dataset_table(
            subset_df, models_config,
            caption=f"Per-subset {metric_name} accuracy.",
            label=f"tab:ablation_subsets_{metric_name.lower()}"
        ))
    
    # 3. Get structure for individual dataset tables
    datasets_subsets = get_datasets_and_subsets(models_config, metric, checkpoint_step,
                                                 apply_merge=apply_merge, benchmark_config=benchmark_config)
    
    print("\n" + "="*60)
    print("DATASETS AND SUBSETS FOUND:")
    print("="*60)
    for ds, subsets in sorted(datasets_subsets.items()):
        print(f"  {ds}: {subsets}")
    
    return dataset_df, subset_df, datasets_subsets


def print_summary(summary_df: pd.DataFrame, models_config: dict, 
                  ablation_name: str, param_key: Optional[str] = None):
    """Print a summary of the ablation study.
    
    Args:
        summary_df: DataFrame with metrics
        models_config: Dict with model configs
        ablation_name: Name of the ablation study
        param_key: Optional key for parameter value in models_config
    """
    baseline_models = [name for name, cfg in models_config.items() if cfg.get('is_baseline', False)]
    baseline_model = baseline_models[0] if baseline_models else None
    
    deltas_df = compute_deltas(summary_df, models_config)
    avg_scores = summary_df['Average'] * 100 if 'Average' in summary_df.columns else summary_df.mean(axis=1) * 100
    
    print("\n" + "="*60)
    print(f"SUMMARY: {ablation_name}")
    print("="*60)
    
    if baseline_model:
        print(f"\nBaseline: {baseline_model}")
    
    print("\nAverage Performance:")
    for model in summary_df.index:
        score = avg_scores[model]
        delta = deltas_df.loc[model, 'Average'] if 'Average' in deltas_df.columns else 0
        marker = "★" if models_config[model].get('is_baseline', False) else " "
        
        param_str = ""
        if param_key and param_key in models_config[model]:
            param_str = f" | {param_key}={models_config[model][param_key]}"
        
        print(f"  {marker} {model}: {score:.1f}% ({delta:+.2f}pp vs baseline){param_str}")
    
    print("\nKey Findings:")
    best_model = avg_scores.idxmax()
    worst_model = avg_scores.idxmin()
    print(f"  - Best: {best_model} ({avg_scores[best_model]:.1f}%)")
    print(f"  - Worst: {worst_model} ({avg_scores[worst_model]:.1f}%)")
    print(f"  - Gap: {avg_scores[best_model] - avg_scores[worst_model]:.1f}pp")

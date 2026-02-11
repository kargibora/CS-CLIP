#!/usr/bin/env python3
"""
Comprehensive Image Half-Truth Analysis Script
===============================================

A highly configurable script for analyzing and visualizing image half-truth
vulnerability results across multiple models.

Usage Examples:
---------------
# Basic usage - load results and show summary
python analyze_image_half_truth.py --results_dir results_image_ht

# Compare specific models
python analyze_image_half_truth.py --results_dir results_image_ht \
    --models "openai-clip,negclip,csclip" \
    --labels "CLIP,NegCLIP,CS-CLIP (Ours)"

# Generate paper figures
python analyze_image_half_truth.py --results_dir results_image_ht \
    --mode paper_figures \
    --output_dir paper_figures/image_ht

# Find interesting cases where our model wins
python analyze_image_half_truth.py --results_dir results_image_ht \
    --mode find_cases \
    --baseline openai-clip \
    --ours csclip

# Browse specific cases
python analyze_image_half_truth.py --results_dir results_image_ht \
    --mode browse \
    --case_indices 0,1,2,3,4

Author: Auto-generated analysis script
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for the analysis."""
    # Paths
    results_dir: Path = Path("results_image_ht")
    samples_json: Path = Path("results.json")
    output_dir: Path = Path("analysis_output")
    image_root: Path = Path("..")
    
    # Model configuration
    models: List[str] = field(default_factory=list)  # Model directory names
    labels: List[str] = field(default_factory=list)  # Display labels
    
    # Analysis mode
    mode: str = "summary"  # summary, comparison, paper_figures, find_cases, browse, sample_viz
    
    # Comparison settings
    baseline_model: Optional[str] = None
    ours_model: Optional[str] = None
    
    # Case browsing
    case_indices: List[int] = field(default_factory=list)
    max_cases: int = 10
    
    # Sample visualization settings
    sample_category: str = "random"  # random, csclip_wins, csclip_wins_all, all_correct, all_wrong
    n_samples: int = 10
    seed: int = 42
    show_models: List[str] = field(default_factory=list)  # Models to show in per-sample viz
    show_labels: List[str] = field(default_factory=list)  # Labels for show_models
    
    # Figure settings
    save_figures: bool = True
    show_figures: bool = True
    figure_format: str = "pdf"  # pdf, png, svg
    dpi: int = 300
    
    # Styling
    compact: bool = False
    
    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.samples_json = Path(self.samples_json)
        self.output_dir = Path(self.output_dir)
        self.image_root = Path(self.image_root)


# =============================================================================
# Color Palette
# =============================================================================

COLORS = {
    'target_correct': '#27AE60',      # Emerald green
    'target_wrong': '#E74C3C',        # Alizarin red
    'distractor': '#95A5A6',          # Gray
    'clip': '#3498DB',                # Blue
    'negclip': '#9B59B6',             # Purple
    'ours': '#E67E22',                # Orange
    'dac': '#1ABC9C',                 # Teal
    'clove': '#E91E63',               # Pink
    'background': '#FAFAFA',
    'text_dark': '#2C3E50',
    'text_light': '#7F8C8D',
    'border_target': '#27AE60',
    'border_distractor': '#BDC3C7',
    'grid': '#ECF0F1',
}

# Model name to color mapping
MODEL_COLORS = {
    'clip': COLORS['clip'],
    'openai': COLORS['clip'],
    'negclip': COLORS['negclip'],
    'csclip': COLORS['ours'],
    'cs-clip': COLORS['ours'],
    'ours': COLORS['ours'],
    'dac': COLORS['dac'],
    'clove': COLORS['clove'],
}


def get_model_color(model_name: str) -> str:
    """Get color for a model based on its name."""
    name_lower = model_name.lower()
    for key, color in MODEL_COLORS.items():
        if key in name_lower:
            return color
    # Default color cycle
    return plt.cm.tab10(hash(model_name) % 10)


# =============================================================================
# Data Loading
# =============================================================================

class DataLoader:
    """Handles loading of results and sample data."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.all_results: Dict[str, Dict] = {}
        self.samples_lookup: Dict[str, Dict] = {}
        self.model_labels: Dict[str, str] = {}
    
    def load_all(self) -> bool:
        """Load all data. Returns True if successful."""
        print("=" * 60)
        print("Loading Data")
        print("=" * 60)
        
        # Load samples.json
        if not self._load_samples():
            return False
        
        # Load results
        if not self._load_results():
            return False
        
        print(f"\n✅ Loaded {len(self.all_results)} models")
        print(f"✅ Loaded {len(self.samples_lookup)} samples")
        return True
    
    def _load_samples(self) -> bool:
        """Load samples.json for image path lookup."""
        samples_path = self.config.samples_json
        
        # Try different locations
        possible_paths = [
            samples_path,
            self.config.results_dir / "samples.json",
            self.config.results_dir.parent / "samples.json",
            Path("samples.json"),
        ]
        
        # Also look inside model subdirectories
        if self.config.results_dir.exists():
            for item in self.config.results_dir.iterdir():
                if item.is_dir() and (item / "samples.json").exists():
                    possible_paths.insert(0, item / "samples.json")
                    break  # Use the first one found
        
        for path in possible_paths:
            if path.exists():
                print(f"📂 Loading samples from: {path}")
                try:
                    with open(path) as f:
                        samples_data = json.load(f)
                    
                    # Handle both list and dict formats
                    if isinstance(samples_data, list):
                        for sample in samples_data:
                            sid = sample.get('sample_id', sample.get('id', ''))
                            if sid:
                                self.samples_lookup[sid] = sample
                    elif isinstance(samples_data, dict):
                        # Could be {sample_id: sample_data} or {key: [samples]}
                        if 'samples' in samples_data:
                            for sample in samples_data['samples']:
                                sid = sample.get('sample_id', sample.get('id', ''))
                                if sid:
                                    self.samples_lookup[sid] = sample
                        else:
                            self.samples_lookup = samples_data
                    
                    print(f"   Loaded {len(self.samples_lookup)} samples")
                    return True
                except Exception as e:
                    print(f"   ⚠️ Error loading {path}: {e}")
                    continue
        
        print("⚠️ Could not find samples.json - image display will be limited")
        return True  # Not fatal
    
    def _load_results(self) -> bool:
        """Load results from the results directory."""
        results_dir = self.config.results_dir
        
        if not results_dir.exists():
            print(f"❌ Results directory not found: {results_dir}")
            return False
        
        # Find model directories or result files
        model_dirs = []
        
        # Check if results_dir contains model subdirectories
        for item in results_dir.iterdir():
            if item.is_dir():
                # Check for results.json in subdirectory
                if (item / "results.json").exists():
                    model_dirs.append(item)
            elif item.name == "results.json":
                # Single results file in results_dir
                model_dirs.append(results_dir)
                break
        
        if not model_dirs:
            # Try loading results.json directly
            if (results_dir / "results.json").exists():
                model_dirs.append(results_dir)
        
        # Filter by specified models if provided
        if self.config.models:
            filtered_dirs = []
            used_dirs = set()  # Track which directories we've already matched
            
            for model_name in self.config.models:
                model_lower = model_name.lower().replace('-', '_').replace(' ', '_')
                best_match = None
                best_score = 0
                
                for d in model_dirs:
                    if d in used_dirs:
                        continue
                    
                    dir_lower = d.name.lower()
                    
                    # Exact patterns to check
                    # Handle special cases for common model names
                    if model_lower in ['openai', 'openai_clip', 'clip']:
                        # Match baseline/openai but NOT negclip, csclip, etc.
                        if 'baseline' in dir_lower or ('openai' in dir_lower and 'cs_clip' not in dir_lower):
                            if 'negclip' not in dir_lower and 'csclip' not in dir_lower:
                                score = 10
                                if score > best_score:
                                    best_score = score
                                    best_match = d
                    
                    elif model_lower in ['negclip', 'neg_clip']:
                        # Match negclip but NOT cs_clip_negclip
                        if 'negclip' in dir_lower and 'cs_clip' not in dir_lower:
                            score = 10
                            if score > best_score:
                                best_score = score
                                best_match = d
                    
                    elif model_lower in ['csclip', 'cs_clip', 'cs_clip_negclip', 'ours']:
                        # Match cs_clip specifically
                        if 'cs_clip' in dir_lower:
                            score = 10
                            if score > best_score:
                                best_score = score
                                best_match = d
                    
                    else:
                        # Generic matching - check if model name is in directory name
                        if model_lower in dir_lower:
                            score = len(model_lower)  # Longer matches are better
                            if score > best_score:
                                best_score = score
                                best_match = d
                
                if best_match:
                    filtered_dirs.append(best_match)
                    used_dirs.add(best_match)
                    print(f"   Matched '{model_name}' -> {best_match.name}")
                else:
                    print(f"   ⚠️ No match found for '{model_name}'")
            
            if filtered_dirs:
                model_dirs = filtered_dirs
            else:
                print("   ⚠️ No models matched, loading all available models")
        
        print(f"\n📊 Found {len(model_dirs)} model directories to load")
        
        # Load each model's results
        for i, model_dir in enumerate(model_dirs):
            results_path = model_dir / "results.json"
            if not results_path.exists():
                continue
            
            model_name = model_dir.name if model_dir != results_dir else "model"
            
            print(f"📂 Loading: {model_name}")
            
            with open(results_path) as f:
                results = json.load(f)
            
            self.all_results[model_name] = results
            
            # Set label
            if self.config.labels and i < len(self.config.labels):
                self.model_labels[model_name] = self.config.labels[i]
            else:
                self.model_labels[model_name] = self._generate_label(model_name)
        
        return len(self.all_results) > 0
    
    def _generate_label(self, model_name: str) -> str:
        """Generate a display label from model name."""
        name_lower = model_name.lower()
        
        # Check cs_clip FIRST (before negclip, since cs_clip_negclip contains both)
        if 'cs_clip' in name_lower or 'csclip' in name_lower or 'cs-clip' in name_lower:
            return 'CS-CLIP (Ours)'
        elif 'baseline' in name_lower or ('openai' in name_lower and 'clip' in name_lower):
            return 'CLIP'
        elif 'negclip' in name_lower or 'neg_clip' in name_lower:
            return 'NegCLIP'
        elif 'dac' in name_lower:
            return 'DAC'
        elif 'clove' in name_lower:
            return 'CLOVE'
        elif 'fsc_clip' in name_lower or 'fscclip' in name_lower:
            return 'FSC-CLIP'
        elif 'laclip' in name_lower:
            return 'LaCLIP'
        elif 'ce_clip' in name_lower or 'ceclip' in name_lower:
            return 'CE-CLIP'
        elif 'con_clip' in name_lower or 'conclip' in name_lower:
            return 'ConCLIP'
        elif 'clic' in name_lower:
            return 'CLIC'
        elif 'degla' in name_lower:
            return 'DEGLA'
        elif 'tsvlc' in name_lower:
            return 'TSVLC'
        elif 'tripletclip' in name_lower:
            return 'TripletCLIP'
        elif 'readclip' in name_lower:
            return 'ReadCLIP'
        else:
            # Clean up the name
            return model_name.replace('_', ' ').replace('-', ' ').title()
    
    def get_label(self, model_name: str) -> str:
        """Get display label for a model."""
        return self.model_labels.get(model_name, model_name)
    
    def get_samples(self, model_name: str) -> List[Dict]:
        """Get per-sample results for a model."""
        if model_name not in self.all_results:
            return []
        results = self.all_results[model_name]
        return results.get('per_sample_results', 
               results.get('sample_results',
               results.get('detailed_results', [])))


# =============================================================================
# Analysis Functions
# =============================================================================

class Analyzer:
    """Performs analysis on loaded data."""
    
    def __init__(self, loader: DataLoader, config: AnalysisConfig):
        self.loader = loader
        self.config = config
    
    def compute_metrics(self, model_name: str) -> Dict[str, float]:
        """Compute metrics for a single model."""
        results = self.loader.all_results.get(model_name, {})
        samples = self.loader.get_samples(model_name)
        
        # Try to get from pre-computed metrics
        metrics = results.get('metrics', {})
        
        if not metrics and samples:
            # Compute from samples
            n_total = len(samples)
            n_correct = sum(1 for s in samples if s.get('is_correct', False))
            
            margins = [s.get('target_image_score', 0) - s.get('max_distractor_score', 0) 
                      for s in samples]
            
            metrics = {
                'accuracy': n_correct / n_total if n_total > 0 else 0,
                'vulnerability': 1 - (n_correct / n_total) if n_total > 0 else 1,
                'avg_margin': np.mean(margins) if margins else 0,
                'std_margin': np.std(margins) if margins else 0,
                'n_samples': n_total,
            }
        
        return metrics
    
    def get_summary_df(self) -> pd.DataFrame:
        """Get summary DataFrame for all models."""
        rows = []
        for model_name in self.loader.all_results:
            metrics = self.compute_metrics(model_name)
            rows.append({
                'Model': self.loader.get_label(model_name),
                'Model Name': model_name,
                'Accuracy (%)': metrics.get('accuracy', 0) * 100,
                'Vulnerability (%)': metrics.get('vulnerability', 0) * 100,
                'Avg Margin': metrics.get('avg_margin', 0),
                'Std Margin': metrics.get('std_margin', 0),
                'N Samples': metrics.get('n_samples', 0),
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Accuracy (%)', ascending=False)
        return df
    
    def find_interesting_cases(
        self,
        baseline_model: str,
        improved_model: str,
        case_type: str = "ours_wins",
        max_cases: int = 20
    ) -> List[Dict]:
        """Find cases where models disagree."""
        baseline_samples = self.loader.get_samples(baseline_model)
        improved_samples = self.loader.get_samples(improved_model)
        
        if not baseline_samples or not improved_samples:
            print("❌ Could not find samples for comparison")
            return []
        
        # Build lookup
        improved_lookup = {}
        for i, s in enumerate(improved_samples):
            sid = s.get('sample_id', s.get('image_key', f'idx_{i}'))
            improved_lookup[sid] = (i, s)
        
        interesting = []
        
        for base_idx, base_sample in enumerate(baseline_samples):
            base_sid = base_sample.get('sample_id', base_sample.get('image_key', ''))
            
            if base_sid not in improved_lookup:
                continue
            
            imp_idx, imp_sample = improved_lookup[base_sid]
            
            base_correct = base_sample.get('is_correct', False)
            imp_correct = imp_sample.get('is_correct', False)
            
            # Filter by case type
            if case_type == "ours_wins" and not (imp_correct and not base_correct):
                continue
            elif case_type == "baseline_wins" and not (base_correct and not imp_correct):
                continue
            elif case_type == "disagree" and (base_correct == imp_correct):
                continue
            
            # Compute margins
            base_margin = (base_sample.get('target_image_score', 0) - 
                          base_sample.get('max_distractor_score', 0))
            imp_margin = (imp_sample.get('target_image_score', 0) - 
                         imp_sample.get('max_distractor_score', 0))
            
            interesting.append({
                'sample_id': base_sid,
                'baseline_index': base_idx,
                'improved_index': imp_idx,
                'caption': base_sample.get('caption', base_sample.get('test_caption', '')),
                'baseline_correct': base_correct,
                'improved_correct': imp_correct,
                'baseline_margin': base_margin,
                'improved_margin': imp_margin,
                'margin_improvement': imp_margin - base_margin,
            })
        
        # Sort by margin improvement
        interesting.sort(key=lambda x: abs(x['margin_improvement']), reverse=True)
        
        return interesting[:max_cases]


# =============================================================================
# Visualization Functions
# =============================================================================

class Visualizer:
    """Handles all visualization."""
    
    def __init__(self, loader: DataLoader, analyzer: Analyzer, config: AnalysisConfig):
        self.loader = loader
        self.analyzer = analyzer
        self.config = config
        self._setup_style()
    
    def _setup_style(self):
        """Set up matplotlib style for publication."""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.linewidth': 0.8,
            'savefig.dpi': self.config.dpi,
            'savefig.bbox': 'tight',
        })
    
    def _save_figure(self, fig, name: str):
        """Save figure to output directory."""
        if self.config.save_figures:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.config.output_dir / f"{name}.{self.config.figure_format}"
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"💾 Saved: {path}")
    
    def _add_image_border(self, ax, color, linewidth=4):
        """Add border to image axis."""
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(linewidth)

    def _get_image_path(self, image_path_str: str) -> Optional[Path]:
        """
        Resolve image path with multiple fallback strategies.
        
        Tries in order:
        1. Direct path
        2. image_root / path
        3. image_root / path (with 'datasets/' prefix stripped)
        4. image_root / path (with 'data/' prefix stripped)
        """
        if not image_path_str:
            return None
        
        path = Path(image_path_str)
        
        # Try direct path
        if path.exists():
            return path
        
        # Try with image_root
        full_path = self.config.image_root / image_path_str
        if full_path.exists():
            return full_path
        
        # Try stripping common prefixes
        for prefix in ['datasets/', 'data/']:
            if image_path_str.startswith(prefix):
                stripped = image_path_str[len(prefix):]
                full_path = self.config.image_root / stripped
                if full_path.exists():
                    return full_path
        
        return None
    
    def _get_distractor_paths(self, distractor_paths: List[str]) -> List[Path]:
        """Resolve multiple distractor image paths."""
        result = []
        for p in distractor_paths:
            resolved = self._get_image_path(p)
            if resolved:
                result.append(resolved)
        return result

    def plot_accuracy_comparison(self) -> plt.Figure:
        """Create accuracy comparison bar chart."""
        df = self.analyzer.get_summary_df()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = df['Model'].tolist()
        accuracies = df['Accuracy (%)'].tolist()
        model_names = df['Model Name'].tolist()
        
        colors = [get_model_color(m) for m in model_names]
        
        bars = ax.bar(range(len(models)), accuracies, color=colors, 
                     edgecolor='white', linewidth=1.5)
        
        # Reference line
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label='Random Chance')
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Image Half-Truth Vulnerability: Model Comparison', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right')
        
        # Value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        self._save_figure(fig, 'accuracy_comparison')
        
        if self.config.show_figures:
            plt.show()
        
        return fig
    
    def plot_margin_distribution(self) -> plt.Figure:
        """Plot margin distributions for all models."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name in self.loader.all_results:
            samples = self.loader.get_samples(model_name)
            if not samples:
                continue
            
            margins = [s.get('target_image_score', 0) - s.get('max_distractor_score', 0)
                      for s in samples]
            
            label = self.loader.get_label(model_name)
            color = get_model_color(model_name)
            
            ax.hist(margins, bins=50, alpha=0.5, label=label, color=color,
                   edgecolor='white', linewidth=0.5)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        ax.set_xlabel('Margin (Target Score - Max Distractor Score)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Margins Across Models', fontweight='bold')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        self._save_figure(fig, 'margin_distribution')
        
        if self.config.show_figures:
            plt.show()
        
        return fig
    
    def plot_case_comparison(
        self,
        case_indices: List[int],
        reference_model: Optional[str] = None,
        models_to_compare: Optional[List[str]] = None,
    ) -> plt.Figure:
        """Create publication-ready multi-case comparison figure."""
        
        # Auto-detect reference model
        if reference_model is None:
            reference_model = list(self.loader.all_results.keys())[0]
        
        # Auto-detect models to compare
        if models_to_compare is None:
            models_to_compare = list(self.loader.all_results.keys())
        
        ref_samples = self.loader.get_samples(reference_model)
        n_cases = len(case_indices)
        
        # Figure setup
        fig_width = 11 if self.config.compact else 13
        row_height = 2.5 if self.config.compact else 3.2
        
        fig = plt.figure(figsize=(fig_width, row_height * n_cases + 0.8), facecolor='white')
        
        gs = fig.add_gridspec(n_cases, 3, width_ratios=[1, 1, 1.8],
                             hspace=0.4, wspace=0.12)
        
        for row_idx, case_idx in enumerate(case_indices):
            if case_idx >= len(ref_samples):
                continue
            
            sample = ref_samples[case_idx]
            sample_id = sample.get('sample_id', sample.get('image_key', ''))
            
            # Get sample info
            sample_info = self.loader.samples_lookup.get(sample_id, {})
            
            # Get image paths using robust resolution
            target_path = None
            distractor_path = None
            
            if sample_info:
                target_path = self._get_image_path(sample_info.get('target_image_path', ''))
                dist_paths = sample_info.get('distractor_image_paths', [])
                if dist_paths:
                    resolved = self._get_distractor_paths(dist_paths)
                    distractor_path = resolved[0] if resolved else None
            
            # Get caption
            caption = sample_info.get('test_caption', sample.get('caption', ''))
            if len(caption) > 80:
                caption = caption[:77] + '...'
            
            # === Target Image ===
            ax_target = fig.add_subplot(gs[row_idx, 0])
            if target_path and target_path.exists():
                img = Image.open(target_path)
                ax_target.imshow(img)
            else:
                ax_target.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center',
                              transform=ax_target.transAxes, fontsize=10)
            ax_target.set_xticks([])
            ax_target.set_yticks([])
            self._add_image_border(ax_target, COLORS['target_correct'], 4)
            ax_target.set_xlabel(f'({chr(97+row_idx)}) Target', fontsize=9, fontweight='bold',
                                color=COLORS['target_correct'], labelpad=3)
            ax_target.set_title(f'"{caption}"', fontsize=9, style='italic',
                               color=COLORS['text_dark'], pad=8, loc='left')
            
            # === Distractor Image ===
            ax_dist = fig.add_subplot(gs[row_idx, 1])
            if distractor_path and distractor_path.exists():
                img = Image.open(distractor_path)
                ax_dist.imshow(img)
            else:
                ax_dist.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center',
                            transform=ax_dist.transAxes, fontsize=10)
            ax_dist.set_xticks([])
            ax_dist.set_yticks([])
            self._add_image_border(ax_dist, COLORS['border_distractor'], 2)
            ax_dist.set_xlabel('Distractor', fontsize=9, color=COLORS['text_light'], labelpad=3)
            
            # === Score Comparison ===
            ax_scores = fig.add_subplot(gs[row_idx, 2])
            
            model_scores = []
            for model_name in models_to_compare:
                m_samples = self.loader.get_samples(model_name)
                
                m_sample = None
                for s in m_samples:
                    if s.get('sample_id', s.get('image_key', '')) == sample_id:
                        m_sample = s
                        break
                
                if m_sample:
                    model_scores.append({
                        'name': self.loader.get_label(model_name),
                        'color': get_model_color(model_name),
                        'target': m_sample.get('target_image_score', 
                                  m_sample.get('target_score', 0)),
                        'distractor': m_sample.get('max_distractor_score', 0),
                        'correct': m_sample.get('is_correct', False)
                    })
            
            if model_scores:
                n_models = len(model_scores)
                y_pos = np.arange(n_models)
                bar_h = 0.35
                
                for i, ms in enumerate(model_scores):
                    # Target bar
                    ax_scores.barh(i + bar_h/2, ms['target'], bar_h,
                                  color=ms['color'], alpha=0.9, edgecolor='white')
                    # Distractor bar
                    ax_scores.barh(i - bar_h/2, ms['distractor'], bar_h,
                                  color=ms['color'], alpha=0.35, hatch='///', edgecolor='white')
                    
                    # Result indicator
                    symbol = '✓' if ms['correct'] else '✗'
                    sym_color = COLORS['target_correct'] if ms['correct'] else COLORS['target_wrong']
                    max_s = max(ms['target'], ms['distractor'])
                    ax_scores.text(max_s + 0.02, i, symbol, fontsize=11,
                                  fontweight='bold', color=sym_color, va='center')
                
                ax_scores.set_yticks(y_pos)
                ax_scores.set_yticklabels([ms['name'] for ms in model_scores], fontsize=8)
                ax_scores.set_xlim(0, 1.05)
                ax_scores.invert_yaxis()
                ax_scores.spines['top'].set_visible(False)
                ax_scores.spines['right'].set_visible(False)
                ax_scores.tick_params(axis='y', length=0)
                ax_scores.xaxis.grid(True, alpha=0.2, linestyle='--')
                
                if row_idx == n_cases - 1:
                    ax_scores.set_xlabel('Similarity', fontsize=9)
        
        # Legend
        legend_elements = [
            Patch(facecolor=COLORS['clip'], alpha=0.9, label='Target Score'),
            Patch(facecolor=COLORS['clip'], alpha=0.35, hatch='///', label='Max Distractor'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2,
                  fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        self._save_figure(fig, f'case_comparison_{"-".join(map(str, case_indices))}')
        
        if self.config.show_figures:
            plt.show()
        
        return fig
    
    def browse_case(self, case_index: int, reference_model: Optional[str] = None) -> plt.Figure:
        """Browse a single case with detailed visualization."""
        return self.plot_case_comparison([case_index], reference_model)

    def plot_single_sample(
        self,
        sample_id: str,
        output_subdir: Optional[str] = None,
        models_to_show: Optional[List[str]] = None,
        filename_prefix: str = "",
    ) -> Optional[plt.Figure]:
        """
        Create a clean, publication-ready per-sample visualization.
        
        Shows:
        - Target image with green border (labeled "Target")
        - Distractor image(s) with gray border
        - Caption prominently displayed
        - Horizontal bar chart comparing model scores
        - Clear visual indication of which models succeed/fail
        
        Args:
            sample_id: The sample ID to visualize
            output_subdir: Subdirectory within output_dir to save (e.g., 'random_samples', 'csclip_wins')
            models_to_show: List of model names to include (default: all)
            filename_prefix: Prefix for the filename (sample_id will be appended)
        
        Returns:
            The figure object, or None if sample not found
        """
        # Get sample info
        sample_info = self.loader.samples_lookup.get(sample_id)
        if not sample_info:
            print(f"⚠️ Sample {sample_id} not found in samples lookup")
            return None
        
        # Get models to show
        if models_to_show is None:
            models_to_show = list(self.loader.all_results.keys())
        
        # Collect model results for this sample - PRESERVE ORDER from models_to_show
        model_results = []
        for model_name in models_to_show:
            samples = self.loader.get_samples(model_name)
            for s in samples:
                if s.get('sample_id', s.get('image_key', '')) == sample_id:
                    model_results.append({
                        'name': self.loader.get_label(model_name),
                        'model_key': model_name,
                        'color': get_model_color(model_name),
                        'target_score': s.get('target_image_score', s.get('target_score', 0)),
                        'distractor_score': s.get('max_distractor_score', 0),
                        'is_correct': s.get('is_correct', False),
                    })
                    break
        
        if not model_results:
            print(f"⚠️ No model results found for sample {sample_id}")
            return None
        
        # Get image paths - only use the FIRST distractor (max distractor)
        target_path = self._get_image_path(sample_info.get('target_image_path', ''))
        distractor_paths = self._get_distractor_paths(sample_info.get('distractor_image_paths', []))
        # Use only first distractor (represents the "max distractor")
        distractor_path = distractor_paths[0] if distractor_paths else None
        
        # Get caption
        caption = sample_info.get('test_caption', '')
        
        # Create figure with 2-column layout: images stacked on left, bar chart on right
        fig = plt.figure(figsize=(10, 5), facecolor='white')
        
        # Layout: 2 columns - left for images (2 rows), right for chart
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1], 
                              wspace=0.15, hspace=0.25)
        
        # === Caption as figure title ===
        fig.suptitle(f'"{caption}"', fontsize=13, fontweight='bold',
                    y=0.98, color=COLORS['text_dark'], style='italic')
        
        # === Target Image (top-left) ===
        ax_target = fig.add_subplot(gs[0, 0])
        if target_path and target_path.exists():
            img = Image.open(target_path)
            ax_target.imshow(img)
        else:
            ax_target.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center',
                          transform=ax_target.transAxes, fontsize=11, color=COLORS['text_light'])
            ax_target.set_facecolor('#F5F5F5')
        
        ax_target.set_xticks([])
        ax_target.set_yticks([])
        self._add_image_border(ax_target, COLORS['target_correct'], 5)
        ax_target.set_title('Target', fontsize=11, fontweight='bold', 
                           color=COLORS['target_correct'], pad=5)
        
        # === Distractor Image (bottom-left) ===
        ax_dist = fig.add_subplot(gs[1, 0])
        if distractor_path and distractor_path.exists():
            img = Image.open(distractor_path)
            ax_dist.imshow(img)
        else:
            ax_dist.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center',
                        transform=ax_dist.transAxes, fontsize=11, color=COLORS['text_light'])
            ax_dist.set_facecolor('#F5F5F5')
        
        ax_dist.set_xticks([])
        ax_dist.set_yticks([])
        self._add_image_border(ax_dist, COLORS['target_wrong'], 3)
        ax_dist.set_xlabel('Distractor', fontsize=11, fontweight='bold',
                          color=COLORS['target_wrong'], labelpad=8)
        
        # === Separator Line between images and chart ===
        # Add a dashed vertical line as separator
        line_x = 0.42  # Position between image column and chart column
        fig.add_artist(plt.Line2D([line_x, line_x], [0.08, 0.88], 
                                   transform=fig.transFigure, 
                                   color='#CCCCCC', linewidth=1.5, 
                                   linestyle='--', zorder=10))
        
        # === Score Comparison - VERTICAL Bar Chart (spans both rows on right) ===
        ax_scores = fig.add_subplot(gs[:, 1])
        
        # DO NOT sort - keep order from models_to_show
        n_models = len(model_results)
        x_positions = np.arange(n_models)
        bar_width = 0.35
        
        # Plot vertical bars - grouped by model
        for i, mr in enumerate(model_results):
            # Target score bar (solid) - left bar of pair
            ax_scores.bar(i - bar_width/2, mr['target_score'], bar_width,
                         color=COLORS['target_correct'], alpha=0.9, edgecolor='white',
                         linewidth=1, label='Target' if i == 0 else '')
            
            # Distractor score bar (hatched) - right bar of pair
            ax_scores.bar(i + bar_width/2, mr['distractor_score'], bar_width,
                         color=COLORS['target_wrong'], alpha=0.7, hatch='///',
                         edgecolor='white', linewidth=1, 
                         label='Max Distractor' if i == 0 else '')
            
            # Result indicator (✓ or ✗) above the bars
            symbol = '✓' if mr['is_correct'] else '✗'
            sym_color = COLORS['target_correct'] if mr['is_correct'] else COLORS['target_wrong']
            max_score = max(mr['target_score'], mr['distractor_score'])
            ax_scores.text(i, max_score + 0.03, symbol, fontsize=14,
                          fontweight='bold', color=sym_color, ha='center', va='bottom')
        
        # Styling
        ax_scores.set_xticks(x_positions)
        ax_scores.set_xticklabels([mr['name'] for mr in model_results], fontsize=9, rotation=0)
        ax_scores.set_ylim(0, 1.12)
        ax_scores.set_ylabel('Similarity Score', fontsize=10)
        ax_scores.spines['top'].set_visible(False)
        ax_scores.spines['right'].set_visible(False)
        ax_scores.yaxis.grid(True, alpha=0.3, linestyle='--', color=COLORS['grid'])
        ax_scores.set_axisbelow(True)
        
        # Add legend
        legend_elements = [
            Patch(facecolor=COLORS['target_correct'], alpha=0.9, label='Target'),
            Patch(facecolor=COLORS['target_wrong'], alpha=0.7, hatch='///', label='Distractor'),
        ]
        ax_scores.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        if self.config.save_figures:
            # Create subdirectory if specified
            if output_subdir:
                save_dir = self.config.output_dir / output_subdir
            else:
                save_dir = self.config.output_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename (include sample_id in name, not in plot)
            if filename_prefix:
                filename = f"{filename_prefix}_{sample_id}"
            else:
                filename = f"sample_{sample_id}"
            
            path = save_dir / f"{filename}.{self.config.figure_format}"
            fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"💾 Saved: {path}")
        
        if self.config.show_figures:
            plt.show()
        else:
            plt.close(fig)
        
        return fig

    def generate_sample_visualizations(
        self,
        category: str = "random",
        n_samples: int = 10,
        baseline_model: Optional[str] = None,
        ours_model: Optional[str] = None,
        models_to_show: Optional[List[str]] = None,
        seed: int = 42,
    ) -> List[str]:
        """
        Generate per-sample visualizations for different categories.
        
        Categories:
        - "random": Random samples
        - "csclip_wins": Cases where CS-CLIP succeeds and baseline(s) fail
        - "csclip_wins_all": Cases where CS-CLIP succeeds and ALL other models fail
        - "all_correct": Cases where all models succeed (easy cases)
        - "all_wrong": Cases where all models fail (hard cases)
        
        Args:
            category: Category of samples to visualize
            n_samples: Maximum number of samples to generate
            baseline_model: Baseline model name for comparison categories
            ours_model: Our model name (typically cs_clip)
            models_to_show: Models to include in the visualization
            seed: Random seed for reproducible sampling
        
        Returns:
            List of sample IDs that were visualized
        """
        np.random.seed(seed)
        
        # Auto-detect models if not provided
        all_models = list(self.loader.all_results.keys())
        
        # Auto-detect our model (CS-CLIP) - check cs_clip FIRST
        if not ours_model:
            for m in all_models:
                m_lower = m.lower()
                if 'cs_clip' in m_lower or 'csclip' in m_lower or 'cs-clip' in m_lower:
                    ours_model = m
                    break
        
        # Auto-detect baseline (CLIP) - must NOT contain cs_clip or negclip
        if not baseline_model:
            for m in all_models:
                m_lower = m.lower()
                if ('baseline' in m_lower or 'openai' in m_lower) and \
                   'negclip' not in m_lower and 'cs_clip' not in m_lower and 'csclip' not in m_lower:
                    baseline_model = m
                    break
        
        if models_to_show is None:
            # Default: show CLIP, NegCLIP, and CS-CLIP (in that order)
            models_to_show = []
            clip_model = None
            negclip_model = None
            csclip_model = None
            
            for m in all_models:
                m_lower = m.lower()
                # Check CS-CLIP first (highest priority to avoid cs_clip_negclip matching negclip)
                if 'cs_clip' in m_lower or 'csclip' in m_lower or 'cs-clip' in m_lower:
                    if csclip_model is None:
                        csclip_model = m
                elif ('baseline' in m_lower or 'openai' in m_lower) and \
                     'negclip' not in m_lower:
                    if clip_model is None:
                        clip_model = m
                elif 'negclip' in m_lower or 'neg_clip' in m_lower:
                    if negclip_model is None:
                        negclip_model = m
            
            # Add in desired order: CLIP, NegCLIP, CS-CLIP
            if clip_model:
                models_to_show.append(clip_model)
            if negclip_model:
                models_to_show.append(negclip_model)
            if csclip_model:
                models_to_show.append(csclip_model)
            
            if not models_to_show:
                models_to_show = all_models[:3]  # Fallback to first 3
        
        print(f"\n📊 Generating '{category}' sample visualizations...")
        print(f"   Models to show: {[self.loader.get_label(m) for m in models_to_show]}")
        if ours_model:
            print(f"   Our model: {self.loader.get_label(ours_model)}")
        if baseline_model:
            print(f"   Baseline: {self.loader.get_label(baseline_model)}")
        
        # Get all sample IDs from reference model
        ref_model = models_to_show[0] if models_to_show else all_models[0]
        ref_samples = self.loader.get_samples(ref_model)
        
        if not ref_samples:
            print("❌ No samples found")
            return []
        
        # Build lookup for all models
        model_lookups = {}
        for model_name in all_models:
            samples = self.loader.get_samples(model_name)
            lookup = {}
            for s in samples:
                sid = s.get('sample_id', s.get('image_key', ''))
                if sid:
                    lookup[sid] = s
            model_lookups[model_name] = lookup
        
        # Select samples based on category
        selected_samples = []
        
        if category == "random":
            all_sids = [s.get('sample_id', s.get('image_key', '')) for s in ref_samples]
            selected_indices = np.random.choice(len(all_sids), min(n_samples, len(all_sids)), replace=False)
            selected_samples = [all_sids[i] for i in selected_indices]
        
        elif category == "csclip_wins":
            # CS-CLIP correct, baseline wrong
            if not ours_model or not baseline_model:
                print("❌ Need both ours_model and baseline_model for csclip_wins category")
                return []
            
            candidates = []
            for s in ref_samples:
                sid = s.get('sample_id', s.get('image_key', ''))
                if not sid:
                    continue
                
                ours_sample = model_lookups.get(ours_model, {}).get(sid)
                base_sample = model_lookups.get(baseline_model, {}).get(sid)
                
                if ours_sample and base_sample:
                    if ours_sample.get('is_correct', False) and not base_sample.get('is_correct', False):
                        # Compute margin improvement for sorting
                        ours_margin = ours_sample.get('target_image_score', 0) - ours_sample.get('max_distractor_score', 0)
                        base_margin = base_sample.get('target_image_score', 0) - base_sample.get('max_distractor_score', 0)
                        candidates.append((sid, ours_margin - base_margin))
            
            # Sort by margin improvement
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected_samples = [c[0] for c in candidates[:n_samples]]
        
        elif category == "csclip_wins_all":
            # CS-CLIP correct, ALL other SHOWN models wrong
            # (uses models_to_show, not all models)
            if not ours_model:
                print("❌ Need ours_model for csclip_wins_all category")
                return []
            
            # Use models_to_show for comparison, excluding our model
            comparison_models = [m for m in models_to_show if m != ours_model]
            
            if not comparison_models:
                print("❌ No comparison models found (need at least one model besides ours)")
                return []
            
            print(f"   Checking CS-CLIP wins over: {[self.loader.get_label(m) for m in comparison_models]}")
            
            candidates = []
            for s in ref_samples:
                sid = s.get('sample_id', s.get('image_key', ''))
                if not sid:
                    continue
                
                ours_sample = model_lookups.get(ours_model, {}).get(sid)
                if not ours_sample or not ours_sample.get('is_correct', False):
                    continue
                
                # Check if all comparison models fail
                all_others_wrong = True
                for other_m in comparison_models:
                    other_sample = model_lookups.get(other_m, {}).get(sid)
                    if other_sample and other_sample.get('is_correct', False):
                        all_others_wrong = False
                        break
                
                if all_others_wrong:
                    ours_margin = ours_sample.get('target_image_score', 0) - ours_sample.get('max_distractor_score', 0)
                    candidates.append((sid, ours_margin))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected_samples = [c[0] for c in candidates[:n_samples]]
        
        elif category == "all_correct":
            # All models correct
            candidates = []
            for s in ref_samples:
                sid = s.get('sample_id', s.get('image_key', ''))
                if not sid:
                    continue
                
                all_correct = True
                for model_name in models_to_show:
                    m_sample = model_lookups.get(model_name, {}).get(sid)
                    if not m_sample or not m_sample.get('is_correct', False):
                        all_correct = False
                        break
                
                if all_correct:
                    candidates.append(sid)
            
            if candidates:
                selected_indices = np.random.choice(len(candidates), min(n_samples, len(candidates)), replace=False)
                selected_samples = [candidates[i] for i in selected_indices]
        
        elif category == "all_wrong":
            # All models wrong
            candidates = []
            for s in ref_samples:
                sid = s.get('sample_id', s.get('image_key', ''))
                if not sid:
                    continue
                
                all_wrong = True
                for model_name in models_to_show:
                    m_sample = model_lookups.get(model_name, {}).get(sid)
                    if m_sample and m_sample.get('is_correct', False):
                        all_wrong = False
                        break
                
                if all_wrong:
                    candidates.append(sid)
            
            if candidates:
                selected_indices = np.random.choice(len(candidates), min(n_samples, len(candidates)), replace=False)
                selected_samples = [candidates[i] for i in selected_indices]
        
        else:
            print(f"❌ Unknown category: {category}")
            return []
        
        print(f"   Found {len(selected_samples)} samples for category '{category}'")
        
        # Generate visualizations
        for i, sample_id in enumerate(selected_samples):
            print(f"   [{i+1}/{len(selected_samples)}] Visualizing {sample_id}...")
            self.plot_single_sample(
                sample_id=sample_id,
                output_subdir=category,
                models_to_show=models_to_show,
                filename_prefix=f"{i+1:03d}",
            )
        
        return selected_samples


# =============================================================================
# Report Generation
# =============================================================================

class ReportGenerator:
    """Generates text reports."""
    
    def __init__(self, loader: DataLoader, analyzer: Analyzer, config: AnalysisConfig):
        self.loader = loader
        self.analyzer = analyzer
        self.config = config
    
    def print_summary(self):
        """Print summary table."""
        df = self.analyzer.get_summary_df()
        
        print("\n" + "=" * 80)
        print("SUMMARY: Image Half-Truth Vulnerability Results")
        print("=" * 80)
        
        # Print table
        print(f"\n{'Model':<25} {'Accuracy':>10} {'Vulnerability':>14} {'Avg Margin':>12} {'N':>8}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"{row['Model']:<25} {row['Accuracy (%)']:>9.1f}% {row['Vulnerability (%)']:>13.1f}% "
                  f"{row['Avg Margin']:>12.4f} {int(row['N Samples']):>8}")
        
        print("-" * 80)
        
        # Best model
        best_idx = df['Accuracy (%)'].idxmax()
        best = df.loc[best_idx]
        print(f"\n🏆 Best Model: {best['Model']} ({best['Accuracy (%)']:.1f}%)")
    
    def print_interesting_cases(self, cases: List[Dict], title: str = "Interesting Cases"):
        """Print interesting cases."""
        if not cases:
            print(f"\n❌ No {title.lower()} found")
            return
        
        print(f"\n{'=' * 80}")
        print(f"{title}")
        print(f"Found {len(cases)} cases")
        print("=" * 80)
        
        for i, case in enumerate(cases):
            base_sym = "✓" if case['baseline_correct'] else "✗"
            imp_sym = "✓" if case['improved_correct'] else "✗"
            
            print(f"\n#{i+1} | Index: {case['baseline_index']}")
            cap = case['caption'][:70] + '...' if len(case['caption']) > 70 else case['caption']
            print(f"   Caption: \"{cap}\"")
            print(f"   Baseline: {base_sym} (margin: {case['baseline_margin']:+.4f})")
            print(f"   Improved: {imp_sym} (margin: {case['improved_margin']:+.4f})")
            print(f"   Δ Margin: {case['margin_improvement']:+.4f}")


# =============================================================================
# Main Application
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Image Half-Truth Vulnerability Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic summary
  python analyze_image_half_truth.py --results_dir results_image_ht

  # Compare specific models with custom labels
  python analyze_image_half_truth.py --results_dir results_image_ht \\
      --models "openai-clip,negclip,csclip" \\
      --labels "CLIP,NegCLIP,CS-CLIP (Ours)"

  # Generate paper figures
  python analyze_image_half_truth.py --results_dir results_image_ht \\
      --mode paper_figures --output_dir paper_figures

  # Find cases where our model wins
  python analyze_image_half_truth.py --results_dir results_image_ht \\
      --mode find_cases --baseline openai-clip --ours csclip

  # Browse specific cases
  python analyze_image_half_truth.py --results_dir results_image_ht \\
      --mode browse --case_indices 0,1,2

  # Generate per-sample visualizations (random samples)
  python analyze_image_half_truth.py --results_dir results_image_ht \\
      --mode sample_viz --sample_category random --n_samples 10

  # Generate per-sample visualizations (CS-CLIP wins over baseline)
  python analyze_image_half_truth.py --results_dir results_image_ht \\
      --mode sample_viz --sample_category csclip_wins --n_samples 20

  # Generate per-sample visualizations (CS-CLIP wins over ALL other models)
  python analyze_image_half_truth.py --results_dir results_image_ht \\
      --mode sample_viz --sample_category csclip_wins_all --n_samples 15
        """
    )
    
    # Paths
    parser.add_argument('--results_dir', type=str, default='results_image_ht',
                       help='Directory containing model results')
    parser.add_argument('--samples_json', type=str, default='samples.json',
                       help='Path to samples.json file')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                       help='Directory for output figures')
    parser.add_argument('--image_root', type=str, default='..',
                       help='Root directory for images')
    
    # Model configuration
    parser.add_argument('--models', type=str, default='',
                       help='Comma-separated list of model names to include')
    parser.add_argument('--labels', type=str, default='',
                       help='Comma-separated list of display labels')
    
    # Mode
    parser.add_argument('--mode', type=str, default='summary',
                       choices=['summary', 'comparison', 'paper_figures', 'find_cases', 'browse', 'sample_viz', 'all'],
                       help='Analysis mode')
    
    # Comparison settings
    parser.add_argument('--baseline', type=str, default=None,
                       help='Baseline model for comparison')
    parser.add_argument('--ours', type=str, default=None,
                       help='Our model for comparison')
    
    # Case browsing
    parser.add_argument('--case_indices', type=str, default='',
                       help='Comma-separated list of case indices to browse')
    parser.add_argument('--max_cases', type=int, default=10,
                       help='Maximum number of cases to show')
    
    # Sample visualization settings
    parser.add_argument('--sample_category', type=str, default='random',
                       choices=['random', 'csclip_wins', 'csclip_wins_all', 'all_correct', 'all_wrong'],
                       help='Category of samples to visualize')
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling')
    parser.add_argument('--show_models', type=str, default='',
                       help='Comma-separated list of model directory names to show in visualizations (e.g., "00_baseline_openai_vitb32,negclip_coco_vitb32,cs_clip_negclip_vitb32")')
    parser.add_argument('--show_labels', type=str, default='',
                       help='Comma-separated list of display labels for --show_models (e.g., "CLIP,NegCLIP,CS-CLIP (Ours)")')
    
    # Figure settings
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save figures')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not show figures')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['pdf', 'png', 'svg'],
                       help='Figure format')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Figure DPI')
    parser.add_argument('--compact', action='store_true',
                       help='Use compact figure layout')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build configuration
    config = AnalysisConfig(
        results_dir=Path(args.results_dir),
        samples_json=Path(args.samples_json),
        output_dir=Path(args.output_dir),
        image_root=Path(args.image_root),
        models=[m.strip() for m in args.models.split(',') if m.strip()],
        labels=[lbl.strip() for lbl in args.labels.split(',') if lbl.strip()],
        mode=args.mode,
        baseline_model=args.baseline,
        ours_model=args.ours,
        case_indices=[int(i.strip()) for i in args.case_indices.split(',') if i.strip()],
        max_cases=args.max_cases,
        sample_category=args.sample_category,
        n_samples=args.n_samples,
        seed=args.seed,
        show_models=[m.strip() for m in args.show_models.split(',') if m.strip()],
        show_labels=[lbl.strip() for lbl in args.show_labels.split(',') if lbl.strip()],
        save_figures=not args.no_save,
        show_figures=not args.no_show,
        figure_format=args.format,
        dpi=args.dpi,
        compact=args.compact,
    )
    
    # Load data
    loader = DataLoader(config)
    if not loader.load_all():
        print("❌ Failed to load data")
        sys.exit(1)
    
    # If show_models specified with show_labels, update model labels
    if config.show_models and config.show_labels:
        for model_name, label in zip(config.show_models, config.show_labels):
            # Find actual model key that matches
            for actual_key in loader.all_results.keys():
                if model_name.lower() in actual_key.lower() or actual_key.lower() in model_name.lower():
                    loader.model_labels[actual_key] = label
                    break
    
    # Create analyzer and visualizer
    analyzer = Analyzer(loader, config)
    visualizer = Visualizer(loader, analyzer, config)
    reporter = ReportGenerator(loader, analyzer, config)
    
    # Execute based on mode
    if config.mode in ['summary', 'all']:
        reporter.print_summary()
    
    if config.mode in ['comparison', 'paper_figures', 'all']:
        visualizer.plot_accuracy_comparison()
        visualizer.plot_margin_distribution()
    
    if config.mode == 'find_cases':
        models = list(loader.all_results.keys())
        
        # Auto-detect baseline model (CLIP without neg/cs prefix)
        if not config.baseline_model:
            for m in models:
                m_lower = m.lower()
                # Match baseline/openai but NOT negclip, csclip
                if ('baseline' in m_lower or 'openai' in m_lower) and \
                   'negclip' not in m_lower and 'cs_clip' not in m_lower and 'csclip' not in m_lower:
                    config.baseline_model = m
                    break
        
        # Auto-detect our model (cs_clip)
        if not config.ours_model:
            for m in models:
                m_lower = m.lower()
                if 'cs_clip' in m_lower or 'csclip' in m_lower:
                    config.ours_model = m
                    break
        
        print("\n🔍 Finding interesting cases...")
        print(f"   Baseline: {config.baseline_model} ({loader.get_label(config.baseline_model) if config.baseline_model else 'Not found'})")
        print(f"   Ours:     {config.ours_model} ({loader.get_label(config.ours_model) if config.ours_model else 'Not found'})")
        
        if config.baseline_model and config.ours_model:
            cases = analyzer.find_interesting_cases(
                config.baseline_model,
                config.ours_model,
                case_type="ours_wins",
                max_cases=config.max_cases
            )
            reporter.print_interesting_cases(cases, 
                f"Cases where {loader.get_label(config.ours_model)} beats {loader.get_label(config.baseline_model)}")
            
            if cases and config.case_indices == []:
                config.case_indices = [c['baseline_index'] for c in cases[:3]]
                visualizer.plot_case_comparison(config.case_indices)
        else:
            print("❌ Could not auto-detect models. Please specify --baseline and --ours")
            print(f"   Available models: {models}")
    
    if config.mode == 'browse':
        if not config.case_indices:
            config.case_indices = [0, 1, 2]
        visualizer.plot_case_comparison(config.case_indices)
    
    if config.mode == 'sample_viz':
        # Resolve show_models to actual model keys
        models_to_show = None
        if config.show_models:
            models_to_show = []
            for show_m in config.show_models:
                show_m_lower = show_m.lower()
                for actual_key in loader.all_results.keys():
                    if show_m_lower in actual_key.lower() or actual_key.lower() in show_m_lower:
                        models_to_show.append(actual_key)
                        break
        
        # Generate per-sample visualizations
        visualizer.generate_sample_visualizations(
            category=config.sample_category,
            n_samples=config.n_samples,
            baseline_model=config.baseline_model,
            ours_model=config.ours_model,
            models_to_show=models_to_show if models_to_show else None,
            seed=config.seed,
        )
    
    if config.mode == 'paper_figures':
        # Generate all paper-ready figures
        print("\n📊 Generating paper figures...")
        
        # 1. Accuracy comparison
        visualizer.plot_accuracy_comparison()
        
        # 2. Margin distribution
        visualizer.plot_margin_distribution()
        
        # 3. Case comparisons (first few cases)
        if not config.case_indices:
            config.case_indices = [0, 1, 2]
        visualizer.plot_case_comparison(config.case_indices)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

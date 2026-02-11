#!/usr/bin/env python3
"""
Multi-Checkpoint Comparison Visualization Tool
Compares multiple model checkpoints across various metrics and datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import warnings
import argparse
import os
from typing import List, Dict, Tuple

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CheckpointComparer:
    def __init__(self, checkpoint_paths: Dict[str, str], metrics: List[str] = None, 
                 output_dir: str = "figures", save_format: str = "png", dpi: int = 300,
                 checkpoint_labels: Dict[str, str] = None):
        """
        Initialize the comparer with checkpoint paths
        
        Args:
            checkpoint_paths: Dict mapping checkpoint names to folder paths
            metrics: List of metrics to focus on (if None, uses all)
            output_dir: Directory to save figures
            save_format: Format to save figures (png, pdf, svg)
            dpi: DPI for saved figures
            checkpoint_labels: Dict mapping checkpoint names to display labels
        """
        self.checkpoint_paths = checkpoint_paths
        self.metrics = metrics or ['all/contrastive_accuracy']
        self.dataframes = {}
        self.combined_df = None
        self.output_dir = output_dir
        self.save_format = save_format
        self.dpi = dpi
        self.figure_counter = 0
        
        # Set checkpoint labels (use names as labels if not provided)
        self.checkpoint_labels = checkpoint_labels or {name: name for name in checkpoint_paths.keys()}
        
        # Create output directory if it doesn't exist
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Figures will be saved to: {self.output_dir}/")
    
    def _save_figure(self, fig, name_suffix: str):
        """Helper method to save figures"""
        if self.output_dir:
            self.figure_counter += 1
            filename = f"{self.figure_counter:02d}_{name_suffix}.{self.save_format}"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        
    def load_checkpoints(self):
        """Load all CSV files from each checkpoint folder"""
        print("Loading checkpoints...")
        
        for name, path in self.checkpoint_paths.items():
            csv_files = glob.glob(os.path.join(path, "*.csv"))
            print(f"  {name}: Found {len(csv_files)} CSV files")
            
            if csv_files:
                dfs = [pd.read_csv(f) for f in csv_files]
                df = pd.concat(dfs, ignore_index=True)
                df['checkpoint_name'] = name
                self.dataframes[name] = df
                print(f"    Shape: {df.shape}")
        
        # Combine all dataframes
        if self.dataframes:
            self.combined_df = pd.concat(list(self.dataframes.values()), ignore_index=True)
            self._clean_data()
            print(f"\nCombined shape: {self.combined_df.shape}")
            print(f"Datasets: {self.combined_df['dataset'].unique()}")
            print(f"Metrics: {self.combined_df['metric'].unique()[:10]}...")  # Show first 10
    
    def _clean_data(self):
        """Clean and prepare data for plotting"""
        df = self.combined_df
        
        # Convert string percentages to numeric
        for col in ['before_alignment', 'after_alignment', 'absolute_difference', 'percent_change']:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(
                        df[col].str.replace('%', '', regex=False), 
                        errors='coerce'
                    )
        
        # Calculate additional metrics
        df['relative_improvement'] = (
            (df['after_alignment'] - df['before_alignment']) / 
            df['before_alignment'].replace(0, np.nan) * 100
        )
        
        self.combined_df = df
    
    def display_improvement_table(self, dataset: str = None, metric_filter: List[str] = None, 
                                 sort_by: str = 'relative_improvement', output_html: bool = False):
        """
        Displays a styled pandas DataFrame summarizing the alignment results.

        Args:
            dataset (str, optional): Filter results for a specific dataset.
            metric_filter (list, optional): Filter for a list of specific metrics.
            sort_by (str, optional): Column to sort the results by.
            output_html (bool): Whether to output HTML for Jupyter notebooks
        """
        if self.combined_df.empty:
            print("Input DataFrame is empty.")
            return

        # --- 1. Data Preparation ---
        table_df = self.combined_df.copy()

        # Filter by dataset if specified
        if dataset:
            table_df = table_df[table_df['dataset'] == dataset]

        # Filter by metric if specified
        if metric_filter:
            table_df = table_df[table_df['metric'].isin(metric_filter)]

        if table_df.empty:
            print(f"No data found for the specified filters (Dataset: {dataset}, Metrics: {metric_filter}).")
            return

        # Select and rename columns for clarity
        table_df = table_df[['dataset', 'subset_name', 'metric', 'checkpoint_name', 
                           'before_alignment', 'after_alignment']].rename(columns={
            'dataset': 'Dataset',
            'subset_name': 'Subset',
            'metric': 'Metric',
            'checkpoint_name': 'Checkpoint',
            'before_alignment': 'Before',
            'after_alignment': 'After'
        })

        # Add checkpoint labels
        table_df['Checkpoint'] = table_df['Checkpoint'].map(self.checkpoint_labels)

        # Calculate improvement metrics
        table_df['Δ Score'] = table_df['After'] - table_df['Before']
        # Handle division by zero for percent change
        table_df['Δ %'] = (100 * table_df['Δ Score'] / table_df['Before'].replace(0, np.nan)).fillna(0)

        # Sort the data
        if sort_by in table_df.columns:
            table_df = table_df.sort_values(by=sort_by, ascending=False)

        # --- 2. Styling ---
        def color_change(val):
            """Colors text green for positive, red for negative."""
            if isinstance(val, (int, float)):
                if val > 0.001:
                    color = 'darkgreen'
                elif val < -0.001:
                    color = 'firebrick'
                else:
                    color = 'black'
                return f'color: {color}'
            return ''

        styled_table = table_df.style \
            .applymap(color_change, subset=['Δ Score', 'Δ %']) \
            .format({
                'Before': '{:.3f}',
                'After': '{:.3f}',
                'Δ Score': '{:+.3f}',
                'Δ %': '{:+.1f}%'
            }) \
            .background_gradient(cmap='Greens', subset=['Δ Score'], vmin=0) \
            .background_gradient(cmap='RdBu_r', subset=['Δ %'], vmin=-20, vmax=20) \
            .set_properties(**{'text-align': 'center'}) \
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center'), ('font-size', '12pt')]},
                {'selector': 'td', 'props': [('font-size', '11pt')]}
            ]) \
            .set_caption(f"Alignment Performance Summary for {dataset if dataset else 'All Datasets'}") \
            .hide(axis='index')

        if output_html:
            try:
                from IPython.display import display
                display(styled_table)
            except ImportError:
                print("IPython not available. Displaying plain table:")
                print(table_df.to_string(index=False))
            except Exception:
                print("Could not display styled table. Displaying plain table:")
                print(table_df.to_string(index=False))
        else:
            print(table_df.to_string(index=False))
        
        return styled_table
    
    def generate_dataset_table_images(self, figsize: Tuple = (14, 10)):
        """
        Generate visually appealing table images for each dataset, sorted by contrastive accuracy delta
        """
        if self.combined_df.empty:
            print("No data available for table generation.")
            return

        # Filter for contrastive accuracy metric
        df = self.combined_df[self.combined_df['metric'] == 'all/contrastive_accuracy'].copy()
        
        if df.empty:
            print("No contrastive accuracy data found.")
            return

        # Get unique datasets
        datasets = df['dataset'].unique()
        
        for dataset in datasets:
            dataset_df = df[df['dataset'] == dataset].copy()
            
            if dataset_df.empty:
                continue
                
            # Prepare data for table
            table_data = []
            for _, row in dataset_df.iterrows():
                checkpoint_label = self.checkpoint_labels.get(row['checkpoint_name'], row['checkpoint_name'])
                table_data.append({
                    'Checkpoint': checkpoint_label,
                    'Subset': row['subset_name'],
                    'Before': row['before_alignment'],
                    'After': row['after_alignment'],
                    'Δ Score': row['after_alignment'] - row['before_alignment'],
                    'Δ %': row['relative_improvement']
                })
            
            table_df = pd.DataFrame(table_data)
            
            # Sort by delta score (descending)
            table_df = table_df.sort_values('Δ Score', ascending=False)
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            ax.axis('tight')
            ax.axis('off')
            
            # Color mapping for delta scores
            def get_cell_colors(df):
                colors = []
                for i, row in df.iterrows():
                    row_colors = ['white'] * len(df.columns)  # Default color
                    
                    # Color delta score column
                    delta_score = row['Δ Score']
                    if delta_score > 0.01:
                        row_colors[4] = '#d4edda'  # Light green
                    elif delta_score < -0.01:
                        row_colors[4] = '#f8d7da'  # Light red
                    
                    # Color delta percentage column
                    delta_pct = row['Δ %']
                    if delta_pct > 1:
                        row_colors[5] = '#d4edda'  # Light green
                    elif delta_pct < -1:
                        row_colors[5] = '#f8d7da'  # Light red
                    
                    colors.append(row_colors)
                return colors
            
            # Format numbers for display
            display_df = table_df.copy()
            display_df['Before'] = display_df['Before'].apply(lambda x: f'{x:.3f}')
            display_df['After'] = display_df['After'].apply(lambda x: f'{x:.3f}')
            display_df['Δ Score'] = display_df['Δ Score'].apply(lambda x: f'{x:+.3f}')
            display_df['Δ %'] = display_df['Δ %'].apply(lambda x: f'{x:+.1f}%')
            
            # Create table
            cell_colors = get_cell_colors(table_df)
            
            table = ax.table(
                cellText=display_df.values,
                colLabels=display_df.columns,
                cellLoc='center',
                loc='center',
                cellColours=cell_colors
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2)
            
            # Style header
            for i in range(len(display_df.columns)):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')
                table[(0, i)].set_height(0.08)
            
            # Style cells
            for i in range(1, len(display_df) + 1):
                for j in range(len(display_df.columns)):
                    table[(i, j)].set_height(0.06)
                    if j in [4, 5]:  # Delta columns
                        table[(i, j)].set_text_props(weight='bold')
            
            # Add title
            title = f'Contrastive Accuracy Improvements - {dataset}'
            plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
            
            # Add subtitle with sorting info
            subtitle = 'Sorted by Δ Score (descending)'
            plt.figtext(0.5, 0.92, subtitle, ha='center', fontsize=12, style='italic')
            
            plt.tight_layout()
            
            # Save the table image
            self._save_figure(fig, f"table_{dataset}_contrastive_accuracy")
            
            plt.show()
            
            # Print summary
            best_improvement = table_df.iloc[0]
            print(f"\n{dataset} - Best Improvement:")
            print(f"  {best_improvement['Checkpoint']} ({best_improvement['Subset']}): {best_improvement['Δ Score']:+.3f} ({best_improvement['Δ %']:+.1f}%)")
    
    def plot_subset_improvements(self, dataset: str = None, metric: str = 'all/contrastive_accuracy', 
                                figsize: Tuple = (16, 12)):
        """
        Create improvement plots for subsets within datasets
        """
        df = self.combined_df.copy()
        
        # Filter by metric
        df = df[df['metric'] == metric]
        
        if df.empty:
            print(f"No data found for metric {metric}")
            return
        
        # If dataset specified, use only that dataset
        if dataset:
            df = df[df['dataset'] == dataset]
            datasets = [dataset]
        else:
            datasets = df['dataset'].unique()
        
        for dataset_name in datasets:
            dataset_df = df[df['dataset'] == dataset_name].copy()
            
            if dataset_df.empty:
                continue
            
            # Get unique subsets
            subsets = dataset_df['subset_name'].unique()
            
            if len(subsets) <= 1:
                print(f"Dataset {dataset_name} has only {len(subsets)} subset(s). Skipping subset plot.")
                continue
            
            # Calculate optimal subplot layout
            n_subsets = len(subsets)
            
            # Use more intelligent layout calculation
            if n_subsets <= 3:
                ncols, nrows = n_subsets, 1
                figsize = (5 * n_subsets, 6)
            elif n_subsets <= 6:
                ncols, nrows = 3, 2
                figsize = (15, 10)
            elif n_subsets <= 9:
                ncols, nrows = 3, 3
                figsize = (15, 12)
            else:
                ncols = 4
                nrows = int(np.ceil(n_subsets / ncols))
                figsize = (20, 4 * nrows)
            
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
            fig.suptitle(f'Subset Improvements - {dataset_name}\nMetric: {metric}', 
                        fontsize=18, fontweight='bold', y=0.95)
            
            # Calculate global y-limits for consistency across subplots
            all_improvements = []
            for subset in subsets:
                subset_df = dataset_df[dataset_df['subset_name'] == subset]
                for checkpoint_name in self.checkpoint_paths.keys():
                    checkpoint_data = subset_df[subset_df['checkpoint_name'] == checkpoint_name]
                    if not checkpoint_data.empty:
                        all_improvements.append(checkpoint_data['relative_improvement'].iloc[0])
            
            if all_improvements:
                global_y_min = min(all_improvements) * 1.2
                global_y_max = max(all_improvements) * 1.2
                # Ensure 0 is visible and somewhat centered
                y_range = max(abs(global_y_min), abs(global_y_max))
                global_y_min = -y_range * 1.1
                global_y_max = y_range * 1.1
            else:
                global_y_min, global_y_max = -5, 5
            
            for idx, subset in enumerate(subsets):
                row, col = idx // ncols, idx % ncols
                ax = axes[row, col]
                
                subset_df = dataset_df[dataset_df['subset_name'] == subset]
                
                # Prepare data for plotting
                checkpoints = []
                improvements = []
                before_scores = []
                after_scores = []
                
                for checkpoint_name in self.checkpoint_paths.keys():
                    checkpoint_data = subset_df[subset_df['checkpoint_name'] == checkpoint_name]
                    if not checkpoint_data.empty:
                        label = self.checkpoint_labels.get(checkpoint_name, checkpoint_name)
                        checkpoints.append(label)
                        improvements.append(checkpoint_data['relative_improvement'].iloc[0])
                        before_scores.append(checkpoint_data['before_alignment'].iloc[0])
                        after_scores.append(checkpoint_data['after_alignment'].iloc[0])
                
                if not checkpoints:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=14, style='italic')
                    ax.set_title(f"{subset}", fontsize=12, fontweight='bold', pad=15)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(global_y_min, global_y_max)
                    continue
                
                # Create bar plot for improvements with better colors
                x_pos = np.arange(len(checkpoints))
                colors = []
                for imp in improvements:
                    if imp > 0.5:
                        colors.append('#2E8B57')  # Sea green for good improvements
                    elif imp > 0:
                        colors.append('#90EE90')  # Light green for small improvements
                    elif imp > -0.5:
                        colors.append('#FFB6C1')  # Light pink for small degradations
                    else:
                        colors.append('#DC143C')  # Crimson for bad degradations
                
                bars = ax.bar(x_pos, improvements, color=colors, alpha=0.8, 
                             edgecolor='black', linewidth=1.2, width=0.6)
                
                # Add value labels on bars with better positioning
                for i, (bar, imp, before, after) in enumerate(zip(bars, improvements, before_scores, after_scores)):
                    height = bar.get_height()
                    
                    # Position text above/below bar depending on sign and magnitude
                    if height > 0:
                        y_pos = height + (global_y_max - global_y_min) * 0.02
                        va = 'bottom'
                    else:
                        y_pos = height - (global_y_max - global_y_min) * 0.02
                        va = 'top'
                    
                    # Main improvement percentage
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{imp:+.1f}%', ha='center', va=va,
                           fontweight='bold', fontsize=11)
                    
                    # Before/after scores at the bottom
                    score_y = global_y_min + (global_y_max - global_y_min) * 0.08
                    ax.text(bar.get_x() + bar.get_width()/2., score_y,
                           f'{before:.3f}→{after:.3f}', ha='center', va='bottom',
                           fontsize=9, style='italic', alpha=0.8)
                
                # Add horizontal line at 0 with better styling
                ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.8, zorder=10)
                
                # Add subtle grid
                ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
                
                # Styling improvements
                ax.set_title(f"{subset}", fontsize=12, fontweight='bold', pad=15)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(checkpoints, rotation=45, ha='right', fontsize=10)
                ax.set_ylabel("Improvement (%)", fontsize=11, fontweight='bold')
                
                # Set consistent y-limits
                ax.set_ylim(global_y_min, global_y_max)
                
                # Remove top and right spines for cleaner look
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                
                # Add subtle background color
                ax.set_facecolor('#FAFAFA')
            
            # Remove empty subplots and style them
            for idx in range(len(subsets), nrows * ncols):
                row, col = idx // ncols, idx % ncols
                ax = axes[row, col]
                ax.axis('off')
            
            # Adjust layout with better spacing
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            
            # Save figure
            self._save_figure(fig, f"subset_improvements_{dataset_name}_{metric.replace('/', '_')}")
            
            plt.show()
    
    def plot_subset_comparison_heatmap(self, metric: str = 'all/contrastive_accuracy', 
                                      figsize: Tuple = (16, 12)):
        """
        Create a heatmap showing improvements across all subsets and checkpoints
        """
        df = self.combined_df.copy()
        df = df[df['metric'] == metric]
        
        if df.empty:
            print(f"No data found for metric {metric}")
            return
        
        # Create dataset_subset combination
        df['dataset_subset'] = df['dataset'] + ' - ' + df['subset_name']
        df['checkpoint_label'] = df['checkpoint_name'].map(self.checkpoint_labels)
        
        # Create pivot table
        pivot_df = df.pivot_table(
            values='relative_improvement',
            index='dataset_subset',
            columns='checkpoint_label',
            aggfunc='mean'
        )
        
        # Sort by mean improvement across checkpoints
        pivot_df['mean_improvement'] = pivot_df.mean(axis=1)
        pivot_df = pivot_df.sort_values('mean_improvement', ascending=False)
        pivot_df = pivot_df.drop('mean_improvement', axis=1)
        
        # Calculate figure size based on data
        n_rows = len(pivot_df.index)
        n_cols = len(pivot_df.columns)
        
        # Dynamic figure sizing
        width = max(12, n_cols * 2)
        height = max(8, n_rows * 0.5)
        figsize = (min(width, 20), min(height, 16))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate center point for better color scaling
        vmin = pivot_df.min().min()
        vmax = pivot_df.max().max()
        vcenter = 0
        
        # Use a better color scheme
        cmap = sns.diverging_palette(10, 150, center='light', as_cmap=True)
        
        # Plot heatmap with improved styling
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt='.1f',
            cmap=cmap,
            center=vcenter,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={
                'label': 'Relative Improvement (%)',
                'shrink': 0.8,
                'aspect': 30
            },
            linewidths=1,
            linecolor='white',
            square=False,
            annot_kws={'fontsize': 10, 'fontweight': 'bold'},
            ax=ax
        )
        
        # Improve title and labels
        ax.set_title(
            f"Subset Improvements Heatmap\nMetric: {metric}\n(Sorted by mean improvement)", 
            fontsize=16, fontweight='bold', pad=25
        )
        ax.set_xlabel("Checkpoint", fontsize=14, fontweight='bold')
        ax.set_ylabel("Dataset - Subset", fontsize=14, fontweight='bold')
        
        # Better tick formatting
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=10)
        
        # Add grid for better readability
        ax.set_facecolor('white')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, f"subset_heatmap_{metric.replace('/', '_')}")
        
        plt.show()
    
    def plot_checkpoint_comparison_bars(self, dataset: str = None, metric: str = None, 
                                       subset: str = None, figsize: Tuple = (14, 8)):
        """
        Compare checkpoints using grouped bar plots
        """
        df = self.combined_df.copy()
        
        # Filter data
        if dataset:
            df = df[df['dataset'] == dataset]
        if metric:
            df = df[df['metric'] == metric]
        else:
            metric = self.metrics[0]
            df = df[df['metric'] == metric]
        if subset:
            df = df[df['subset_name'] == subset]
        
        if df.empty:
            print("No data matches the filters")
            return
        
        # Prepare data for plotting
        plot_data = []
        for _, row in df.iterrows():
            checkpoint_label = self.checkpoint_labels.get(row['checkpoint_name'], row['checkpoint_name'])
            plot_data.append({
                'Checkpoint': checkpoint_label,
                'Stage': 'Before',
                'Score': row['before_alignment'],
                'Dataset': row['dataset'],
                'Subset': row.get('subset_name', 'all')
            })
            plot_data.append({
                'Checkpoint': checkpoint_label,
                'Stage': 'After',
                'Score': row['after_alignment'],
                'Dataset': row['dataset'],
                'Subset': row.get('subset_name', 'all')
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot
        sns.barplot(
            data=plot_df,
            x='Checkpoint',
            y='Score',
            hue='Stage',
            palette=['#FF6B6B', '#4ECDC4'],
            edgecolor='black',
            linewidth=1.5,
            ax=ax
        )
        
        # Annotate improvements
        checkpoint_labels = plot_df['Checkpoint'].unique()
        for i, checkpoint_label in enumerate(checkpoint_labels):
            # Find the original checkpoint name for this label
            original_name = None
            for name, label in self.checkpoint_labels.items():
                if label == checkpoint_label:
                    original_name = name
                    break
            
            if original_name:
                checkpoint_data = df[df['checkpoint_name'] == original_name].iloc[0]
                before = checkpoint_data['before_alignment']
                after = checkpoint_data['after_alignment']
                improvement = checkpoint_data['relative_improvement']
                
                # Add improvement text above bars
                y_pos = max(before, after) * 1.05
                ax.text(i, y_pos, f'+{improvement:.1f}%' if improvement > 0 else f'{improvement:.1f}%',
                       ha='center', fontweight='bold',
                       color='green' if improvement > 0 else 'red')
        
        # Styling
        title = "Checkpoint Comparison"
        if dataset:
            title += f" - {dataset}"
        if subset:
            title += f" ({subset})"
        title += f"\nMetric: {metric}"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Checkpoint", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.legend(title="Stage", fontsize=11, title_fontsize=12)
        plt.xticks(rotation=45, ha='right')
        sns.despine()
        plt.tight_layout()
        
        # Save figure
        name = f"checkpoint_comparison_{dataset or 'all'}_{metric.replace('/', '_')}"
        if subset:
            name += f"_{subset}"
        self._save_figure(fig, name)
        
        plt.show()
    
    def plot_relative_improvements_heatmap(self, metric: str = None, figsize: Tuple = (16, 10)):
        """
        Create a heatmap showing relative improvements across checkpoints and datasets
        """
        df = self.combined_df.copy()
        
        if metric:
            df = df[df['metric'] == metric]
        else:
            metric = self.metrics[0]
            df = df[df['metric'] == metric]
        
        # Map checkpoint names to labels
        df['checkpoint_label'] = df['checkpoint_name'].map(self.checkpoint_labels)
        
        # Create pivot table
        pivot_df = df.pivot_table(
            values='relative_improvement',
            index='dataset',
            columns='checkpoint_label',
            aggfunc='mean'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Relative Improvement (%)'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        
        ax.set_title(f"Relative Improvements Across Checkpoints\nMetric: {metric}", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Checkpoint", fontsize=12)
        ax.set_ylabel("Dataset", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, f"heatmap_improvements_{metric.replace('/', '_')}")
        
        plt.show()
    
    def plot_subset_comparison(self, dataset: str, metric: str = None, figsize: Tuple = (18, 10)):
        """
        Compare checkpoints across all subsets of a dataset
        """
        df = self.combined_df.copy()
        df = df[df['dataset'] == dataset]
        
        if metric:
            df = df[df['metric'] == metric]
        else:
            metric = self.metrics[0]
            df = df[df['metric'] == metric]
        
        if df.empty:
            print(f"No data for dataset {dataset} and metric {metric}")
            return
        
        subsets = df['subset_name'].unique()
        n_subsets = len(subsets)
        
        # Setup subplot grid
        ncols = min(4, n_subsets)
        nrows = int(np.ceil(n_subsets / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        
        for idx, subset in enumerate(subsets):
            ax = axes[idx // ncols, idx % ncols]
            subset_df = df[df['subset_name'] == subset]
            
            # Prepare data
            x_pos = np.arange(len(self.checkpoint_paths))
            width = 0.35
            
            before_vals = []
            after_vals = []
            improvements = []
            labels = []
            
            for checkpoint_name in self.checkpoint_paths.keys():
                checkpoint_data = subset_df[subset_df['checkpoint_name'] == checkpoint_name]
                checkpoint_label = self.checkpoint_labels.get(checkpoint_name, checkpoint_name)
                labels.append(checkpoint_label)
                
                if not checkpoint_data.empty:
                    before_vals.append(checkpoint_data['before_alignment'].iloc[0])
                    after_vals.append(checkpoint_data['after_alignment'].iloc[0])
                    improvements.append(checkpoint_data['relative_improvement'].iloc[0])
                else:
                    before_vals.append(0)
                    after_vals.append(0)
                    improvements.append(0)
            
            # Plot bars
            bars1 = ax.bar(x_pos - width/2, before_vals, width, label='Before', 
                          color='lightcoral', edgecolor='black')
            bars2 = ax.bar(x_pos + width/2, after_vals, width, label='After',
                          color='lightgreen', edgecolor='black')
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Add improvement percentages
            for i, (imp, after) in enumerate(zip(improvements, after_vals)):
                if after > 0:
                    ax.text(i, after * 1.05, f'{imp:+.1f}%',
                           ha='center', fontweight='bold', fontsize=9,
                           color='green' if imp > 0 else 'red')
            
            ax.set_title(f"Subset: {subset}", fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel("Score", fontsize=10)
            ax.legend(fontsize=8)
            sns.despine(ax=ax)
        
        # Remove empty subplots
        for j in range(idx + 1, nrows * ncols):
            fig.delaxes(axes[j // ncols, j % ncols])
        
        fig.suptitle(f"Checkpoint Comparison Across Subsets\nDataset: {dataset}, Metric: {metric}",
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, f"subset_comparison_{dataset}_{metric.replace('/', '_')}")
        
        plt.show()
    
    def plot_improvement_distribution(self, metric: str = None, figsize: Tuple = (14, 8)):
        """
        Plot distribution of improvements for each checkpoint
        """
        df = self.combined_df.copy()
        
        if metric:
            df = df[df['metric'] == metric]
        else:
            metric = self.metrics[0]
            df = df[df['metric'] == metric]
        
        # Map checkpoint names to labels
        df['checkpoint_label'] = df['checkpoint_name'].map(self.checkpoint_labels)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot
        ax1 = axes[0]
        sns.boxplot(
            data=df,
            x='checkpoint_label',
            y='relative_improvement',
            palette='Set2',
            ax=ax1
        )
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_title(f"Distribution of Relative Improvements\nMetric: {metric}", 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel("Checkpoint", fontsize=11)
        ax1.set_ylabel("Relative Improvement (%)", fontsize=11)
        ax1.tick_params(axis='x', rotation=45)
        
        # Violin plot
        ax2 = axes[1]
        sns.violinplot(
            data=df,
            x='checkpoint_label',
            y='relative_improvement',
            palette='muted',
            ax=ax2
        )
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title(f"Density of Relative Improvements\nMetric: {metric}", 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel("Checkpoint", fontsize=11)
        ax2.set_ylabel("Relative Improvement (%)", fontsize=11)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, f"improvement_distribution_{metric.replace('/', '_')}")
        
        plt.show()
    
    def plot_absolute_differences(self, dataset: str = None, metric: str = 'all/contrastive_accuracy', 
                                 figsize: Tuple = (14, 8)):
        """
        Plot absolute differences (after - before) for each checkpoint
        """
        df = self.combined_df.copy()
        
        # Filter by metric
        df = df[df['metric'] == metric]
        
        if df.empty:
            print(f"No data found for metric {metric}")
            return
        
        # If dataset specified, use only that dataset
        if dataset:
            df = df[df['dataset'] == dataset]
            datasets = [dataset]
        else:
            datasets = df['dataset'].unique()
        
        for dataset_name in datasets:
            dataset_df = df[df['dataset'] == dataset_name].copy()
            
            if dataset_df.empty:
                continue
            
            # Get unique subsets
            subsets = dataset_df['subset_name'].unique()
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Prepare data for plotting
            x_pos = np.arange(len(subsets))
            width = 0.8 / len(self.checkpoint_paths)
            
            # Plot bars for each checkpoint
            for i, checkpoint_name in enumerate(self.checkpoint_paths.keys()):
                checkpoint_label = self.checkpoint_labels.get(checkpoint_name, checkpoint_name)
                abs_diffs = []
                
                for subset in subsets:
                    subset_data = dataset_df[
                        (dataset_df['subset_name'] == subset) & 
                        (dataset_df['checkpoint_name'] == checkpoint_name)
                    ]
                    
                    if not subset_data.empty:
                        # Calculate absolute difference: after - before
                        abs_diff = subset_data['after_alignment'].iloc[0] - subset_data['before_alignment'].iloc[0]
                        abs_diffs.append(abs_diff)
                    else:
                        abs_diffs.append(0)
                
                # Color bars based on positive/negative values
                colors = ['green' if v >= 0 else 'red' for v in abs_diffs]
                bars = ax.bar(x_pos + i*width, abs_diffs, width, 
                            label=checkpoint_label, color=colors, alpha=0.7, 
                            edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for bar, val in zip(bars, abs_diffs):
                    height = bar.get_height()
                    if height != 0:
                        ax.text(bar.get_x() + bar.get_width()/2., 
                               height + (0.01 if height > 0 else -0.01),
                               f'{val:+.3f}', ha='center', 
                               va='bottom' if height > 0 else 'top',
                               fontweight='bold', fontsize=10)
            
            # Styling
            ax.axhline(0, color='black', linewidth=1.5, alpha=0.8)
            ax.set_title(f'Absolute Score Differences - {dataset_name}\nMetric: {metric}', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Subset', fontsize=12, fontweight='bold')
            ax.set_ylabel('Absolute Difference (After - Before)', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos + width * (len(self.checkpoint_paths) - 1) / 2)
            ax.set_xticklabels(subsets, rotation=45, ha='right')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            # Save figure
            self._save_figure(fig, f"absolute_differences_{dataset_name}_{metric.replace('/', '_')}")
            
            plt.show()

    def plot_absolute_vs_relative_comparison(self, dataset: str = None, metric: str = 'all/contrastive_accuracy',
                                           figsize: Tuple = (16, 8)):
        """
        Side-by-side comparison of absolute differences and relative improvements
        """
        df = self.combined_df.copy()
        df = df[df['metric'] == metric]
        
        if df.empty:
            print(f"No data found for metric {metric}")
            return
        
        # If dataset specified, use only that dataset
        if dataset:
            df = df[df['dataset'] == dataset]
            datasets = [dataset]
        else:
            datasets = df['dataset'].unique()
        
        for dataset_name in datasets:
            dataset_df = df[df['dataset'] == dataset_name].copy()
            
            if dataset_df.empty:
                continue
            
            # Get unique subsets
            subsets = dataset_df['subset_name'].unique()
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Prepare data
            x_pos = np.arange(len(subsets))
            width = 0.8 / len(self.checkpoint_paths)
            
            # Plot absolute differences (left subplot)
            for i, checkpoint_name in enumerate(self.checkpoint_paths.keys()):
                checkpoint_label = self.checkpoint_labels.get(checkpoint_name, checkpoint_name)
                abs_diffs = []
                rel_improvements = []
                
                for subset in subsets:
                    subset_data = dataset_df[
                        (dataset_df['subset_name'] == subset) & 
                        (dataset_df['checkpoint_name'] == checkpoint_name)
                    ]
                    
                    if not subset_data.empty:
                        abs_diff = subset_data['after_alignment'].iloc[0] - subset_data['before_alignment'].iloc[0]
                        rel_imp = subset_data['relative_improvement'].iloc[0]
                        abs_diffs.append(abs_diff)
                        rel_improvements.append(rel_imp)
                    else:
                        abs_diffs.append(0)
                        rel_improvements.append(0)
                
                # Absolute differences plot
                colors1 = ['green' if v >= 0 else 'red' for v in abs_diffs]
                bars1 = ax1.bar(x_pos + i*width, abs_diffs, width, 
                              label=checkpoint_label, color=colors1, alpha=0.7,
                              edgecolor='black', linewidth=1)
                
                # Relative improvements plot
                colors2 = ['green' if v >= 0 else 'red' for v in rel_improvements]
                bars2 = ax2.bar(x_pos + i*width, rel_improvements, width,
                              label=checkpoint_label, color=colors2, alpha=0.7,
                              edgecolor='black', linewidth=1)
                
                # Add value labels
                for bar, val in zip(bars1, abs_diffs):
                    if val != 0:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., 
                               height + (0.005 if height > 0 else -0.005),
                               f'{val:+.3f}', ha='center', 
                               va='bottom' if height > 0 else 'top',
                               fontweight='bold', fontsize=9)
                
                for bar, val in zip(bars2, rel_improvements):
                    if val != 0:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., 
                               height + (0.5 if height > 0 else -0.5),
                               f'{val:+.1f}%', ha='center', 
                               va='bottom' if height > 0 else 'top',
                               fontweight='bold', fontsize=9)
            
            # Style absolute differences plot
            ax1.axhline(0, color='black', linewidth=1.5, alpha=0.8)
            ax1.set_title('Absolute Differences\n(After - Before)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Subset', fontsize=12)
            ax1.set_ylabel('Score Difference', fontsize=12)
            ax1.set_xticks(x_pos + width * (len(self.checkpoint_paths) - 1) / 2)
            ax1.set_xticklabels(subsets, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            # Style relative improvements plot
            ax2.axhline(0, color='black', linewidth=1.5, alpha=0.8)
            ax2.set_title('Relative Improvements\n(Percentage Change)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Subset', fontsize=12)
            ax2.set_ylabel('Improvement (%)', fontsize=12)
            ax2.set_xticks(x_pos + width * (len(self.checkpoint_paths) - 1) / 2)
            ax2.set_xticklabels(subsets, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # Add legend to the second subplot
            ax2.legend(loc='best', fontsize=10)
            
            # Main title
            fig.suptitle(f'{dataset_name} - Absolute vs Relative Differences\nMetric: {metric}', 
                        fontsize=16, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            
            # Save figure
            self._save_figure(fig, f"abs_vs_rel_comparison_{dataset_name}_{metric.replace('/', '_')}")
            
            plt.show()

    def plot_scatter_comparison(self, metric: str = None, figsize: Tuple = (12, 10)):
        """
        Scatter plot comparing before vs after alignment for all checkpoints
        """
        df = self.combined_df.copy()
        
        if metric:
            df = df[df['metric'] == metric]
        else:
            metric = self.metrics[0]
            df = df[df['metric'] == metric]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot scatter for each checkpoint
        for checkpoint_name in self.checkpoint_paths.keys():
            checkpoint_df = df[df['checkpoint_name'] == checkpoint_name]
            checkpoint_label = self.checkpoint_labels.get(checkpoint_name, checkpoint_name)
            ax.scatter(
                checkpoint_df['before_alignment'],
                checkpoint_df['after_alignment'],
                label=checkpoint_label,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
        
        # Add diagonal line (no change)
        min_val = df[['before_alignment', 'after_alignment']].min().min()
        max_val = df[['before_alignment', 'after_alignment']].max().max()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='No change')
        
        # Styling
        ax.set_xlabel("Before Alignment", fontsize=12)
        ax.set_ylabel("After Alignment", fontsize=12)
        ax.set_title(f"Before vs After Alignment Comparison\nMetric: {metric}", 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add improvement region shading
        ax.fill_between([min_val, max_val], [min_val, max_val], max_val,
                       color='green', alpha=0.1, label='Improvement region')
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, f"scatter_comparison_{metric.replace('/', '_')}")
        
        plt.show()
    
    def plot_metric_comparison_radar(self, dataset: str, checkpoint_names: List[str] = None, 
                                    figsize: Tuple = (12, 12)):
        """
        Radar chart comparing multiple metrics across checkpoints for a specific dataset
        """
        try:
            import matplotlib.pyplot as plt
            from math import pi
        except ImportError as e:
            print(f"Failed to import required modules for radar plot: {e}")
            return
            
        try:
            df = self.combined_df.copy()
            df = df[df['dataset'] == dataset]
            
            if checkpoint_names is None:
                checkpoint_names = list(self.checkpoint_paths.keys())
            
            # Filter for specified checkpoints
            df = df[df['checkpoint_name'].isin(checkpoint_names)]
            
            if df.empty:
                print(f"No data found for dataset {dataset} and checkpoints {checkpoint_names}")
                return
            
            # Get common metrics across all checkpoints
            metrics_count = df.groupby('metric')['checkpoint_name'].nunique()
            common_metrics = metrics_count[metrics_count >= len(checkpoint_names)].index
            
            if len(common_metrics) < 3:
                print(f"Not enough common metrics for radar chart (found {len(common_metrics)}). Need at least 3.")
                print(f"Available metrics: {list(common_metrics)}")
                return
            
            # Limit to reasonable number of metrics for visibility
            common_metrics = list(common_metrics)[:8]
            
            # Prepare data
            N = len(common_metrics)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Create the plot
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
            
            # Plot each checkpoint
            for checkpoint_name in checkpoint_names:
                checkpoint_df = df[df['checkpoint_name'] == checkpoint_name]
                checkpoint_label = self.checkpoint_labels.get(checkpoint_name, checkpoint_name)
                values = []
                
                for metric in common_metrics:
                    metric_data = checkpoint_df[checkpoint_df['metric'] == metric]
                    if not metric_data.empty:
                        # Use relative improvement
                        improvement = metric_data['relative_improvement'].iloc[0]
                        values.append(improvement)
                    else:
                        values.append(0)
                
                # Close the plot
                values += values[:1]
                
                # Plot
                ax.plot(angles, values, 'o-', linewidth=2, label=checkpoint_label, markersize=4)
                ax.fill(angles, values, alpha=0.15)
            
            # Customize the plot
            ax.set_xticks(angles[:-1])
            # Clean up metric names for display
            metric_labels = []
            for metric in common_metrics:
                if '/' in metric:
                    label = metric.split('/')[-1]
                else:
                    label = metric
                # Truncate long labels
                if len(label) > 15:
                    label = label[:12] + "..."
                metric_labels.append(label)
            
            ax.set_xticklabels(metric_labels, fontsize=9)
            
            # Set y-axis limits based on data
            all_values = []
            for checkpoint_name in checkpoint_names:
                checkpoint_df = df[df['checkpoint_name'] == checkpoint_name]
                for metric in common_metrics:
                    metric_data = checkpoint_df[checkpoint_df['metric'] == metric]
                    if not metric_data.empty:
                        all_values.append(metric_data['relative_improvement'].iloc[0])
            
            if all_values:
                y_min = min(all_values) - 5
                y_max = max(all_values) + 5
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_ylim(-20, 50)
            
            # Add grid lines at key values
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # Labels and title
            ax.set_ylabel("Relative Improvement (%)", fontsize=10, labelpad=20)
            ax.set_title(f"Multi-Metric Comparison Radar Chart\nDataset: {dataset}", 
                        fontsize=14, fontweight='bold', pad=30)
            
            # Legend
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
            
            plt.tight_layout()
            
            # Save figure
            self._save_figure(fig, f"radar_chart_{dataset}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating radar plot: {e}")
            print("This might be due to missing data or matplotlib configuration issues.")
            return
    
    def generate_summary_report(self, output_file: str = "checkpoint_comparison_summary.txt"):
        """
        Generate a text summary report of the comparison
        """
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CHECKPOINT COMPARISON SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for checkpoint_name in self.checkpoint_paths.keys():
                checkpoint_label = self.checkpoint_labels.get(checkpoint_name, checkpoint_name)
                f.write(f"\nCheckpoint: {checkpoint_label} ({checkpoint_name})\n")
                f.write("-" * 40 + "\n")
                
                checkpoint_df = self.combined_df[self.combined_df['checkpoint_name'] == checkpoint_name]
                
                for metric in self.metrics:
                    metric_df = checkpoint_df[checkpoint_df['metric'] == metric]
                    if not metric_df.empty:
                        f.write(f"\n  Metric: {metric}\n")
                        f.write(f"    Mean improvement: {metric_df['relative_improvement'].mean():.2f}%\n")
                        f.write(f"    Std improvement: {metric_df['relative_improvement'].std():.2f}%\n")
                        f.write(f"    Max improvement: {metric_df['relative_improvement'].max():.2f}%\n")
                        f.write(f"    Min improvement: {metric_df['relative_improvement'].min():.2f}%\n")
                        f.write(f"    Datasets evaluated: {len(metric_df['dataset'].unique())}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("COMPARATIVE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            for metric in self.metrics:
                f.write(f"\nMetric: {metric}\n")
                f.write("-" * 40 + "\n")
                
                metric_df = self.combined_df[self.combined_df['metric'] == metric]
                summary = metric_df.groupby('checkpoint_name')['relative_improvement'].agg(['mean', 'std', 'max', 'min'])
                f.write(summary.to_string())
                f.write("\n")
        
        print(f"Summary report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple model checkpoints")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of checkpoint paths in format name:path (e.g., model1:/path/to/csv/folder1)"
    )
    parser.add_argument(
        "--checkpoint-labels",
        nargs="+",
        default=None,
        help="List of labels for checkpoints in format name:label (e.g., model1:'Model A')"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all/contrastive_accuracy"],
        help="List of metrics to analyze"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to focus on"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Specific subset to focus on"
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        default=["all"],
        choices=["bars", "heatmap", "subset", "distribution", "scatter", "radar", "subset_improvements", "subset_heatmap", "absolute_differences", "abs_vs_rel", "all"],
        help="Types of plots to generate"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save a summary report"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save figures (default: figures)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "jpg"],
        help="Format to save figures (default: png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display figures, only save them"
    )
    parser.add_argument(
        "--show-table",
        action="store_true",
        help="Show improvement table with relative improvements"
    )
    parser.add_argument(
        "--table-metric-filter",
        nargs="+",
        default=None,
        help="Filter metrics for improvement table (e.g., 'all/contrastive_accuracy')"
    )
    parser.add_argument(
        "--generate-table-images",
        action="store_true",
        help="Generate visual table images for each dataset"
    )
    
    args = parser.parse_args()
    
    # Configure matplotlib
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    
    # Parse checkpoint paths
    checkpoint_paths = {}
    for checkpoint_str in args.checkpoints:
        name, path = checkpoint_str.split(":")
        checkpoint_paths[name] = path
    
    # Parse checkpoint labels if provided
    checkpoint_labels = {}
    if args.checkpoint_labels:
        for label_str in args.checkpoint_labels:
            name, label = label_str.split(":", 1)  # Use split with maxsplit=1 to handle colons in labels
            checkpoint_labels[name] = label
    else:
        checkpoint_labels = None
    
    # Initialize comparer with save options
    comparer = CheckpointComparer(
        checkpoint_paths, 
        args.metrics,
        output_dir=args.output_dir,
        save_format=args.format,
        dpi=args.dpi,
        checkpoint_labels=checkpoint_labels
    )
    comparer.load_checkpoints()
    
    # Generate plots based on selected types
    plot_types = args.plot_types
    if "all" in plot_types:
        plot_types = ["bars", "heatmap", "subset", "distribution", "scatter", "radar", "subset_improvements", "subset_heatmap", "absolute_differences", "abs_vs_rel"]
    
    for plot_type in plot_types:
        print(f"\nGenerating {plot_type} plot...")
        
        if plot_type == "bars":
            for metric in args.metrics:
                comparer.plot_checkpoint_comparison_bars(
                    dataset=args.dataset,
                    metric=metric,
                    subset=args.subset
                )
        
        elif plot_type == "heatmap":
            for metric in args.metrics:
                comparer.plot_relative_improvements_heatmap(metric=metric)
        
        elif plot_type == "subset" and args.dataset:
            for metric in args.metrics:
                comparer.plot_subset_comparison(args.dataset, metric=metric)
        
        elif plot_type == "distribution":
            for metric in args.metrics:
                comparer.plot_improvement_distribution(metric=metric)
        
        elif plot_type == "scatter":
            for metric in args.metrics:
                comparer.plot_scatter_comparison(metric=metric)
        
        elif plot_type == "radar" and args.dataset:
            comparer.plot_metric_comparison_radar(args.dataset)
        
        elif plot_type == "subset_improvements":
            for metric in args.metrics:
                comparer.plot_subset_improvements(dataset=args.dataset, metric=metric)
        
        elif plot_type == "subset_heatmap":
            for metric in args.metrics:
                comparer.plot_subset_comparison_heatmap(metric=metric)
        
        elif plot_type == "absolute_differences":
            for metric in args.metrics:
                comparer.plot_absolute_differences(dataset=args.dataset, metric=metric)
        
        elif plot_type == "abs_vs_rel":
            for metric in args.metrics:
                comparer.plot_absolute_vs_relative_comparison(dataset=args.dataset, metric=metric)
    
    # Show improvement table if requested
    if args.show_table:
        print("\nGenerating improvement table...")
        comparer.display_improvement_table(
            dataset=args.dataset,
            metric_filter=args.table_metric_filter,
            sort_by='relative_improvement'
        )
    
    # Generate table images if requested
    if args.generate_table_images:
        print("\nGenerating table images...")
        comparer.generate_dataset_table_images()
    
    # Generate summary report
    if args.save_report:
        comparer.generate_summary_report()


if __name__ == "__main__":
    main()
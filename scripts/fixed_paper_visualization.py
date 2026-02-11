#!/usr/bin/env python3
"""
Fixed Paper-Quality Visualization for Dataset Samples

Fixes all issues:
1. Correct caption extraction from BLA and SPEC_I2T
2. Better spacing between samples
3. Proper checkmark rendering (using text instead of unicode)
4. Fixed COCO_Order initialization
5. Shows ALL candidate captions with clear correct/wrong indicators
6. Per-dataset plots showing all subsets
7. Global overview plot with dataset/subset pairs
"""

import os
import sys
import random
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
import torch

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_loading import get_dataset_class


class FixedBenchmarkSampler:
    """Fixed sampler that correctly extracts captions from all datasets."""
    
    def __init__(self, data_root_base: str = "./datasets"):
        self.data_root_base = data_root_base
        
        # Fixed dataset configurations
        self.dataset_configs = {
            'Winoground': {'path': 'Winoground', 'subsets': ['all']},
            'VG_Attribution': {'path': 'VG_Attr', 'subsets': ['all']},
            'VG_Relation': {'path': 'VG_Rel', 'subsets': ['all']},
            'COCO_Order': {'path': 'COCO_Order', 'subsets': ['all']},
            'Flickr30k_Order': {'path': 'Flickr30k_Order', 'subsets': ['all']},
            'VALSE': {'path': 'VALSE', 'subsets': ['all']},
            'VL_CheckList': {'path': 'VL-CheckList', 'subsets': ['hake_action', 'swig_action', 'vg_action', 'vg_color', 'vaw_action', 'vaw_color']},
            'SugarCrepe': {'path': 'SugarCrepe', 'subsets': ['add_att', 'add_obj', 'replace_att', 'replace_obj', 'replace_rel', 'swap_att', 'swap_obj']},
            'ColorSwap': {'path': 'ColorSwap', 'subsets': ['all']},
            'ColorFoil': {'path': 'ColorFoil', 'subsets': ['all']},
            'COCO_Counterfactuals': {'path': 'COCO-Counterfactuals', 'subsets': ['all']},
            'ControlledImages': {'path': 'WhatsUp', 'subsets': ['A']},
            'VisMin': {'path': 'VisMin', 'subsets': ['object', 'attribute', 'counting']},
            'BLA': {'path': 'BLA_Benchmark', 'subsets': ['ap', 'co', 'rc']},
            'SPEC_I2T': {'path': 'SPEC', 'subsets': ['absolute_size', 'absolute_spatial', 'count', 'existence', 'relative_size', 'relative_spatial']},
            'CC3M': {'path': 'CC3M', 'subsets': ['all']},
            'NegBench': {'path': 'negbench', 'subsets': ['all']},
        }
    
    def load_dataset(self, dataset_name: str, subset_name: str):
        """Load dataset with correct parameters."""
        try:
            dataset_class = get_dataset_class(dataset_name)
            if dataset_class is None:
                return None
                
            config = self.dataset_configs.get(dataset_name)
            if not config:
                return None
                
            data_path = os.path.join(self.data_root_base, config['path'])
            
            # ARO datasets (COCO_Order, Flickr30k_Order, VG_Attribution, VG_Relation)
            if dataset_name in ['COCO_Order', 'Flickr30k_Order']:
                dataset = dataset_class(
                    image_preprocess=None,
                    download=False
                )
            elif dataset_name in ['VG_Attribution', 'VG_Relation']:
                dataset = dataset_class(
                    image_preprocess=None,
                    download=False
                )
            elif dataset_name == 'Winoground':
                dataset = dataset_class(
                    data_root=data_path,
                    image_preprocess=None,
                    use_auth_token=None
                )
            elif dataset_name == 'BLA':
                dataset = dataset_class(
                    data_root=data_path,
                    subset=subset_name,
                    image_preprocess=None
                )
            elif dataset_name == 'SPEC_I2T':
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=subset_name,
                    image_preprocess=None
                )
            elif dataset_name == 'SugarCrepe':
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=subset_name,
                    coco_root=data_path,
                    image_preprocess=None
                )
            elif dataset_name == 'ControlledImages':
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=subset_name,
                    image_preprocess=None
                )
            elif dataset_name == 'VL_CheckList':
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=subset_name,
                    image_preprocess=None
                )
            elif dataset_name == 'VisMin':
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=subset_name,
                    image_preprocess=None
                )
            elif dataset_name == 'VALSE':
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=subset_name,
                    image_preprocess=None
                )
            elif dataset_name == 'NegBench':
                dataset = dataset_class(
                    data_path=data_path,
                    subset_name=subset_name,
                    image_preprocess=None
                )
            else:
                # Generic loading
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=subset_name,
                    image_preprocess=None
                )
            
            return dataset
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}/{subset_name}: {e}")
            return None
    
    def extract_sample_data(self, dataset, dataset_name: str, idx: int, subset_name: str) -> Dict:
        """Extract sample data with ALL candidate captions correctly."""
        try:
            sample = dataset[idx]
            
            # Initialize return structure
            result = {
                'image': None,
                'positive_caption': '',
                'negative_captions': [],
                'all_captions': [],  # All candidates with labels
                'subset': subset_name,
                'dataset': dataset_name,
                'idx': idx
            }
            
            # Extract image - handle different formats
            if 'image' in sample:
                result['image'] = sample['image']
            elif 'image_options' in sample:
                img = sample['image_options']
                # image_options might be a single image or a list
                if isinstance(img, list):
                    result['image'] = img[0] if img else None
                else:
                    result['image'] = img
            else:
                return None
            
            # BLA format: {'caption': str, 'foil': str, ...}
            if dataset_name == 'BLA':
                if 'caption' in sample and 'foil' in sample:
                    result['positive_caption'] = sample['caption']
                    result['negative_captions'] = [sample['foil']]
                    result['all_captions'] = [
                        {'text': sample['caption'], 'correct': True},
                        {'text': sample['foil'], 'correct': False}
                    ]
                else:
                    print(f"⚠️  BLA sample {idx} missing caption/foil: {sample.keys()}")
                    return None
            
            # SPEC_I2T format: {'caption_options': [...], 'label': int}
            elif dataset_name == 'SPEC_I2T':
                if 'caption_options' in sample:
                    captions = sample['caption_options']
                    label = sample.get('label', 0)
                    
                    result['positive_caption'] = captions[label]
                    result['negative_captions'] = [c for i, c in enumerate(captions) if i != label]
                    result['all_captions'] = [
                        {'text': c, 'correct': (i == label)}
                        for i, c in enumerate(captions)
                    ]
                else:
                    print(f"⚠️  SPEC_I2T sample {idx} missing caption_options: {sample.keys()}")
                    return None
            
            # Standard format with caption_options
            elif 'caption_options' in sample:
                captions = sample['caption_options']
                label = sample.get('label', 0)
                
                if isinstance(captions, list) and len(captions) > 0:
                    result['positive_caption'] = captions[label] if label < len(captions) else captions[0]
                    result['negative_captions'] = [c for i, c in enumerate(captions) if i != label]
                    result['all_captions'] = [
                        {'text': c, 'correct': (i == label)}
                        for i, c in enumerate(captions)
                    ]
                else:
                    return None
            
            # Fallback: single positive/negative pair
            elif 'caption' in sample:
                pos_cap = sample['caption']
                neg_caps = []
                
                if 'negative_caption' in sample:
                    neg_caps = [sample['negative_caption']]
                elif 'negative_captions' in sample:
                    neg_caps = sample['negative_captions']
                
                result['positive_caption'] = pos_cap
                result['negative_captions'] = neg_caps
                result['all_captions'] = [{'text': pos_cap, 'correct': True}]
                result['all_captions'].extend([{'text': nc, 'correct': False} for nc in neg_caps])
            
            else:
                print(f"⚠️  Unknown format for {dataset_name} sample {idx}: {sample.keys()}")
                return None
            
            return result
            
        except Exception as e:
            print(f"⚠️  Error extracting sample {idx} from {dataset_name}: {e}")
            return None
    
    def sample_from_datasets(self, config: Dict[str, int], random_seed: int = 42) -> Dict[str, List[Dict]]:
        """Sample from all datasets, organized by dataset/subset."""
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        all_samples = {}
        
        for dataset_name in config.keys():
            if dataset_name not in self.dataset_configs:
                continue
            
            dataset_config = self.dataset_configs[dataset_name]
            subsets = dataset_config['subsets']
            
            # Sample from each subset
            for subset_name in subsets:
                key = f"{dataset_name}/{subset_name}" if subset_name != 'all' else dataset_name
                
                print(f"📊 Sampling from {key}...")
                
                dataset = self.load_dataset(dataset_name, subset_name)
                if dataset is None:
                    all_samples[key] = []
                    continue
                
                # Sample 1 representative sample per subset
                n_samples = 1
                if len(dataset) == 0:
                    all_samples[key] = []
                    continue
                
                if len(dataset) < n_samples:
                    indices = list(range(len(dataset)))
                else:
                    indices = random.sample(range(len(dataset)), n_samples)
                
                samples = []
                for idx in indices:
                    sample_data = self.extract_sample_data(dataset, dataset_name, idx, subset_name)
                    if sample_data is not None:
                        samples.append(sample_data)
                
                all_samples[key] = samples
                print(f"   ✅ Got {len(samples)} samples")
        
        return all_samples


class PaperVisualizer:
    """Creates publication-quality visualizations with proper spacing and rendering."""
    
    def __init__(self, output_dir: str = "./paper_figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._setup_style()
    
    def _setup_style(self):
        """Setup clean publication style."""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 9,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'figure.titlesize': 12,
            'text.usetex': False,
        })
        
        self.colors = {
            'correct': '#2E7D32',    # Darker green (easier to read)
            'wrong': '#C62828',      # Darker red (easier to read)
            'border': '#E0E0E0',     # Light Gray
            'background': '#FFFFFF'
        }
        
        # Category colors - muted professional palette
        self.category_colors = {
            'Attribute Binding': '#5C6BC0',      # Indigo
            'Spatial Relations': '#26A69A',      # Teal
            'Quantitative': '#FFA726',           # Orange
            'Object & Role': '#AB47BC',          # Purple
            'Linguistic': '#66BB6A',             # Green
            'Compositional': '#EC407A',          # Pink
            'Attribute Recognition': '#42A5F5',  # Blue
        }
    
    def _convert_image(self, image):
        """Convert image to PIL format."""
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = torch.clamp(image, 0, 1)
            image = (image.cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(image)
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255 if image.max() <= 1.0 else image).astype(np.uint8)
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            return Image.fromarray(image)
        return None
    
    def create_dataset_plot(self, dataset_name: str, subset_samples: Dict[str, List[Dict]]):
        """Create one plot per dataset showing all its subsets."""
        # Filter samples for this dataset
        relevant_samples = {k: v for k, v in subset_samples.items() 
                          if k.startswith(dataset_name)}
        
        if not relevant_samples:
            return
        
        # Flatten all samples
        all_samples = []
        for key, samples in relevant_samples.items():
            subset = key.split('/')[-1] if '/' in key else 'all'
            for sample in samples:
                sample['display_subset'] = subset
                all_samples.append(sample)
        
        if not all_samples:
            return
        
        n_samples = len(all_samples)
        n_cols = min(3, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        # Increased spacing: 4 inches per sample height
        fig_width = 15
        fig_height = 5 * n_rows  # More vertical space
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        if n_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Lighter title
        fig.suptitle(f'{dataset_name} Dataset Samples', 
                    fontsize=13, fontweight='normal', y=0.98)
        
        for idx, sample in enumerate(all_samples):
            ax = axes[idx]
            
            # Display image
            try:
                img = self._convert_image(sample['image'])
                if img:
                    ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Image Error", ha='center', va='center')
            
            ax.axis('off')
            
            # Subset label
            subset_label = sample.get('display_subset', 'all')
            if subset_label != 'all':
                ax.set_title(f"Subset: {subset_label}", fontsize=10, pad=10)
            
            # Format captions with clear correct/wrong indicators
            caption_lines = []
            for cap_info in sample.get('all_captions', []):
                text = cap_info['text']
                if len(text) > 70:
                    text = text[:67] + '...'
                
                # Use simple text indicators instead of unicode
                if cap_info['correct']:
                    indicator = "[CORRECT]"
                    color = self.colors['correct']
                else:
                    indicator = "[WRONG]"
                    color = self.colors['wrong']
                
                caption_lines.append((f"{indicator} {text}", color))
            
            # Create caption text box with better spacing
            y_pos = -0.15
            for line_text, line_color in caption_lines:
                ax.text(0.5, y_pos, line_text,
                       transform=ax.transAxes, ha='center', va='top',
                       fontsize=8, color=line_color,
                       bbox=dict(boxstyle="round,pad=0.5", 
                               facecolor='white', 
                               edgecolor=line_color,
                               linewidth=1.5, alpha=0.9))
                y_pos -= 0.12  # More spacing between captions
        
        # Hide unused axes
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')
        
        # Better spacing
        plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=4.0, w_pad=2.0)
        
        # Save
        safe_name = dataset_name.lower().replace('_', '-')
        base_path = os.path.join(self.output_dir, f"{safe_name}_samples")
        
        plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        print(f"✅ Saved {dataset_name} plot: {base_path}.{{png,pdf}}")
        plt.close()
    
    def create_global_overview(self, all_samples: Dict[str, List[Dict]]):
        """Create modern category-based overview with better layout."""
        
        # Dataset to category mapping
        dataset_categories = {
            'Attribute Binding': ['VG_Attribution', 'ColorSwap', 'SugarCrepe', 'VisMin'],
            'Spatial Relations': ['VG_Relation', 'SPEC_I2T', 'ControlledImages', 'VALSE'],
            'Quantitative': ['SPEC_I2T', 'VisMin', 'VALSE'],
            'Object & Role': ['VisMin', 'SugarCrepe', 'VL_CheckList', 'BLA', 'COCO_Counterfactuals'],
            'Linguistic': ['COCO_Order', 'Flickr30k_Order', 'VALSE', 'BLA'],
            'Compositional': ['Winoground', 'SugarCrepe'],
            'Attribute Recognition': ['ColorFoil', 'VL_CheckList'],
        }
        
        # Organize samples by category
        categorized_samples = {}
        for category, datasets in dataset_categories.items():
            category_items = []
            for key, samples in all_samples.items():
                if not samples:
                    continue
                dataset_name = key.split('/')[0]
                if dataset_name in datasets:
                    category_items.append((key, samples[0]))
            
            if category_items:
                categorized_samples[category] = category_items
        
        # Create category-based plot: each row is a category, columns are samples
        n_categories = len(categorized_samples)
        max_samples_per_category = max(len(items) for items in categorized_samples.values())
        
        # Figure dimensions
        fig_width = min(18, 3 * max_samples_per_category)
        fig_height = 3.5 * n_categories
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.suptitle('Evaluation Datasets Organized by Capability', 
                    fontsize=14, fontweight='normal', y=0.995)
        
        # Create grid: rows = categories, cols = max samples
        gs = GridSpec(n_categories, max_samples_per_category, figure=fig,
                     hspace=0.4, wspace=0.15,
                     top=0.96, bottom=0.03, left=0.08, right=0.98)
        
        for cat_idx, (category, items) in enumerate(categorized_samples.items()):
            cat_color = self.category_colors.get(category, '#757575')
            
            # Add category label on the left
            if max_samples_per_category > 0:
                ax_label = fig.add_subplot(gs[cat_idx, 0])
                ax_label.text(-0.1, 0.5, category, 
                            transform=ax_label.transAxes,
                            rotation=0, fontsize=11, fontweight='bold',
                            color=cat_color, ha='right', va='center')
                ax_label.axis('off')
            
            # Plot samples for this category
            for sample_idx, (key, sample) in enumerate(items):
                ax = fig.add_subplot(gs[cat_idx, sample_idx])
                
                # Display image
                try:
                    img = self._convert_image(sample['image'])
                    if img:
                        ax.imshow(img)
                except Exception:
                    pass
                
                ax.axis('off')
                
                # Add colored border to indicate category
                for spine in ['top', 'bottom', 'left', 'right']:
                    ax.spines[spine].set_color(cat_color)
                    ax.spines[spine].set_linewidth(3)
                    ax.spines[spine].set_visible(True)
                
                # Dataset/subset name as title
                title_text = key.replace('/', '\n')
                ax.set_title(title_text, fontsize=8, pad=5, color=cat_color)
                
                # Show one caption pair compactly
                if sample.get('all_captions'):
                    correct_cap = next((c['text'] for c in sample['all_captions'] if c['correct']), '')
                    wrong_cap = next((c['text'] for c in sample['all_captions'] if not c['correct']), '')
                    
                    # Truncate for overview
                    if len(correct_cap) > 45:
                        correct_cap = correct_cap[:42] + '...'
                    if len(wrong_cap) > 45:
                        wrong_cap = wrong_cap[:42] + '...'
                    
                    ax.text(0.5, -0.08, f"✓ {correct_cap}",
                           transform=ax.transAxes, ha='center', va='top',
                           fontsize=6.5, color=self.colors['correct'],
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='white', 
                                   edgecolor=self.colors['correct'],
                                   linewidth=0.8, alpha=0.9))
                    ax.text(0.5, -0.18, f"✗ {wrong_cap}",
                           transform=ax.transAxes, ha='center', va='top',
                           fontsize=6.5, color=self.colors['wrong'],
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='white', 
                                   edgecolor=self.colors['wrong'],
                                   linewidth=0.8, alpha=0.9))
            
            # Hide empty slots in this category row
            for empty_idx in range(len(items), max_samples_per_category):
                ax_empty = fig.add_subplot(gs[cat_idx, empty_idx])
                ax_empty.axis('off')
        
        # Save
        overview_path = os.path.join(self.output_dir, "00_category_overview")
        plt.savefig(f"{overview_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f"{overview_path}.pdf", format='pdf', bbox_inches='tight', facecolor='white')
        
        print(f"✅ Saved category-based overview: {overview_path}.{{png,pdf}}")
        plt.close()
    
    def create_all_figures(self, all_samples: Dict[str, List[Dict]]):
        """Generate all figures."""
        print("\n📊 Creating visualizations...")
        print(f"📁 Output directory: {self.output_dir}\n")
        
        # Get unique datasets
        datasets = set(key.split('/')[0] for key in all_samples.keys())
        
        # Create per-dataset plots
        for dataset_name in sorted(datasets):
            self.create_dataset_plot(dataset_name, all_samples)
        
        # Create global overview
        self.create_global_overview(all_samples)
        
        print(f"\n✅ All figures created in: {self.output_dir}/")


def main():
    """Main execution."""
    # Initialize sampler
    sampler = FixedBenchmarkSampler()
    
    # Sample configuration - one sample per subset
    sample_config = {
        'SugarCrepe': 1,
        'SPEC_I2T': 1,
        'BLA': 1,
        'VL_CheckList': 1,
        'VisMin': 1,
        'VALSE': 1,
        'Winoground': 1,
        'VG_Attribution': 1,
        'VG_Relation': 1,
        'COCO_Order': 1,
        'Flickr30k_Order': 1,
        'ColorSwap': 1,
        'ColorFoil': 1,
        'COCO_Counterfactuals': 1,
        'ControlledImages': 1,
        'CC3M': 1,
        'NegBench': 1,
    }
    
    print("🚀 Generating fixed paper visualizations...")
    
    # Sample from datasets
    all_samples = sampler.sample_from_datasets(sample_config, random_seed=42)
    
    # Create visualizations
    visualizer = PaperVisualizer(output_dir="./paper_figures_fixed")
    visualizer.create_all_figures(all_samples)


if __name__ == "__main__":
    main()

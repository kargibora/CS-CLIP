#!/usr/bin/env python3
"""
Enhanced Paper-Quality Visualization for Dataset Samples

Creates publication-ready visualizations organized by capability categories
for appendix figures in research papers.

Key improvements over basic plots:
1. Organized by capability categories (not just datasets)
2. Cleaner layout with better typography
3. Highlights what each dataset tests (capability annotations)
4. Color-coded by capability type
5. Better caption formatting with clear positive/negative distinction
6. Export in multiple formats (PNG, PDF, SVG for papers)
7. Shows original caption vs foil caption for each sample
8. Includes all datasets with proper subset handling
"""

import os
import sys
import textwrap
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

from generate_sampling_plot import BenchmarkSampler

# Updated capability taxonomy with proper dataset-subset mapping
# Note: ARO is a display name - actual datasets are VG_Attribution, VG_Relation, COCO_Order, Flickr30k_Order
CAPABILITY_CATEGORIES = {
    'Attribute Recognition': {
        'description': 'Recognizing single attributes (color, material, etc.) without multi-object binding',
        'color': '#E74C3C',  # Red
        'icon': '🎨',
        'datasets': {
            'ColorFoil': ['all'],
            'SugarCrepe': ['replace_att', 'add_att'],
            'SugarCrepe_PP': ['replace_att'],
            'VL_CheckList': ['vg_color', 'vaw_color'],
        }
    },
    'Attribute Binding': {
        'description': 'Multi-object attribute binding (which attribute belongs to which object)',
        'color': '#3498DB',  # Blue
        'icon': '🔗',
        'datasets': {
            'VG_Attribution': ['all'],  # ARO subset - actual dataset name
            'ColorSwap': ['all'],
            'SugarCrepe': ['swap_att'],
            'SugarCrepe_PP': ['swap_att'],
            'VisMin': ['attribute'],
            'Winoground': ['all']
        }
    },
    'Relations': {
        'description': 'Spatial & Relational Understanding',
        'color': '#2ECC71',  # Green
        'icon': '📐',
        'datasets': {
            'VG_Relation': ['all'],  # ARO subset - actual dataset name
            'SPEC_I2T': ['relative_spatial', 'absolute_position', 'absolute_spatial'],  # Added absolute_spatial
            'VisMin': ['relation'],
            'SugarCrepe': ['replace_rel', 'swap_rel'],
            'SugarCrepe_PP': ['replace_rel', 'swap_rel'],
            'VL_CheckList': ['vg_spatial'],
            'VALSE': ['relations'],
            'ControlledImages': ['A', 'B', 'VG-One', 'VG-Two', 'COCO-One', 'COCO-Two'],
            'COLA': ['multi_objects']
        }
    },
    'Quantitative': {
        'description': 'Counting, Plurality & Size',
        'color': '#F39C12',  # Orange
        'icon': '🔢',
        'datasets': {
            'SPEC_I2T': ['count', 'absolute_size', 'relative_size'],
            'VisMin': ['counting'],
            'VALSE': ['counting', 'plurals'],
        }
    },
    'Existence & Negation': {
        'description': 'Object Existence & Logical Negation',
        'color': '#E91E63',  # Pink
        'icon': '🚫',
        'datasets': {
            'SPEC_I2T': ['existence'],
            'VALSE': ['existence'],
            'NegBench': ['msr_vtt_mcq_rephrased_llama', 'COCO_val_mcq_llama3.1_rephrased'],
        }
    },
    'Object & Role': {
        'description': 'Object Recognition & Role Assignment',
        'color': '#9B59B6',  # Purple
        'icon': '👤',
        'datasets': {
            'VisMin': ['object'],
            'SugarCrepe': ['replace_obj', 'swap_obj', 'add_obj'],
            'SugarCrepe_PP': ['replace_obj', 'swap_obj', 'add_obj'],
            'VL_CheckList': ['hake_action', 'swig_action', 'vg_action', 'vaw_action'],
            'VALSE': ['actions', 'coreference', 'noun phrases'],
            'COCO_Counterfactuals': ['all']  # Fixed name
        }
    },
    'Linguistic': {
        'description': 'Word Order, Paraphrase & Syntax',
        'color': '#1ABC9C',  # Teal
        'icon': '📝',
        'datasets': {
            'ColorSwap': ['all'],
            'BLA': ['ap', 'co', 'rc'],
            'COCO_Order': ['all'],  # ARO subset - actual dataset name
            'Flickr30k_Order': ['all'],  # ARO subset - actual dataset name
        }
    }
}


class PaperQualityVisualizer:
    """Creates publication-quality visualizations organized by capability."""
    
    def __init__(self, output_dir: str = "./paper_figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Publication-ready style
        self._setup_publication_style()
        
    def _setup_publication_style(self):
        """Configure publication-quality plotting style."""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'text.usetex': False,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.5,
            'lines.linewidth': 2.0,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
        self.colors = {
            'positive': '#27AE60',
            'negative': '#E74C3C',
            'neutral': '#95A5A6',
            'background': '#F8F9FA',
            'text': '#2C3E50',
            'border': '#BDC3C7',
            'highlight': '#3498DB'
        }
        
    def wrap_text(self, text: str, width: int = 50) -> str:
        """Wrap text to specified width."""
        return '\n'.join(textwrap.wrap(text, width=width, break_long_words=False))
    
    def _convert_image(self, image) -> Image.Image:
        """Convert various image formats to PIL Image."""
        # Handle list of images (take first one)
        if isinstance(image, list):
            if len(image) > 0:
                image = image[0]
            else:
                raise ValueError("Empty image list")
        
        if isinstance(image, Image.Image):
            return image
            
        if isinstance(image, torch.Tensor):
            image = torch.clamp(image, 0, 1)
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = (image.cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(image)
            
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            return Image.fromarray(image)
        
        # Handle dict-like objects
        if hasattr(image, 'get') or isinstance(image, dict):
            if 'image' in image:
                return self._convert_image(image['image'])
            
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _get_capability_for_dataset(self, dataset_name: str, subset: str = None) -> Tuple[str, Dict]:
        """Get the capability category for a dataset/subset combination."""
        for capability, info in CAPABILITY_CATEGORIES.items():
            if dataset_name in info['datasets']:
                # Check if subset matches or if 'all' is specified
                subsets = info['datasets'][dataset_name]
                # Match if: 1) 'all' in category config, 2) subset is None, 3) subset in list, 
                # 4) subset is 'all' (dataset returns all subsets combined)
                if 'all' in subsets or subset is None or subset in subsets or subset == 'all':
                    return capability, info
        # No "Other" category - return None to skip unmatched datasets
        return None, None
    
    def create_capability_category_plot(self, capability: str, samples_by_dataset: Dict[str, List[Dict]]) -> None:
        """Create a beautiful plot for one capability category showing multiple dataset examples."""
        
        if not samples_by_dataset:
            return
            
        cap_info = CAPABILITY_CATEGORIES[capability]
        cap_color = cap_info['color']
        cap_icon = cap_info['icon']
        
        # Collect all samples across datasets for this capability
        all_samples = []
        for dataset_name, samples in samples_by_dataset.items():
            for sample in samples:
                sample['dataset_name'] = dataset_name
                all_samples.append(sample)
        
        if not all_samples:
            return
        
        # Limit to 6 samples for clean visualization
        n_samples = min(6, len(all_samples))
        selected_samples = all_samples[:n_samples]
        
        # Create figure with better layout
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        
        # Main title with icon (no description subtitle - images should be self-explanatory)
        fig.suptitle(f'{cap_icon} {capability}', 
                    fontsize=22, fontweight='bold', 
                    color=cap_color, y=0.96)  # Adjusted y from 0.98
        
        # Create grid: 2 rows x 3 columns
        n_cols = 3
        n_rows = 2
        
        # Increased vertical spacing to prevent caption overlap
        gs = GridSpec(n_rows, n_cols, figure=fig,
                     hspace=0.65, wspace=0.25,  # Increased hspace from 0.35 to 0.65
                     top=0.92, bottom=0.06, left=0.04, right=0.96)  # Adjusted top from 0.88 to 0.92
        
        for i, sample in enumerate(selected_samples):
            row = i // n_cols
            col = i % n_cols
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Display image with border
            try:
                img = self._convert_image(sample['image'])
                ax.imshow(img)
                
                # Add colored border
                for spine in ax.spines.values():
                    spine.set_edgecolor(cap_color)
                    spine.set_linewidth(3)
                    
            except Exception as e:
                ax.text(0.5, 0.5, f"Image Error\n{str(e)[:40]}", 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=9, color=self.colors['neutral'])
            
            ax.axis('off')
            
            # Add dataset label at top
            dataset_label = sample.get('dataset_name', 'Unknown')
            subset = sample.get('subset', '')
            if subset and subset not in ['all', 'main']:
                dataset_label = f"{dataset_label}\n({subset})"
            
            ax.text(0.5, 1.08, dataset_label,
                   transform=ax.transAxes, ha='center', va='bottom',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.4", 
                           facecolor=cap_color, 
                           edgecolor='none',
                           alpha=0.2))
            
            # Format captions - show original vs foil clearly
            # Handle different dataset formats
            if 'positive_caption' in sample:
                pos_cap = sample['positive_caption']
                neg_caps = sample.get('negative_captions', [])
            elif 'caption' in sample:
                # BLA format: caption (correct) and foil (incorrect)
                pos_cap = sample['caption']
                foil = sample.get('foil', '')
                neg_caps = [foil] if foil else []
            else:
                pos_cap = "No correct caption found"
                neg_caps = []
            
            # Wrap captions
            pos_wrapped = self.wrap_text(pos_cap, width=45)
            
            # Create caption box below image - adjusted positioning to avoid overlap
            caption_y = -0.05  # Moved further down from -0.02
            
            # Positive caption (with checkmark)
            ax.text(0.02, caption_y, '✓ Original:',
                   transform=ax.transAxes, ha='left', va='top',
                   fontsize=10, fontweight='bold',
                   color=self.colors['positive'])
            
            ax.text(0.02, caption_y - 0.05, pos_wrapped,  # Reduced gap from 0.06 to 0.05
                   transform=ax.transAxes, ha='left', va='top',
                   fontsize=9, color='#2C3E50',
                   bbox=dict(boxstyle="round,pad=0.5", 
                           facecolor='#D5F4E6',
                           edgecolor=self.colors['positive'],
                           linewidth=1.5, alpha=0.9))
            
            # Negative captions (with X mark)
            if neg_caps:
                neg_cap = neg_caps[0]  # Show first negative
                neg_wrapped = self.wrap_text(neg_cap, width=45)
                
                neg_y = caption_y - 0.18  # Adjusted from -0.20
                
                ax.text(0.02, neg_y, '✗ Foil:',
                       transform=ax.transAxes, ha='left', va='top',
                       fontsize=10, fontweight='bold',
                       color=self.colors['negative'])
                
                ax.text(0.02, neg_y - 0.05, neg_wrapped,  # Reduced gap from 0.06 to 0.05
                       transform=ax.transAxes, ha='left', va='top',
                       fontsize=9, color='#2C3E50',
                       bbox=dict(boxstyle="round,pad=0.5", 
                               facecolor='#FADBD8',
                               edgecolor=self.colors['negative'],
                               linewidth=1.5, alpha=0.9))
        
        # Save figure
        safe_name = capability.lower().replace(' ', '_').replace('&', 'and')
        base_path = os.path.join(self.output_dir, f"capability_{safe_name}")
        
        plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        print(f"✅ Saved {capability}: {base_path}.{{png,pdf}}")
        plt.close()
    
    def create_overview_summary(self, samples_by_capability: Dict[str, Dict[str, List[Dict]]]) -> None:
        """Create a beautiful overview showing all capabilities."""
        
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        
        fig.suptitle('Compositional Vision-Language Understanding: Capability Categories',
                    fontsize=24, fontweight='bold', y=0.98)
        
        fig.text(0.5, 0.95, 
                'Evaluating multi-object attribute binding, spatial relations, and compositional reasoning',
                ha='center', fontsize=14, style='italic', color='#34495E', alpha=0.8)
        
        n_caps = len(samples_by_capability)
        n_cols = 3
        n_rows = (n_caps + n_cols - 1) // n_cols
        
        gs = GridSpec(n_rows, n_cols, figure=fig,
                     hspace=0.35, wspace=0.25,
                     top=0.90, bottom=0.05, left=0.04, right=0.96)
        
        for idx, (capability, datasets_dict) in enumerate(sorted(samples_by_capability.items())):
            if not datasets_dict:
                continue
                
            row = idx // n_cols
            col = idx % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
            
            cap_info = CAPABILITY_CATEGORIES[capability]
            cap_color = cap_info['color']
            cap_icon = cap_info['icon']
            
            # Title with icon
            ax.text(0.5, 0.95, f"{cap_icon} {capability}",
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=16, fontweight='bold', color=cap_color)
            
            # Description
            ax.text(0.5, 0.85, cap_info['description'],
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=11, style='italic', color='#34495E',
                   wrap=True)
            
            # List datasets
            dataset_names = list(datasets_dict.keys())
            total_samples = sum(len(samples) for samples in datasets_dict.values())
            
            datasets_text = ', '.join(dataset_names[:4])
            if len(dataset_names) > 4:
                datasets_text += f'\n+ {len(dataset_names)-4} more'
            
            ax.text(0.5, 0.65, f"Datasets ({len(dataset_names)}):",
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=10, fontweight='bold', color='#2C3E50')
            
            ax.text(0.5, 0.55, datasets_text,
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=9, color='#7F8C8D')
            
            # Sample count
            ax.text(0.5, 0.25, f"{total_samples} samples",
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5",
                           facecolor=cap_color, alpha=0.2,
                           edgecolor=cap_color, linewidth=2))
            
            # Add colored background box
            box = mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                         transform=ax.transAxes,
                                         boxstyle="round,pad=0.02",
                                         facecolor=cap_color, alpha=0.05,
                                         edgecolor=cap_color, linewidth=2)
            ax.add_patch(box)
        
        # Save
        overview_path = os.path.join(self.output_dir, "00_overview_summary")
        plt.savefig(f"{overview_path}.png", dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.savefig(f"{overview_path}.pdf", format='pdf', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✅ Saved overview summary: {overview_path}.{{png,pdf}}")
        plt.close()
    
    def create_all_paper_figures(self, all_samples: Dict[str, List[Dict]]) -> None:
        """Generate all paper-quality figures organized by capability."""
        print("\n📊 Creating enhanced paper-quality visualizations...")
        print(f"📁 Output directory: {self.output_dir}\n")
        
        # Organize samples by capability
        samples_by_capability = {}
        
        for capability in CAPABILITY_CATEGORIES.keys():
            samples_by_capability[capability] = {}
        
        # Group samples by capability and dataset
        for dataset_name, samples in all_samples.items():
            if not samples:
                continue
                
            for sample in samples:
                subset = sample.get('subset', None)
                capability, _ = self._get_capability_for_dataset(dataset_name, subset)
                
                # Skip samples from datasets not in any capability category
                if capability is None:
                    print(f"⚠️  Warning: Dataset '{dataset_name}' (subset: {subset}) not found in any capability category - skipping")
                    continue
                
                if capability not in samples_by_capability:
                    samples_by_capability[capability] = {}
                
                if dataset_name not in samples_by_capability[capability]:
                    samples_by_capability[capability][dataset_name] = []
                
                samples_by_capability[capability][dataset_name].append(sample)
        
        # Create overview
        self.create_overview_summary(samples_by_capability)
        
        # Create one figure per capability
        for capability, datasets_dict in samples_by_capability.items():
            if datasets_dict:
                self.create_capability_category_plot(capability, datasets_dict)
        
        print(f"\n✅ All figures created in: {self.output_dir}/")
        print("📝 Files include both PNG (high-res) and PDF (vector) formats")
        print(f"📊 Generated {len(samples_by_capability)} capability-based visualizations")


def main():
    """Main execution function."""
    # Initialize sampler
    sampler = BenchmarkSampler()
    
    # Enhanced sample configuration with correct dataset names
    # Note: Use actual dataset class names (VG_Attribution, VG_Relation, etc.)
    # not display names (ARO)
    sample_config = {
        # Attribute Recognition
        'ColorFoil': 3,
        'SugarCrepe': 8,  # Will sample from multiple subsets
        'VL_CheckList': 6,
        
        # Attribute Binding
        'VG_Attribution': 4,  # ARO subset (actual dataset name)
        'ColorSwap': 3,
        'VisMin': 6,  # Multiple subsets
        'Winoground': 3,
        
        # Relations
        'VG_Relation': 4,  # ARO subset (actual dataset name)
        'SPEC_I2T': 8,  # Multiple spatial/relation subsets
        'VALSE': 10,  # Multiple subsets
        'ControlledImages': 6,  # VG-One, VG-Two, COCO-One, COCO-Two, A, B
        'COLA': 3,
        
        # Quantitative (covered by SPEC_I2T, VisMin, VALSE above)
        
        # Existence & Negation
        'NegBench': 4,  # msr_vtt, COCO subsets
        
        # Object & Role
        'COCO_Counterfactuals': 3,  # Correct name
        
        # Linguistic
        'BLA': 4,  # ap, co, rc subsets
        'COCO_Order': 3,  # ARO subset (actual dataset name)
        'Flickr30k_Order': 3,  # ARO subset (actual dataset name)
        
        # Additional datasets
        'CC3M': 2,
    }
    
    print("🚀 Generating enhanced paper-quality dataset visualizations...")
    print("📊 Including all capability categories with proper subset handling")
    print("ℹ️  Note: ARO subsets mapped to VG_Attribution, VG_Relation, COCO_Order, Flickr30k_Order\n")
    
    # Sample from datasets
    all_samples = sampler.sample_from_datasets(sample_config, random_seed=42)
    
    # Print sampling summary
    print("\n📋 Sampling Summary:")
    print("="*60)
    for dataset_name, samples in sorted(all_samples.items()):
        if samples:
            subsets = set(s.get('subset', 'main') for s in samples)
            subset_str = f" (subsets: {', '.join(sorted(subsets))})" if len(subsets) > 1 or 'main' not in subsets else ""
            print(f"  {dataset_name}: {len(samples)} samples{subset_str}")
    print("="*60)
    
    # Create visualizations
    visualizer = PaperQualityVisualizer(output_dir="./paper_figures_enhanced")
    visualizer.create_all_paper_figures(all_samples)
    
    print("\n🎨 Visualization Features:")
    print("  ✓ Organized by capability categories")
    print("  ✓ Color-coded by capability type")
    print("  ✓ Clear original vs foil caption distinction")
    print("  ✓ Dataset and subset labels")
    print("  ✓ High-resolution PNG + vector PDF formats")
    print("\n🎉 Done! Check ./paper_figures_enhanced/ for results")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive Dataset Subset Analysis and Sampling Script

This script automatically discovers all available subsets for each dataset and 
creates optimized sampling configurations to get representative samples from 
every subset acro        # Multi-subset datasets        # Summary statistics
        total_datasets = len(self.dataset_subsets)
        total_samples = sum(len(info['su        print(f"\n🎉 Comprehensive visualization complete!")
        print(f"📁 Check output directory: {output_dir}")
        print(f"🎯 Total samples visualized: {total_samples}")
        
        return all_samples']) * info['samples_per_subset'] 
                          for info in self.dataset_subsets.values())
        
        print("\n" + "=" * 50)
        print("📋 SUMMARY STATISTICS:")
        print("=" * 50)
        print(f"📊 Total datasets: {total_datasets}")
        print(f"🧩 Multi-subset datasets: {len(multi_subset)}")
        print(f"🎯 Single-subset datasets: {len(single_subset)}")
        print(f"📈 Total subsets: {sum(len(info['subsets']) for info in self.dataset_subsets.values())}")
        print(f"🎲 Total recommended samples: {total_samples}")
        
        print("\n🎯 SAMPLING STRATEGY:")
        print("  • Multi-subset datasets: 1 sample per subset for diversity")
        print("  • Single datasets: 2-4 samples for statistical reliability")
        print(f"  Total visualization: {total_samples} samples across {total_datasets} benchmarks")🔍 MULTI-SUBSET DATASETS:")
        print("-" * 50)
        total_subsets = 0
        for dataset, info in multi_subset.items():
            num_subsets = len(info['subsets'])
            print(f"📁 {dataset}")
            print(f"   └── {num_subsets} subsets: {', '.join(info['subsets'])}")
            print(f"   └── {info['description']}")
            print(f"   └── Recommended: {info['samples_per_subset']} sample per subset")
            total_subsets += num_subsetschmarks.
"""

import os
import sys
from typing import Dict

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from generate_sampling_plot import BenchmarkSampler, ModernBenchmarkVisualizer

class DatasetSubsetAnalyzer:
    """Analyzes and maps all available subsets for each dataset."""
    
    def __init__(self):
        # Comprehensive mapping of datasets and their subsets (updated for working datasets)
        self.dataset_subsets = {
            # Datasets confirmed to work
            'VALSE': {
                'type': 'internal_detection',
                'subsets': [
                    'existence', 'plurality', 'counting', 'spatial', 'action_replace',
                    'obj_replace', 'att_replace', 'relation', 'foil_it', 'foil_obj',
                    'foil_att', 'foil_action'
                ],
                'total_subsets': 12,
                'samples_per_subset': 1,
                'description': 'Linguistic phenomena detection across 12 categories'
            },
            
            'SugarCrepe': {
                'type': 'file_based_subsets',
                'subsets': [
                    'add_att', 'add_obj', 'replace_att', 'replace_obj', 
                    'replace_rel', 'swap_att', 'swap_obj'
                ],
                'total_subsets': 7,
                'samples_per_subset': 1,
                'description': 'Compositional reasoning across 7 hard negative strategies'
            },
            
            'Winoground': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 4,
                'description': 'Visio-linguistic compositional reasoning'
            },
            
            # Datasets with extraction issues - reduced sample count for testing
            'VL_CheckList': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 3,
                'description': 'Compositional reasoning across attributes and relations'
            },
            
            'ColorFoil': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Color-based foil detection'
            },
            
            'COCO_Counterfactuals': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Counterfactual reasoning with COCO'
            },
            
            # Add other working datasets as available
            'VG_Attribution': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Visual attribute recognition'
            },
            
            'VG_Relation': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Visual relation understanding'
            },
        }
        self.dataset_subsets = {
            # Datasets with detected internal subsets (via sample analysis)
            'VALSE': {
                'type': 'internal_detection',
                'subsets': [
                    'existence', 'plurality', 'counting', 'spatial', 'action_replace',
                    'obj_replace', 'att_replace', 'relation', 'foil_it', 'foil_obj',
                    'foil_att', 'foil_action'
                ],
                'samples_per_subset': 1,
                'description': 'Linguistic phenomena detection'
            },
            
            # Datasets with file-based subsets (separate files per subset)
            'SugarCrepe': {
                'type': 'file_based',
                'subsets': ['add_att', 'add_obj', 'replace_att', 'replace_obj', 'replace_rel', 'swap_att', 'swap_obj'],
                'samples_per_subset': 1,
                'description': 'Hard negatives generation strategies'
            },
            
            'SPEC_I2T': {
                'type': 'file_based',
                'subsets': ['count', 'relative_spatial', 'relative_size', 'absolute_size', 'absolute_spatial', 'existence'],
                'samples_per_subset': 1,
                'description': 'Spatial and numerical reasoning'
            },
            
            'BLA': {
                'type': 'file_based',
                'subsets': ['ap', 'co', 'rc'],  # active_passive, coordination, relative_clause
                'samples_per_subset': 1,
                'description': 'Basic language abilities'
            },
            
            'VL_CheckList': {
                'type': 'file_based',
                'subsets': [
                    # Attribute types
                    'vaw_action', 'vg_action', 'vaw_color', 'vg_color', 
                    'vaw_material', 'vg_material', 'vaw_size', 'vg_size',
                    'vaw_state', 'vg_state',
                    # Relation types  
                    'hake_action', 'swig_action', 'vg_action_relation', 'vg_spatial'
                ],
                'samples_per_subset': 1,
                'description': 'Compositional reasoning across attributes and relations'
            },
            
            # Datasets with single or minimal subsets
            'Winoground': {
                'type': 'single',
                'subsets': ['all'],
                'samples_per_subset': 4,
                'description': 'Compositional visio-linguistic reasoning'
            },
            
            'VG_Attribution': {
                'type': 'single',
                'subsets': ['all'],
                'samples_per_subset': 3,
                'description': 'Visual attributes in Visual Genome'
            },
            
            'VG_Relation': {
                'type': 'single',
                'subsets': ['all'],
                'samples_per_subset': 3,
                'description': 'Visual relations in Visual Genome'
            },
            
            'COCO_Order': {
                'type': 'single',
                'subsets': ['all'],
                'samples_per_subset': 2,
                'description': 'Word order understanding with COCO'
            },
            
            'Flickr30k_Order': {
                'type': 'single',
                'subsets': ['all'],
                'samples_per_subset': 2,
                'description': 'Word order understanding with Flickr30k'
            },
            
            'ColorSwap': {
                'type': 'single',
                'subsets': ['all'],
                'samples_per_subset': 2,
                'description': 'Color understanding and swapping'
            },
            
            'ColorFoil': {
                'type': 'single',
                'subsets': ['all'],
                'samples_per_subset': 2,
                'description': 'Color-based foil detection'
            },
            
            'COCO_Counterfactuals': {
                'type': 'single',
                'subsets': ['all'],
                'samples_per_subset': 2,
                'description': 'Counterfactual reasoning with COCO'
            },
            
            'ControlledImages': {
                'type': 'single',
                'subsets': ['all'],
                'samples_per_subset': 2,
                'description': 'Controlled image generation evaluation'
            },
            
            'VisMin': {
                'type': 'single',
                'subsets': ['all'],
                'samples_per_subset': 2,
                'description': 'Visual minimal pairs'
            }
        }
    
    def get_total_samples_for_dataset(self, dataset_name: str) -> int:
        """Get total recommended samples for a dataset."""
        info = self.dataset_subsets.get(dataset_name, {})
        return info.get('total_subsets', 1) * info.get('samples_per_subset', 2)
        """Calculate total samples needed for a dataset based on its subsets."""
        if dataset_name not in self.dataset_subsets:
            return 2  # Default fallback
        
        info = self.dataset_subsets[dataset_name]
        return len(info['subsets']) * info['samples_per_subset']
    
    def generate_optimal_config(self) -> Dict[str, int]:
        """Generate optimal configuration for comprehensive sampling."""
        config = {}
        for dataset, info in self.dataset_subsets.items():
            config[dataset] = info['total_subsets'] * info['samples_per_subset']
        return config
        """Generate optimal sampling configuration for all datasets."""
        config = {}
        for dataset_name, info in self.dataset_subsets.items():
            config[dataset_name] = self.get_total_samples_for_dataset(dataset_name)
        return config
    
    def print_analysis(self):
        """Print comprehensive analysis of all datasets."""
        print("=" * 80)
        print("📊 COMPREHENSIVE DATASET SUBSET ANALYSIS")
        print("=" * 80)
        
        # Group by type
        multi_subset = {}
        single_subset = {}
        
        for dataset, info in self.dataset_subsets.items():
            total_subsets = len(info['subsets'])
            if total_subsets > 1:
                multi_subset[dataset] = info
            else:
                single_subset[dataset] = info
        
        # Multi-subset datasets
        print("\n🔍 MULTI-SUBSET DATASETS:")
        print("-" * 50)
        total_subsets = 0
        for dataset, info in multi_subset.items():
            print(f"📁 {dataset}")
            print(f"   └── {info['total_subsets']} subsets: {', '.join(info['subsets'][:3])}{'...' if len(info['subsets']) > 3 else ''}")
            print(f"   └── {info['description']}")
            print(f"   └── Recommended: {info['samples_per_subset']} sample per subset")
            total_subsets += info['total_subsets']
        
        print(f"\n   📈 Total subsets across multi-subset datasets: {total_subsets}")
        
        # Single-subset datasets  
        print("\n🎯 SINGLE-SUBSET DATASETS:")
        print("-" * 50)
        for dataset, info in single_subset.items():
            print(f"📁 {dataset}")
            print(f"   └── {info['description']}")
            print(f"   └── Recommended: {info['samples_per_subset']} samples")
        
        # Summary statistics
        total_datasets = len(self.dataset_subsets)
        total_samples = sum(info['total_subsets'] * info['samples_per_subset'] 
                          for info in self.dataset_subsets.values())
        
        print("\n" + "=" * 50)
        print("📋 SUMMARY STATISTICS:")
        print("=" * 50)
        print(f"📊 Total datasets: {total_datasets}")
        print(f"🧩 Multi-subset datasets: {len(multi_subset)}")
        print(f"🎯 Single-subset datasets: {len(single_subset)}")
        print(f"📈 Total subsets: {sum(info['total_subsets'] for info in self.dataset_subsets.values())}")
        print(f"🎲 Total recommended samples: {total_samples}")
        
        print("\n🎯 SAMPLING STRATEGY:")
        print("  • Multi-subset datasets: 1 sample per subset for diversity")
        print("  • Single datasets: 2-4 samples for statistical reliability")
        print(f"  Total visualization: {total_samples} samples across {total_datasets} benchmarks")
        
        return total_datasets, total_samples
        """Print detailed analysis of all datasets and their subsets."""
        print("🔍 DATASET SUBSET ANALYSIS")
        print("=" * 80)
        
        total_datasets = len(self.dataset_subsets)
        total_subsets = sum(len(info['subsets']) for info in self.dataset_subsets.values())
        total_samples = sum(self.get_total_samples_for_dataset(name) for name in self.dataset_subsets.keys())
        
        print(f"📊 Overview: {total_datasets} datasets, {total_subsets} total subsets, {total_samples} total samples")
        print()
        
        # Group by type
        by_type = {}
        for dataset_name, info in self.dataset_subsets.items():
            dataset_type = info['type']
            if dataset_type not in by_type:
                by_type[dataset_type] = []
            by_type[dataset_type].append((dataset_name, info))
        
        for dataset_type, datasets in by_type.items():
            print(f"📂 {dataset_type.upper().replace('_', ' ')} DATASETS:")
            for dataset_name, info in datasets:
                n_subsets = len(info['subsets'])
                n_samples = self.get_total_samples_for_dataset(dataset_name)
                print(f"  🎯 {dataset_name:<18}: {n_subsets:2d} subsets → {n_samples:2d} samples")
                print(f"     📝 {info['description']}")
                
                # Show subsets for multi-subset datasets
                if n_subsets > 1:
                    subset_str = ', '.join(info['subsets'][:6])  # Show first 6
                    if n_subsets > 6:
                        subset_str += f", ... (+{n_subsets-6} more)"
                    print(f"     📋 Subsets: {subset_str}")
                print()
        
        print("🎯 SAMPLING STRATEGY:")
        print(f"  • Multi-subset datasets: 1 sample per subset for diversity")
        print(f"  • Single datasets: 2-4 samples for statistical reliability")
        print(f"  • Total visualization: {total_samples} samples across {total_datasets} benchmarks")


def create_comprehensive_sampling_script():
    """Create and run comprehensive sampling across all dataset subsets."""
    
    print("🚀 Starting comprehensive benchmark sampling...")
    
    # Configuration for comprehensive dataset sampling with new datasets
    config = {
        'VALSE': 12,           # 12 linguistic phenomena subsets
        'SugarCrepe': 7,       # 7 strategy subsets  
        'SugarCrepe_PP': 2,    # 2 samples from preprocessed version
        'VL_CheckList': 14,    # 14 attribute/relation subsets (1 sample from each)
        'Winoground': 4,       # 4 samples for reliability
        'ColorFoil': 2,        # 2 samples
        'COCO_Counterfactuals': 2,  # 2 samples
        'VG_Attribution': 2,   # 2 samples (fixed parameters)
        'VG_Relation': 2,      # 2 samples (fixed parameters)
        'BLA': 3,              # 3 subsets (ap, co, rc)
        'SPEC_I2T': 6,        # 6 reasoning types 
        'ColorSwap': 2,        # 2 samples
        'ControlledImages': 2, # 2 samples
    }
    
    print("📋 Sampling configuration:")
    for dataset, sample_count in config.items():
        print(f"   {dataset}: {sample_count} samples")
    
    try:
        # Initialize sampler and visualizer
        sampler = BenchmarkSampler(data_root_base="./datasets")
        visualizer = ModernBenchmarkVisualizer()
        
        # Sample all datasets using the correct method
        print("\n🔄 Sampling datasets...")
        all_samples = sampler.sample_from_datasets(config, random_seed=42)
        
        # Print results
        for dataset, samples in all_samples.items():
            if samples:
                print(f"   ✅ {dataset}: {len(samples)} samples collected")
            else:
                print(f"   ❌ {dataset}: Failed to sample")
        
        # Create visualization
        total_samples = sum(len(samples) for samples in all_samples.values())
        print(f"\n🎨 Creating visualization with {total_samples} total samples...")
        output_dir = "./comprehensive_benchmark_plots"
        visualizer.create_all_plots(all_samples, output_dir)
        
        print(f"\n🎉 Comprehensive visualization complete!")
        print("� Check output directory: ./comprehensive_benchmark_plots/")
        print(f"🎯 Total samples visualized: {total_samples}")
        
        return all_samples
        
    except Exception as e:
        print(f"❌ Error during comprehensive sampling: {e}")
        return None


def create_custom_sampling_script():
    """Create a custom sampling configuration script."""
    
    print("\n" + "🛠️ " * 20)
    print("🛠️  CUSTOM CONFIGURATION GENERATOR")
    print("🛠️ " * 20)
    
    analyzer = DatasetSubsetAnalyzer()
    
    print("\nAvailable datasets and their subsets:")
    for dataset_name, info in analyzer.dataset_subsets.items():
        print(f"  {dataset_name:<15}: {info['total_subsets']} subsets")
        n_subsets = len(info['subsets'])
        print(f"  {dataset_name:<18}: {n_subsets} subsets")
    
    print(f"\n💡 Usage examples:")
    print(f"  # Sample from all subsets of key datasets:")
    print(f"  python generate_sampling_plot.py VALSE=12 SugarCrepe=7 VL_CheckList=14 SPEC=6")
    print(f"  ")
    print(f"  # Sample from specific datasets:")
    print(f"  python generate_sampling_plot.py Winoground=4 VALSE=6 SugarCrepe=3")
    print(f"  ")
    print(f"  # Use comprehensive configuration:")
    config = analyzer.generate_optimal_config()
    config_str = ' '.join([f"{k}={v}" for k, v in list(config.items())[:3]])
    print(f"  python generate_sampling_plot.py {config_str} ...")


def main():
    """Main function - just run comprehensive sampling."""
    create_comprehensive_sampling_script()


if __name__ == "__main__":
    main()

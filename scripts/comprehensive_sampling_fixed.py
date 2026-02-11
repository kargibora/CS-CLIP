#!/usr/bin/env python3
"""
Comprehensive Dataset Subset Analysis and Sampling Script

This script automatically discovers all available subsets for each dataset and 
creates optimized sampling configurations to get representative samples from 
every subset across all benchmarks.

UPDATED TO MATCH ACTUAL DATASETS IN CODEBASE:
- Corrected SPEC subsets to match filesystem (6 spatial/size categories)
- Fixed BLA subsets to use actual file codes (ap, co, rc)
- Updated VL_CheckList to use actual corpus configs (14 attribute/relation types)
- Added all available datasets with negative captions from data_loading/__init__.py
- Proper subset names for single datasets ('all' or specific like 'A' for WhatsUp)

For paper visualizations: Each sample shows Image + Positive Caption + Negative Caption
"""

import os
import sys
from typing import Dict
import random
# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import after path setup
from generate_sampling_plot import BenchmarkSampler, ModernBenchmarkVisualizer


class DatasetSubsetAnalyzer:
    """Analyzes and maps all available subsets for each dataset."""
    
    def __init__(self):
        # Comprehensive mapping of datasets and their subsets based on actual codebase
        self.dataset_subsets = {
            # Datasets with file-based subsets (each subset is a separate file/config)
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
            
            'SPEC_I2T': {
                'type': 'directory_based_subsets',
                'subsets': [
                    'absolute_size', 'absolute_spatial', 'count', 
                    'existence', 'relative_size', 'relative_spatial'
                ],
                'total_subsets': 6,
                'samples_per_subset': 1,
                'description': 'Systematic reasoning across 6 spatial and object properties'
            },
            
            'BLA': {
                'type': 'file_based_subsets', 
                'subsets': ['ap', 'co', 'rc'],  # active_passive, coordination, relative_clause
                'total_subsets': 3,
                'samples_per_subset': 1,
                'description': 'Language abilities: active/passive, coordination, relative clauses'
            },
            
            'VL_CheckList': {
                'type': 'config_based_subsets',
                'subsets': [
                    'hake_action', 'swig_action', 'vg_action', 
                    'vg_color', 'vaw_action', 'vaw_color'
                ],
                'total_subsets': 6,
                'samples_per_subset': 1,
                'description': 'Systematic evaluation across 6 available attribute and relation types'
            },
            
            'VisMin': {
                'type': 'internal_subsets',
                'subsets': ['object', 'attribute', 'counting'],
                'total_subsets': 3,
                'samples_per_subset': 1,
                'description': 'Visual minimalism across 3 categories'
            },
            
            # Datasets with linguistic phenomena filtering (VALSE style)
            'VALSE': {
                'type': 'linguistic_phenomena',
                'subsets': [
                    'existence', 'plurality', 'counting', 'spatial', 'actions',
                    'coreference', 'comparative', 'foil_it', 'relations',
                    'coordination', 'passivization', 'actant_swap'
                ],
                'total_subsets': 12,
                'samples_per_subset': 1,
                'description': 'Linguistic phenomena detection across 12 categories'
            },
            
            # Single-subset datasets (no internal divisions)
            'Winoground': {
                'type': 'single_dataset',
                'subsets': ['all'],
                'total_subsets': 1,
                'samples_per_subset': 4,
                'description': 'Visio-linguistic compositional reasoning'
            },
            
            'VG_Attribution': {
                'type': 'single_dataset',
                'subsets': ['all'],
                'total_subsets': 1,
                'samples_per_subset': 3,
                'description': 'Visual attribute recognition from Visual Genome'
            },
            
            'VG_Relation': {
                'type': 'single_dataset',
                'subsets': ['all'],
                'total_subsets': 1,
                'samples_per_subset': 3,
                'description': 'Visual relation understanding from Visual Genome'
            },
            
            'COCO_Order': {
                'type': 'single_dataset',
                'subsets': ['all'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Word order sensitivity in COCO captions'
            },
            
            'Flickr30k_Order': {
                'type': 'single_dataset',
                'subsets': ['all'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Word order sensitivity in Flickr30k captions'
            },
            
            'ColorSwap': {
                'type': 'single_dataset',
                'subsets': ['all'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Color attribute swapping detection'
            },
            
            'ColorFoil': {
                'type': 'single_dataset',
                'subsets': ['all'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Color-based hard negative detection'
            },
            
            'COCO_Counterfactuals': {
                'type': 'single_dataset',
                'subsets': ['all'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Counterfactual reasoning in COCO images'
            },
            
            'ControlledImages': {
                'type': 'single_dataset',
                'subsets': ['A'],  # WhatsUp dataset
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Controlled image understanding (WhatsUp)'
            },
            
            'CC3M': {
                'type': 'single_dataset',
                'subsets': ['all'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Conceptual Captions 3M samples'
            },
            
            'NegBench': {
                'type': 'single_dataset',
                'subsets': ['all'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Negative caption benchmark'
            },
        }
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get comprehensive information about a dataset."""
        return self.dataset_subsets.get(dataset_name, {})
    
    def get_optimal_config(self) -> Dict[str, int]:
        """Generate optimal configuration for comprehensive sampling."""
        config = {}
        for dataset, info in self.dataset_subsets.items():
            config[dataset] = info['total_subsets'] * info['samples_per_subset']
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
            if info['total_subsets'] > 1:
                multi_subset[dataset] = info
            else:
                single_subset[dataset] = info
        
        # Multi-subset datasets
        print("\n🔍 MULTI-SUBSET DATASETS:")
        print("-" * 80)
        total_subsets = 0
        for dataset, info in sorted(multi_subset.items()):
            print(f"\n📁 {dataset} ({info['type']})")
            print(f"   ├── Type: {info['description']}")
            print(f"   ├── Subsets ({info['total_subsets']}): {', '.join(info['subsets'])}")
            print(f"   └── Strategy: {info['samples_per_subset']} sample{'s' if info['samples_per_subset'] > 1 else ''} per subset")
            total_subsets += info['total_subsets']
        
        print(f"\n   📈 Total subsets across multi-subset datasets: {total_subsets}")
        
        # Single-subset datasets  
        print("\n🎯 SINGLE-SUBSET DATASETS:")
        print("-" * 80)
        for dataset, info in sorted(single_subset.items()):
            print(f"\n📁 {dataset}")
            print(f"   ├── {info['description']}")
            print(f"   └── Strategy: {info['samples_per_subset']} samples for paper visualization")
        
        # Summary statistics
        total_datasets = len(self.dataset_subsets)
        total_samples = sum(info['total_subsets'] * info['samples_per_subset'] 
                          for info in self.dataset_subsets.values())
        
        print("\n" + "=" * 80)
        print("📋 SUMMARY STATISTICS:")
        print("=" * 80)
        print(f"📊 Total datasets: {total_datasets}")
        print(f"🧩 Multi-subset datasets: {len(multi_subset)}")
        print(f"🎯 Single-subset datasets: {len(single_subset)}")
        print(f"📈 Total unique subsets: {sum(info['total_subsets'] for info in self.dataset_subsets.values())}")
        print(f"🎲 Total recommended samples: {total_samples}")
        
        print("\n🎯 SAMPLING STRATEGY FOR PAPER:")
        print("  • Multi-subset: 1 sample per subset → comprehensive coverage")
        print("  • Single datasets: 2-4 samples → show variety & quality")
        print(f"  • Total visualization: {total_samples} samples across {total_datasets} benchmarks")
        print("  • Each sample shows: Image + Positive Caption + Negative Caption")
        
        return total_datasets, total_samples
    
    def get_subset_command_args(self) -> str:
        """Generate command line arguments for optimal sampling."""
        args = []
        for dataset, info in self.dataset_subsets.items():
            total_samples = info['total_subsets'] * info['samples_per_subset']
            args.append(f"{dataset}={total_samples}")
        return " ".join(args)


def run_comprehensive_sampling():
    """Run comprehensive sampling across all datasets."""
    analyzer = DatasetSubsetAnalyzer()
    
    print("🚀 Starting comprehensive benchmark sampling...")
    print("\n🎲 Random seed: 42 (set via random.seed)")
    print("📁 Output directory: ./comprehensive_benchmark_plots")
    
    # Set random seed
    random.seed(42)
    
    # Get optimal configuration
    config = analyzer.get_optimal_config()
    
    try:
        # Initialize sampler and visualizer
        sampler = BenchmarkSampler()
        visualizer = ModernBenchmarkVisualizer(style='modern')
        
        print("\n📋 Sampling configuration:")
        for dataset, sample_count in config.items():
            info = analyzer.get_dataset_info(dataset)
            if info['total_subsets'] > 1:
                print(f"   {dataset}: {sample_count} samples ({info['total_subsets']} subsets × {info['samples_per_subset']} each)")
            else:
                print(f"   {dataset}: {sample_count} samples (single dataset)")
        
        # Sample all datasets
        print("\n🔄 Sampling datasets...")
        all_samples = sampler.sample_from_datasets(config, random_seed=42)
        
        print("\n📈 Detailed results:")
        total_samples = 0
        for dataset, samples in all_samples.items():
            info = analyzer.get_dataset_info(dataset)
            total_samples += len(samples)
            if info and info['total_subsets'] > 1:
                subset_info = f"across {info['total_subsets']} subsets"
            else:
                subset_info = "single dataset"
            print(f"   📊 {dataset}: {len(samples)} samples {subset_info}")
        
        # Create visualization
        output_dir = "./comprehensive_benchmark_plots"
        print(f"\n🎨 Creating visualization with {total_samples} total samples...")
        visualizer.create_all_plots(all_samples, output_dir=output_dir)
        
        print("\n🎉 Comprehensive visualization complete!")
        print(f"📁 Check output directory: {output_dir}/")
        print(f"🎯 Total samples visualized: {total_samples}")
        
        return all_samples
        
    except Exception as e:
        print(f"❌ Error during comprehensive sampling: {e}")
        return None


def main():
    """Main function with argument parsing."""
    analyzer = DatasetSubsetAnalyzer()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        # Analysis mode
        analyzer.print_analysis()
    elif len(sys.argv) > 1 and sys.argv[1] == "--help":
        # Help mode
        print("📚 Comprehensive Dataset Sampling Tool")
        print("=" * 80)
        print("This tool analyzes vision-language benchmark datasets and creates")
        print("optimized sampling configurations for thesis/paper visualization.\n")
        
        print("� Available Datasets:")
        print("-" * 80)
        analyzer.print_analysis()
        
        print("\n�💡 Usage examples:")
        print("  # Sample from all subsets of key multi-subset datasets:")
        print("  python generate_sampling_plot.py VALSE=12 SugarCrepe=7 VL_CheckList=14 SPEC_I2T=6")
        print("")
        print("  # Sample from specific datasets with custom counts:")
        print("  python generate_sampling_plot.py Winoground=4 VALSE=6 SugarCrepe=3")
        print("  ")
        print("  # Use comprehensive configuration (all datasets, all subsets):")
        
        args = analyzer.get_subset_command_args()
        print(f"  python generate_sampling_plot.py {args}")
        
        print("\n🔍 Available commands:")
        print("  --analyze    Show detailed dataset subset analysis")
        print("  --help       Show this help message")
        print("  (no args)    Run comprehensive sampling across all datasets")
        
        print("\n📝 Output:")
        print("  • Creates visualizations in ./comprehensive_benchmark_plots/")
        print("  • Each sample shows: Image + Positive Caption + Negative Caption")
        print("  • Organized by dataset/subset for easy paper figure creation")
        
    else:
        # Sampling mode
        run_comprehensive_sampling()


if __name__ == "__main__":
    main()

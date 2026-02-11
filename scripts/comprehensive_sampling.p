#!/usr/bin/env python3
"""
Comprehensive Dataset Subset Analysis and Sampling Script

This script automatically discovers all available subsets for each dataset and 
creates optimized sampling configurations to get representative samples from 
every subset across all benchmarks.
"""

import os
import sys
from typing import Dict

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import after path setup
from generate_sampling_plot import BenchmarkSampler, ModernBenchmarkVisualizer


class DatasetSubsetAnalyzer:
    """Analyzes and maps all available subsets for each dataset."""
    
    def __init__(self):
        # Comprehensive mapping of datasets and their subsets
        self.dataset_subsets = {
            # Datasets with detected internal subsets (via sample analysis)
            'VALSE': {
                'type': 'internal_subsets',
                'subsets': [
                    'existence', 'plurality', 'counting', 'spatial', 'actions',
                    'coreference', 'comparative', 'foil_it', 'relations',
                    'coordination', 'passivization', 'actant_swap'
                ],
                'total_subsets': 12,
                'samples_per_subset': 1,
                'description': 'Linguistic phenomena detection across 12 categories'
            },
            
            # Datasets with multi-file structure (different subset files)
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
            
            'SPEC': {
                'type': 'internal_subsets',
                'subsets': [
                    'object', 'attribute', 'relation', 'count', 
                    'comparative', 'action'
                ],
                'total_subsets': 6,
                'samples_per_subset': 1,
                'description': 'Systematic reasoning across 6 semantic types'
            },
            
            'VL_CheckList': {
                'type': 'internal_subsets',
                'subsets': [
                    'obj', 'attr_color', 'attr_material', 'attr_state', 'attr_size',
                    'attr_shape', 'rel_spatial', 'rel_action', 'rel_comparative',
                    'rel_part_whole', 'scene', 'count', 'exist', 'plurals'
                ],
                'total_subsets': 14,
                'samples_per_subset': 1,
                'description': 'Systematic evaluation across 14 attribute and relation types'
            },
            
            'BLA': {
                'type': 'internal_subsets', 
                'subsets': ['basic', 'logical', 'advanced'],
                'total_subsets': 3,
                'samples_per_subset': 1,
                'description': 'Language abilities across 3 complexity levels'
            },
            
            # Single-subset datasets (no internal divisions)
            'Winoground': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 4,
                'description': 'Visio-linguistic compositional reasoning'
            },
            
            'VG_Attribution': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 3,
                'description': 'Visual attribute recognition'
            },
            
            'VG_Relation': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 3,
                'description': 'Visual relation understanding'
            },
            
            'COCO_Order': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Word order sensitivity in COCO'
            },
            
            'Flickr30k_Order': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Word order sensitivity in Flickr30k'
            },
            
            'ColorSwap': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Color attribute swapping detection'
            },
            
            'ColorFoil': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Color-based hard negative detection'
            },
            
            'COCO_Counterfactuals': {
                'type': 'single_dataset',
                'subsets': ['main'],
                'total_subsets': 1,
                'samples_per_subset': 2,
                'description': 'Counterfactual reasoning in COCO'
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
        print("-" * 50)
        total_subsets = 0
        for dataset, info in multi_subset.items():
            print(f"📁 {dataset}")
            print(f"   └── {info['total_subsets']} subsets: {', '.join(info['subsets'])}")
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
    print("\n🎲 Random seed: 42")
    print("📁 Output directory: ./comprehensive_benchmark_plots")
    
    # Get optimal configuration
    config = analyzer.get_optimal_config()
    
    try:
        # Initialize sampler and visualizer
        sampler = BenchmarkSampler(seed=42)
        visualizer = ModernBenchmarkVisualizer(
            output_dir="./comprehensive_benchmark_plots",
            dpi=300
        )
        
        print("\n📋 Sampling configuration:")
        for dataset, sample_count in config.items():
            info = analyzer.get_dataset_info(dataset)
            if info['total_subsets'] > 1:
                print(f"   {dataset}: {sample_count} samples ({info['total_subsets']} subsets × {info['samples_per_subset']} each)")
            else:
                print(f"   {dataset}: {sample_count} samples (single dataset)")
        
        # Sample all datasets
        print("\n🔄 Sampling datasets...")
        all_samples = {}
        
        for dataset, sample_count in config.items():
            try:
                samples = sampler.sample_from_benchmark(dataset, sample_count)
                all_samples[dataset] = samples
                print(f"   ✅ {dataset}: {len(samples)} samples collected")
            except Exception as e:
                print(f"   ❌ {dataset}: Error - {e}")
                continue
        
        print("\n📈 Detailed results:")
        total_samples = 0
        for dataset, samples in all_samples.items():
            info = analyzer.get_dataset_info(dataset)
            total_samples += len(samples)
            if info['total_subsets'] > 1:
                subset_info = f"across {info['total_subsets']} subsets"
            else:
                subset_info = "single dataset"
            print(f"   📊 {dataset}: {len(samples)} samples {subset_info}")
        
        # Create visualization
        print(f"\n🎨 Creating visualization with {total_samples} total samples...")
        visualizer.create_comprehensive_plot(all_samples)
        
        print("\n🎉 Comprehensive visualization complete!")
        print("📁 Check output directory: ./comprehensive_benchmark_plots/")
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
        print("=" * 40)
        print("This tool analyzes vision-language benchmark datasets and creates")
        print("optimized sampling configurations for thesis visualization.\n")
        
        print("💡 Usage examples:")
        print("  # Sample from all subsets of key datasets:")
        print("  python generate_sampling_plot.py VALSE=12 SugarCrepe=7 VL_CheckList=14 SPEC=6")
        print("")
        print("  # Sample from specific datasets:")
        print("  python generate_sampling_plot.py Winoground=4 VALSE=6 SugarCrepe=3")
        print("  ")
        print("  # Use comprehensive configuration:")
        
        args = analyzer.get_subset_command_args()
        print(f"  python generate_sampling_plot.py {args}")
        
        print("\n🔍 Available commands:")
        print("  --analyze    Show detailed dataset subset analysis")
        print("  --help       Show this help message")
        print("  (no args)    Run comprehensive sampling")
        
    else:
        # Sampling mode
        run_comprehensive_sampling()


if __name__ == "__main__":
    main()

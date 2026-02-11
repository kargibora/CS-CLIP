#!/usr/bin/env python3
"""
Quick test script for the enhanced paper visualization
Generates a small sample to verify everything works
"""

import os
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from enhanced_paper_visualization import PaperQualityVisualizer, CAPABILITY_CATEGORIES
from generate_sampling_plot import BenchmarkSampler


def quick_test():
    """Quick test with minimal samples."""
    print("🧪 Running quick test of enhanced visualization...\n")
    
    # Print capability summary
    print("📊 Capability Categories:")
    print("="*70)
    for cap_name, cap_info in CAPABILITY_CATEGORIES.items():
        icon = cap_info['icon']
        color = cap_info['color']
        desc = cap_info['description']
        n_datasets = len(cap_info['datasets'])
        print(f"{icon} {cap_name:25s} | {n_datasets} datasets | {desc}")
    print("="*70)
    
    # Sample just a few datasets for quick testing
    sampler = BenchmarkSampler()
    
    sample_config = {
        'ColorFoil': 2,
        'SugarCrepe': 3,
        'ARO': 2,
        'VALSE': 3,
        'NegBench': 2,
        'ControlledImages': 3,
    }
    
    print("\n🎯 Sampling datasets...")
    all_samples = sampler.sample_from_datasets(sample_config, random_seed=42)
    
    # Print summary
    print("\n📋 Sample Summary:")
    for dataset_name, samples in sorted(all_samples.items()):
        if samples:
            subsets = set(s.get('subset', 'main') for s in samples)
            print(f"  {dataset_name}: {len(samples)} samples (subsets: {', '.join(sorted(subsets))})")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    output_dir = "./test_figures"
    visualizer = PaperQualityVisualizer(output_dir=output_dir)
    visualizer.create_all_paper_figures(all_samples)
    
    print(f"\n✅ Test complete! Check {output_dir}/ for results")
    print("\nGenerated files:")
    if os.path.exists(output_dir):
        for f in sorted(os.listdir(output_dir)):
            print(f"  - {f}")


if __name__ == "__main__":
    quick_test()

#!/usr/bin/env python3
"""
Test CLIPBenchmarkDataset class with full evaluation

Tests both classification and retrieval tasks:
- Classification: wds_imagenet1k, wds_cifar10, wds_vtab-flowers
- Retrieval: wds_mscoco_captions2017, wds_flickr8k, wds_flickr30k
"""

import sys
import argparse
from pathlib import Path

import torch
import open_clip

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loading.clip_benchmark import CLIPBenchmarkDataset


def test_dataset_with_evaluation(
    dataset_name: str,
    model_name: str = 'ViT-B-32',
    pretrained: str = 'openai',
    batch_size: int = 64,
    max_samples: int = None,
    device: str = None
):
    """
    Test CLIPBenchmarkDataset with full evaluation
    
    Args:
        dataset_name: Name of the dataset (e.g., 'wds_mscoco_captions2017', 'wds_cifar10')
        model_name: CLIP model architecture
        pretrained: Pretrained weights to use
        batch_size: Batch size for evaluation
        max_samples: Limit number of samples (for quick testing)
        device: Device to run on (auto-detect if None)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"Testing CLIPBenchmarkDataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"Model: {model_name} ({pretrained})")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print(f"{'='*80}\n")
    
    try:
        # Load model
        print("📦 Loading CLIP model...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully\n")
        
        # Load dataset
        print(f"📦 Loading dataset: {dataset_name}")
        dataset = CLIPBenchmarkDataset(
            dataset_name=dataset_name,
            image_preprocess=preprocess,
            data_root='./datasets/clip_benchmark',
            split='test',
            download=True,
            verbose=True
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        print(f"  Task type: {dataset.task}\n")
        
        # Limit samples for quick testing
        if max_samples and max_samples < len(dataset):
            print(f"⚠️  Limiting to {max_samples} samples for quick test\n")
            indices = torch.randperm(len(dataset))[:max_samples].tolist()
            from torch.utils.data import Subset
            dataset = Subset(dataset, indices)
        
        # Evaluate
        print("🔍 Running evaluation...")
        results, embeddings = dataset.evaluate(
            embedding_model=model,
            device=device,
            batch_size=batch_size,
            verbose=True
        )
        
        # Display results
        print(f"\n{'='*80}")
        print(f"RESULTS: {dataset_name}")
        print(f"{'='*80}")
        
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        print(f"{'='*80}\n")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Test CLIP Benchmark datasets with evaluation')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['wds_cifar10', 'wds_mscoco_captions2017'],
        help='Dataset names to test (default: wds_cifar10 wds_mscoco_captions2017)'
    )
    parser.add_argument(
        '--model',
        default='ViT-B-32',
        help='CLIP model architecture (default: ViT-B-32)'
    )
    parser.add_argument(
        '--pretrained',
        default='openai',
        help='Pretrained weights (default: openai)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for evaluation (default: 64)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit number of samples for quick testing (default: None = use all)'
    )
    parser.add_argument(
        '--device',
        default=None,
        help='Device to run on (default: auto-detect)'
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available dataset names and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_datasets:
        print("\n" + "="*80)
        print("Available CLIP Benchmark Datasets")
        print("="*80)
        print("\nClassification Datasets:")
        classification = [
            'wds_imagenet1k', 'wds_imagenetv2', 'wds_imagenet_sketch',
            'wds_imagenet-a', 'wds_imagenet-r', 'wds_cifar10', 'wds_cifar100',
            'wds_vtab-flowers', 'wds_vtab-pets', 'wds_vtab-caltech101',
            'wds_vtab-dtd', 'wds_food101', 'wds_sun397', 'wds_cars',
            'wds_fgvc_aircraft', 'wds_mnist', 'wds_stl10', 'wds_gtsrb'
        ]
        for ds in classification:
            print(f"  - {ds}")
        
        print("\nRetrieval Datasets:")
        retrieval = [
            'wds_mscoco_captions2017', 'wds_flickr8k', 'wds_flickr30k'
        ]
        for ds in retrieval:
            print(f"  - {ds}")
        print()
        return
    
    # Test each dataset
    all_results = {}
    for dataset_name in args.datasets:
        results = test_dataset_with_evaluation(
            dataset_name=dataset_name,
            model_name=args.model,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=args.device
        )
        if results:
            all_results[dataset_name] = results
    
    # Summary
    if all_results:
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL TESTS")
        print(f"{'='*80}")
        for dataset_name, results in all_results.items():
            print(f"\n{dataset_name}:")
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
        print()


if __name__ == "__main__":
    main()

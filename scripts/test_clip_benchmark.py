#!/usr/bin/env python3
"""
Test script for CLIP Benchmark Dataset Loader

This script tests loading and evaluating CLIP benchmark datasets from HuggingFace.
Tests both classification and retrieval tasks with real models.

Usage:
    # Test specific datasets
    python scripts/test_clip_benchmark.py --datasets wds/mscoco_captions wds/vtab/flowers
    
    # Test with custom model
    python scripts/test_clip_benchmark.py --model ViT-L-14 --pretrained laion2b_s32b_b82k
    
    # Quick test with subset
    python scripts/test_clip_benchmark.py --datasets wds/vtab/cifar10 --num_samples 100
    
    # Dry run (just load datasets, no evaluation)
    python scripts/test_clip_benchmark.py --dry_run
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import open_clip
    from data_loading.clip_benchmark import CLIPBenchmarkDataset
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install required packages:")
    print("  pip install open_clip_torch clip-benchmark")
    sys.exit(1)


def test_dataset_loading(dataset_name: str, verbose: bool = True):
    """
    Test loading a single dataset.
    
    Args:
        dataset_name: Name of the dataset to test
        verbose: Print detailed info
        
    Returns:
        tuple: (success: bool, dataset: CLIPBenchmarkDataset or None, error: str or None)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing Dataset: {dataset_name}")
        print(f"{'='*80}")
    
    try:
        # Load with minimal preprocessing (just to test)
        dataset = CLIPBenchmarkDataset(
            dataset_name=dataset_name,
            task='auto',  # Auto-detect task
            data_root='datasets/clip_benchmark',
            split='test',
            image_preprocess=None,  # Will use default
            download=True,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n✅ Successfully loaded: {dataset_name}")
            print(f"   Task: {dataset.task}")
            print(f"   Samples: {len(dataset)}")
            
            if dataset.task == 'zeroshot_classification':
                print(f"   Classes: {len(dataset.classes)}")
                print(f"   Templates: {len(dataset.templates)}")
                if len(dataset.classes) <= 10:
                    print(f"   Class names: {dataset.classes}")
            elif dataset.task == 'zeroshot_retrieval':
                print(f"   Captions: {len(dataset.captions)}")
        
        # Test getting a sample
        try:
            sample = dataset[0]
            if verbose:
                print(f"   Sample format: {type(sample)}")
                if isinstance(sample, tuple):
                    print(f"   Sample length: {len(sample)}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Could not get sample: {e}")
        
        return True, dataset, None
        
    except Exception as e:
        if verbose:
            print(f"\n❌ Failed to load: {dataset_name}")
            print(f"   Error: {str(e)}")
        return False, None, str(e)


def test_dataset_evaluation(
    dataset: CLIPBenchmarkDataset,
    model,
    preprocess,
    device: str = 'cuda',
    batch_size: int = 32,
    num_samples: int = None,
    verbose: bool = True
):
    """
    Test evaluating a model on a dataset.
    
    Args:
        dataset: CLIPBenchmarkDataset instance
        model: CLIP model
        preprocess: Image preprocessing function
        device: Device to run on
        batch_size: Batch size for evaluation
        num_samples: If set, only evaluate on this many samples
        verbose: Print detailed info
        
    Returns:
        tuple: (success: bool, results: dict or None, error: str or None)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Evaluating: {dataset.dataset_name}")
        print(f"{'='*80}")
    
    try:
        # Create dataset with proper preprocessing
        dataset.image_preprocess = preprocess
        
        # Optionally subsample for faster testing
        indices = None
        if num_samples is not None and len(dataset) > num_samples:
            import random
            indices = random.sample(range(len(dataset)), num_samples)
            if verbose:
                print(f"   Testing on {num_samples}/{len(dataset)} samples")
        
        # Run evaluation
        start_time = time.time()
        results, embeddings = dataset.evaluate(
            embedding_model=model,
            device=device,
            batch_size=batch_size,
            indices=indices,
        )
        eval_time = time.time() - start_time
        
        if verbose:
            print(f"\n✅ Evaluation completed in {eval_time:.2f}s")
            print("\n📊 Results:")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
        
        return True, results, None
        
    except Exception as e:
        if verbose:
            print(f"\n❌ Evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
        return False, None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Test CLIP Benchmark Dataset Loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test MSCOCO retrieval
  python scripts/test_clip_benchmark.py --datasets wds/mscoco_captions
  
  # Test flowers classification
  python scripts/test_clip_benchmark.py --datasets wds/vtab/flowers
  
  # Test multiple datasets
  python scripts/test_clip_benchmark.py --datasets wds/mscoco_captions wds/vtab/flowers wds/vtab/cifar10
  
  # Quick test with 100 samples
  python scripts/test_clip_benchmark.py --datasets wds/vtab/flowers --num_samples 100
  
  # List all available datasets
  python scripts/test_clip_benchmark.py --list_datasets
  
  # Dry run (just load, don't evaluate)
  python scripts/test_clip_benchmark.py --datasets wds/mscoco_captions --dry_run
        """
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['wds/mscoco_captions', 'wds/vtab/flowers'],
        help='List of datasets to test (default: mscoco_captions and flowers)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='ViT-B-32',
        help='OpenCLIP model architecture (default: ViT-B-32)'
    )
    
    parser.add_argument(
        '--pretrained',
        type=str,
        default='openai',
        help='Pretrained weights (default: openai)'
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        default='datasets/clip_benchmark',
        help='Root directory for dataset cache'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (default: cuda if available)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Only test loading, skip evaluation'
    )
    
    parser.add_argument(
        '--list_datasets',
        action='store_true',
        help='List all available datasets and exit'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed information'
    )
    
    args = parser.parse_args()
    
    # List datasets and exit
    if args.list_datasets:
        print("\n" + "="*80)
        print("Available CLIP Benchmark Datasets")
        print("="*80 + "\n")
        
        print("🔍 RETRIEVAL DATASETS:")
        print("-" * 80)
        retrieval_datasets = [
            'wds/mscoco_captions',
            'wds/flickr8k',
            'wds/flickr30k',
        ]
        for ds in retrieval_datasets:
            print(f"  • {ds}")
        
        print("\n🎯 CLASSIFICATION DATASETS:")
        print("-" * 80)
        classification_datasets = [
            'wds/imagenet1k',
            'wds/imagenetv2',
            'wds/imagenet_sketch',
            'wds/imagenet-a',
            'wds/imagenet-r',
            'wds/imagenet-o',
            'wds/objectnet',
            'wds/fer2013',
            'wds/voc2007',
            'wds/voc2007_multilabel',
            'wds/sun397',
            'wds/cars',
            'wds/fgvc_aircraft',
            'wds/mnist',
            'wds/stl10',
            'wds/gtsrb',
            'wds/country211',
            'wds/renderedsst2',
        ]
        for ds in classification_datasets:
            print(f"  • {ds}")
        
        print("\n🔬 VTAB DATASETS:")
        print("-" * 80)
        vtab_datasets = [
            'wds/vtab/caltech101',
            'wds/vtab/cifar10',
            'wds/vtab/cifar100',
            'wds/vtab/clevr_count_all',
            'wds/vtab/clevr_closest_object_distance',
            'wds/vtab/diabetic_retinopathy',
            'wds/vtab/dmlab',
            'wds/vtab/dsprites_label_orientation',
            'wds/vtab/dsprites_label_x_position',
            'wds/vtab/dsprites_label_y_position',
            'wds/vtab/dtd',
            'wds/vtab/eurosat',
            'wds/vtab/kitti_closest_vehicle_distance',
            'wds/vtab/flowers',
            'wds/vtab/pets',
            'wds/vtab/pcam',
            'wds/vtab/resisc45',
            'wds/vtab/smallnorb_label_azimuth',
            'wds/vtab/smallnorb_label_elevation',
            'wds/vtab/svhn',
        ]
        for ds in vtab_datasets:
            print(f"  • {ds}")
        
        print("\n" + "="*80)
        print(f"Total: {len(retrieval_datasets) + len(classification_datasets) + len(vtab_datasets)} datasets")
        print("="*80 + "\n")
        return
    
    # Print test configuration
    print("\n" + "="*80)
    print("CLIP Benchmark Test Configuration")
    print("="*80)
    print(f"  Model: {args.model}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data root: {args.data_root}")
    print(f"  Datasets to test: {len(args.datasets)}")
    for ds in args.datasets:
        print(f"    • {ds}")
    if args.num_samples:
        print(f"  Samples per dataset: {args.num_samples}")
    if args.dry_run:
        print("  Mode: DRY RUN (loading only)")
    print("="*80 + "\n")
    
    # Load model if not dry run
    model = None
    preprocess = None
    if not args.dry_run:
        print("Loading CLIP model...")
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                args.model,
                pretrained=args.pretrained,
                device=args.device
            )
            model = model.eval()
            print(f"✅ Model loaded: {args.model} ({args.pretrained})\n")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)
    
    # Test each dataset
    results_summary = []
    
    for dataset_name in args.datasets:
        # Test loading
        success, dataset, error = test_dataset_loading(dataset_name, verbose=args.verbose)
        
        if not success:
            results_summary.append({
                'dataset': dataset_name,
                'loading': 'FAILED',
                'evaluation': 'SKIPPED',
                'error': error
            })
            continue
        
        results_summary.append({
            'dataset': dataset_name,
            'loading': 'SUCCESS',
            'task': dataset.task,
            'samples': len(dataset),
        })
        
        # Test evaluation if not dry run
        if not args.dry_run:
            success, eval_results, error = test_dataset_evaluation(
                dataset=dataset,
                model=model,
                preprocess=preprocess,
                device=args.device,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                verbose=args.verbose
            )
            
            if success:
                results_summary[-1]['evaluation'] = 'SUCCESS'
                results_summary[-1]['results'] = eval_results
            else:
                results_summary[-1]['evaluation'] = 'FAILED'
                results_summary[-1]['error'] = error
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")
    
    for result in results_summary:
        print(f"Dataset: {result['dataset']}")
        print(f"  Loading: {result['loading']}")
        
        if result['loading'] == 'SUCCESS':
            print(f"  Task: {result['task']}")
            print(f"  Samples: {result['samples']}")
            
            if not args.dry_run:
                print(f"  Evaluation: {result.get('evaluation', 'N/A')}")
                
                if result.get('evaluation') == 'SUCCESS' and 'results' in result:
                    eval_results = result['results']
                    print("  Metrics:")
                    
                    # Print key metrics
                    if 'acc1' in eval_results:
                        print(f"    • Top-1 Accuracy: {eval_results['acc1']:.4f}")
                    if 'acc5' in eval_results:
                        print(f"    • Top-5 Accuracy: {eval_results['acc5']:.4f}")
                    if 'mean_per_class_recall' in eval_results:
                        print(f"    • Mean Per-Class Recall: {eval_results['mean_per_class_recall']:.4f}")
                    
                    # Retrieval metrics
                    for k in [1, 5, 10]:
                        if f'image_retrieval_recall@{k}' in eval_results:
                            print(f"    • Image R@{k}: {eval_results[f'image_retrieval_recall@{k}']:.4f}")
                        if f'text_retrieval_recall@{k}' in eval_results:
                            print(f"    • Text R@{k}: {eval_results[f'text_retrieval_recall@{k}']:.4f}")
        
        if 'error' in result:
            print(f"  Error: {result['error']}")
        
        print()
    
    # Count successes
    loading_success = sum(1 for r in results_summary if r['loading'] == 'SUCCESS')
    if not args.dry_run:
        eval_success = sum(1 for r in results_summary if r.get('evaluation') == 'SUCCESS')
        print(f"Loading: {loading_success}/{len(results_summary)} successful")
        print(f"Evaluation: {eval_success}/{len(results_summary)} successful")
    else:
        print(f"Loading: {loading_success}/{len(results_summary)} successful")
    
    print("\n" + "="*80)
    
    # Exit with appropriate code
    if loading_success == len(results_summary):
        if args.dry_run or eval_success == len(results_summary):
            print("✅ All tests passed!")
            print("="*80 + "\n")
            sys.exit(0)
    
    print("⚠️  Some tests failed - see above for details")
    print("="*80 + "\n")
    sys.exit(1)


if __name__ == '__main__':
    main()

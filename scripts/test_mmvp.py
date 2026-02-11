#!/usr/bin/env python
"""
Test script for MMVP-VLM Dataset

This script tests:
1. Dataset loading from HuggingFace
2. Sample structure and content
3. Pattern type filtering
4. Evaluation with a CLIP model (optional)

Usage:
    python scripts/test_mmvp.py                    # Basic loading test
    python scripts/test_mmvp.py --evaluate         # Run evaluation with CLIP
    python scripts/test_mmvp.py --subset Orientation  # Test specific subset
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_loading(data_root: str = None, verbose: bool = True):
    """Test basic dataset loading."""
    from data_loading.mmvp import MMVPDataset
    
    print("=" * 60)
    print("Testing MMVP Dataset Loading")
    print("=" * 60)
    
    # Load full dataset
    print("\n[1] Loading full dataset...")
    dataset = MMVPDataset(
        data_root=data_root,
        subset_name="all",
        verbose=verbose,
    )
    
    print("\n✓ Dataset loaded successfully!")
    print(f"  Total samples: {len(dataset)}")
    
    # Check integrity
    print("\n[2] Checking dataset integrity...")
    stats = dataset.check_dataset_integrity()
    print("  Pattern distribution:")
    for pattern, count in sorted(stats['pattern_counts'].items()):
        print(f"    - {pattern}: {count}")
    print(f"  Missing images: {stats['missing_images']}")
    print(f"  Missing captions: {stats['missing_captions']}")
    
    # Test sample access
    print("\n[3] Testing sample access...")
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Caption: '{sample['caption'][:50]}...' " if len(sample['caption']) > 50 else f"  Caption: '{sample['caption']}'")
    print(f"  Foil: '{sample['foil'][:50]}...' " if len(sample['foil']) > 50 else f"  Foil: '{sample['foil']}'")
    print(f"  Pattern type: {sample['pattern_type']}")
    print(f"  Pair ID: {sample['pair_id']}")
    print(f"  Label: {sample['label']}")
    
    # Test paired samples
    print("\n[4] Testing paired samples (should have opposite captions)...")
    sample1 = dataset[0]
    sample2 = dataset[1]
    if sample1['pair_id'] == sample2['pair_id']:
        print(f"  ✓ Same pair ID: {sample1['pair_id']}")
        print(f"  ✓ Sample 1 caption matches Sample 2 foil: {sample1['caption'] == sample2['foil']}")
        print(f"  ✓ Sample 1 foil matches Sample 2 caption: {sample1['foil'] == sample2['caption']}")
    else:
        print("  ! Different pair IDs (may be filtered)")
    
    return dataset


def test_subset_loading(subset_name: str, data_root: str = None):
    """Test loading a specific subset."""
    from data_loading.mmvp import MMVPDataset
    
    print("\n" + "=" * 60)
    print(f"Testing Subset Loading: {subset_name}")
    print("=" * 60)
    
    dataset = MMVPDataset(
        data_root=data_root,
        subset_name=subset_name,
        verbose=True,
    )
    
    print(f"\n✓ Subset '{subset_name}' loaded with {len(dataset)} samples")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("  First sample:")
        print(f"    Caption: {sample['caption']}")
        print(f"    Pattern: {sample['pattern_type']}")
    
    return dataset


def test_with_preprocess(data_root: str = None):
    """Test dataset with CLIP preprocessing."""
    print("\n" + "=" * 60)
    print("Testing with CLIP Preprocessing")
    print("=" * 60)
    
    try:
        import clip
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        from data_loading.mmvp import MMVPDataset
        
        dataset = MMVPDataset(
            data_root=data_root,
            subset_name="all",
            image_preprocess=preprocess,
            verbose=True,
        )
        
        sample = dataset[0]
        print("\n✓ Image preprocessed successfully!")
        print(f"  Image tensor shape: {sample['image'].shape}")
        print(f"  Image dtype: {sample['image'].dtype}")
        
        return dataset, model, preprocess
        
    except ImportError as e:
        print(f"\n! Could not load CLIP: {e}")
        return None, None, None


def test_dataloader(data_root: str = None):
    """Test DataLoader with collate function."""
    print("\n" + "=" * 60)
    print("Testing DataLoader")
    print("=" * 60)
    
    try:
        import clip
        import torch
        from torch.utils.data import DataLoader
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, preprocess = clip.load("ViT-B/32", device=device)
        
        from data_loading.mmvp import MMVPDataset
        
        dataset = MMVPDataset(
            data_root=data_root,
            subset_name="all",
            image_preprocess=preprocess,
            verbose=False,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset._collate_fn,
        )
        
        batch = next(iter(dataloader))
        print("\n✓ DataLoader batch loaded successfully!")
        print(f"  Batch images shape: {batch['images'].shape}")
        print(f"  Batch captions: {len(batch['captions'])} items")
        print(f"  Batch foils: {len(batch['foils'])} items")
        print(f"  Batch pattern types: {batch['pattern_types']}")
        
        return dataloader
        
    except Exception as e:
        print(f"\n! DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_evaluation(data_root: str = None, batch_size: int = 16):
    """Test evaluation with CLIP model."""
    print("\n" + "=" * 60)
    print("Testing Evaluation")
    print("=" * 60)
    
    try:
        import clip
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nUsing device: {device}")
        
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        
        from data_loading.mmvp import MMVPDataset
        
        dataset = MMVPDataset(
            data_root=data_root,
            subset_name="all",
            image_preprocess=preprocess,
            verbose=True,
        )
        
        print(f"\nRunning evaluation on {len(dataset)} samples...")
        
        # Evaluate on a small subset for speed
        indices = list(range(min(100, len(dataset))))
        
        results, embeddings = dataset.evaluate(
            embedding_model=model,
            device=device,
            batch_size=batch_size,
            indices=indices,
        )
        
        print("\n✓ Evaluation completed!")
        print("\nResults:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\nEmbeddings:")
        for key, value in embeddings.items():
            print(f"  {key}: shape {value.shape}")
        
        return results, embeddings
        
    except Exception as e:
        print(f"\n! Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_split_dataset(data_root: str = None):
    """Test dataset splitting."""
    print("\n" + "=" * 60)
    print("Testing Dataset Split")
    print("=" * 60)
    
    from data_loading.mmvp import MMVPDataset
    
    dataset = MMVPDataset(
        data_root=data_root,
        subset_name="all",
        verbose=False,
    )
    
    # Test random split
    print("\n[1] Testing random split...")
    splits = dataset.split_dataset(val_ratio=0.2, test_ratio=0.1, split_type="random")
    print(f"  Train: {len(splits['train']['indices'])} samples")
    print(f"  Val: {len(splits['val']['indices'])} samples")
    print(f"  Test: {len(splits['test']['indices'])} samples")
    
    # Test pair split
    print("\n[2] Testing pair split (keeps image pairs together)...")
    splits = dataset.split_dataset(val_ratio=0.2, test_ratio=0.1, split_type="pair")
    print(f"  Train: {len(splits['train']['indices'])} samples")
    print(f"  Val: {len(splits['val']['indices'])} samples")
    print(f"  Test: {len(splits['test']['indices'])} samples")
    
    return splits


def main():
    parser = argparse.ArgumentParser(description="Test MMVP Dataset")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Cache directory for dataset")
    parser.add_argument("--subset", type=str, default=None,
                        help="Test a specific subset (e.g., 'Orientation', 'Color')")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation with CLIP model")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--all-tests", action="store_true",
                        help="Run all tests")
    args = parser.parse_args()
    
    # Basic loading test
    _ = test_basic_loading(args.data_root)
    
    # Test specific subset if requested
    if args.subset:
        test_subset_loading(args.subset, args.data_root)
    
    if args.all_tests or args.evaluate:
        # Test with preprocessing
        test_with_preprocess(args.data_root)
        
        # Test DataLoader
        test_dataloader(args.data_root)
        
        # Test splitting
        test_split_dataset(args.data_root)
    
    if args.evaluate:
        # Run evaluation
        test_evaluation(args.data_root, args.batch_size)
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

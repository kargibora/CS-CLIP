#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for COCOComponentsDataset.

This script tests:
1. Basic dataset loading
2. Component sampling (positive/negative)
3. Relation sampling
4. Mixed sampling
5. Training wrapper (COCOComponentsNeg)
6. Tokenization with different tokenizers (CLIP/BLIP/FLAVA)

Usage:
    python scripts/test_coco_components.py \\
        --coco_root /path/to/coco \\
        --annotation_file coco_with_components.json \\
        --test_split train
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image

from data_loading.coco_components import (
    COCOComponentsDataset,
    COCOComponentsNeg,
    load_karpathy_splits
)


def test_basic_loading(coco_root: str, annotation_file: str, split: str = "train"):
    """Test basic dataset loading."""
    print("\n" + "="*80)
    print("TEST 1: Basic Dataset Loading")
    print("="*80)
    
    dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split=split,
        sample_relations=False,
        sample_components=False
    )
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Split: {split}")
    
    # Get first sample
    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Image shape: {sample['image'].size if isinstance(sample['image'], Image.Image) else sample['image'].shape}")
    print(f"  Caption: {sample['caption'][:100]}...")
    print(f"  Components: {sample.get('components', [])[:5]}")
    print(f"  Relations: {len(sample.get('relations', []))}")
    
    # Test multiple samples
    print(f"\nTesting first 5 samples...")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        assert 'image' in sample, f"Sample {i} missing 'image'"
        assert 'caption' in sample, f"Sample {i} missing 'caption'"
        print(f"  Sample {i}: ✓")
    
    print("\n✅ Basic loading test passed!")
    return dataset


def test_component_sampling(coco_root: str, annotation_file: str, split: str = "train"):
    """Test component sampling."""
    print("\n" + "="*80)
    print("TEST 2: Component Sampling")
    print("="*80)
    
    # Test positive components
    print("\n[Positive Components]")
    dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split=split,
        sample_components=True,
        sample_positive_components=True,
        sample_negative_components=False,
        max_positive_components=3,
        sample_relations=False
    )
    
    sample = dataset[0]
    print(f"  Caption: {sample['caption']}")
    print(f"  Type: {type(sample['caption'])}")
    
    # Test negative components
    print("\n[Negative Components]")
    dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split=split,
        sample_components=True,
        sample_positive_components=False,
        sample_negative_components=True,
        max_negative_components=2,
        sample_relations=False
    )
    
    sample = dataset[0]
    print(f"  Caption: {sample['caption']}")
    print(f"  Type: {type(sample['caption'])}")
    
    # Test mixed components
    print("\n[Mixed Components]")
    dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split=split,
        sample_components=True,
        sample_positive_components=True,
        sample_negative_components=True,
        max_positive_components=2,
        max_negative_components=2,
        sample_relations=False
    )
    
    for i in range(3):
        sample = dataset[i]
        print(f"\n  Sample {i}:")
        print(f"    Caption: {sample['caption']}")
    
    print("\n✅ Component sampling test passed!")


def test_relation_sampling(coco_root: str, annotation_file: str, split: str = "train"):
    """Test relation sampling."""
    print("\n" + "="*80)
    print("TEST 3: Relation Sampling")
    print("="*80)
    
    dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split=split,
        sample_relations=True,
        relation_probability=1.0,  # Always sample relations
        max_relations=2,
        sample_components=False
    )
    
    print(f"  Relation probability: 1.0")
    print(f"  Max relations: 2")
    
    for i in range(3):
        sample = dataset[i]
        print(f"\n  Sample {i}:")
        print(f"    Caption: {sample['caption']}")
    
    print("\n✅ Relation sampling test passed!")


def test_mixed_sampling(coco_root: str, annotation_file: str, split: str = "train"):
    """Test mixed sampling (components + relations)."""
    print("\n" + "="*80)
    print("TEST 4: Mixed Sampling (Components + Relations)")
    print("="*80)
    
    dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split=split,
        sample_relations=True,
        sample_relation_or_components="mixed",
        relation_probability=0.5,
        max_relations=1,
        sample_components=True,
        max_positive_components=2,
        max_negative_components=1
    )
    
    print(f"  Mode: mixed")
    print(f"  Relation probability: 0.5")
    print(f"  Max relations: 1")
    print(f"  Max positive components: 2")
    print(f"  Max negative components: 1")
    
    for i in range(5):
        sample = dataset[i]
        print(f"\n  Sample {i}:")
        print(f"    Caption: {sample['caption']}")
    
    print("\n✅ Mixed sampling test passed!")


def test_negative_sampling(coco_root: str, annotation_file: str, split: str = "train"):
    """Test negative caption generation."""
    print("\n" + "="*80)
    print("TEST 5: Negative Sampling")
    print("="*80)
    
    # Test relation negatives
    print("\n[Relation Negatives]")
    dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split=split,
        sample_relations=True,
        negative_relation_prob=1.0,  # Always create negatives
        sample_components=False
    )
    
    for i in range(3):
        sample = dataset[i]
        if 'negative_captions' in sample and sample['negative_captions']:
            print(f"\n  Sample {i}:")
            print(f"    Original: {sample['caption']}")
            print(f"    Negatives: {sample['negative_captions'][:2]}")
    
    # Test inplace negatives
    print("\n[Inplace Negatives]")
    dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split=split,
        sample_components=True,
        negative_inplace_prob=1.0,  # Always create negatives
        sample_relations=False
    )
    
    for i in range(3):
        sample = dataset[i]
        if 'negative_captions' in sample and sample['negative_captions']:
            print(f"\n  Sample {i}:")
            print(f"    Original: {sample['caption']}")
            print(f"    Negatives: {sample['negative_captions'][:2]}")
    
    print("\n✅ Negative sampling test passed!")


def test_training_wrapper(coco_root: str, annotation_file: str, split: str = "train"):
    """Test COCOComponentsNeg training wrapper."""
    print("\n" + "="*80)
    print("TEST 6: Training Wrapper (COCOComponentsNeg)")
    print("="*80)
    
    # Create base dataset
    base_dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split=split,
        sample_relations=True,
        sample_relation_or_components="mixed",
        sample_components=True,
        negative_relation_prob=0.5,
        negative_inplace_prob=0.5
    )
    
    # Create training wrapper
    indices = list(range(min(100, len(base_dataset))))
    train_dataset = COCOComponentsNeg(
        dataset=base_dataset,
        indices=indices,
        tokenizer=None,  # Use default CLIP tokenizer
        max_length=77
    )
    
    print(f"✓ Training wrapper created")
    print(f"  Base dataset size: {len(base_dataset)}")
    print(f"  Training subset size: {len(train_dataset)}")
    
    # Test __getitem__
    sample = train_dataset[0]
    print(f"\nSample structure:")
    print(f"  Type: {type(sample)}")
    print(f"  Length: {len(sample)}")
    
    if len(sample) == 4:
        image, pos_tokens, rand_neg, all_neg_tokens = sample
        print(f"\n  Image shape: {image.shape if torch.is_tensor(image) else 'PIL Image'}")
        print(f"  Positive tokens shape: {pos_tokens.shape if torch.is_tensor(pos_tokens) else pos_tokens}")
        print(f"  Random negative: {rand_neg}")
        print(f"  All negative tokens shape: {all_neg_tokens.shape if torch.is_tensor(all_neg_tokens) else all_neg_tokens}")
    
    # Test multiple samples
    print(f"\nTesting first 5 samples from training wrapper...")
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        assert len(sample) == 4, f"Sample {i} should return 4 items"
        print(f"  Sample {i}: ✓")
    
    print("\n✅ Training wrapper test passed!")


def test_karpathy_splits(coco_root: str, annotation_file: str):
    """Test Karpathy split loading."""
    print("\n" + "="*80)
    print("TEST 7: Karpathy Splits")
    print("="*80)
    
    train_dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split="train",
        sample_relations=False,
        sample_components=False
    )
    
    val_dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split="val",
        sample_relations=False,
        sample_components=False
    )
    
    test_dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split="test",
        sample_relations=False,
        sample_components=False
    )
    
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Val size: {len(val_dataset)}")
    print(f"  Test size: {len(test_dataset)}")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    
    print("\n✅ Karpathy splits test passed!")


def test_with_custom_tokenizer(coco_root: str, annotation_file: str, split: str = "train"):
    """Test with custom tokenizers (BLIP/FLAVA)."""
    print("\n" + "="*80)
    print("TEST 8: Custom Tokenizers")
    print("="*80)
    
    base_dataset = COCOComponentsDataset(
        coco_root=coco_root,
        annotation_file=annotation_file,
        split=split,
        sample_components=True,
        max_positive_components=2
    )
    
    indices = list(range(min(10, len(base_dataset))))
    
    # Test with custom tokenizer function
    def custom_tokenizer(texts, max_length=77):
        """Mock tokenizer for testing."""
        if isinstance(texts, str):
            texts = [texts]
        # Return mock tokens
        return torch.randint(0, 1000, (len(texts), max_length))
    
    train_dataset = COCOComponentsNeg(
        dataset=base_dataset,
        indices=indices,
        tokenizer=custom_tokenizer,
        max_length=77
    )
    
    sample = train_dataset[0]
    image, pos_tokens, rand_neg, all_neg_tokens = sample
    
    print(f"  Custom tokenizer output shape: {pos_tokens.shape}")
    print(f"  Expected shape: torch.Size([77]) or similar")
    
    print("\n✅ Custom tokenizer test passed!")


def main():
    parser = argparse.ArgumentParser(description="Test COCOComponentsDataset")
    parser.add_argument("--coco_root", type=str, required=True,
                       help="Root directory containing COCO images")
    parser.add_argument("--annotation_file", type=str, required=True,
                       help="Path to COCO components JSON file")
    parser.add_argument("--test_split", type=str, default="train",
                       choices=["train", "val", "test"],
                       help="Which split to test")
    parser.add_argument("--tests", type=str, nargs="+",
                       default=["all"],
                       choices=["all", "loading", "components", "relations",
                               "mixed", "negatives", "wrapper", "splits", "tokenizer"],
                       help="Which tests to run")
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.coco_root):
        print(f"Error: COCO root not found: {args.coco_root}")
        return
    
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file not found: {args.annotation_file}")
        return
    
    print("="*80)
    print("COCO Components Dataset Test Suite")
    print("="*80)
    print(f"COCO root: {args.coco_root}")
    print(f"Annotation file: {args.annotation_file}")
    print(f"Test split: {args.test_split}")
    print(f"Tests to run: {', '.join(args.tests)}")
    
    # Run tests
    tests_to_run = args.tests
    if "all" in tests_to_run:
        tests_to_run = ["loading", "components", "relations", "mixed",
                       "negatives", "wrapper", "splits", "tokenizer"]
    
    try:
        if "loading" in tests_to_run:
            test_basic_loading(args.coco_root, args.annotation_file, args.test_split)
        
        if "components" in tests_to_run:
            test_component_sampling(args.coco_root, args.annotation_file, args.test_split)
        
        if "relations" in tests_to_run:
            test_relation_sampling(args.coco_root, args.annotation_file, args.test_split)
        
        if "mixed" in tests_to_run:
            test_mixed_sampling(args.coco_root, args.annotation_file, args.test_split)
        
        if "negatives" in tests_to_run:
            test_negative_sampling(args.coco_root, args.annotation_file, args.test_split)
        
        if "wrapper" in tests_to_run:
            test_training_wrapper(args.coco_root, args.annotation_file, args.test_split)
        
        if "splits" in tests_to_run:
            test_karpathy_splits(args.coco_root, args.annotation_file)
        
        if "tokenizer" in tests_to_run:
            test_with_custom_tokenizer(args.coco_root, args.annotation_file, args.test_split)
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

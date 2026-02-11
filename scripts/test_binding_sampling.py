#!/usr/bin/env python3
"""
Test script for binding negative sampling.

Usage:
    python scripts/test_binding_sampling.py --json_file path/to/sample.json
    python scripts/test_binding_sampling.py --json_dir swap_pos_json/coco_train --num_samples 100
    python scripts/test_binding_sampling.py --demo
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sampler import StructuredSampler


def load_samples_from_file(json_file: str, max_samples: int = None) -> list:
    """Load samples from a JSON file."""
    with open(json_file, 'r') as f:
        samples = json.load(f)
    
    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
    
    return samples


def load_samples_from_dir(json_dir: str, max_samples: int = 100) -> list:
    """Load samples from multiple JSON files in a directory."""
    json_dir = Path(json_dir)
    all_samples = []
    
    for json_file in sorted(json_dir.glob("*.json"))[:5]:  # Load from first 5 files
        with open(json_file, 'r') as f:
            samples = json.load(f)
            all_samples.extend(samples)
    
    if len(all_samples) > max_samples:
        all_samples = random.sample(all_samples, max_samples)
    
    return all_samples


def get_demo_samples() -> list:
    """Get demo samples with binding negatives."""
    return [
        {
            "sample_id": "demo_1",
            "original_caption": "A blue dog and a red car",
            "positive_components": ["blue dog", "red car"],
            "negative_components": {
                "blue dog": [{"negative": "green dog", "change_type": "attribute_change"}],
                "red car": [{"negative": "red truck", "change_type": "object_change"}]
            },
            "relations": [],
            "binding_negatives": [
                {
                    "component_1": "blue dog",
                    "component_2": "red car",
                    "binding_neg_1": "blue car",
                    "binding_neg_2": "red dog",
                    "swap_type": "noun_swap"
                }
            ]
        },
        {
            "sample_id": "demo_2",
            "original_caption": "A large wooden table with small metal chairs",
            "positive_components": ["large wooden table", "small metal chairs"],
            "negative_components": {
                "large wooden table": [{"negative": "small wooden table", "change_type": "attribute_change"}],
                "small metal chairs": [{"negative": "large metal chairs", "change_type": "attribute_change"}]
            },
            "relations": [
                {
                    "subject": "small metal chairs",
                    "relation_type": "are near",
                    "object": "large wooden table",
                    "negatives": [
                        {"relation_type": "are far from", "change_type": "antonym"}
                    ]
                }
            ],
            "binding_negatives": [
                {
                    "component_1": "large wooden table",
                    "component_2": "small metal chairs",
                    "binding_neg_1": "large wooden chairs",
                    "binding_neg_2": "small metal table",
                    "swap_type": "noun_swap"
                }
            ]
        },
        {
            "sample_id": "demo_3",
            "original_caption": "White cat on black couch near brown dog",
            "positive_components": ["white cat", "black couch", "brown dog"],
            "negative_components": {
                "white cat": [{"negative": "gray cat", "change_type": "attribute_change"}],
                "black couch": [{"negative": "white couch", "change_type": "attribute_change"}],
                "brown dog": [{"negative": "black dog", "change_type": "attribute_change"}]
            },
            "relations": [],
            "binding_negatives": [
                {
                    "component_1": "white cat",
                    "component_2": "brown dog",
                    "binding_neg_1": "white dog",
                    "binding_neg_2": "brown cat",
                    "swap_type": "noun_swap"
                },
                {
                    "component_1": "black couch",
                    "component_2": "brown dog",
                    "binding_neg_1": "black dog",
                    "binding_neg_2": "brown couch",
                    "swap_type": "noun_swap"
                }
            ]
        },
        {
            "sample_id": "demo_4_no_binding",
            "original_caption": "A restaurant with chairs",
            "positive_components": ["restaurant", "chairs"],
            "negative_components": {
                "restaurant": [{"negative": "cafe", "change_type": "object_change"}],
                "chairs": [{"negative": "tables", "change_type": "object_change"}]
            },
            "relations": [],
            "binding_negatives": []  # No binding negatives (no attributes)
        }
    ]


def test_binding_sampling(samples: list, binding_prob: float = 1.0, num_trials: int = 5):
    """Test binding negative sampling on samples."""
    
    print("=" * 80)
    print("BINDING NEGATIVE SAMPLING TEST")
    print("=" * 80)
    print(f"Binding probability: {binding_prob}")
    print(f"Trials per sample: {num_trials}")
    print(f"Total samples: {len(samples)}")
    print("=" * 80)
    
    # Create sampler with high binding probability for testing
    sampler = StructuredSampler(
        structured_relation_prob=0.5,
        use_context_in_component_pairs=True,
        binding_negative_prob=binding_prob,
    )
    
    stats = {
        "total_samples": len(samples),
        "samples_with_binding": 0,
        "binding_sampled": 0,
        "relation_sampled": 0,
        "component_sampled": 0,
        "failed": 0,
    }
    
    for sample in samples:
        sample_id = sample.get("sample_id", "unknown")
        caption = sample.get("original_caption", sample.get("caption", ""))
        binding_negatives = sample.get("binding_negatives", [])
        
        print(f"\n{'─' * 60}")
        print(f"📝 Sample: {sample_id}")
        print(f"   Caption: {caption}")
        print(f"   Components: {sample.get('positive_components', [])}")
        print(f"   Binding negatives available: {len(binding_negatives)}")
        
        if binding_negatives:
            stats["samples_with_binding"] += 1
            print(f"   Pre-generated bindings:")
            for bn in binding_negatives[:3]:  # Show first 3
                print(f"      • {bn.get('component_1')} ↔ {bn.get('component_2')}")
                print(f"        → {bn.get('binding_neg_1')}, {bn.get('binding_neg_2')}")
        
        print(f"\n   Sampling results ({num_trials} trials):")
        
        for trial in range(num_trials):
            pos, neg, meta = sampler.sample_structured_positive_and_negative(sample)
            pair_type = meta.get("pair_type", "none")
            
            if pair_type == "binding":
                stats["binding_sampled"] += 1
                emoji = "🔗"
            elif pair_type == "relation":
                stats["relation_sampled"] += 1
                emoji = "↔️"
            elif pair_type == "component":
                stats["component_sampled"] += 1
                emoji = "🧩"
            else:
                stats["failed"] += 1
                emoji = "❌"
            
            if pos and neg:
                print(f"      {emoji} [{pair_type}] pos: \"{pos}\" → neg: \"{neg}\"")
            else:
                print(f"      {emoji} [failed] No valid pair found")
    
    # Print summary statistics
    total_trials = len(samples) * num_trials
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total samples:                {stats['total_samples']}")
    print(f"Samples with binding negs:    {stats['samples_with_binding']}")
    print(f"Total trials:                 {total_trials}")
    print(f"")
    print(f"Pair type distribution:")
    print(f"  🔗 Binding:    {stats['binding_sampled']:4d} ({100*stats['binding_sampled']/total_trials:.1f}%)")
    print(f"  ↔️  Relation:   {stats['relation_sampled']:4d} ({100*stats['relation_sampled']/total_trials:.1f}%)")
    print(f"  🧩 Component:  {stats['component_sampled']:4d} ({100*stats['component_sampled']/total_trials:.1f}%)")
    print(f"  ❌ Failed:     {stats['failed']:4d} ({100*stats['failed']/total_trials:.1f}%)")
    print("=" * 80)
    
    return stats


def test_probability_sweep(samples: list):
    """Test different binding probabilities."""
    
    print("\n" + "=" * 80)
    print("BINDING PROBABILITY SWEEP")
    print("=" * 80)
    
    # Filter to samples with binding negatives
    samples_with_binding = [s for s in samples if s.get("binding_negatives")]
    
    if not samples_with_binding:
        print("No samples with binding negatives found!")
        return
    
    print(f"Testing on {len(samples_with_binding)} samples with binding negatives")
    print("")
    
    for prob in [0.0, 0.25, 0.5, 0.75, 1.0]:
        sampler = StructuredSampler(
            structured_relation_prob=0.5,
            use_context_in_component_pairs=True,
            binding_negative_prob=prob,
        )
        
        counts = {"binding": 0, "relation": 0, "component": 0, "failed": 0}
        num_trials = 100
        
        for _ in range(num_trials):
            sample = random.choice(samples_with_binding)
            pos, neg, meta = sampler.sample_structured_positive_and_negative(sample)
            pair_type = meta.get("pair_type", "failed")
            counts[pair_type] = counts.get(pair_type, 0) + 1
        
        print(f"binding_prob={prob:.2f}: "
              f"binding={counts['binding']:3d} ({100*counts['binding']/num_trials:5.1f}%), "
              f"relation={counts['relation']:3d} ({100*counts['relation']/num_trials:5.1f}%), "
              f"component={counts['component']:3d} ({100*counts['component']/num_trials:5.1f}%), "
              f"failed={counts['failed']:3d}")


def main():
    parser = argparse.ArgumentParser(description="Test binding negative sampling")
    parser.add_argument("--json_file", type=str, help="Path to a single JSON file")
    parser.add_argument("--json_dir", type=str, help="Path to directory with JSON files")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--num_trials", type=int, default=5, help="Trials per sample")
    parser.add_argument("--binding_prob", type=float, default=1.0, help="Binding probability")
    parser.add_argument("--demo", action="store_true", help="Run with demo samples")
    parser.add_argument("--sweep", action="store_true", help="Run probability sweep test")
    
    args = parser.parse_args()
    
    # Load samples
    if args.demo:
        samples = get_demo_samples()
        print("Using demo samples")
    elif args.json_file:
        samples = load_samples_from_file(args.json_file, args.num_samples)
        print(f"Loaded {len(samples)} samples from {args.json_file}")
    elif args.json_dir:
        samples = load_samples_from_dir(args.json_dir, args.num_samples)
        print(f"Loaded {len(samples)} samples from {args.json_dir}")
    else:
        # Default to demo
        samples = get_demo_samples()
        print("No input specified, using demo samples")
    
    # Run tests
    test_binding_sampling(samples, args.binding_prob, args.num_trials)
    
    if args.sweep:
        test_probability_sweep(samples)


if __name__ == "__main__":
    main()
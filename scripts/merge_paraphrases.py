#!/usr/bin/env python3
# filepath: /mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally/merge_paraphrases.py
"""
Merge paraphrased captions with processed COCO JSON files.

This script:
1. Reads processed JSON files from swap_pos_json/coco/
2. Reads paraphrase mapping from datasets/COCO/zero.json
3. Matches captions by image_id and caption index
4. Outputs merged JSON with paraphrased_caption field

Usage:
    python merge_paraphrases.py \
        --input-dir swap_pos_json/coco/ \
        --paraphrase-file datasets/COCO/zero.json \
        --output-dir swap_pos_json/coco_with_paraphrase/
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm


def extract_image_id_from_path(image_path: str) -> Optional[str]:
    """
    Extract COCO image ID from image path.
    
    Examples:
        "datasets/COCO/train2014/COCO_train2014_000000057870.jpg" -> "57870"
        "datasets/COCO/val2014/COCO_val2014_000000384029.jpg" -> "384029"
    """
    # Match pattern like COCO_train2014_000000057870.jpg
    match = re.search(r'COCO_(?:train|val)\d{4}_0*(\d+)\.jpg', image_path)
    if match:
        return match.group(1)
    
    # Try simpler pattern - just extract numbers before .jpg
    match = re.search(r'(\d+)\.jpg$', image_path)
    if match:
        # Remove leading zeros
        return str(int(match.group(1)))
    
    return None


def load_paraphrase_mapping(paraphrase_file: str) -> Dict[str, List[str]]:
    """
    Load paraphrase mapping from JSON file.
    
    Expected format:
    {
        "57870": [
            "The dining establishment features contemporary wooden tables...",
            "A spacious dining table accompanied by charming...",
            ...
        ],
        "384029": [...],
        ...
    }
    
    Returns:
        Dict mapping image_id -> list of paraphrased captions
    """
    print(f"Loading paraphrase file: {paraphrase_file}")
    
    with open(paraphrase_file, 'r', encoding='utf-8') as f:
        paraphrase_data = json.load(f)
    
    print(f"  Loaded {len(paraphrase_data)} image entries with paraphrases")
    
    # Verify format
    sample_key = next(iter(paraphrase_data.keys()))
    sample_value = paraphrase_data[sample_key]
    print(f"  Sample entry: image_id={sample_key}, num_paraphrases={len(sample_value)}")
    
    return paraphrase_data


def load_processed_jsons(input_dir: str) -> List[Dict[str, Any]]:
    """
    Load all processed JSON files from input directory.
    
    Returns:
        List of all samples from all JSON files
    """
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    all_samples = []
    
    for json_file in tqdm(json_files, desc="Loading JSON files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, list):
                for sample in data:
                    sample['_source_file'] = json_file.name
                all_samples.extend(data)
            elif isinstance(data, dict):
                # If it's a dict with samples as values
                for key, sample in data.items():
                    if isinstance(sample, dict):
                        sample['_source_file'] = json_file.name
                        all_samples.append(sample)
        except Exception as e:
            print(f"  Warning: Failed to load {json_file}: {e}")
    
    print(f"Loaded {len(all_samples)} total samples")
    return all_samples


def group_samples_by_image(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group samples by image_id, preserving order.
    
    This is needed because COCO has multiple captions per image,
    and we need to match them with paraphrases by index.
    """
    grouped = defaultdict(list)
    
    for sample in samples:
        image_path = sample.get('image_path', '')
        image_id = extract_image_id_from_path(image_path)
        
        if image_id:
            grouped[image_id].append(sample)
        else:
            print(f"  Warning: Could not extract image_id from: {image_path}")
    
    print(f"Grouped samples into {len(grouped)} unique images")
    
    return grouped


def merge_paraphrases(
    samples: List[Dict[str, Any]],
    paraphrase_mapping: Dict[str, List[str]],
    strict: bool = False
) -> List[Dict[str, Any]]:
    """
    Merge paraphrased captions into samples.
    
    Matching logic:
    1. Extract image_id from sample's image_path
    2. Group samples by image_id (maintaining order)
    3. Match each sample with corresponding paraphrase by index
    
    Args:
        samples: List of processed samples
        paraphrase_mapping: Dict mapping image_id -> list of paraphrases
        strict: If True, raise error on missing paraphrases
        
    Returns:
        List of samples with 'paraphrased_caption' field added
    """
    # Group samples by image_id
    grouped = group_samples_by_image(samples)
    
    merged_samples = []
    stats = {
        'total': 0,
        'matched': 0,
        'missing_image': 0,
        'index_out_of_range': 0,
    }
    
    for image_id, image_samples in tqdm(grouped.items(), desc="Merging paraphrases"):
        paraphrases = paraphrase_mapping.get(image_id, [])
        
        for idx, sample in enumerate(image_samples):
            stats['total'] += 1
            
            # Create a copy to avoid modifying original
            merged_sample = sample.copy()
            
            if not paraphrases:
                stats['missing_image'] += 1
                merged_sample['paraphrased_caption'] = None
                if strict:
                    raise ValueError(f"No paraphrases found for image_id: {image_id}")
            elif idx >= len(paraphrases):
                stats['index_out_of_range'] += 1
                merged_sample['paraphrased_caption'] = None
                print(f"  Warning: Image {image_id} has {len(image_samples)} samples but only {len(paraphrases)} paraphrases")
            else:
                stats['matched'] += 1
                merged_sample['paraphrased_caption'] = paraphrases[idx]
            
            # Remove temporary field
            merged_sample.pop('_source_file', None)
            
            merged_samples.append(merged_sample)
    
    # Print statistics
    print("\n" + "=" * 50)
    print("MERGE STATISTICS")
    print("=" * 50)
    print(f"Total samples:           {stats['total']}")
    print(f"Successfully matched:    {stats['matched']} ({100*stats['matched']/stats['total']:.1f}%)")
    print(f"Missing image in paraph: {stats['missing_image']}")
    print(f"Index out of range:      {stats['index_out_of_range']}")
    print("=" * 50)
    
    return merged_samples


def save_merged_samples(
    samples: List[Dict[str, Any]],
    output_dir: str,
    samples_per_file: int = 50000
):
    """
    Save merged samples to output directory.
    
    Args:
        samples: List of merged samples
        output_dir: Output directory path
        samples_per_file: Number of samples per output file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split into chunks if needed
    num_files = (len(samples) + samples_per_file - 1) // samples_per_file
    
    for i in range(num_files):
        start_idx = i * samples_per_file
        end_idx = min((i + 1) * samples_per_file, len(samples))
        chunk = samples[start_idx:end_idx]
        
        if num_files == 1:
            output_file = output_path / "coco_with_paraphrases.json"
        else:
            output_file = output_path / f"coco_with_paraphrases_{i:03d}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunk)} samples to {output_file}")


def verify_merge(samples: List[Dict[str, Any]], num_samples: int = 5):
    """
    Print some sample entries to verify the merge.
    """
    print("\n" + "=" * 50)
    print("SAMPLE VERIFICATION")
    print("=" * 50)
    
    # Get samples that have paraphrases
    samples_with_paraphrase = [s for s in samples if s.get('paraphrased_caption')]
    
    for i, sample in enumerate(samples_with_paraphrase[:num_samples]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Image path: {sample.get('image_path', 'N/A')}")
        print(f"Original:   {sample.get('original_caption', 'N/A')[:80]}...")
        print(f"Paraphrase: {sample.get('paraphrased_caption', 'N/A')[:80]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Merge paraphrased captions with processed COCO JSON files"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="swap_pos_json/coco/",
        help="Directory containing processed JSON files"
    )
    parser.add_argument(
        "--paraphrase-file",
        type=str,
        default="datasets/COCO/zero.json",
        help="Path to paraphrase JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="swap_pos_json/coco_with_paraphrase/",
        help="Output directory for merged JSON files"
    )
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=50000,
        help="Number of samples per output file"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error if paraphrase is missing"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Print sample entries after merge"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("PARAPHRASE MERGER")
    print("=" * 50)
    print(f"Input directory:   {args.input_dir}")
    print(f"Paraphrase file:   {args.paraphrase_file}")
    print(f"Output directory:  {args.output_dir}")
    print("=" * 50 + "\n")
    
    # Load data
    paraphrase_mapping = load_paraphrase_mapping(args.paraphrase_file)
    samples = load_processed_jsons(args.input_dir)
    
    # Merge
    merged_samples = merge_paraphrases(
        samples, 
        paraphrase_mapping, 
        strict=args.strict
    )
    
    # Verify
    if args.verify:
        verify_merge(merged_samples)
    
    # Save
    save_merged_samples(
        merged_samples,
        args.output_dir,
        samples_per_file=args.samples_per_file
    )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
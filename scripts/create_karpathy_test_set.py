#!/usr/bin/env python3
"""
Create a test set for Karpathy split by sampling images from COCO train2014/val2014
that are NOT already in the Karpathy split (dataset_coco.json).

This script:
1. Loads the Karpathy split (dataset_coco.json) to get existing image IDs
2. Loads COCO annotations (train2014 and val2014) to get all available images with captions
3. Filters out images already in Karpathy split
4. Randomly samples N images from the remaining pool
5. Outputs in the same format as dataset_coco.json

Usage:
    python create_karpathy_test_set.py \
        --karpathy_json /path/to/dataset_coco.json \
        --coco_train_captions /path/to/captions_train2014.json \
        --coco_val_captions /path/to/captions_val2014.json \
        --output /path/to/output_test_set.json \
        --n_samples 5000 \
        --seed 42
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_karpathy_split(karpathy_json_path):
    """Load Karpathy split and extract all image filenames/IDs."""
    print(f"Loading Karpathy split from {karpathy_json_path}...")
    
    with open(karpathy_json_path, 'r') as f:
        data = json.load(f)
    
    # Extract all image filenames and IDs
    existing_filenames = set()
    existing_ids = set()
    
    for img in data.get('images', []):
        filename = img.get('filename', '')
        if filename:
            existing_filenames.add(filename)
        
        # Also extract image ID from filename (e.g., COCO_val2014_000000391895.jpg -> 391895)
        if filename.startswith('COCO_'):
            try:
                img_id = int(filename.split('_')[-1].replace('.jpg', ''))
                existing_ids.add(img_id)
            except (ValueError, IndexError):
                pass
    
    print(f"  Found {len(existing_filenames)} images in Karpathy split")
    print(f"  Extracted {len(existing_ids)} image IDs")
    
    return existing_filenames, existing_ids, data


def load_coco_captions(captions_json_path, split_name):
    """Load COCO captions annotation file."""
    print(f"Loading COCO {split_name} captions from {captions_json_path}...")
    
    with open(captions_json_path, 'r') as f:
        data = json.load(f)
    
    # Build image ID to image info mapping
    images = {img['id']: img for img in data.get('images', [])}
    
    # Build image ID to captions mapping
    captions_by_image = defaultdict(list)
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        caption = ann['caption']
        captions_by_image[img_id].append({
            'id': ann['id'],
            'caption': caption
        })
    
    print(f"  Found {len(images)} images with {len(data.get('annotations', []))} captions")
    
    return images, captions_by_image, split_name


def create_karpathy_format_entry(img_info, captions, split_name, imgid, sentid_start):
    """Create a single image entry in Karpathy format."""
    
    # Determine filepath based on source
    if 'train' in split_name.lower():
        filepath = 'train2014'
    else:
        filepath = 'val2014'
    
    # Get filename
    filename = img_info.get('file_name', '')
    
    # Create sentences
    sentences = []
    sentids = []
    
    for i, cap in enumerate(captions):
        sentid = sentid_start + i
        sentids.append(sentid)
        
        # Tokenize (simple whitespace + lowercase)
        raw_caption = cap['caption'].strip()
        tokens = raw_caption.lower().replace('.', ' .').replace(',', ' ,').replace('!', ' !').replace('?', ' ?').split()
        # Clean up tokens
        tokens = [t.strip() for t in tokens if t.strip()]
        
        sentences.append({
            'tokens': tokens,
            'raw': raw_caption,
            'imgid': imgid,
            'sentid': sentid
        })
    
    entry = {
        'filepath': filepath,
        'sentids': sentids,
        'filename': filename,
        'imgid': imgid,
        'split': 'test',  # Mark as test split
        'sentences': sentences,
        'cocoid': img_info['id']  # Keep original COCO ID for reference
    }
    
    return entry, sentid_start + len(captions)


def main():
    parser = argparse.ArgumentParser(description='Create test set from COCO images not in Karpathy split')
    
    parser.add_argument('--karpathy_json', type=str, required=True,
                        help='Path to dataset_coco.json (Karpathy split)')
    parser.add_argument('--coco_train_captions', type=str, default=None,
                        help='Path to COCO captions_train2014.json')
    parser.add_argument('--coco_val_captions', type=str, default=None,
                        help='Path to COCO captions_val2014.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON path')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of images to sample (default: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--source', type=str, default='both',
                        choices=['train', 'val', 'both'],
                        help='Source to sample from: train, val, or both')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load Karpathy split
    existing_filenames, existing_ids, karpathy_data = load_karpathy_split(args.karpathy_json)
    
    # Load COCO captions
    all_candidates = []  # List of (img_info, captions, split_name)
    
    if args.source in ['train', 'both'] and args.coco_train_captions:
        train_images, train_captions, train_split = load_coco_captions(
            args.coco_train_captions, 'train2014'
        )
        
        # Filter out images already in Karpathy split
        for img_id, img_info in train_images.items():
            filename = img_info.get('file_name', '')
            if filename not in existing_filenames and img_id not in existing_ids:
                caps = train_captions.get(img_id, [])
                if caps:  # Only include if has captions
                    all_candidates.append((img_info, caps, train_split))
        
        print(f"  {len([c for c in all_candidates if c[2] == 'train2014'])} train images available (not in Karpathy)")
    
    if args.source in ['val', 'both'] and args.coco_val_captions:
        val_images, val_captions, val_split = load_coco_captions(
            args.coco_val_captions, 'val2014'
        )
        
        # Filter out images already in Karpathy split
        for img_id, img_info in val_images.items():
            filename = img_info.get('file_name', '')
            if filename not in existing_filenames and img_id not in existing_ids:
                caps = val_captions.get(img_id, [])
                if caps:  # Only include if has captions
                    all_candidates.append((img_info, caps, val_split))
        
        print(f"  {len([c for c in all_candidates if c[2] == 'val2014'])} val images available (not in Karpathy)")
    
    print(f"\nTotal available candidates: {len(all_candidates)}")
    
    if len(all_candidates) == 0:
        print("ERROR: No candidate images found. Check your file paths.")
        return
    
    # Sample
    n_to_sample = min(args.n_samples, len(all_candidates))
    print(f"Sampling {n_to_sample} images...")
    
    sampled = random.sample(all_candidates, n_to_sample)
    
    # Create output in Karpathy format
    output_images = []
    sentid_counter = 0
    
    for imgid, (img_info, captions, split_name) in enumerate(tqdm(sampled, desc="Creating entries")):
        entry, sentid_counter = create_karpathy_format_entry(
            img_info, captions, split_name, imgid, sentid_counter
        )
        output_images.append(entry)
    
    # Create output structure
    output_data = {
        'images': output_images,
        'dataset': 'coco_test_holdout'
    }
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("OUTPUT STATISTICS")
    print(f"{'='*60}")
    print(f"Total images: {len(output_images)}")
    print(f"Total sentences: {sentid_counter}")
    print(f"Average captions per image: {sentid_counter / len(output_images):.2f}")
    
    from_train = sum(1 for img in output_images if img['filepath'] == 'train2014')
    from_val = sum(1 for img in output_images if img['filepath'] == 'val2014')
    print(f"From train2014: {from_train}")
    print(f"From val2014: {from_val}")
    
    print(f"\nSaved to: {output_path}")
    
    # Show sample entry
    print(f"\n{'='*60}")
    print("SAMPLE ENTRY")
    print(f"{'='*60}")
    sample = output_images[0]
    print(json.dumps(sample, indent=2)[:1000] + "...")


if __name__ == '__main__':
    main()

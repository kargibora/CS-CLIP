#!/usr/bin/env python
"""
Process COCO negatives JSON files with multiprocessing support:
1. Filter out invalid samples (missing positive/negative content)
2. Generate swap negatives using TextShuffler
3. Validate image paths exist
4. Output cleaned JSON ready for training/testing

Usage:
    # Process a folder of JSON files in parallel
    python scripts/process_coco_negatives.py \
        --input_dir /path/to/coco_jsons \
        --output_dir /path/to/processed \
        --image_root /path/to/coco/images \
        --num_processes 8

    # Process a single JSON file
    python scripts/process_coco_negatives.py \
        --input_json /path/to/coco_negatives.json \
        --output_json /path/to/coco_negatives_processed.json \
        --image_root /path/to/coco/images

The input JSON should have samples with structure:
{
    "sample_id": "coco_61199_822730",
    "original_caption": "A red double decker bus...",
    "positive_components": ["red double decker bus", "stop"],
    "negative_components": {...},
    "relations": [
        {
            "subject": "cat",
            "predicate": "sitting on", 
            "object": "mat",
            "negative_relations": [
                {"subject": "cat", "predicate": "under", "object": "mat"},
                {"subject": "mat", "predicate": "sitting on", "object": "cat"}
            ]
        },
        ...
    ],
    "image_path": "datasets/COCO/train2014/COCO_train2014_000000237309.jpg"
}

Note: negative_relations are now embedded WITHIN each relation object, not as a separate top-level field.
"""
import os
import sys
import json
import glob
import argparse
import logging
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.perturbations import TextShuffler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Global variable to hold TextShuffler instance per worker process
_shuffler = None


def _init_worker():
    """Initialize TextShuffler once per worker process."""
    global _shuffler
    print(f"   [Worker {os.getpid()}] Initializing TextShuffler...")
    _shuffler = TextShuffler()


def get_shuffler():
    """Get or initialize the global TextShuffler instance."""
    global _shuffler
    if _shuffler is None:
        print("Initializing TextShuffler...")
        _shuffler = TextShuffler()
    return _shuffler


def is_valid_sample(item):
    """
    Check if a sample has at least one positive and one negative option.
    
    Requirements:
    1. Must have at least one of: positive_components OR relations
    2. Must have at least one of: negative_components OR negative_relations (embedded in relations)
    
    Note: negative_relations are now embedded WITHIN each relation object, not as a separate top-level field.
    
    Args:
        item: JSON sample dict
        
    Returns:
        bool: True if sample meets requirements, False otherwise
    """
    # Check for positive content
    positive_components = item.get("positive_components", [])
    relations = item.get("relations", [])
    has_positive = (positive_components and len(positive_components) > 0) or \
                   (relations and len(relations) > 0)
    
    # Check for negative content
    negative_components = item.get("negative_components", {})
    
    # negative_components is a dict, check if it has any non-empty values
    has_negative_components = False
    if negative_components:
        for comp, negs in negative_components.items():
            if negs and len(negs) > 0:
                has_negative_components = True
                break
    
    # Check for negative_relations embedded within each relation object
    has_negative_relations = False
    if relations:
        for relation in relations:
            neg_rels = relation.get("negative_relations", [])
            if neg_rels and len(neg_rels) > 0:
                has_negative_relations = True
                break
    
    # Also check for legacy top-level negative_relations (backward compatibility)
    top_level_neg_relations = item.get("negative_relations", [])
    if top_level_neg_relations and len(top_level_neg_relations) > 0:
        has_negative_relations = True
    
    has_negative = has_negative_components or has_negative_relations
    
    return has_positive and has_negative


def generate_swap_negatives(caption: str, shuffler: TextShuffler) -> list:
    """
    Generate swap negatives for a caption using TextShuffler.
    
    Args:
        caption: Original caption text
        shuffler: TextShuffler instance
        
    Returns:
        List of dicts with 'swap_type' and 'negative' keys
    """
    shuffler_methods = [
        ('shuffle_nouns_and_adj', shuffler.shuffle_nouns_and_adj),
        ('shuffle_allbut_nouns_and_adj', shuffler.shuffle_allbut_nouns_and_adj),
        ('shuffle_within_trigrams', shuffler.shuffle_within_trigrams),
        ('shuffle_trigrams', shuffler.shuffle_trigrams),
    ]
    
    swap_negatives = []
    for swap_type, func in shuffler_methods:
        try:
            shuffled = func(caption)
            # Only add if it's actually different from the original
            if shuffled != caption:
                swap_negatives.append({
                    'swap_type': swap_type,
                    'negative': shuffled
                })
        except Exception as e:
            logging.debug(f"Shuffle failed for '{caption[:50]}...': {e}")
            continue
    
    return swap_negatives


def generate_swap_negatives_batch(captions: list, shuffler: TextShuffler, batch_size: int = 1000) -> list:
    """
    Generate swap negatives for multiple captions in batch (FASTER).
    
    Uses spaCy's pipe() for batch processing which is much faster than
    processing captions one at a time.
    
    Args:
        captions: List of caption texts
        shuffler: TextShuffler instance
        batch_size: Number of captions to process at once in spaCy pipe
        
    Returns:
        List of lists, where each inner list contains dicts with 'swap_type' and 'negative' keys
    """
    import random
    
    if not captions:
        return []
    
    # Pre-process all captions with spaCy in batch
    docs = list(shuffler.nlp.pipe(captions, batch_size=batch_size))
    
    all_swap_negatives = []
    
    for caption, doc in zip(captions, docs):
        swap_negatives = []
        
        # Method 1: shuffle_nouns_and_adj
        try:
            nouns_and_adj = [token for token in doc if token.pos_ in ['NOUN', 'ADJ', 'PROPN']]
            if len(nouns_and_adj) > 1:
                shuffled_tokens = list(doc)
                positions = [i for i, token in enumerate(doc) if token.pos_ in ['NOUN', 'ADJ', 'PROPN']]
                shuffled_nouns_adj = random.sample(nouns_and_adj, len(nouns_and_adj))
                for pos, token in zip(positions, shuffled_nouns_adj):
                    shuffled_tokens[pos] = token
                shuffled = ' '.join([t.text for t in shuffled_tokens])
                if shuffled != caption:
                    swap_negatives.append({
                        'swap_type': 'shuffle_nouns_and_adj',
                        'negative': shuffled
                    })
        except Exception:
            pass
        
        # Method 2: shuffle_allbut_nouns_and_adj
        try:
            others = [token for token in doc if token.pos_ not in ['NOUN', 'ADJ', 'PROPN']]
            if len(others) > 1:
                shuffled_tokens = list(doc)
                positions = [i for i, token in enumerate(doc) if token.pos_ not in ['NOUN', 'ADJ', 'PROPN']]
                shuffled_others = random.sample(others, len(others))
                for pos, token in zip(positions, shuffled_others):
                    shuffled_tokens[pos] = token
                shuffled = ' '.join([t.text for t in shuffled_tokens])
                if shuffled != caption:
                    swap_negatives.append({
                        'swap_type': 'shuffle_allbut_nouns_and_adj',
                        'negative': shuffled
                    })
        except Exception:
            pass
        
        # Method 3 & 4: Use original methods (these don't benefit as much from batching)
        try:
            shuffled = shuffler.shuffle_within_trigrams(caption)
            if shuffled != caption:
                swap_negatives.append({
                    'swap_type': 'shuffle_within_trigrams',
                    'negative': shuffled
                })
        except Exception:
            pass
        
        try:
            shuffled = shuffler.shuffle_trigrams(caption)
            if shuffled != caption:
                swap_negatives.append({
                    'swap_type': 'shuffle_trigrams',
                    'negative': shuffled
                })
        except Exception:
            pass
        
        all_swap_negatives.append(swap_negatives)
    
    return all_swap_negatives


def process_single_json_worker(args):
    """
    Process a single JSON file. Designed to be called by multiprocessing.Pool.
    
    Args:
        args: Tuple of (input_path, output_path, image_root, batch_size, validate_images, 
                        add_swap_negatives, filter_invalid)
        
    Returns:
        dict: Statistics about the processing
    """
    global _shuffler
    
    (input_path, output_path, image_root, batch_size, 
     validate_images, add_swap_negatives, filter_invalid) = args
    
    basename = os.path.basename(input_path)
    
    if not os.path.exists(input_path):
        return {
            'filename': basename,
            'status': 'error',
            'message': f"File not found: {input_path}"
        }
    
    if os.path.getsize(input_path) == 0:
        return {
            'filename': basename,
            'status': 'error',
            'message': f"File is empty: {input_path}"
        }
    
    print(f"🔹 [Worker {os.getpid()}] Processing {basename}...")
    
    # Use the global shuffler initialized per worker process
    shuffler = _shuffler
    
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {
            'filename': basename,
            'status': 'error',
            'message': f"Invalid JSON: {e}"
        }
    
    # Process samples
    stats = {
        'filename': basename,
        'status': 'success',
        'total_input': len(data),
        'invalid_skipped': 0,
        'missing_image_skipped': 0,
        'duplicates_skipped': 0,
        'total_swap_negatives': 0,
        'swap_negatives_failed': 0,
    }
    
    seen_sample_ids = set()
    samples_to_process = []
    
    # First pass: filter and validate
    for item in data:
        sample_id = item.get("sample_id", "")
        
        # Skip duplicates
        if sample_id in seen_sample_ids:
            stats['duplicates_skipped'] += 1
            continue
        seen_sample_ids.add(sample_id)
        
        # Filter invalid samples
        if filter_invalid and not is_valid_sample(item):
            stats['invalid_skipped'] += 1
            continue
        
        # Validate image exists
        if validate_images and image_root:
            image_path = item.get("image_path", "")
            if image_path:
                if not os.path.isabs(image_path):
                    full_path = os.path.join(image_root, image_path)
                else:
                    full_path = image_path
                
                if not os.path.exists(full_path):
                    stats['missing_image_skipped'] += 1
                    continue
        
        samples_to_process.append(item)
    
    # Second pass: generate swap negatives in batch
    if add_swap_negatives and samples_to_process and shuffler is not None:
        captions = [item.get('original_caption', '') for item in samples_to_process]
        
        for batch_start in range(0, len(captions), batch_size):
            batch_end = min(batch_start + batch_size, len(captions))
            batch_captions = captions[batch_start:batch_end]
            
            non_empty_indices = []
            non_empty_captions = []
            for i, caption in enumerate(batch_captions):
                if caption:
                    non_empty_indices.append(batch_start + i)
                    non_empty_captions.append(caption)
                else:
                    samples_to_process[batch_start + i]['swap_negatives'] = []
                    stats['swap_negatives_failed'] += 1
            
            if non_empty_captions:
                try:
                    batch_swap_negatives = generate_swap_negatives_batch(
                        non_empty_captions, shuffler, batch_size=batch_size
                    )
                    
                    for local_idx, swap_negs in zip(non_empty_indices, batch_swap_negatives):
                        samples_to_process[local_idx]['swap_negatives'] = swap_negs
                        stats['total_swap_negatives'] += len(swap_negs)
                        
                except Exception as e:
                    logging.warning(f"Batch processing failed, using fallback: {e}")
                    for local_idx, caption in zip(non_empty_indices, non_empty_captions):
                        try:
                            swap_negs = generate_swap_negatives(caption, shuffler)
                            samples_to_process[local_idx]['swap_negatives'] = swap_negs
                            stats['total_swap_negatives'] += len(swap_negs)
                        except Exception:
                            samples_to_process[local_idx]['swap_negatives'] = []
                            stats['swap_negatives_failed'] += 1
    
    stats['total_output'] = len(samples_to_process)
    stats['retention_rate'] = (len(samples_to_process) / len(data) * 100) if data else 0
    stats['avg_swap_negatives'] = (
        stats['total_swap_negatives'] / len(samples_to_process) 
        if samples_to_process else 0
    )
    
    # Save output JSON
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(samples_to_process, f)
    
    print(f"✅ [Worker {os.getpid()}] Finished {basename}: "
          f"{stats['total_output']}/{stats['total_input']} samples "
          f"({stats['retention_rate']:.1f}%)")
    
    return stats


def process_coco_folder(
    input_dir: str,
    output_dir: str,
    image_root: str = None,
    file_pattern: str = "*.json",
    num_processes: int = None,
    batch_size: int = 1000,
    validate_images: bool = True,
    add_swap_negatives: bool = True,
    filter_invalid: bool = True,
    overwrite: bool = True,
):
    """
    Process all COCO JSON files in a folder using multiprocessing.
    
    Args:
        input_dir: Directory containing COCO JSON files
        output_dir: Directory to save processed JSON files
        image_root: Root directory for COCO images (for validation)
        file_pattern: Glob pattern for input files (default: "*.json")
        num_processes: Number of worker processes (default: CPU count)
        batch_size: Batch size for swap negatives generation
        validate_images: Whether to check if image files exist
        add_swap_negatives: Whether to generate swap negatives
        filter_invalid: Whether to filter invalid samples
        overwrite: If False, skip already processed files
        
    Returns:
        dict: Combined statistics from all files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if num_processes is None:
        num_processes = cpu_count()
    
    # Find all matching JSON files
    input_files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
    
    if not input_files:
        print(f"❌ No files matching '{file_pattern}' found in {input_dir}")
        return {}
    
    print(f"📂 Found {len(input_files)} JSON files in {input_dir}")
    
    # Filter out already processed files if overwrite=False
    skipped_existing = []
    if not overwrite:
        filtered_files = []
        for input_path in input_files:
            basename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, basename)
            marker_file = os.path.join(output_dir, f".{basename}.done")
            
            if os.path.exists(marker_file) or os.path.exists(output_path):
                skipped_existing.append(basename)
            else:
                filtered_files.append(input_path)
        
        if skipped_existing:
            print(f"⏭️  Skipping {len(skipped_existing)} already processed files")
        
        input_files = filtered_files
        
        if not input_files:
            print("✅ All files already processed. Nothing to do.")
            return {}
    
    print(f"🚀 Starting parallel processing with {num_processes} processes...")
    print(f"   Batch size for swap negatives: {batch_size}")
    
    # Prepare arguments for each worker
    args_list = []
    for input_path in input_files:
        basename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, basename)
        args_list.append((
            input_path, output_path, image_root, batch_size,
            validate_images, add_swap_negatives, filter_invalid
        ))
    
    # Process in parallel with worker initializer
    with Pool(processes=num_processes, initializer=_init_worker) as pool:
        results = pool.map(process_single_json_worker, args_list)
    
    # Create marker files for successfully processed files
    if not overwrite:
        for result in results:
            if result.get('status') == 'success':
                marker_file = os.path.join(output_dir, f".{result['filename']}.done")
                with open(marker_file, 'w') as f:
                    f.write(f"Processed on {datetime.now().isoformat()}\n")
    
    # Print summary
    print("\n" + "="*80)
    print("🎉 ALL FILES PROCESSED - SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'error']
    
    if skipped_existing:
        print(f"⏭️  Skipped (already processed): {len(skipped_existing)} files")
    
    if successful:
        total_input = sum(r['total_input'] for r in successful)
        total_output = sum(r['total_output'] for r in successful)
        total_duplicates = sum(r['duplicates_skipped'] for r in successful)
        total_invalid = sum(r['invalid_skipped'] for r in successful)
        total_missing = sum(r['missing_image_skipped'] for r in successful)
        total_swap_negatives = sum(r['total_swap_negatives'] for r in successful)
        avg_swap_negs = total_swap_negatives / total_output if total_output > 0 else 0
        total_swap_failed = sum(r['swap_negatives_failed'] for r in successful)
        retention = (total_output / total_input * 100) if total_input > 0 else 0
        
        print(f"✅ Successfully processed: {len(successful)} files")
        print(f"📝 Total samples: {total_output}/{total_input} ({retention:.1f}% retention)")
        print(f"🔄 Total swap negatives: {total_swap_negatives} (avg {avg_swap_negs:.2f}/sample)")
        if total_swap_failed > 0:
            print(f"⚠️  Swap negatives failed: {total_swap_failed}")
        print(f"⚠️  Duplicates skipped: {total_duplicates}")
        print(f"⚠️  Invalid samples skipped: {total_invalid}")
        if validate_images:
            print(f"⚠️  Missing images skipped: {total_missing}")
    
    if failed:
        print(f"\n❌ Failed to process: {len(failed)} files")
        for r in failed:
            print(f"   - {r['filename']}: {r['message']}")
    
    print("="*80)
    
    return {
        'successful': len(successful),
        'failed': len(failed),
        'skipped': len(skipped_existing),
        'results': results
    }


def process_coco_negatives(
    input_json: str,
    output_json: str,
    image_root: str = None,
    batch_size: int = 1000,
    validate_images: bool = True,
    add_swap_negatives: bool = True,
    filter_invalid: bool = True,
):
    """
    Process COCO negatives JSON file.
    
    Args:
        input_json: Path to input JSON file with COCO negatives
        output_json: Path to output processed JSON file
        image_root: Root directory for COCO images (for validation)
        batch_size: Batch size for swap negatives generation
        validate_images: Whether to check if image files exist
        add_swap_negatives: Whether to generate swap negatives
        filter_invalid: Whether to filter out samples without positive/negative content
    
    Returns:
        dict: Statistics about the processing
    """
    print(f"🔹 Processing COCO negatives: {input_json}")
    
    # Load input JSON
    print(f"   Loading JSON...")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"   Loaded {len(data)} samples")
    
    # Initialize shuffler if needed
    shuffler = None
    if add_swap_negatives:
        shuffler = get_shuffler()
    
    # Process samples
    valid_samples = []
    stats = {
        'total_input': len(data),
        'invalid_skipped': 0,
        'missing_image_skipped': 0,
        'duplicates_skipped': 0,
        'total_swap_negatives': 0,
        'swap_negatives_failed': 0,
    }
    
    seen_sample_ids = set()
    samples_to_process = []
    
    # First pass: filter and validate
    print(f"   Filtering and validating samples...")
    for item in tqdm(data, desc="Filtering"):
        sample_id = item.get("sample_id", "")
        
        # Skip duplicates
        if sample_id in seen_sample_ids:
            stats['duplicates_skipped'] += 1
            continue
        seen_sample_ids.add(sample_id)
        
        # Filter invalid samples
        if filter_invalid and not is_valid_sample(item):
            stats['invalid_skipped'] += 1
            continue
        
        # Validate image exists
        if validate_images and image_root:
            image_path = item.get("image_path", "")
            if image_path:
                # Handle relative paths
                if not os.path.isabs(image_path):
                    full_path = os.path.join(image_root, image_path)
                else:
                    full_path = image_path
                
                if not os.path.exists(full_path):
                    stats['missing_image_skipped'] += 1
                    continue
        
        samples_to_process.append(item)
    
    print(f"   {len(samples_to_process)} samples passed filtering")
    
    # Second pass: generate swap negatives in batch
    if add_swap_negatives and samples_to_process:
        print(f"   Generating swap negatives for {len(samples_to_process)} samples...")
        
        captions = [item.get('original_caption', '') for item in samples_to_process]
        
        # Process in batches
        for batch_start in tqdm(range(0, len(captions), batch_size), desc="Generating swaps"):
            batch_end = min(batch_start + batch_size, len(captions))
            batch_captions = captions[batch_start:batch_end]
            
            # Filter out empty captions
            non_empty_indices = []
            non_empty_captions = []
            for i, caption in enumerate(batch_captions):
                if caption:
                    non_empty_indices.append(batch_start + i)
                    non_empty_captions.append(caption)
                else:
                    samples_to_process[batch_start + i]['swap_negatives'] = []
                    stats['swap_negatives_failed'] += 1
            
            if non_empty_captions:
                try:
                    batch_swap_negatives = generate_swap_negatives_batch(
                        non_empty_captions, shuffler, batch_size=batch_size
                    )
                    
                    for local_idx, swap_negs in zip(non_empty_indices, batch_swap_negatives):
                        samples_to_process[local_idx]['swap_negatives'] = swap_negs
                        stats['total_swap_negatives'] += len(swap_negs)
                        
                except Exception as e:
                    logging.warning(f"Batch processing failed, using fallback: {e}")
                    # Fallback: process individually
                    for local_idx, caption in zip(non_empty_indices, non_empty_captions):
                        try:
                            swap_negs = generate_swap_negatives(caption, shuffler)
                            samples_to_process[local_idx]['swap_negatives'] = swap_negs
                            stats['total_swap_negatives'] += len(swap_negs)
                        except Exception as e2:
                            logging.debug(f"Failed to generate swap negatives: {e2}")
                            samples_to_process[local_idx]['swap_negatives'] = []
                            stats['swap_negatives_failed'] += 1
    
    valid_samples = samples_to_process
    stats['total_output'] = len(valid_samples)
    stats['retention_rate'] = (len(valid_samples) / len(data) * 100) if data else 0
    stats['avg_swap_negatives'] = (
        stats['total_swap_negatives'] / len(valid_samples) 
        if valid_samples else 0
    )
    
    # Save output JSON
    print(f"   Saving processed JSON to: {output_json}")
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(valid_samples, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 PROCESSING SUMMARY")
    print("="*60)
    print(f"   Input samples:          {stats['total_input']}")
    print(f"   Output samples:         {stats['total_output']}")
    print(f"   Retention rate:         {stats['retention_rate']:.1f}%")
    print(f"   ─────────────────────────────────────")
    print(f"   Duplicates skipped:     {stats['duplicates_skipped']}")
    print(f"   Invalid skipped:        {stats['invalid_skipped']}")
    print(f"   Missing images skipped: {stats['missing_image_skipped']}")
    if add_swap_negatives:
        print(f"   ─────────────────────────────────────")
        print(f"   Swap negatives added:   {stats['total_swap_negatives']}")
        print(f"   Average per sample:     {stats['avg_swap_negatives']:.2f}")
        if stats['swap_negatives_failed'] > 0:
            print(f"   Swap negatives failed:  {stats['swap_negatives_failed']}")
    print("="*60)
    
    return stats


def process_coco_negatives_split(
    input_json: str,
    output_dir: str,
    image_root: str = None,
    batch_size: int = 1000,
    validate_images: bool = True,
    add_swap_negatives: bool = True,
    filter_invalid: bool = True,
):
    """
    Process COCO negatives and split by dataset split (train/val/test).
    
    Detects split from image_path (e.g., "train2014", "val2014", "train2017").
    
    Args:
        input_json: Path to input JSON file
        output_dir: Directory to save split JSON files
        image_root: Root directory for COCO images
        batch_size: Batch size for swap negatives generation
        validate_images: Whether to check if image files exist
        add_swap_negatives: Whether to generate swap negatives
        filter_invalid: Whether to filter invalid samples
    """
    print(f"🔹 Processing COCO negatives with split: {input_json}")
    
    # Load input JSON
    print(f"   Loading JSON...")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"   Loaded {len(data)} samples")
    
    # Group by split
    splits = {}
    for item in data:
        image_path = item.get("image_path", "")
        
        # Detect split from path
        split = "unknown"
        if "train" in image_path.lower():
            split = "train"
        elif "val" in image_path.lower():
            split = "val"
        elif "test" in image_path.lower():
            split = "test"
        
        if split not in splits:
            splits[split] = []
        splits[split].append(item)
    
    print(f"   Found splits: {list(splits.keys())}")
    for split, items in splits.items():
        print(f"      {split}: {len(items)} samples")
    
    # Process each split
    os.makedirs(output_dir, exist_ok=True)
    all_stats = {}
    
    for split, items in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")
        
        # Save temporary file for processing
        temp_input = os.path.join(output_dir, f"_temp_{split}.json")
        with open(temp_input, 'w') as f:
            json.dump(items, f)
        
        output_json = os.path.join(output_dir, f"coco_{split}_negatives.json")
        
        stats = process_coco_negatives(
            input_json=temp_input,
            output_json=output_json,
            image_root=image_root,
            batch_size=batch_size,
            validate_images=validate_images,
            add_swap_negatives=add_swap_negatives,
            filter_invalid=filter_invalid,
        )
        
        all_stats[split] = stats
        
        # Clean up temp file
        os.remove(temp_input)
    
    # Print overall summary
    print("\n" + "="*60)
    print("🎉 ALL SPLITS PROCESSED")
    print("="*60)
    for split, stats in all_stats.items():
        print(f"   {split}: {stats['total_output']}/{stats['total_input']} samples "
              f"({stats['retention_rate']:.1f}%)")
    print("="*60)
    
    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process COCO negatives JSON files with multiprocessing support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a folder of JSON files in parallel (RECOMMENDED)
    python scripts/process_coco_negatives.py \\
        --input_dir /path/to/coco_jsons \\
        --output_dir /path/to/processed \\
        --image_root /path/to/coco/images \\
        --num_processes 8

    # Process single JSON file
    python scripts/process_coco_negatives.py \\
        --input_json coco_negatives.json \\
        --output_json coco_negatives_processed.json \\
        --image_root /path/to/coco/images

    # Process and split by train/val
    python scripts/process_coco_negatives.py \\
        --input_json coco_negatives.json \\
        --output_dir coco_negatives_processed/ \\
        --image_root /path/to/coco/images \\
        --split_by_dataset

    # Skip image validation (faster)
    python scripts/process_coco_negatives.py \\
        --input_dir /path/to/coco_jsons \\
        --output_dir /path/to/processed \\
        --no_validate_images

    # Skip swap negatives generation
    python scripts/process_coco_negatives.py \\
        --input_dir /path/to/coco_jsons \\
        --output_dir /path/to/processed \\
        --no_swap_negatives

    # Resume processing (skip already processed files)
    python scripts/process_coco_negatives.py \\
        --input_dir /path/to/coco_jsons \\
        --output_dir /path/to/processed \\
        --no_overwrite
        """
    )
    
    # Input options (mutually exclusive: folder vs single file)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=str,
                             help="Directory containing COCO JSON files to process in parallel")
    input_group.add_argument("--input_json", type=str,
                             help="Path to single input COCO negatives JSON file")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for output files (required for --input_dir, optional for --split_by_dataset)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to output processed JSON file (for single file mode)")
    
    # Common options
    parser.add_argument("--image_root", type=str, default=None,
                        help="Root directory for COCO images (for validation)")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for swap negatives generation")
    parser.add_argument("--file_pattern", type=str, default="*.json",
                        help="Glob pattern for input files when using --input_dir (default: *.json)")
    
    # Processing options
    parser.add_argument("--num_processes", type=int, default=None,
                        help="Number of worker processes (default: CPU count)")
    parser.add_argument("--split_by_dataset", action="store_true",
                        help="Split output by train/val/test based on image_path (single file mode only)")
    parser.add_argument("--no_validate_images", action="store_true",
                        help="Skip checking if image files exist")
    parser.add_argument("--no_swap_negatives", action="store_true",
                        help="Skip generating swap negatives")
    parser.add_argument("--no_filter_invalid", action="store_true",
                        help="Keep samples without positive/negative content")
    parser.add_argument("--no_overwrite", action="store_true",
                        help="Skip already processed files (uses marker files)")
    
    args = parser.parse_args()
    
    # Folder-based parallel processing mode
    if args.input_dir:
        if not args.output_dir:
            parser.error("--output_dir is required when using --input_dir")
        
        process_coco_folder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            image_root=args.image_root,
            file_pattern=args.file_pattern,
            num_processes=args.num_processes,
            batch_size=args.batch_size,
            validate_images=not args.no_validate_images,
            add_swap_negatives=not args.no_swap_negatives,
            filter_invalid=not args.no_filter_invalid,
            overwrite=not args.no_overwrite,
        )
    
    # Single file mode with split by dataset
    elif args.split_by_dataset:
        if not args.output_dir:
            parser.error("--output_dir is required when using --split_by_dataset")
        
        process_coco_negatives_split(
            input_json=args.input_json,
            output_dir=args.output_dir,
            image_root=args.image_root,
            batch_size=args.batch_size,
            validate_images=not args.no_validate_images,
            add_swap_negatives=not args.no_swap_negatives,
            filter_invalid=not args.no_filter_invalid,
        )
    
    # Single file mode
    else:
        if not args.output_json:
            # Default output name
            base = os.path.splitext(args.input_json)[0]
            args.output_json = f"{base}_processed.json"
        
        process_coco_negatives(
            input_json=args.input_json,
            output_json=args.output_json,
            image_root=args.image_root,
            batch_size=args.batch_size,
            validate_images=not args.no_validate_images,
            add_swap_negatives=not args.no_swap_negatives,
            filter_invalid=not args.no_filter_invalid,
        )

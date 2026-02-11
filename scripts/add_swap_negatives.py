"""
Add swap negatives to LAION JSON files using TextShuffler with BATCH PROCESSING.

Processes JSON files in parallel using multiprocessing, adding "swap_negatives" field
to each sample by applying TextShuffler methods to the original caption.

PERFORMANCE: Uses spaCy's pipe() for batch processing, which is 2-5x faster than
processing captions one at a time. Typical speed: 5000-15000 samples/second.

NEW: Each swap negative includes the swap_type for tracking which method was used.

Output JSON Format:
    {
        "sample_id": "...",
        "original_caption": "A cat sitting on a mat",
        "swap_negatives": [
            {
                "swap_type": "shuffle_nouns_and_adj",
                "negative": "A mat sitting on a cat"
            },
            {
                "swap_type": "shuffle_allbut_nouns_and_adj",
                "negative": "sitting A cat on mat a"
            },
            {
                "swap_type": "shuffle_within_trigrams",
                "negative": "sitting A cat mat a on"
            },
            {
                "swap_type": "shuffle_trigrams",
                "negative": "on a mat A cat sitting"
            }
        ]
    }

Usage:
    # Default (batch_size=1000)
    python scripts/add_swap_negatives.py
    
    # Larger batches for more speed (uses more memory)
    python scripts/add_swap_negatives.py --batch_size 5000
    
    # Process specific tar range (useful for distributed processing)
    python scripts/add_swap_negatives.py --tar_start 0 --tar_end 1000
    python scripts/add_swap_negatives.py --tar_start 1000 --tar_end 2000
    python scripts/add_swap_negatives.py --tar_start 2000 --tar_end 3000
    
    # Custom paths with 8 processes
    python scripts/add_swap_negatives.py \
        --input_dir /path/to/input \
        --output_dir /path/to/output \
        --num_processes 8 \
        --batch_size 2000 \
        --tar_start 0 \
        --tar_end 5000
"""

import os
import json
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.perturbations import TextShuffler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def generate_swap_negatives(caption: str, shuffler: TextShuffler) -> list:
    """
    Generate swap negatives for a caption using TextShuffler.
    
    Args:
        caption: Original caption text
        shuffler: TextShuffler instance
        
    Returns:
        List of dicts with 'swap_type' and 'negative' keys (up to 4 variants, excluding failed/identical ones)
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
            # This comparison is fast (string comparison is O(n) in Python, highly optimized)
            if shuffled != caption:
                swap_negatives.append({
                    'swap_type': swap_type,
                    'negative': shuffled
                })
        except Exception as e:
            # If shuffling fails, log warning and skip
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
    if not captions:
        return []
    
    # Pre-process all captions with spaCy in batch
    # This is the key optimization - spaCy's pipe() is much faster than calling nlp() in a loop
    docs = list(shuffler.nlp.pipe(captions, batch_size=batch_size))
    
    # Now generate negatives using pre-parsed docs
    all_swap_negatives = []
    
    for caption, doc in zip(captions, docs):
        swap_negatives = []
        
        # Method 1: shuffle_nouns_and_adj
        try:
            nouns_and_adj = [token for token in doc if token.pos_ in ['NOUN', 'ADJ', 'PROPN']]
            if len(nouns_and_adj) > 1:
                import random
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
                import random
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


def process_tar_range(args):
    """
    Process a range of tar files in a single process.
    
    Args:
        args: Tuple of (input_dir, output_dir, tar_start, tar_end, process_id, dry_run, batch_size, file_pattern)
        
    Returns:
        dict: Statistics about the processing
    """
    input_dir, output_dir, tar_start, tar_end, process_id, dry_run, batch_size, file_pattern = args
    
    try:
        # Initialize TextShuffler for this process
        logging.info(f"[Process {process_id}] Initializing TextShuffler...")
        shuffler = TextShuffler()
        
        # Find all JSON files in input directory
        input_files = sorted(Path(input_dir).glob(file_pattern))
        
        # Get the subset for this process
        my_files = input_files[tar_start:tar_end]
        total_files_for_this_process = len(my_files)
        
        if not my_files:
            logging.warning(f"[Process {process_id}] No files to process in range [{tar_start}:{tar_end}]")
            return {
                'process_id': process_id,
                'status': 'success',
                'tar_range': f"{tar_start}-{tar_end}",
                'files_processed': 0,
                'total_samples': 0,
                'success_count': 0,
                'failed_count': 0,
                'total_negatives': 0,
                'elapsed_time': 0,
            }
        
        logging.info(
            f"[Process {process_id}] Processing tar range {tar_start}-{tar_end-1} "
            f"({total_files_for_this_process} files)"
        )
        
        # Process statistics
        start_time = time.time()
        files_processed = 0
        total_samples = 0
        total_success = 0
        total_failed = 0
        total_negatives = 0
        
        # Process each file in this range
        for file_idx, input_path in enumerate(my_files):
            file_start_time = time.time()
            
            # Update progress
            files_processed += 1
            logging.info(
                f"[Process {process_id}] Processing tar {files_processed}/{total_files_for_this_process}: "
                f"{input_path.name}"
            )
            
            try:
                # Load input JSON
                with open(input_path, 'r') as f:
                    data = json.load(f)
                
                num_samples = len(data)
                total_samples += num_samples
                
                # Process in batches
                for batch_start in range(0, num_samples, batch_size):
                    batch_end = min(batch_start + batch_size, num_samples)
                    batch_samples = data[batch_start:batch_end]
                    
                    # Extract captions for batch processing
                    batch_captions = []
                    batch_indices = []
                    for i, sample in enumerate(batch_samples):
                        caption = sample.get('original_caption', '')
                        if caption:
                            batch_captions.append(caption)
                            batch_indices.append(batch_start + i)
                        else:
                            # Empty caption - add empty list directly
                            data[batch_start + i]['swap_negatives'] = []
                            total_failed += 1
                    
                    # Generate swap negatives for entire batch at once (FAST!)
                    if batch_captions:
                        try:
                            batch_swap_negatives = generate_swap_negatives_batch(
                                batch_captions, shuffler, batch_size=batch_size
                            )
                            
                            # Assign results back to samples
                            for idx, swap_negatives in zip(batch_indices, batch_swap_negatives):
                                data[idx]['swap_negatives'] = swap_negatives
                                total_success += 1
                                total_negatives += len(swap_negatives)
                                
                        except Exception as e:
                            logging.warning(f"[Process {process_id}] Batch failed, using fallback: {e}")
                            # Fallback: process individually
                            for idx, caption in zip(batch_indices, batch_captions):
                                try:
                                    swap_negatives = generate_swap_negatives(caption, shuffler)
                                    data[idx]['swap_negatives'] = swap_negatives
                                    total_success += 1
                                    total_negatives += len(swap_negatives)
                                except Exception as e2:
                                    logging.debug(f"[Process {process_id}] Failed sample at {idx}: {e2}")
                                    data[idx]['swap_negatives'] = []
                                    total_failed += 1
                
                # Save output JSON (unless dry run)
                if not dry_run:
                    output_path = Path(output_dir) / input_path.name
                    with open(output_path, 'w') as f:
                        json.dump(data, f)
                
                file_elapsed = time.time() - file_start_time
                logging.info(
                    f"[Process {process_id}] ✅ Completed {input_path.name}: "
                    f"{num_samples} samples in {file_elapsed:.1f}s "
                    f"({num_samples/file_elapsed:.1f} samples/s)"
                )
                
            except Exception as e:
                logging.error(f"[Process {process_id}] ❌ Error processing {input_path.name}: {e}")
                continue
        
        elapsed = time.time() - start_time
        
        stats = {
            'process_id': process_id,
            'status': 'success',
            'tar_range': f"{tar_start}-{tar_end-1}",
            'files_processed': files_processed,
            'total_samples': total_samples,
            'success_count': total_success,
            'failed_count': total_failed,
            'total_negatives': total_negatives,
            'avg_negatives_per_sample': total_negatives / total_success if total_success > 0 else 0,
            'elapsed_time': elapsed,
            'speed': total_samples / elapsed if elapsed > 0 else 0,
        }
        
        logging.info(
            f"[Process {process_id}] 🎉 FINISHED tar range {tar_start}-{tar_end-1}: "
            f"{files_processed} files, {total_samples} samples, "
            f"{elapsed:.1f}s ({stats['speed']:.1f} samples/s)"
        )
        
        return stats
        
    except Exception as e:
        logging.error(f"[Process {process_id}] ❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'process_id': process_id,
            'status': 'error',
            'tar_range': f"{tar_start}-{tar_end}",
            'error_message': str(e),
        }


def add_swap_negatives_parallel(
    input_dir: str,
    output_dir: str,
    num_processes: int = None,
    file_pattern: str = "*.json",
    dry_run: bool = False,
    batch_size: int = 1000,
    tar_start: int = None,
    tar_end: int = None,
):
    """
    Add swap negatives to JSON files in parallel with BATCH PROCESSING.
    
    Divides the tar file range across processes, where each process handles
    its own subset of files sequentially with per-process progress tracking.
    
    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory to save output JSON files
        num_processes: Number of parallel processes (default: CPU count)
        file_pattern: Glob pattern for JSON files to process
        dry_run: If True, don't save output (for speed testing)
        batch_size: Number of captions to process in each batch (larger = faster but more memory)
        tar_start: Start index for tar range (inclusive)
        tar_end: End index for tar range (exclusive)
    """
    # Create output directory
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
    else:
        logging.info("🚀 DRY RUN MODE - will not save output files")
    
    # Find all JSON files
    input_files = sorted(Path(input_dir).glob(file_pattern))
    
    if not input_files:
        logging.error(f"No JSON files found matching '{file_pattern}' in {input_dir}")
        return
    
    total_files = len(input_files)
    logging.info(f"Found {total_files} JSON files in {input_dir}")
    
    # Apply tar range filtering
    if tar_start is not None or tar_end is not None:
        tar_start = tar_start if tar_start is not None else 0
        tar_end = tar_end if tar_end is not None else total_files
        
        if tar_start < 0:
            logging.error(f"tar_start must be >= 0, got {tar_start}")
            return
        if tar_end > total_files:
            logging.warning(f"tar_end {tar_end} exceeds total files {total_files}, using {total_files}")
            tar_end = total_files
        if tar_start >= tar_end:
            logging.error(f"tar_start ({tar_start}) must be < tar_end ({tar_end})")
            return
        
        logging.info(f"Tar range filter: processing files {tar_start} to {tar_end-1}")
        files_to_process = tar_end - tar_start
    else:
        tar_start = 0
        tar_end = total_files
        files_to_process = total_files
    
    # Determine number of processes
    if num_processes is None:
        num_processes = min(cpu_count(), files_to_process)
    
    num_processes = min(num_processes, files_to_process)  # Don't use more processes than files
    
    logging.info(f"Using {num_processes} parallel processes")
    logging.info(f"Batch size: {batch_size} captions per batch")
    
    # Divide tar range across processes
    files_per_process = files_to_process // num_processes
    remainder = files_to_process % num_processes
    
    args_list = []
    current_start = tar_start
    
    for process_id in range(num_processes):
        # Distribute remainder across first few processes
        process_file_count = files_per_process + (1 if process_id < remainder else 0)
        process_end = current_start + process_file_count
        
        args_list.append((
            input_dir,
            output_dir,
            current_start,
            process_end,
            process_id,
            dry_run,
            batch_size,
            file_pattern
        ))
        
        logging.info(
            f"Process {process_id}: will process tar range {current_start}-{process_end-1} "
            f"({process_file_count} files)"
        )
        
        current_start = process_end
    
    # Process in parallel
    print("\n" + "="*80)
    print("🚀 STARTING PARALLEL PROCESSING")
    print("="*80)
    print(f"Total files to process: {files_to_process}")
    print(f"Processes: {num_processes}")
    print(f"Files per process: ~{files_per_process}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_tar_range, args_list)
    
    total_elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("🎉 PROCESSING COMPLETE - SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    if successful:
        total_files_processed = sum(r['files_processed'] for r in successful)
        total_samples = sum(r['total_samples'] for r in successful)
        total_success = sum(r['success_count'] for r in successful)
        total_failed = sum(r['failed_count'] for r in successful)
        total_negatives = sum(r['total_negatives'] for r in successful)
        avg_negatives = total_negatives / total_success if total_success > 0 else 0
        
        print(f"✅ Successfully completed: {len(successful)}/{num_processes} processes")
        print(f"📁 Files processed: {total_files_processed}/{files_to_process}")
        print(f"📝 Total samples: {total_samples:,}")
        print(f"   - Success: {total_success:,} ({100*total_success/total_samples:.1f}%)")
        print(f"   - Failed: {total_failed:,} ({100*total_failed/total_samples:.1f}%)")
        print(f"🔄 Total swap negatives generated: {total_negatives:,}")
        print(f"   - Average per sample: {avg_negatives:.2f}")
        print(f"⏱️  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
        print(f"   - Speed: {total_samples / total_elapsed:.1f} samples/second")
        print(f"   - Speed: {total_files_processed / total_elapsed:.2f} files/second")
        
        # Per-process breakdown
        print(f"\n📊 Per-Process Breakdown:")
        print("-" * 80)
        for r in successful:
            print(f"  Process {r['process_id']}: "
                  f"{r['files_processed']} files, "
                  f"{r['total_samples']:,} samples, "
                  f"{r['elapsed_time']:.1f}s, "
                  f"{r['speed']:.1f} samples/s")
    
    if failed:
        print(f"\n❌ Failed processes: {len(failed)}")
        for r in failed:
            print(f"   - Process {r['process_id']} (range {r['tar_range']}): "
                  f"{r.get('error_message', 'Unknown error')}")
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No files were saved")
    else:
        print("\n" + "="*80)
        print(f"📁 Output saved to: {output_dir}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Add swap negatives to LAION JSON files using TextShuffler"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components",
        help="Directory containing input JSON files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components_with_swaps",
        help="Directory to save output JSON files with swap negatives"
    )
    
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: use all CPUs)"
    )
    
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.json",
        help="Glob pattern for JSON files to process (default: *.json)"
    )
    
    parser.add_argument(
        "--test_first",
        type=int,
        default=None,
        help="Process only first N files (for testing)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't save output files (for speed testing)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of captions to process per batch (larger = faster, more memory). Default: 1000"
    )
    
    parser.add_argument(
        "--tar_start",
        type=int,
        default=None,
        help="Start index for tar range (inclusive). E.g., --tar_start 0 --tar_end 1000 processes tars 0-999"
    )
    
    parser.add_argument(
        "--tar_end",
        type=int,
        default=None,
        help="End index for tar range (exclusive). E.g., --tar_start 0 --tar_end 1000 processes tars 0-999"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logging.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Find files to determine total count
    input_files = sorted(Path(args.input_dir).glob(args.file_pattern))
    
    if not input_files:
        logging.error(f"No JSON files found in {args.input_dir}")
        return
    
    total_files = len(input_files)
    logging.info(f"Found {total_files} JSON files matching pattern '{args.file_pattern}'")
    
    # Determine tar range
    tar_start = args.tar_start if args.tar_start is not None else 0
    tar_end = args.tar_end if args.tar_end is not None else total_files
    
    # Validate tar range
    if tar_start < 0:
        logging.error(f"tar_start must be >= 0, got {tar_start}")
        return
    if tar_end > total_files:
        logging.warning(f"tar_end {tar_end} exceeds total files {total_files}, using {total_files}")
        tar_end = total_files
    if tar_start >= tar_end:
        logging.error(f"tar_start ({tar_start}) must be < tar_end ({tar_end})")
        return
    
    # Apply test_first after tar range
    if args.test_first is not None:
        tar_end = min(tar_start + args.test_first, tar_end)
        logging.info(f"Testing mode: limiting to first {args.test_first} files from tar_start")
    
    # Run processing
    add_swap_negatives_parallel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_processes=args.num_processes,
        file_pattern=args.file_pattern,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        tar_start=tar_start,
        tar_end=tar_end,
    )


if __name__ == "__main__":
    main()

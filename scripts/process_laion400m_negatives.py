import os
import sys
import json
import glob
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


def is_valid_sample(item):
    """
    Check if a sample has at least one positive and one negative option.
    
    Requirements:
    1. Must have at least one of: positive_components OR relations
    2. Must have at least one of: negative_components OR negative_relations
    
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
    negative_relations = item.get("negative_relations", [])
    
    # negative_components is a dict, check if it has any non-empty values
    has_negative_components = False
    if negative_components:
        for comp, negs in negative_components.items():
            if negs and len(negs) > 0:
                has_negative_components = True
                break
    
    has_negative_relations = negative_relations and len(negative_relations) > 0
    has_negative = has_negative_components or has_negative_relations
    
    return has_positive and has_negative


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
    import random
    
    if not captions:
        return []
    
    # Pre-process all captions with spaCy in batch
    docs = list(shuffler.nlp.pipe(captions, batch_size=batch_size))
    
    # Now generate negatives using pre-parsed docs
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


def process_single_json(args):
    """
    Process a single JSON file. Designed to be called by multiprocessing.Pool.
    
    Args:
        args: Tuple of (prefix, input_dir, output_dir, batch_size)
        
    Returns:
        dict: Statistics about the processing
    """
    global _shuffler
    prefix, input_dir, output_dir, batch_size = args
    
    input_path = os.path.join(input_dir, f"{prefix}.json")
    if not os.path.exists(input_path):
        return {
            'prefix': prefix,
            'status': 'error',
            'message': f"File not found: {input_path}"
        }
    
    # Check if file is empty
    if os.path.getsize(input_path) == 0:
        return {
            'prefix': prefix,
            'status': 'error',
            'message': f"File is empty: {input_path}"
        }

    print(f"🔹 Processing {input_path} ...")
    
    # Use the global shuffler initialized per worker process
    shuffler = _shuffler
    
    try:
        with open(input_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {
            'prefix': prefix,
            'status': 'error',
            'message': f"Invalid JSON in {input_path}: {e}"
        }

    # Group by tar number with deduplication and validation
    tar_groups = defaultdict(list)
    seen_sample_ids = defaultdict(set)  # Track seen sample_ids per tar
    duplicates_skipped = 0
    invalid_samples_skipped = 0
    
    # First pass: filter and group valid samples
    valid_items = []
    valid_item_indices = []  # Track original indices for assignment
    
    for idx, item in enumerate(tqdm(data, desc=f"Grouping {prefix} by tar")):
        # Extract tar number from sample_id ("00000.tar::000000007")
        if "sample_id" in item and ".tar::" in item["sample_id"]:
            tar_str = item["sample_id"].split(".tar::")[0]
            # Handle both basename only (e.g., "00000") and full paths
            tar_basename = os.path.basename(tar_str)
            tar_num = int(tar_basename)
            sample_id = item["sample_id"]
        elif "image_url" in item and ".tar" in str(item["image_url"]):
            # Fallback: extract from image_url if sample_id not found
            tar_str = os.path.basename(str(item["image_url"])).split(".tar")[0]
            tar_num = int(tar_str)
            sample_id = item.get("sample_id", item["image_url"])
        else:
            raise ValueError(f"Could not find tar number in sample_id or image_url: {item}")

        # Skip duplicate sample_ids within the same tar
        if sample_id in seen_sample_ids[tar_num]:
            duplicates_skipped += 1
            continue
        
        # Skip invalid samples (no positive/negative content)
        if not is_valid_sample(item):
            invalid_samples_skipped += 1
            continue
        
        seen_sample_ids[tar_num].add(sample_id)
        
        # Remove image_url to save space (optional - keeps file smaller)
        item.pop("image_url", None)
        
        # Store for batch processing
        item['_tar_num'] = tar_num  # Temporarily store tar_num
        valid_items.append(item)
        valid_item_indices.append(idx)
    
    # Generate swap negatives in batch for all valid samples
    print(f"   Generating swap negatives for {len(valid_items)} valid samples...")
    captions = [item.get('original_caption', '') for item in valid_items]
    total_swap_negatives = 0
    swap_negatives_failed = 0
    
    # Process in batches
    for batch_start in range(0, len(captions), batch_size):
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
                valid_items[batch_start + i]['swap_negatives'] = []
                swap_negatives_failed += 1
        
        if non_empty_captions:
            try:
                batch_swap_negatives = generate_swap_negatives_batch(
                    non_empty_captions, shuffler, batch_size=batch_size
                )
                
                for local_idx, swap_negs in zip(non_empty_indices, batch_swap_negatives):
                    valid_items[local_idx]['swap_negatives'] = swap_negs
                    total_swap_negatives += len(swap_negs)
                    
            except Exception as e:
                logging.warning(f"Batch processing failed, using fallback: {e}")
                # Fallback: process individually
                for local_idx, caption in zip(non_empty_indices, non_empty_captions):
                    try:
                        swap_negs = generate_swap_negatives(caption, shuffler)
                        valid_items[local_idx]['swap_negatives'] = swap_negs
                        total_swap_negatives += len(swap_negs)
                    except Exception as e2:
                        logging.debug(f"Failed to generate swap negatives: {e2}")
                        valid_items[local_idx]['swap_negatives'] = []
                        swap_negatives_failed += 1
    
    # Group valid items by tar number
    for item in valid_items:
        tar_num = item.pop('_tar_num')  # Remove temporary field
        tar_groups[tar_num].append(item)

    # Write per-tar JSON files
    for tar_num, items in tqdm(tar_groups.items(), desc=f"Saving {prefix} per-tar JSONs"):
        out_path = os.path.join(output_dir, f"{tar_num:05d}.json")
        with open(out_path, "w") as f:
            json.dump(items, f)

    # Calculate statistics
    total_samples_written = sum(len(items) for items in tar_groups.values())
    total_processed = len(data)
    avg_swap_negs = total_swap_negatives / total_samples_written if total_samples_written > 0 else 0
    
    print(f"✅ Finished {prefix} → {len(tar_groups)} tar files created in {output_dir}")
    print("📊 Statistics:")
    print(f"   Total samples processed: {total_processed}")
    print(f"   Valid samples written: {total_samples_written}")
    print(f"   🔄 Total swap negatives generated: {total_swap_negatives}")
    print(f"      Average per sample: {avg_swap_negs:.2f}")
    if swap_negatives_failed > 0:
        print(f"   ⚠️  Swap negatives failed: {swap_negatives_failed}")
    if duplicates_skipped > 0:
        print(f"   ⚠️  Skipped {duplicates_skipped} duplicate samples (based on sample_id)")
    if invalid_samples_skipped > 0:
        print(f"   ⚠️  Skipped {invalid_samples_skipped} invalid samples (missing positive/negative content)")
    retention_rate = (total_samples_written / total_processed * 100) if total_processed > 0 else 0
    print(f"   Retention rate: {retention_rate:.2f}%")
    
    return {
        'prefix': prefix,
        'status': 'success',
        'tar_files_created': len(tar_groups),
        'total_processed': total_processed,
        'total_written': total_samples_written,
        'duplicates_skipped': duplicates_skipped,
        'invalid_skipped': invalid_samples_skipped,
        'retention_rate': retention_rate,
        'total_swap_negatives': total_swap_negatives,
        'avg_swap_negatives_per_sample': avg_swap_negs,
        'swap_negatives_failed': swap_negatives_failed
    }


def split_jsons_by_tar(
    input_dir,
    input_prefixes,
    output_dir,
    num_processes=None,
    batch_size=1000,
    overwrite=True
):
    """
    Splits large LAION JSON files into per-tar JSON files using multiple processes.
    Also generates swap negatives for each valid sample.

    Args:
        input_dir (str): Path to folder with big JSON files (e.g., laion400m-train-0-256.json).
        input_prefixes (list of str): Filenames (without .json) to process, e.g. ["laion400m-train-0-256"].
        output_dir (str): Path to folder where per-tar JSONs will be saved.
        num_processes (int, optional): Number of processes to use. Defaults to CPU count.
        batch_size (int): Number of captions to process per batch for swap negatives. Default: 1000.
        overwrite (bool): If False, skip input files that have already been processed. Default: True.
    """
    os.makedirs(output_dir, exist_ok=True)

    if num_processes is None:
        num_processes = cpu_count()
    
    # Filter out already processed files if overwrite=False
    skipped_existing = []
    if not overwrite:
        filtered_prefixes = []
        for prefix in input_prefixes:
            # Check if any output files from this prefix exist
            # We use a marker file to indicate completion
            marker_file = os.path.join(output_dir, f".{prefix}.done")
            if os.path.exists(marker_file):
                skipped_existing.append(prefix)
            else:
                filtered_prefixes.append(prefix)
        
        if skipped_existing:
            print(f"⏭️  Skipping {len(skipped_existing)} already processed files (overwrite=False)")
        
        input_prefixes = filtered_prefixes
        
        if not input_prefixes:
            print("✅ All files already processed. Nothing to do.")
            return
    
    print(f"🚀 Starting parallel processing with {num_processes} processes for {len(input_prefixes)} JSON files...")
    print(f"   Batch size for swap negatives: {batch_size}")
    print("   Initializing TextShuffler in each worker process...")
    
    # Prepare arguments for each process
    args_list = [(prefix, input_dir, output_dir, batch_size) for prefix in input_prefixes]
    
    # Process in parallel with worker initializer (TextShuffler loaded once per worker)
    with Pool(processes=num_processes, initializer=_init_worker) as pool:
        results = pool.map(process_single_json, args_list)
    
    # Create marker files for successfully processed prefixes
    if not overwrite:
        for result in results:
            if result['status'] == 'success':
                marker_file = os.path.join(output_dir, f".{result['prefix']}.done")
                with open(marker_file, 'w') as f:
                    f.write(f"Processed on {datetime.now().isoformat()}\n")
    
    # Print summary
    print("\n" + "="*80)
    print("🎉 ALL FILES PROCESSED - SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    if skipped_existing:
        print(f"⏭️  Skipped (already processed): {len(skipped_existing)} files")
    
    if successful:
        total_tar_files = sum(r['tar_files_created'] for r in successful)
        total_samples = sum(r['total_written'] for r in successful)
        total_duplicates = sum(r['duplicates_skipped'] for r in successful)
        total_invalid = sum(r['invalid_skipped'] for r in successful)
        total_swap_negatives = sum(r['total_swap_negatives'] for r in successful)
        avg_swap_negs = total_swap_negatives / total_samples if total_samples > 0 else 0
        total_swap_failed = sum(r['swap_negatives_failed'] for r in successful)
        
        print(f"✅ Successfully processed: {len(successful)} files")
        print(f"📁 Total tar files created: {total_tar_files}")
        print(f"📝 Total samples written: {total_samples}")
        print(f"🔄 Total swap negatives generated: {total_swap_negatives}")
        print(f"   Average per sample: {avg_swap_negs:.2f}")
        if total_swap_failed > 0:
            print(f"⚠️  Total swap negatives failed: {total_swap_failed}")
        print(f"⚠️  Total duplicates skipped: {total_duplicates}")
        print(f"⚠️  Total invalid samples skipped: {total_invalid}")
    
    if failed:
        print(f"\n❌ Failed to process: {len(failed)} files")
        for r in failed:
            print(f"   - {r['prefix']}: {r['message']}")
    
    print("="*80)


if __name__ == "__main__":
    # Example usage
    input_dir = "/mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally/new_component_jsons"
    output_dir = "/mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components"

    # List of big JSON prefixes to process
    # They are all the files in the input_dir that start with "laion400m-train-"
    big_jsons = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(input_dir, "laion400m-train-*.json"))
    ]

    # Use all available CPU cores by default, or specify a number
    # num_processes = 4  # Uncomment to use a specific number of processes
    num_processes = None  # Uses all available CPU cores
    
    # Batch size for swap negatives generation (larger = faster but more memory)
    batch_size = 1000
    
    # Set to False to skip already processed files (uses .done marker files)
    overwrite = False
    
    split_jsons_by_tar(
        input_dir, big_jsons, output_dir, 
        num_processes=num_processes, 
        batch_size=batch_size,
        overwrite=overwrite
    )

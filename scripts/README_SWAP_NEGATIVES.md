# Swap Negatives Generation

Scripts to add swap negatives to LAION JSON files using TextShuffler with **BATCH PROCESSING** for maximum speed.

## Overview

These scripts process your LAION JSON files and add a `swap_negatives` field containing text-shuffled versions of the original caption. This provides an efficient way to generate negatives without runtime processing during training.

**⚡ PERFORMANCE:** Uses spaCy's `pipe()` for batch processing, achieving **2-5x speedup** compared to processing captions one at a time. Typical speed: **5,000-15,000 samples/second** depending on batch size and hardware.

## Files

1. **`test_swap_negatives.py`** - Test suite to verify TextShuffler works correctly
2. **`add_swap_negatives.py`** - Main script with BATCH PROCESSING for speed
3. **`benchmark_batch_processing.py`** - Benchmark to compare batch vs non-batch speed
4. **`generate_tar_range_jobs.py`** - Generate commands for distributed processing
5. **`example_tar_range_processing.sh`** - Example script showing tar range usage

## Quick Links

- **[TAR_RANGE_GUIDE.md](TAR_RANGE_GUIDE.md)** - Complete guide to distributed processing with tar ranges
- **[BATCH_PROCESSING_EXPLAINED.md](BATCH_PROCESSING_EXPLAINED.md)** - Deep dive into batch processing optimization

## Step 0: Benchmark Batch Processing (Optional)

See the speedup from batch processing:

```bash
python scripts/benchmark_batch_processing.py
```

This will show you the performance difference between processing captions one-by-one vs in batches. Typical results:
- **Non-batch:** ~50-100 captions/second  
- **Batch (1000):** ~5,000-15,000 captions/second
- **Speedup:** 2-5x faster

## Step 1: Run Tests

First, verify everything works with the test script:

```bash
cd /mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally

python scripts/test_swap_negatives.py
```

This will run 5 tests:
1. **Basic TextShuffler functionality** - Verify all shuffle methods work
2. **Swap negative generation** - Test complete generation pipeline
3. **Processing speed** - Benchmark on 100 captions
4. **Real JSON file** - Test with actual data from your dataset
5. **Memory efficiency** - Check TextShuffler memory usage

Expected output:
```
🧪 ========================================================================== 🧪
   SWAP NEGATIVES TESTING SUITE
🧪 ========================================================================== 🧪

TEST 1: Basic TextShuffler Functionality
...
✅ Basic functionality test complete

TEST 2: Swap Negative Generation
...
✅ Swap negative generation test complete

TEST 3: Processing Speed
...
   Speed: 50.2 captions/second
...
✅ Processing complete

TEST 4: Real JSON File Processing
...
✅ Real JSON file test complete

TEST 5: Memory Efficiency
...
   TextShuffler memory: 450.2 MB
✅ Memory efficiency test complete

🎉 ========================================================================== 🎉
   ALL TESTS PASSED!
🎉 ========================================================================== 🎉
```

## Step 2: Test on Small Sample (with Dry Run for Speed Check)

First, test speed without saving files:

```bash
python scripts/add_swap_negatives.py \
    --test_first 10 \
    --dry_run \
    --num_processes 8 \
    --batch_size 1000
```

This processes 10 files without saving to check speed. Then test with actual saving:

```bash
python scripts/add_swap_negatives.py \
    --input_dir /mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components \
    --output_dir /mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components_with_swaps \
    --test_first 5 \
    --num_processes 4 \
    --batch_size 1000
```

This processes only the first 5 JSON files with 4 processes and batch size of 1000.

Check the output:
```bash
# View a sample
python -c "
import json
with open('/mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components_with_swaps/00000.json', 'r') as f:
    data = json.load(f)
    sample = data[0]
    print('Original caption:', sample['original_caption'])
    print('Swap negatives:', sample['swap_negatives'])
"
```

Expected output format:
```json
{
    "sample_id": "03187.tar::031870011",
    "original_caption": "Green Grass Texture - Foto Stock",
    "positive_components": ["Green Grass Texture", "Foto Stock"],
    "negative_components": {...},
    "relations": [...],
    "swap_negatives": [
        {
            "swap_type": "shuffle_nouns_and_adj",
            "negative": "Texture Green Grass - Stock Foto"
        },
        {
            "swap_type": "shuffle_allbut_nouns_and_adj",
            "negative": "Foto Stock - Green Grass Texture"
        },
        {
            "swap_type": "shuffle_within_trigrams",
            "negative": "Green Texture Grass - Foto Stock"
        },
        {
            "swap_type": "shuffle_trigrams",
            "negative": "Stock Foto - Texture Grass Green"
        }
    ]
}
```

**Note:** The `swap_negatives` field now contains objects with:
- `swap_type`: The shuffling method used (e.g., "shuffle_nouns_and_adj")
- `negative`: The shuffled caption text

## Step 3: Process All Files with Batch Processing

Once verified, process all files with maximum parallelism and optimal batch size:

```bash
# Recommended: batch_size=1000-2000 for good balance
python scripts/add_swap_negatives.py \
    --input_dir /mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components \
    --output_dir /mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components_with_swaps \
    --num_processes 8 \
    --batch_size 1000

# For even more speed (if you have plenty of RAM):
python scripts/add_swap_negatives.py \
    --num_processes 8 \
    --batch_size 5000
```

**Performance Tips:**
- **batch_size=1000**: Good default, ~5,000-8,000 samples/sec
- **batch_size=2000**: Faster, ~8,000-12,000 samples/sec, uses more RAM
- **batch_size=5000**: Maximum speed, ~10,000-15,000 samples/sec, requires lots of RAM
- **num_processes=8**: Match your GPU count for balanced resource usage

Expected output:
```
Found 10700 JSON files to process
Using 8 parallel processes
🚀 Starting parallel processing...
[Process 0] Loading /mnt/.../00000.json...
[Process 1] Loading /mnt/.../00001.json...
...

================================================================================
🎉 PROCESSING COMPLETE - SUMMARY
================================================================================
✅ Successfully processed: 10700/10700 files
📝 Total samples: 12,500,000
   - Success: 12,450,000 (99.6%)
   - Failed: 50,000 (0.4%)
🔄 Total swap negatives generated: 49,800,000
   - Average per sample: 4.00
⏱️  Total time: 3600.0s
   - Speed: 3472.2 samples/second
   - Speed: 2.97 files/second
================================================================================
📁 Output saved to: /mnt/.../laion400m_negatives_components_with_swaps
================================================================================
```

## Usage in Training

Once you have the files with swap negatives, you can use them in two ways:

### Option 1: Use Pre-generated Swap Negatives (No Runtime Processing)

```python
# In your dataset __getitem__:
def __getitem__(self, idx):
    sample = self.data[idx]
    
    # Get captions
    original_caption = sample['original_caption']
    swap_negatives = sample.get('swap_negatives', [])
    
    # Use first swap negative if available
    if swap_negatives:
        # Each swap negative is now a dict with 'swap_type' and 'negative'
        negative_obj = swap_negatives[0]  # Or random.choice(swap_negatives)
        negative = negative_obj['negative']
        swap_type = negative_obj['swap_type']
        
        print(f"Using negative from: {swap_type}")
    else:
        # Fallback to other negatives
        negative = original_caption
    
    return {
        'image': img,
        'caption_options': [original_caption, negative],
        ...
    }
```

### Option 2: Filter by Swap Type

```python
# Use only specific types of swap negatives
def __getitem__(self, idx):
    sample = self.data[idx]
    original_caption = sample['original_caption']
    swap_negatives = sample.get('swap_negatives', [])
    
    # Filter for noun/adjective swaps only
    noun_swaps = [
        neg for neg in swap_negatives 
        if neg['swap_type'] in ['shuffle_nouns_and_adj', 'shuffle_allbut_nouns_and_adj']
    ]
    
    if noun_swaps:
        negative = random.choice(noun_swaps)['negative']
    else:
        negative = original_caption
    
    return {'image': img, 'caption': original_caption, 'negative': negative}
```

### Option 3: Combine with Batch-Level Swapping

```python
# Use pre-generated swaps AND batch-level swapping for diversity
from data_loading.batch_negative_collator import create_batch_negative_collator

collate_fn = create_batch_negative_collator(
    strategy="mixed",  # Mix pre-generated and in-batch swaps
    num_negatives=2,   # One from pre-gen, one from batch
)

dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)
```

## Step 4: Distributed Processing (Optional)

For very large datasets, you can split processing across multiple jobs using tar ranges:

### Method 1: Manual Tar Ranges

Process different ranges in separate jobs/nodes:

```bash
# Node 1: Process first 1000 files
python scripts/add_swap_negatives.py --tar_start 0 --tar_end 1000

# Node 2: Process next 1000 files
python scripts/add_swap_negatives.py --tar_start 1000 --tar_end 2000

# Node 3: Process next 1000 files
python scripts/add_swap_negatives.py --tar_start 2000 --tar_end 3000
# ... etc
```

### Method 2: Auto-Generate Commands

Use the helper script to generate all commands:

```bash
# Generate commands for 10700 files, 1000 files per job
python scripts/generate_tar_range_jobs.py \
    --total_files 10700 \
    --files_per_job 1000

# Or split into exactly 11 jobs
python scripts/generate_tar_range_jobs.py \
    --total_files 10700 \
    --num_jobs 11
```

This will output:
- Individual commands for each tar range
- SLURM array job script
- GNU Parallel commands

### Method 3: SLURM Array Job

The generator creates a SLURM script you can use:

```bash
# Generate the script
python scripts/generate_tar_range_jobs.py \
    --total_files 10700 \
    --num_jobs 11 > submit_swap_negatives.sh

# Submit array job
sbatch submit_swap_negatives.sh
```

This will run all 11 jobs in parallel on your cluster.

## Performance Notes

- **Speed**: ~5,000-15,000 samples/second with batch processing (depends on batch_size and CPU)
- **Memory**: Each TextShuffler uses ~450MB (spaCy model), plus batch processing memory
- **Processes**: Using 8 processes is optimal for 8 GPUs
- **Time**: Processing ~12M samples takes ~5-10 minutes with batch processing + 8 processes
- **Batch Size**: Larger batches (2000-5000) are faster but use more memory

## Troubleshooting

### Test fails with import error
```bash
# Make sure you're in the correct directory
cd /mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally
```

### TextShuffler initialization fails
Check that spaCy is installed and the model is downloaded:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Out of memory
Reduce the number of processes:
```bash
python scripts/add_swap_negatives.py --num_processes 4
```

### Processing too slow
Increase the number of processes (up to CPU count):
```bash
python scripts/add_swap_negatives.py --num_processes 16
```

## Command Reference

```bash
# Full help
python scripts/add_swap_negatives.py --help

# Test mode (first 10 files)
python scripts/add_swap_negatives.py --test_first 10

# Dry run (no file saving, for speed testing)
python scripts/add_swap_negatives.py --test_first 10 --dry_run

# Custom batch size for more speed
python scripts/add_swap_negatives.py --batch_size 5000

# Process specific tar range (useful for distributed processing)
python scripts/add_swap_negatives.py --tar_start 0 --tar_end 1000

# Custom directories
python scripts/add_swap_negatives.py \
    --input_dir /path/to/input \
    --output_dir /path/to/output

# Specific file pattern
python scripts/add_swap_negatives.py \
    --file_pattern "0000*.json"  # Only process files starting with 0000

# Generate distributed job commands
python scripts/generate_tar_range_jobs.py --total_files 10700 --files_per_job 1000
```

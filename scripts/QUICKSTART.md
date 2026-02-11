# Quick Start: Swap Negatives Generation

## TL;DR - Three Commands

```bash
# 1. Run tests
python scripts/test_swap_negatives.py

# 2. Speed test (10 files, no saving)
python scripts/add_swap_negatives.py --test_first 10 --dry_run --num_processes 8

# 3. Process everything
python scripts/add_swap_negatives.py --num_processes 8
```

## Or Use the Automated Script

```bash
chmod +x scripts/quickstart_swap_negatives.sh
./scripts/quickstart_swap_negatives.sh
```

This runs all steps automatically with confirmation before processing.

---

## Detailed Steps

### Step 1: Run Tests (30 seconds)

```bash
cd /mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally

python scripts/test_swap_negatives.py
```

**What it does:**
- Tests TextShuffler on sample captions
- Measures processing speed (~50 captions/second)
- Tests with your actual JSON files
- Checks memory usage (~450MB per TextShuffler)

**Expected output:**
```
🧪 ========================================================================== 🧪
   SWAP NEGATIVES TESTING SUITE
🧪 ========================================================================== 🧪

TEST 1: Basic TextShuffler Functionality
...
✅ All tests passed!
```

### Step 2: Speed Test - Dry Run (1-2 minutes)

Test on a few files WITHOUT saving (to measure speed):

```bash
python scripts/add_swap_negatives.py \
    --test_first 10 \
    --dry_run \
    --num_processes 8
```

**What it does:**
- Processes first 10 JSON files
- Generates swap negatives
- Does NOT save (dry run)
- Shows processing speed

**Expected output:**
```
Found 10 JSON files to process
Using 8 parallel processes
🚀 DRY RUN MODE - will not save output files
...
✅ Successfully processed: 10/10 files
⏱️  Total time: 120.5s
   - Speed: 3500.2 samples/second

⚠️  DRY RUN MODE - No files were saved
```

**Adjust processes if needed:**
- If too slow: increase `--num_processes` (up to CPU count)
- If out of memory: decrease `--num_processes`

### Step 3: Process All Files (~1 hour)

Once satisfied with the speed, process everything:

```bash
python scripts/add_swap_negatives.py \
    --input_dir /mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components \
    --output_dir /mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components_with_swaps \
    --num_processes 8
```

**What it does:**
- Processes ALL JSON files (10,000+ files)
- Adds `swap_negatives` field to each sample
- Saves to new directory

**Expected output:**
```
Found 10700 JSON files to process
Using 8 parallel processes
🚀 Starting parallel processing...
[Process 0] Processing 1234 samples from 00000.json...
...

🎉 PROCESSING COMPLETE - SUMMARY
================================================================================
✅ Successfully processed: 10700/10700 files
📝 Total samples: 12,500,000
   - Success: 12,450,000 (99.6%)
🔄 Total swap negatives generated: 49,800,000
⏱️  Total time: 3600.0s
   - Speed: 3472.2 samples/second
📁 Output saved to: .../laion400m_negatives_components_with_swaps
```

---

## Common Usage Patterns

### Test on specific files
```bash
# Process only files starting with "00"
python scripts/add_swap_negatives.py \
    --file_pattern "00*.json" \
    --dry_run
```

### Test with fewer processes
```bash
# Use only 4 processes
python scripts/add_swap_negatives.py \
    --test_first 5 \
    --num_processes 4 \
    --dry_run
```

### Resume interrupted processing
```bash
# If processing was interrupted, just run again
# It will overwrite incomplete files
python scripts/add_swap_negatives.py --num_processes 8
```

### Check output format
```bash
# View a sample with swap negatives
python -c "
import json
with open('/mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components_with_swaps/00000.json', 'r') as f:
    data = json.load(f)
    sample = data[0]
    print('Original:', sample['original_caption'])
    print('Swaps:', sample['swap_negatives'])
"
```

---

## Troubleshooting

### Tests fail
```bash
# Check spaCy is installed
pip install spacy
python -m spacy download en_core_web_sm

# Run tests in verbose mode
python scripts/test_swap_negatives.py 2>&1 | tee test_output.log
```

### Out of memory
```bash
# Reduce processes
python scripts/add_swap_negatives.py --num_processes 4
```

### Too slow
```bash
# Increase processes (up to CPU count)
python scripts/add_swap_negatives.py --num_processes 16
```

### Check progress during processing
```bash
# Watch the output directory
watch -n 5 'ls -lh /mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components_with_swaps | wc -l'
```

---

## Performance Estimates

For ~12M samples across 10,700 files:

| Processes | Time | Speed |
|-----------|------|-------|
| 4 | ~2 hours | 1,700 samples/s |
| 8 | ~1 hour | 3,500 samples/s |
| 16 | ~30 min | 7,000 samples/s |

*Actual performance depends on CPU speed and disk I/O*

---

## What Gets Added

Before:
```json
{
    "sample_id": "03187.tar::031870011",
    "original_caption": "Green Grass Texture - Foto Stock",
    "positive_components": ["Green Grass Texture", "Foto Stock"],
    ...
}
```

After:
```json
{
    "sample_id": "03187.tar::031870011",
    "original_caption": "Green Grass Texture - Foto Stock",
    "positive_components": ["Green Grass Texture", "Foto Stock"],
    "swap_negatives": [
        "Texture Green Grass - Stock Foto",
        "Foto Stock - Green Grass Texture",
        "Green Texture Grass - Foto Stock",
        "Stock Foto - Texture Grass Green"
    ],
    ...
}
```

Each sample gets 4 swap negative variants (one per shuffle method).

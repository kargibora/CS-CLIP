# Tar Range Processing Guide

## Overview

The `--tar_start` and `--tar_end` arguments allow you to process specific subsets of files, enabling distributed processing across multiple jobs or machines.

## Use Cases

### 1. Distributed Processing Across Cluster Nodes

Split work across multiple SLURM jobs:

```bash
# Node 1
python scripts/add_swap_negatives.py --tar_start 0 --tar_end 1000

# Node 2
python scripts/add_swap_negatives.py --tar_start 1000 --tar_end 2000

# Node 3
python scripts/add_swap_negatives.py --tar_start 2000 --tar_end 3000
```

### 2. Incremental Processing

Process in chunks over time:

```bash
# Day 1: Process first 2000
python scripts/add_swap_negatives.py --tar_start 0 --tar_end 2000

# Day 2: Process next 2000
python scripts/add_swap_negatives.py --tar_start 2000 --tar_end 4000

# Day 3: Process remaining
python scripts/add_swap_negatives.py --tar_start 4000 --tar_end 10700
```

### 3. Resume After Failure

If a job fails, resume from where it left off:

```bash
# Initial job fails at file 5432
python scripts/add_swap_negatives.py --tar_start 0 --tar_end 10700

# Resume from file 5432
python scripts/add_swap_negatives.py --tar_start 5432 --tar_end 10700
```

### 4. Testing Specific Ranges

Test on specific file ranges:

```bash
# Test on middle files
python scripts/add_swap_negatives.py --tar_start 5000 --tar_end 5010 --dry_run
```

## How It Works

1. **File Discovery**: Script finds all JSON files matching the pattern
2. **Sorting**: Files are sorted alphabetically (e.g., 00000.json, 00001.json, ...)
3. **Range Selection**: Files at indices [tar_start:tar_end] are selected
4. **Processing**: Only the selected files are processed

**Example with 10700 files:**
- `--tar_start 0 --tar_end 1000` → processes files 00000.json through 00999.json
- `--tar_start 1000 --tar_end 2000` → processes files 01000.json through 01999.json
- `--tar_start 10000 --tar_end 10700` → processes files 10000.json through 10699.json

## Helper Script: generate_tar_range_jobs.py

Automatically generate commands for distributed processing:

```bash
# Split 10700 files into jobs of 1000 files each
python scripts/generate_tar_range_jobs.py \
    --total_files 10700 \
    --files_per_job 1000
```

**Output:**
1. Individual commands for each tar range
2. SLURM array job script
3. GNU Parallel commands

### Split by Number of Jobs

```bash
# Split into exactly 20 jobs
python scripts/generate_tar_range_jobs.py \
    --total_files 10700 \
    --num_jobs 20
```

This will create 20 jobs of ~535 files each.

### Customize Parameters

```bash
python scripts/generate_tar_range_jobs.py \
    --total_files 10700 \
    --files_per_job 500 \
    --num_processes 8 \
    --batch_size 2000 \
    --dry_run
```

## SLURM Array Job

The generator creates a SLURM script for you:

```bash
# Generate SLURM script (redirects to stdout, capture it)
python scripts/generate_tar_range_jobs.py \
    --total_files 10700 \
    --num_jobs 11 \
    > submit_swap_negatives.sh

# Make it executable
chmod +x submit_swap_negatives.sh

# Submit to SLURM
sbatch submit_swap_negatives.sh
```

**SLURM Script Features:**
- Array job for parallel execution
- Automatic tar range calculation per job
- Configurable resources (CPUs, memory, time)
- Separate logs for each job

**Example SLURM script:**
```bash
#!/bin/bash
#SBATCH --job-name=swap_negatives
#SBATCH --array=0-10          # 11 jobs (0 through 10)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/swap_negatives_%A_%a.out
#SBATCH --error=logs/swap_negatives_%A_%a.err

# Calculate tar range for this job
TAR_START=$((SLURM_ARRAY_TASK_ID * 1000))
TAR_END=$(((SLURM_ARRAY_TASK_ID + 1) * 1000))

# Run processing
python scripts/add_swap_negatives.py \
    --tar_start $TAR_START \
    --tar_end $TAR_END \
    --num_processes 8 \
    --batch_size 1000
```

## GNU Parallel

For parallel execution on a single machine:

```bash
# Generate commands
python scripts/generate_tar_range_jobs.py \
    --total_files 10700 \
    --files_per_job 1000 \
    > tar_range_commands.txt

# Run 4 jobs in parallel
parallel -j 4 < tar_range_commands.txt
```

This will:
1. Run 4 jobs simultaneously
2. As each job completes, start the next one
3. Continue until all jobs are done

## Best Practices

### 1. Choose Appropriate Range Size

**Small ranges (100-500 files):**
- ✅ More granular control
- ✅ Easier to resume failed jobs
- ❌ More overhead (more jobs to manage)

**Medium ranges (1000-2000 files):**
- ✅ Good balance
- ✅ Recommended for most use cases
- ✅ ~5-10 minutes per job

**Large ranges (5000+ files):**
- ✅ Fewer jobs to manage
- ❌ Harder to resume if failed
- ❌ Long-running jobs (hours)

### 2. Match Resources to Range Size

```bash
# Small range: fewer processes
python scripts/add_swap_negatives.py \
    --tar_start 0 --tar_end 100 \
    --num_processes 4

# Large range: more processes
python scripts/add_swap_negatives.py \
    --tar_start 0 --tar_end 5000 \
    --num_processes 16
```

### 3. Test Before Full Run

Always test with a small range first:

```bash
# Test dry run
python scripts/add_swap_negatives.py \
    --tar_start 0 --tar_end 10 \
    --dry_run

# Test actual processing
python scripts/add_swap_negatives.py \
    --tar_start 0 --tar_end 10
```

### 4. Monitor Progress

Use separate output directories or logging to track progress:

```bash
# Job-specific output directory
python scripts/add_swap_negatives.py \
    --tar_start 0 --tar_end 1000 \
    --output_dir /path/to/output/job_0
```

Or check SLURM logs:
```bash
tail -f logs/swap_negatives_12345_0.out
```

## Common Patterns

### Pattern 1: Divide and Conquer (Parallel Nodes)

```bash
# Run these on different cluster nodes simultaneously
sbatch --wrap="python scripts/add_swap_negatives.py --tar_start 0 --tar_end 2000"
sbatch --wrap="python scripts/add_swap_negatives.py --tar_start 2000 --tar_end 4000"
sbatch --wrap="python scripts/add_swap_negatives.py --tar_start 4000 --tar_end 6000"
sbatch --wrap="python scripts/add_swap_negatives.py --tar_start 6000 --tar_end 8000"
sbatch --wrap="python scripts/add_swap_negatives.py --tar_start 8000 --tar_end 10700"
```

### Pattern 2: Sequential Processing (One Node)

```bash
# Process sequentially with checkpoints
for start in {0..10000..1000}; do
    end=$((start + 1000))
    echo "Processing $start to $end"
    python scripts/add_swap_negatives.py \
        --tar_start $start \
        --tar_end $end \
        --num_processes 8
done
```

### Pattern 3: Priority Processing (Important Files First)

```bash
# Process critical files first
python scripts/add_swap_negatives.py --tar_start 0 --tar_end 100

# Then process rest in background
python scripts/add_swap_negatives.py --tar_start 100 --tar_end 10700 &
```

## Troubleshooting

### Files Missing After Distributed Processing

Check that all ranges were processed:

```bash
# Expected files
total_files=10700

# Actual files in output
actual_files=$(ls /path/to/output/*.json | wc -l)

echo "Expected: $total_files"
echo "Actual: $actual_files"
echo "Missing: $((total_files - actual_files))"
```

### Overlapping Ranges

**Don't do this:**
```bash
# BAD: Overlapping ranges will process same files twice
python scripts/add_swap_negatives.py --tar_start 0 --tar_end 1500
python scripts/add_swap_negatives.py --tar_start 1000 --tar_end 2000  # Overlaps!
```

**Do this:**
```bash
# GOOD: Non-overlapping ranges
python scripts/add_swap_negatives.py --tar_start 0 --tar_end 1000
python scripts/add_swap_negatives.py --tar_start 1000 --tar_end 2000
```

### Finding How Many Files You Have

```bash
# Count JSON files
ls /path/to/input/*.json | wc -l

# Or use Python
python -c "from pathlib import Path; print(len(list(Path('/path/to/input').glob('*.json'))))"
```

## Performance Considerations

**Tar range does NOT affect per-job speed**, it only determines which files to process.

Speed depends on:
- `--num_processes`: More processes = faster (up to CPU count)
- `--batch_size`: Larger batches = faster (but more memory)
- Hardware: CPU speed, memory, disk I/O

**Example timing:**
- 1000 files with 8 processes and batch_size=1000: ~5-10 minutes
- 5000 files with 8 processes and batch_size=1000: ~25-50 minutes
- 10700 files with 8 processes and batch_size=1000: ~50-100 minutes

**With distributed processing (11 jobs of 1000 files each, running in parallel):**
- Total time: ~5-10 minutes (same as single job!)

This is the power of distributed processing! ⚡

## Summary

**Tar ranges enable:**
- ✅ Distributed processing across multiple nodes
- ✅ Incremental processing over time
- ✅ Easy resumption after failures
- ✅ Flexible resource allocation
- ✅ Testing on specific file ranges

**Quick commands:**
```bash
# Process specific range
python scripts/add_swap_negatives.py --tar_start 0 --tar_end 1000

# Generate all commands
python scripts/generate_tar_range_jobs.py --total_files 10700 --files_per_job 1000

# Submit SLURM array job
python scripts/generate_tar_range_jobs.py --total_files 10700 --num_jobs 11 > submit.sh
sbatch submit.sh
```

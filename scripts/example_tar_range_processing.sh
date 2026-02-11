#!/bin/bash
# Example: Process files in tar ranges for distributed processing

# This script demonstrates how to split processing into multiple jobs
# using tar ranges. Useful for distributing work across nodes or running
# multiple jobs in parallel.

# Assuming you have 10700 JSON files to process, split into 11 jobs of ~1000 files each

echo "Starting distributed tar range processing..."
echo "Total files: 10700"
echo "Jobs: 11 (each processing ~1000 files)"
echo ""

# Job 0: Files 0-999
echo "Starting Job 0: Files 0-999"
python scripts/add_swap_negatives.py \
    --tar_start 0 \
    --tar_end 1000 \
    --num_processes 8 \
    --batch_size 1000 &

# Job 1: Files 1000-1999
echo "Starting Job 1: Files 1000-1999"
python scripts/add_swap_negatives.py \
    --tar_start 1000 \
    --tar_end 2000 \
    --num_processes 8 \
    --batch_size 1000 &

# Job 2: Files 2000-2999
echo "Starting Job 2: Files 2000-2999"
python scripts/add_swap_negatives.py \
    --tar_start 2000 \
    --tar_end 3000 \
    --num_processes 8 \
    --batch_size 1000 &

# Wait for first 3 jobs to complete
wait

echo ""
echo "First batch of jobs complete!"
echo ""
echo "You can run more jobs in parallel like this, or use SLURM array jobs"
echo "for automatic distribution across cluster nodes."
echo ""
echo "To generate all commands automatically, run:"
echo "python scripts/generate_tar_range_jobs.py --total_files 10700 --files_per_job 1000"

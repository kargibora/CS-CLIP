"""
Generate commands for distributed tar range processing.

This script helps you split the processing of JSON files into multiple jobs
that can run in parallel on different nodes or at different times.

Usage:
    python scripts/generate_tar_range_jobs.py --total_files 10700 --files_per_job 1000
    python scripts/generate_tar_range_jobs.py --total_files 10700 --num_jobs 11
"""

import argparse
import math


def generate_tar_range_commands(
    total_files: int,
    files_per_job: int = None,
    num_jobs: int = None,
    input_dir: str = None,
    output_dir: str = None,
    num_processes: int = 8,
    batch_size: int = 1000,
    dry_run: bool = False,
):
    """
    Generate commands for tar range processing.
    
    Args:
        total_files: Total number of JSON files to process
        files_per_job: Number of files per job (alternative to num_jobs)
        num_jobs: Number of jobs to split into (alternative to files_per_job)
        input_dir: Input directory path (optional)
        output_dir: Output directory path (optional)
        num_processes: Number of processes per job
        batch_size: Batch size for processing
        dry_run: Whether to add --dry_run flag
    """
    # Determine split strategy
    if files_per_job is not None:
        num_jobs = math.ceil(total_files / files_per_job)
        actual_files_per_job = files_per_job
    elif num_jobs is not None:
        actual_files_per_job = math.ceil(total_files / num_jobs)
    else:
        raise ValueError("Must specify either files_per_job or num_jobs")
    
    print("="*80)
    print("TAR RANGE JOB GENERATOR")
    print("="*80)
    print(f"Total files: {total_files}")
    print(f"Number of jobs: {num_jobs}")
    print(f"Files per job: ~{actual_files_per_job}")
    print(f"Processes per job: {num_processes}")
    print(f"Batch size: {batch_size}")
    if dry_run:
        print("Mode: DRY RUN (no file saving)")
    print("="*80)
    print()
    
    # Generate commands
    commands = []
    for job_idx in range(num_jobs):
        tar_start = job_idx * actual_files_per_job
        tar_end = min((job_idx + 1) * actual_files_per_job, total_files)
        
        # Build command
        cmd_parts = [
            "python scripts/add_swap_negatives.py",
            f"--tar_start {tar_start}",
            f"--tar_end {tar_end}",
            f"--num_processes {num_processes}",
            f"--batch_size {batch_size}",
        ]
        
        if input_dir:
            cmd_parts.append(f"--input_dir {input_dir}")
        if output_dir:
            cmd_parts.append(f"--output_dir {output_dir}")
        if dry_run:
            cmd_parts.append("--dry_run")
        
        cmd = " \\\n    ".join(cmd_parts)
        commands.append((job_idx, tar_start, tar_end, cmd))
    
    # Print commands
    print("COMMANDS TO RUN:")
    print("="*80)
    print()
    
    for job_idx, tar_start, tar_end, cmd in commands:
        num_files = tar_end - tar_start
        print(f"# Job {job_idx}: Process files {tar_start}-{tar_end-1} ({num_files} files)")
        print(cmd)
        print()
    
    print("="*80)
    print()
    
    # Generate SLURM script if requested
    print("SLURM ARRAY JOB TEMPLATE:")
    print("="*80)
    print(generate_slurm_script(
        num_jobs=num_jobs,
        files_per_job=actual_files_per_job,
        total_files=total_files,
        input_dir=input_dir or "/mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components",
        output_dir=output_dir or "/mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components_with_swaps",
        num_processes=num_processes,
        batch_size=batch_size,
        dry_run=dry_run,
    ))
    print("="*80)
    print()
    
    # Generate parallel commands
    print("PARALLEL EXECUTION (GNU Parallel):")
    print("="*80)
    print("# Save commands to file")
    print("cat > tar_range_commands.txt << 'EOF'")
    for _, _, _, cmd in commands:
        print(cmd.replace(" \\\n    ", " "))
    print("EOF")
    print()
    print("# Run in parallel (e.g., 4 jobs at a time)")
    print("parallel -j 4 < tar_range_commands.txt")
    print("="*80)
    print()
    
    return commands


def generate_slurm_script(
    num_jobs: int,
    files_per_job: int,
    total_files: int,
    input_dir: str,
    output_dir: str,
    num_processes: int,
    batch_size: int,
    dry_run: bool,
):
    """Generate SLURM array job script."""
    
    dry_run_flag = "--dry_run" if dry_run else ""
    
    script = f"""#!/bin/bash
#SBATCH --job-name=swap_negatives
#SBATCH --array=0-{num_jobs-1}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={num_processes}
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/swap_negatives_%A_%a.out
#SBATCH --error=logs/swap_negatives_%A_%a.err

# Calculate tar range for this job
TAR_START=$((SLURM_ARRAY_TASK_ID * {files_per_job}))
TAR_END=$(((SLURM_ARRAY_TASK_ID + 1) * {files_per_job}))

# Ensure we don't exceed total files
if [ $TAR_END -gt {total_files} ]; then
    TAR_END={total_files}
fi

echo "Job $SLURM_ARRAY_TASK_ID: Processing files $TAR_START to $((TAR_END-1))"

# Run processing
python scripts/add_swap_negatives.py \\
    --input_dir {input_dir} \\
    --output_dir {output_dir} \\
    --tar_start $TAR_START \\
    --tar_end $TAR_END \\
    --num_processes {num_processes} \\
    --batch_size {batch_size} {dry_run_flag}

echo "Job $SLURM_ARRAY_TASK_ID: Complete"
"""
    return script


def main():
    parser = argparse.ArgumentParser(
        description="Generate commands for distributed tar range processing"
    )
    
    parser.add_argument(
        "--total_files",
        type=int,
        required=True,
        help="Total number of JSON files to process"
    )
    
    parser.add_argument(
        "--files_per_job",
        type=int,
        default=None,
        help="Number of files to process per job (alternative to --num_jobs)"
    )
    
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=None,
        help="Number of jobs to split into (alternative to --files_per_job)"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input directory path (optional, uses default if not specified)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory path (optional, uses default if not specified)"
    )
    
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes per job (default: 8)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Add --dry_run flag to commands (for testing)"
    )
    
    args = parser.parse_args()
    
    if args.files_per_job is None and args.num_jobs is None:
        parser.error("Must specify either --files_per_job or --num_jobs")
    
    generate_tar_range_commands(
        total_files=args.total_files,
        files_per_job=args.files_per_job,
        num_jobs=args.num_jobs,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_processes=args.num_processes,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

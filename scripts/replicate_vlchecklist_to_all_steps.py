#!/usr/bin/env python3
"""
Script to replicate VL_CheckList results from step 0 to all other steps in ablation CSV files.

This handles the case where VL_CheckList was only evaluated at step 0 (base model)
and needs to be replicated to other steps for consistent analysis.

Usage:
    python replicate_vlchecklist_to_all_steps.py [--dry-run] [--file FILE] [--dir DIR]
    
Arguments:
    --dry-run    Show what would be done without modifying files
    --file FILE  Process a single file
    --dir DIR    Process all CSV files in a directory (default: ablations folder)
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime


def get_unique_steps(df: pd.DataFrame) -> list:
    """Get all unique steps in the dataframe."""
    return sorted(df['step'].unique().tolist())


def check_vlchecklist_status(df: pd.DataFrame) -> dict:
    """
    Check the status of VL_CheckList data in the dataframe.
    
    Returns:
        dict with keys:
            - has_vlchecklist: bool
            - vlchecklist_steps: list of steps that have VL_CheckList data
            - all_steps: list of all steps in the file
            - needs_replication: bool
    """
    all_steps = get_unique_steps(df)
    vlchecklist_df = df[df['dataset'] == 'VL_CheckList']
    vlchecklist_steps = sorted(vlchecklist_df['step'].unique().tolist()) if not vlchecklist_df.empty else []
    
    has_vlchecklist = len(vlchecklist_steps) > 0
    needs_replication = has_vlchecklist and vlchecklist_steps == [0] and len(all_steps) > 1
    
    return {
        'has_vlchecklist': has_vlchecklist,
        'vlchecklist_steps': vlchecklist_steps,
        'all_steps': all_steps,
        'needs_replication': needs_replication
    }


def replicate_vlchecklist_to_steps(df: pd.DataFrame, target_steps: list) -> pd.DataFrame:
    """
    Replicate VL_CheckList rows from step 0 to all target steps.
    
    Args:
        df: Original dataframe
        target_steps: List of steps to replicate VL_CheckList data to
        
    Returns:
        New dataframe with replicated VL_CheckList rows appended
    """
    # Get VL_CheckList rows at step 0
    vlchecklist_step0 = df[(df['dataset'] == 'VL_CheckList') & (df['step'] == 0)].copy()
    
    if vlchecklist_step0.empty:
        print("  No VL_CheckList data at step 0 to replicate")
        return df
    
    # Create replicated rows for each target step
    new_rows = []
    for step in target_steps:
        if step == 0:
            continue  # Skip step 0, already exists
            
        replicated = vlchecklist_step0.copy()
        replicated['step'] = step
        # Update timestamp to current time to indicate replication
        replicated['timestamp'] = datetime.now().isoformat()
        new_rows.append(replicated)
    
    if new_rows:
        # Concatenate all new rows
        new_rows_df = pd.concat(new_rows, ignore_index=True)
        # Append to original dataframe
        result_df = pd.concat([df, new_rows_df], ignore_index=True)
        return result_df
    
    return df


def process_file(filepath: Path, dry_run: bool = False) -> bool:
    """
    Process a single CSV file.
    
    Args:
        filepath: Path to the CSV file
        dry_run: If True, don't modify the file
        
    Returns:
        True if file was modified (or would be in dry run), False otherwise
    """
    print(f"\nProcessing: {filepath.name}")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return False
    
    status = check_vlchecklist_status(df)
    
    print(f"  All steps: {status['all_steps']}")
    print(f"  VL_CheckList steps: {status['vlchecklist_steps']}")
    
    if not status['has_vlchecklist']:
        print("  No VL_CheckList data found - skipping")
        return False
    
    if not status['needs_replication']:
        if status['vlchecklist_steps'] == status['all_steps']:
            print("  VL_CheckList already exists for all steps - skipping")
        else:
            print(f"  VL_CheckList exists for steps {status['vlchecklist_steps']} - no replication needed")
        return False
    
    # Need to replicate
    steps_to_add = [s for s in status['all_steps'] if s != 0]
    print(f"  Will replicate VL_CheckList from step 0 to steps: {steps_to_add}")
    
    if dry_run:
        print("  [DRY RUN] Would modify file")
        return True
    
    # Perform replication
    new_df = replicate_vlchecklist_to_steps(df, status['all_steps'])
    
    # Save back to file
    new_df.to_csv(filepath, index=False)
    
    # Verify
    verify_df = pd.read_csv(filepath)
    new_status = check_vlchecklist_status(verify_df)
    print(f"  ✓ File updated. VL_CheckList now at steps: {new_status['vlchecklist_steps']}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Replicate VL_CheckList results from step 0 to all other steps in ablation CSV files.'
    )
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be done without modifying files')
    parser.add_argument('--file', type=str, 
                        help='Process a single file')
    parser.add_argument('--dir', type=str, 
                        help='Process all CSV files in a directory')
    
    args = parser.parse_args()
    
    # Default directory
    default_dir = Path('/mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally/experiments/evaluation/ablations')
    
    if args.file:
        # Process single file
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            return
        process_file(filepath, dry_run=args.dry_run)
    else:
        # Process directory
        target_dir = Path(args.dir) if args.dir else default_dir
        
        if not target_dir.exists():
            print(f"Error: Directory not found: {target_dir}")
            return
        
        csv_files = sorted(target_dir.glob('*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {target_dir}")
            return
        
        print(f"Found {len(csv_files)} CSV files in {target_dir}")
        if args.dry_run:
            print("[DRY RUN MODE - No files will be modified]")
        
        modified_count = 0
        for filepath in csv_files:
            if process_file(filepath, dry_run=args.dry_run):
                modified_count += 1
        
        print(f"\n{'='*60}")
        if args.dry_run:
            print(f"Summary: {modified_count} files would be modified")
        else:
            print(f"Summary: {modified_count} files modified")


if __name__ == '__main__':
    main()

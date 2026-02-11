#!/usr/bin/env python3
"""
Add binding negatives to JSON files by swapping attributes between component pairs.

Binding negatives test attribute-object binding by swapping attributes between
two components that both have attributes. For example:
    "blue dog" + "red car" → 
    - binding_neg for "blue dog": "red dog" (swapped attribute from "red car")
    - binding_neg for "red car": "blue car" (swapped attribute from "blue dog")

This is different from regular component negatives which change attributes independently.

Input JSON Format (from unified generation):
    {
        "sample_id": "...",
        "original_caption": "A blue dog and a red car",
        "positive_components": ["blue dog", "red car"],
        "negative_components": {...},
        ...
    }

Output adds "binding_negatives" field:
    {
        ...
        "binding_negatives": [
            {
                "component_1": "blue dog",
                "component_2": "red car", 
                "binding_neg_1": "red dog",
                "binding_neg_2": "blue car",
                "swap_type": "attribute_swap"
            }
        ]
    }

Usage:
    # Process COCO JSON files
    python scripts/add_binding_negatives.py \
        --input_dir swap_pos_json/coco_train \
        --output_dir swap_pos_json/coco_train_with_binding \
        --k_pairs 3
    
    # Process LAION JSON files
    python scripts/add_binding_negatives.py \
        --input_dir /path/to/laion_negatives \
        --output_dir /path/to/laion_with_binding \
        --k_pairs 2 \
        --num_processes 8
    
    # Process single JSON file
    python scripts/add_binding_negatives.py \
        --input_file sample.json \
        --output_file sample_with_binding.json \
        --k_pairs 3

    # Dry run to test without saving
    python scripts/add_binding_negatives.py \
        --input_dir swap_pos_json/coco_train \
        --dry_run \
        --k_pairs 3
"""

import os
import sys
import json
import argparse
import logging
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import combinations
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import spacy
except ImportError:
    print("ERROR: spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


@dataclass
class ComponentParts:
    """Parsed component with attributes and head noun."""
    original: str
    attributes: List[str]  # Adjectives, numbers (in order they appear)
    head_noun: str  # Main noun phrase (can be compound like "place mats")
    
    def reconstruct_with_new_attributes(self, new_attributes: List[str]) -> str:
        """Reconstruct component with new attributes, preserving structure."""
        if not new_attributes:
            return self.head_noun
        # Put attributes before the head noun
        return ' '.join(new_attributes + [self.head_noun])
    
    def reconstruct_with_new_noun(self, new_noun: str) -> str:
        """Reconstruct component with new head noun, preserving attributes."""
        if not self.attributes:
            return new_noun
        # Put original attributes before the new noun
        return ' '.join(self.attributes + [new_noun])


class BindingNegativeGenerator:
    """Generate binding negatives by swapping attributes between component pairs."""
    
    def __init__(self):
        """Initialize spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def parse_component(self, component: str, caption_doc=None) -> ComponentParts:
        """
        Parse a component into attributes and head noun.
        
        Strategy:
        - If caption_doc is provided, find the component span within it for better POS tags
        - Otherwise parse the component directly
        - Use simple heuristic: last token is head noun, everything before is attributes
        
        Examples:
            "blue dog" → attrs=["blue"], head="dog"
            "small metal chairs" → attrs=["small", "metal"], head="chairs"
            "white cat" → attrs=["white"], head="cat"
        """
        # Try to find component in caption for better parsing context
        if caption_doc is not None:
            caption_text = caption_doc.text.lower()
            comp_lower = component.lower()
            
            # Find the component span in the caption
            start_char = caption_text.find(comp_lower)
            if start_char != -1:
                end_char = start_char + len(comp_lower)
                # Get the tokens that overlap with this span
                tokens = []
                for token in caption_doc:
                    if token.idx >= start_char and token.idx < end_char:
                        tokens.append(token)
                    elif token.idx + len(token.text) > start_char and token.idx < end_char:
                        tokens.append(token)
                
                if tokens:
                    return self._extract_parts_from_tokens(component, tokens)
        
        # Fallback: parse component directly
        doc = self.nlp(component)
        return self._extract_parts_from_tokens(component, list(doc))
    
    def _extract_parts_from_tokens(self, original: str, tokens: list) -> ComponentParts:
        """
        Extract attributes and head noun from a list of spaCy tokens.
        
        Simple robust strategy:
        - The last token is the head noun (the main object)
        - All preceding tokens are attributes
        - This works regardless of POS tag errors
        """
        if not tokens:
            return ComponentParts(original=original, attributes=[], head_noun=original)
        
        if len(tokens) == 1:
            return ComponentParts(original=original, attributes=[], head_noun=tokens[0].text)
        
        # Simple: last token is head, rest are attributes
        head_noun = tokens[-1].text
        attributes = [t.text for t in tokens[:-1]]
        
        return ComponentParts(
            original=original,
            attributes=attributes,
            head_noun=head_noun
        )
    
    def can_swap_nouns(self, comp1: ComponentParts, comp2: ComponentParts) -> bool:
        """
        Check if two components can have their nouns swapped.
        
        Noun swap tests binding: keep attributes, swap the object.
        E.g., "large wooden table" + "small metal chairs" 
              -> "large wooden chairs" + "small metal table"
        
        Requirements:
        - At least one component must have attributes
        - Both must have different head nouns
        """
        # At least one must have attributes for binding test to be meaningful
        has_attrs = len(comp1.attributes) > 0 or len(comp2.attributes) > 0
        # Both must have different head nouns
        different_nouns = (comp1.head_noun.lower() != comp2.head_noun.lower() and
                          comp1.head_noun and comp2.head_noun)
        return has_attrs and different_nouns
    
    def swap_nouns(
        self, 
        comp1: ComponentParts, 
        comp2: ComponentParts
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Swap nouns between two components (keep each component's own attributes).
        
        Example:
            "large wooden table" + "small metal chairs"
            -> "large wooden chairs" (comp1's attrs + comp2's noun)
            -> "small metal table" (comp2's attrs + comp1's noun)
        
        Returns:
            Tuple of (new_comp1, new_comp2) with swapped nouns,
            or (None, None) if swap not possible.
        """
        if not self.can_swap_nouns(comp1, comp2):
            return None, None
        
        # Swap nouns: each component keeps its own attributes but gets the other's noun
        new_comp1 = comp1.reconstruct_with_new_noun(comp2.head_noun)
        new_comp2 = comp2.reconstruct_with_new_noun(comp1.head_noun)
        
        # Check that the swapped versions are different from originals
        if new_comp1.lower() == comp1.original.lower():
            return None, None
        if new_comp2.lower() == comp2.original.lower():
            return None, None
        
        return new_comp1, new_comp2
    
    def generate_binding_pair(
        self,
        component1: str,
        component2: str,
        caption: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a binding negative pair from two components by swapping nouns.
        
        Binding negatives test attribute-object binding by keeping attributes
        but swapping the objects between two components.
        
        Args:
            component1: First component string
            component2: Second component string  
            caption: Optional original caption for better parsing context
        
        Example:
            "large wooden table" + "small metal chairs"
            -> "large wooden chairs" + "small metal table"
        
        Returns:
            Dict with binding pair info, or None if no swap possible.
        """
        # Parse caption once for context if provided
        caption_doc = self.nlp(caption) if caption else None
        
        comp1_parts = self.parse_component(component1, caption_doc)
        comp2_parts = self.parse_component(component2, caption_doc)
        
        # Try noun swap (keeps attributes, swaps objects)
        new_comp1, new_comp2 = self.swap_nouns(comp1_parts, comp2_parts)
        if new_comp1 and new_comp2:
            return {
                'component_1': component1,
                'component_2': component2,
                'binding_neg_1': new_comp1,
                'binding_neg_2': new_comp2,
                'swap_type': 'noun_swap',
                'original_attrs_1': comp1_parts.attributes,
                'original_attrs_2': comp2_parts.attributes,
                'swapped_noun_1': comp2_parts.head_noun,
                'swapped_noun_2': comp1_parts.head_noun,
            }
        
        return None
    
    def generate_binding_negatives(
        self,
        components: List[str],
        k_pairs: int = 3,
        max_attempts: int = 50,
        caption: str = None
    ) -> List[Dict[str, Any]]:
        """
        Generate k binding negative pairs from a list of components.
        
        Randomly samples component pairs and attempts to create binding negatives.
        Stops when k successful pairs are found or max_attempts reached.
        
        Args:
            components: List of component strings
            k_pairs: Target number of successful binding pairs
            max_attempts: Maximum number of pair attempts
            caption: Optional original caption for better parsing context
            
        Returns:
            List of binding pair dicts (up to k_pairs)
        """
        if len(components) < 2:
            return []
        
        # Get all possible pairs
        all_pairs = list(combinations(range(len(components)), 2))
        
        if not all_pairs:
            return []
        
        # Shuffle pairs for random sampling
        random.shuffle(all_pairs)
        
        binding_pairs = []
        attempts = 0
        used_pairs = set()
        
        for i, j in all_pairs:
            if len(binding_pairs) >= k_pairs:
                break
            if attempts >= max_attempts:
                break
            
            attempts += 1
            
            # Skip if this pair was already used
            pair_key = (min(i, j), max(i, j))
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)
            
            # Try to generate binding pair
            result = self.generate_binding_pair(components[i], components[j], caption=caption)
            if result:
                binding_pairs.append(result)
        
        return binding_pairs
    
    def process_sample(
        self,
        sample: Dict[str, Any],
        k_pairs: int = 3
    ) -> Dict[str, Any]:
        """
        Add binding negatives to a single sample.
        
        Args:
            sample: Sample dict with 'positive_components' and 'original_caption' fields
            k_pairs: Target number of binding pairs
            
        Returns:
            Sample dict with added 'binding_negatives' field
        """
        components = sample.get('positive_components', [])
        caption = sample.get('original_caption', '')
        
        # Generate binding negatives
        binding_negatives = self.generate_binding_negatives(
            components, k_pairs=k_pairs, caption=caption
        )
        
        # Add to sample (don't overwrite existing binding_pairs from LLM)
        sample['binding_negatives'] = binding_negatives
        
        return sample
    
    def process_samples_batch(
        self,
        samples: List[Dict[str, Any]],
        k_pairs: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Process multiple samples in batch.
        
        Args:
            samples: List of sample dicts
            k_pairs: Target number of binding pairs per sample
            
        Returns:
            List of samples with binding negatives added
        """
        results = []
        for sample in samples:
            results.append(self.process_sample(sample, k_pairs=k_pairs))
        return results


def process_json_file(args) -> Dict[str, Any]:
    """
    Process a single JSON file and add binding negatives.
    
    Args:
        args: Tuple of (input_path, output_path, k_pairs, dry_run)
        
    Returns:
        Statistics dict
    """
    input_path, output_path, k_pairs, dry_run = args
    
    try:
        # Initialize generator for this process
        generator = BindingNegativeGenerator()
        
        # Load input file
        logging.info(f"Loading: {os.path.basename(input_path)}")
        with open(input_path, 'r') as f:
            samples = json.load(f)
        
        if not isinstance(samples, list):
            samples = [samples]
        
        # Process samples
        total_samples = len(samples)
        total_binding_pairs = 0
        samples_with_bindings = 0
        
        logging.info(f"Processing {total_samples} samples in {os.path.basename(input_path)}")
        for idx, sample in enumerate(samples):
            sample = generator.process_sample(sample, k_pairs=k_pairs)
            n_pairs = len(sample.get('binding_negatives', []))
            total_binding_pairs += n_pairs
            if n_pairs > 0:
                samples_with_bindings += 1
            
            # Log progress every 1000 samples
            if (idx + 1) % 1000 == 0:
                logging.info(f"  {os.path.basename(input_path)}: {idx+1}/{total_samples} samples")
        
        # Save output
        if not dry_run:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(samples, f, indent=2)
        
        return {
            'status': 'success',
            'input_file': str(input_path),
            'total_samples': total_samples,
            'samples_with_bindings': samples_with_bindings,
            'total_binding_pairs': total_binding_pairs,
            'avg_pairs_per_sample': total_binding_pairs / total_samples if total_samples > 0 else 0,
        }
        
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'input_file': str(input_path),
            'error': str(e),
        }


def process_directory(
    input_dir: str,
    output_dir: str,
    k_pairs: int = 3,
    num_processes: int = None,
    file_pattern: str = "*.json",
    dry_run: bool = False,
    tar_start: int = None,
    tar_end: int = None,
):
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Input directory with JSON files
        output_dir: Output directory for processed files
        k_pairs: Target binding pairs per sample
        num_processes: Number of parallel processes
        file_pattern: Glob pattern for JSON files
        dry_run: If True, don't save output
        tar_start: Start index for file range
        tar_end: End index for file range
    """
    # Find input files
    input_files = sorted(Path(input_dir).glob(file_pattern))
    
    if not input_files:
        logging.error(f"No JSON files found matching '{file_pattern}' in {input_dir}")
        return
    
    total_files = len(input_files)
    logging.info(f"Found {total_files} JSON files")
    
    # Apply range filtering
    tar_start = tar_start if tar_start is not None else 0
    tar_end = tar_end if tar_end is not None else total_files
    tar_end = min(tar_end, total_files)
    
    input_files = input_files[tar_start:tar_end]
    files_to_process = len(input_files)
    
    logging.info(f"Processing files {tar_start} to {tar_end-1} ({files_to_process} files)")
    
    # Create output directory
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    args_list = []
    for input_path in input_files:
        output_path = Path(output_dir) / input_path.name
        args_list.append((str(input_path), str(output_path), k_pairs, dry_run))
    
    # Determine number of processes
    if num_processes is None:
        num_processes = min(cpu_count(), files_to_process)
    num_processes = max(1, min(num_processes, files_to_process))
    
    logging.info(f"Using {num_processes} processes")
    
    # Process files
    start_time = time.time()
    
    if num_processes == 1:
        # Single process
        results = []
        for args in tqdm(args_list, desc="Processing files"):
            results.append(process_json_file(args))
    else:
        # Multi-process
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_json_file, args_list),
                total=len(args_list),
                desc="Processing files"
            ))
    
    elapsed = time.time() - start_time
    
    # Print summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print("\n" + "=" * 80)
    print("BINDING NEGATIVES GENERATION COMPLETE")
    print("=" * 80)
    
    if successful:
        total_samples = sum(r['total_samples'] for r in successful)
        total_with_bindings = sum(r['samples_with_bindings'] for r in successful)
        total_pairs = sum(r['total_binding_pairs'] for r in successful)
        
        print(f"✅ Files processed: {len(successful)}/{files_to_process}")
        print(f"📝 Total samples: {total_samples:,}")
        print(f"🔗 Samples with binding pairs: {total_with_bindings:,} ({100*total_with_bindings/total_samples:.1f}%)")
        print(f"🔄 Total binding pairs: {total_pairs:,}")
        print(f"   Average per sample: {total_pairs/total_samples:.2f}")
        print(f"⏱️  Time: {elapsed:.1f}s ({total_samples/elapsed:.1f} samples/s)")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} files")
        for r in failed[:5]:
            print(f"   - {r['input_file']}: {r.get('error', 'Unknown error')}")
    
    if dry_run:
        print("\n⚠️  DRY RUN - No files saved")
    else:
        print(f"\n📁 Output saved to: {output_dir}")
    
    print("=" * 80)


def process_single_file(
    input_file: str,
    output_file: str,
    k_pairs: int = 3,
    dry_run: bool = False,
):
    """Process a single JSON file."""
    result = process_json_file((input_file, output_file, k_pairs, dry_run))
    
    if result['status'] == 'success':
        print(f"✅ Processed {result['total_samples']} samples")
        print(f"🔗 Generated {result['total_binding_pairs']} binding pairs")
        print(f"   Average: {result['avg_pairs_per_sample']:.2f} pairs/sample")
        if not dry_run:
            print(f"📁 Saved to: {output_file}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")


def demo_mode():
    """Run a quick demo with sample data."""
    print("\n" + "=" * 80)
    print("BINDING NEGATIVES DEMO")
    print("=" * 80)
    
    generator = BindingNegativeGenerator()
    
    # Demo components
    demo_samples = [
        {
            "sample_id": "demo_1",
            "original_caption": "A blue dog and a red car",
            "positive_components": ["blue dog", "red car"],
        },
        {
            "sample_id": "demo_2", 
            "original_caption": "Three red apples in a blue bowl",
            "positive_components": ["three red apples", "blue bowl"],
        },
        {
            "sample_id": "demo_3",
            "original_caption": "A large wooden table with small metal chairs",
            "positive_components": ["large wooden table", "small metal chairs"],
        },
        {
            "sample_id": "demo_4",
            "original_caption": "Young woman with old man",
            "positive_components": ["young woman", "old man"],
        },
        {
            "sample_id": "demo_5",
            "original_caption": "White cat on black couch near brown dog",
            "positive_components": ["white cat", "black couch", "brown dog"],
        },
    ]
    
    for sample in demo_samples:
        print(f"\n📝 Caption: {sample['original_caption']}")
        print(f"   Components: {sample['positive_components']}")
        
        result = generator.process_sample(sample, k_pairs=3)
        
        if result['binding_negatives']:
            print(f"   ✅ Binding negatives:")
            for pair in result['binding_negatives']:
                print(f"      - {pair['component_1']} ↔ {pair['component_2']}")
                print(f"        → {pair['binding_neg_1']}, {pair['binding_neg_2']}")
                print(f"        ({pair['swap_type']})")
        else:
            print(f"   ⚠️  No binding pairs found")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Add binding negatives to JSON files by swapping attributes between components"
    )
    
    # Input/output options
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing input JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output JSON files"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Single input JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Single output JSON file"
    )
    
    # Generation options
    parser.add_argument(
        "--k_pairs",
        type=int,
        default=3,
        help="Target number of binding pairs per sample (default: 3)"
    )
    
    # Processing options
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: CPU count)"
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.json",
        help="Glob pattern for JSON files (default: *.json)"
    )
    parser.add_argument(
        "--tar_start",
        type=int,
        default=None,
        help="Start index for file range"
    )
    parser.add_argument(
        "--tar_end",
        type=int,
        default=None,
        help="End index for file range"
    )
    
    # Other options
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't save output files (for testing)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with sample data"
    )
    
    args = parser.parse_args()
    
    # Demo mode
    if args.demo:
        demo_mode()
        return
    
    # Single file mode
    if args.input_file:
        output_file = args.output_file or args.input_file.replace('.json', '_with_binding.json')
        process_single_file(
            args.input_file,
            output_file,
            k_pairs=args.k_pairs,
            dry_run=args.dry_run
        )
        return
    
    # Directory mode
    if args.input_dir:
        output_dir = args.output_dir or args.input_dir + "_with_binding"
        process_directory(
            input_dir=args.input_dir,
            output_dir=output_dir,
            k_pairs=args.k_pairs,
            num_processes=args.num_processes,
            file_pattern=args.file_pattern,
            dry_run=args.dry_run,
            tar_start=args.tar_start,
            tar_end=args.tar_end,
        )
        return
    
    # No input specified - show help
    parser.print_help()
    print("\n💡 Tip: Run with --demo to see examples")


if __name__ == "__main__":
    main()

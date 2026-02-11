import argparse
import pandas as pd
import json
import uuid
import os

from .parsing import parse_caption
from .generation import (
    generate_joint_negatives_batched,
    generate_lexicon_attribute_neg,
    perform_concept_based_swapping,
    generate_positive_captions_batched,
)
from .component_negatives import (
    generate_mixed_negatives,
)
from .relational_generation import (
    extract_relational_components_batched,
    generate_component_negatives_batched,  # Use the unified version from relational_generation
    generate_relational_negatives_batched,
)
from .unified_generation import (
    generate_unified_negatives_batched,
)
from .llm_utils import VLLMWrapper
from tqdm import tqdm
import webdataset as wds

def safe_decode(caption):
    # Accepts either bytes or str, returns str
    if isinstance(caption, bytes):
        return caption.decode('utf-8', errors='replace')
    return caption

def open_tsv(fname, folder):
    df = pd.read_csv(fname, sep='\t', names=["caption", "url"])
    df['folder'] = folder
    return df

def filter_caption_and_image(sample):
    return 'txt' in sample and (
        'jpg' in sample or 'png' in sample or 'jpeg' in sample
    )

def get_caption_key_dataloader(input_shards, batch_size=1024, num_workers=4):
    pipeline = [
        wds.SimpleShardList(input_shards),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.select(filter_caption_and_image),
        wds.to_tuple("txt", "__key__", "__url__"),
        wds.batched(batch_size, partial=False),
    ]

    dataset = wds.DataPipeline(*pipeline)
    data_loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    return data_loader

def get_shard_from_key(key):
    # If key looks like shardname/key, split it out
    if "::" in key:
        return key
    if "/" in key:
        # e.g. laion400m-data/00001/abcde12345
        split = key.split("/", 1)
        return f"{split[0]}::{split[1]}"
    return key

def main():
    parser = argparse.ArgumentParser(description="Generate hard negatives for image captions (modular version)")
    parser.add_argument('--input', type=str, help="CSV file with captions or a plain .txt")
    parser.add_argument('--output', type=str, default="negatives.json")
    parser.add_argument('--concepts', type=str, nargs='+', default=["color", "size", "shape", "material", "orientation", "relative_size"])
    parser.add_argument('--use_lexicons', action='store_true')
    parser.add_argument('--use_llm', action='store_true')
    parser.add_argument('--use_concept_detection', action='store_true')
    parser.add_argument('--use_object_replacement', action='store_true')
    parser.add_argument('--use_attribute_swapping', action='store_true')
    parser.add_argument('--n_neg', type=int, default=1)
    parser.add_argument('--llm_name', type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument('--llm_batch', type=int, default=4)
    parser.add_argument('--difficulty', type=str, choices=['easy', 'medium', 'hard'], default='hard')
    parser.add_argument('--attribute_lexicons', type=str, default="attribute_lexicons.json")
    parser.add_argument('--replace_any_attribute', action='store_true',
        help="Replace any attribute (adjective/adverb) with any visually plausible alternative, ignoring specific concepts.")
    parser.add_argument('--subset', type=int, nargs=2, metavar=('START', 'END'), help="Subset the dataset with start and end indices")
    parser.add_argument('--max_n_pairs', type=int, default=None, help="For each caption, use at most this many (attribute, object) pairs")
    parser.add_argument('--shards', type=str, default=None,
        help="WebDataset shards path, e.g. '/mnt/lustre/datasets/laion400m/laion400m-data/{00000..00010}.tar'")
    parser.add_argument('--coco_karpathy', type=str, default=None,
        help="Path to COCO Karpathy split JSON file (e.g., 'datasets/COCO/dataset_coco.json')")
    parser.add_argument('--coco_images_root', type=str, default=None,
        help="Root path for COCO images (e.g., 'datasets/COCO/'). Images are at {root}/{filepath}/{filename}")
    parser.add_argument('--coco_split', type=str, default=None, choices=['train', 'val', 'test', 'restval', None],
        help="Which COCO split to use (train/val/test/restval). If None, uses all splits.")
    parser.add_argument('--generate_positives', action='store_true',
        help="Generate positive captions by extracting and reconstructing visual components")
    parser.add_argument('--positives_output', type=str, default="positives.json",
        help="Output file for positive captions (default: positives.json)")
    
    # Component-based negative generation
    parser.add_argument('--use_component_negatives', action='store_true',
        help="Use component-based negative generation (modifies positive components)")
    parser.add_argument('--n_neg_per_component', type=int, default=2,
        help="Number of negative variants per component (default: 2)")
    parser.add_argument('--component_mixing_strategy', type=str, 
        choices=['random', 'single', 'all', 'mixed'], default='random',
        help="Strategy for mixing positive/negative components: "
             "random (mix random number), single (change one at a time), "
             "all (change all), mixed (alias for random)")
    parser.add_argument('--n_mixed_negatives', type=int, default=3,
        help="Number of mixed negative captions to generate per sample (default: 3)")
    
# After the component negatives arguments, add:
    parser.add_argument('--use_relational_extraction', action='store_true',
                    help='Extract components with their spatial/relational information')
    parser.add_argument('--use_relational_negatives', action='store_true',
                    help='Generate negatives by modifying spatial relations')
    parser.add_argument('--use_attribute_binding_negatives', action='store_true',
                    help='Generate negatives by swapping attributes between components (color/size/quantity swaps)')
    parser.add_argument('--n_relational_negatives', type=int, default=3,
                    help='Number of relational negatives per sample')
    parser.add_argument('--relational_change_types', nargs='+',
                    default=['relation_reversal', 'spatial_swap', 'attribute_change'],
                    choices=['relation_reversal', 'spatial_swap', 'attribute_change', 'relation_removal'],
                    help='Types of relational changes to apply')
    parser.add_argument('--relational_output', type=str, default=None,
                    help='Output file for relational extraction results (JSON)')
    
    # UNIFIED GENERATION (NEW - combines all steps into one)
    parser.add_argument('--use_unified_generation', action='store_true',
                    help='[FAST] Generate ALL negatives in a SINGLE LLM call (extraction + component negatives + binding pairs + relational negatives). Replaces --use_relational_extraction + --use_component_negatives + --use_relational_negatives.')
    parser.add_argument('--unified_n_component_neg', type=int, default=2,
                    help='Number of component negatives per component in unified mode (default: 2)')
    parser.add_argument('--unified_n_binding_pairs', type=int, default=2,
                    help='Number of binding pairs per caption in unified mode (default: 2)')
    parser.add_argument('--unified_n_relational_neg', type=int, default=3,
                    help='Number of relational negatives per caption in unified mode (default: 3)')

    args = parser.parse_args()

    # --- Load attribute lexicons (if present) ---
    try:
        with open(args.attribute_lexicons) as f:
            attribute_lexicons = json.load(f)
    except FileNotFoundError:
        print("Warning: attribute_lexicons.json not found. Lexicon-based methods will be unavailable.")
        attribute_lexicons = {}

    # --- Load captions ---
    if args.input:
        if args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
            if not 'caption' in df.columns:
                raise ValueError("CSV must have a 'caption' column")
            captions = df.to_dict(orient='records')
        elif args.input.endswith('.txt'):
            with open(args.input) as f:
                captions = [{"caption": line.strip()} for line in f if line.strip()]
        elif args.input.endswith('.tsv'):
            df = open_tsv(args.input, 'train')
            captions = df.to_dict(orient='records')
        else:
            raise ValueError("Input file must be .csv, .txt, or .tsv")
        for row in captions:
            if 'url' in row:
                row['sample_id'] = f"url::{row['url']}"
            elif 'image_path' in row:
                row['sample_id'] = f"path::{row['image_path']}"
            else:
                row['sample_id'] = str(uuid.uuid4())
    elif args.coco_karpathy is not None:
        # Load COCO Karpathy split format
        print(f"Loading COCO Karpathy split from: {args.coco_karpathy}")
        with open(args.coco_karpathy, 'r') as f:
            coco_data = json.load(f)
        
        captions = []
        images_root = args.coco_images_root or os.path.dirname(args.coco_karpathy)
        
        for img in tqdm(coco_data['images'], desc="Loading COCO captions"):
            # Filter by split if specified
            if args.coco_split is not None and img['split'] != args.coco_split:
                continue
            
            # Build image path: {images_root}/{filepath}/{filename}
            image_path = os.path.join(images_root, img['filepath'], img['filename'])
            imgid = img['imgid']
            
            # Each image can have multiple captions (sentences)
            for sent in img['sentences']:
                sentid = sent['sentid']
                caption_text = sent['raw'].strip()
                
                # Create unique sample_id: imgid_sentid (like wds_key format)
                sample_id = f"coco_{imgid}_{sentid}"
                
                captions.append({
                    "caption": caption_text,
                    "sample_id": sample_id,
                    "image_url": image_path,  # Full path to image
                    "wds_key": f"{imgid}_{sentid}",  # Unique key per caption
                    "image_path": image_path,
                    "imgid": imgid,
                    "sentid": sentid,
                    "split": img['split'],
                })
        
        print(f"Loaded {len(captions)} captions from COCO Karpathy split")
        if args.coco_split:
            print(f"  (filtered to split='{args.coco_split}')")
        
    elif args.shards is not None:
        print(f"Loading captions and keys from shards: {args.shards}")
        data_loader = get_caption_key_dataloader(args.shards, batch_size=4096, num_workers=4)
        caption_list, key_list = [], []
        for batch in tqdm(data_loader, desc="Extracting captions/keys from shards"):
            # Some WebDatasets may not have __url__, ignore if not present
            if len(batch) == 3:
                batch_captions, batch_keys, batch_urls = batch
            else:
                batch_captions, batch_keys = batch
                batch_urls = [None] * len(batch_captions)
            for caption, key, url in zip(batch_captions, batch_keys, batch_urls):
                # COMBINE shard::key if url is present
                if url:
                    shard = os.path.basename(url)
                    sample_id = f"{shard}::{key}"
                else:
                    sample_id = key

                # For this case, url = filename (.tar)
                caption_list.append({
                    "caption": safe_decode(caption),
                    "sample_id": sample_id,
                    "url": url if url is not None else "",
                    "__key__": key
                })
        captions = caption_list
        print(f"Loaded {len(captions)} captions from shards.")
    else:
        list_captions = [
            "a photo of two giraffe.",
            "the toaster is on the left side of the truck.",
            "the broccoli is smaller in size/scale than the bench.",
            "a dog and a blue cat.",
            "Blue and yellow umbrellas line the sunny beach.",
            "Three striped shirts are hanging on a rack.",
            "a red apple next to two green pears.",
            "A round white plate sits beneath a golden brown pancake.",
            "old book on top of a wooden desk.",
            "An excited child wearing a large red hat.",
            "glass bottle with cold milk on a table.",
            "five black birds flying above the lake.",
            "Small orange ball and a big blue ball.",
            "Yellow tulips in a tall glass vase.",
            "A shiny silver spoon lies beside the plate.",
            "THE ELEPHANT IS LARGER THAN THE HORSE.",
            "one puppy and three kittens playing together.",
            "A bright, patterned rug covers the wooden floor.",
            "Woman in green dress under a red umbrella.",
            "CUP OF COFFEE WITH FOAM ART ON THE SURFACE.",
            "Stack of old brown books beside a tiny white candle.",
            "A soft, fluffy pillow sits behind the cat.",
            "Green plant with long, thin leaves in a brown pot.",
            "Several round cookies on a square black tray.",
            "a tall glass of cold lemonade with two straws.",
            "silver car parked between a blue van and a yellow truck.",
            "Delicate crystal vase contains a single pink rose.",
            "boy in blue jeans and a striped t-shirt jumps high.",
            "An open laptop with a bright screen sits on the desk.",
            "A small, brown dog runs ahead of a large white dog."
        ]
        captions = [{"caption": caption, "sample_id": str(uuid.uuid4())} for caption in list_captions]

    # --- Subset if requested ---
    if args.subset:
        start, end = args.subset
        captions = captions[start:end]

    # --- LLM setup ---
    llm = None
    if args.use_llm or args.use_concept_detection or args.use_object_replacement:
        llm = VLLMWrapper(
            model_name=args.llm_name,
            batch_size=args.llm_batch
        )

    # --- Concept override ---
    if args.replace_any_attribute:
        concepts = ['any']
    else:
        concepts = args.concepts

    # --------------------------
    #      JOINT PIPELINE
    # --------------------------
    all_captions = []
    all_adjectives = []
    all_objects = []
    all_concepts = []
    all_sample_ids = []
    all_urls = []
    all_keys = []
    all_paths = []
    for caption_row in tqdm(captions, desc="Preparing joint negatives batch"):
        caption = caption_row['caption']
        sample_id = caption_row['sample_id']
        parsed = parse_caption(caption, concepts)
        for concept in concepts:
            pairs = [(adj, obj) for (adj, obj) in parsed['attributes'][concept] if adj and obj]
            if args.max_n_pairs is not None:
                pairs = pairs[:args.max_n_pairs]
            for (adj, obj) in pairs:
                all_captions.append(caption)
                all_adjectives.append(adj)
                all_objects.append(obj)
                all_concepts.append(concept)
                all_sample_ids.append(sample_id)
                all_urls.append(caption_row.get('url', ''))
                all_paths.append(caption_row.get('image_path', ''))
                all_keys.append(caption_row.get('__key__', ''))

    results = []
    if llm is not None and (args.use_concept_detection or args.use_object_replacement):
        print(f"Generating joint (attribute/object) negatives for {len(all_captions)} pairs...")
        joint_negatives = generate_joint_negatives_batched(
            llm,
            captions=all_captions,
            adjectives=all_adjectives,
            objects=all_objects,
            concepts=all_concepts,
            n_neg=args.n_neg,
            batch_size=args.llm_batch,
            hardness=args.difficulty
        )

        for idx, neg_list in enumerate(joint_negatives):
            caption = all_captions[idx]
            adj = all_adjectives[idx]
            obj = all_objects[idx]
            concept = all_concepts[idx]
            sample_id = all_sample_ids[idx]
            url = all_urls[idx]
            key = all_keys[idx]
            path = all_paths[idx]
            for entry in neg_list:
                if entry.get("attribute_negative"):
                    result = {
                        "sample_id": sample_id,
                        "caption": caption,
                        "negative": entry["attribute_negative"],
                        "type": "attribute_replacement",
                        "concept": concept if concept != 'any' else entry.get("attribute_concept", "unknown"),
                        "original_attribute": adj,
                        "original_object": obj,
                        "generation_method": "joint_attribute",
                    }
                    if url:
                        result["image_url"] = url
                    if path:
                        result["image_path"] = path
                    results.append(result)
                if entry.get("object_negative"):
                    result = {
                        "sample_id": sample_id,
                        "caption": caption,
                        "negative": entry["object_negative"],
                        "type": "object_replacement",
                        "concept": concept if concept != 'any' else entry.get("attribute_concept", "unknown"),
                        "original_attribute": adj,
                        "original_object": obj,
                        "generation_method": "joint_object",
                    }
                    if url:
                        result["image_url"] = url
                    if key:
                        result["wds_key"] = key
                    if path:
                        result["image_path"] = path
                    results.append(result)
                if entry.get("swap_negative"):
                    result = {
                        "sample_id": sample_id,
                        "caption": caption,
                        "negative": entry["swap_negative"],
                        "type": "attribute_swap",
                        "concept": concept if concept != 'any' else entry.get("attribute_concept", "unknown"),
                        "original_attribute": adj,
                        "original_object": obj,
                        "generation_method": "joint_swap",
                    }
                    if url:
                        result["image_url"] = url
                    if key:
                        result["wds_key"] = key
                    if path:
                        result["image_path"] = path
                    results.append(result)

    # --- Optionally, add lexicon-based negatives (unchanged) ---
    if args.use_lexicons:
        for caption_row in captions:
            caption = caption_row['caption']
            sample_id = caption_row['sample_id']
            parsed = parse_caption(caption, concepts)
            for concept in concepts:
                for (adj, obj) in parsed['attributes'][concept]:
                    if concept in attribute_lexicons:
                        neg = generate_lexicon_attribute_neg(caption, concept, adj, obj, attribute_lexicons)
                        if neg:
                            results.append({
                                "sample_id": sample_id,
                                "caption": caption,
                                "negative": neg,
                                "type": f"lexicon_{concept}",
                                "original_attribute": adj,
                                "bound_object": obj,
                            })

    # --- Optionally, attribute/object swapping or other extras here ---
    if args.use_attribute_swapping and llm is not None:
        print("Performing concept-based swapping...")
        swap_results = perform_concept_based_swapping(captions, concepts, {}, {c['caption']: c['sample_id'] for c in captions})
        results.extend(swap_results)
        print(f"Generated {len(swap_results)} swap-based negatives")

    # --- Generate positive captions if requested ---
    if args.generate_positives:
        if llm is None:
            print("Warning: --generate_positives requires LLM. Initializing LLM...")
            llm = VLLMWrapper(
                model_name=args.llm_name,
                batch_size=args.llm_batch
            )
        
        print("Generating positive captions by extracting and reconstructing components...")
        caption_texts = [c['caption'] for c in captions]
        positive_results = generate_positive_captions_batched(
            llm,
            captions=caption_texts,
            batch_size=args.llm_batch
        )
        
        # Add metadata to positive results
        for i, pos_result in enumerate(positive_results):
            caption_row = captions[i]
            pos_result['sample_id'] = caption_row['sample_id']
            if 'url' in caption_row:
                pos_result['image_url'] = caption_row['url']
            if '__key__' in caption_row:
                pos_result['wds_key'] = caption_row['__key__']
            if 'image_path' in caption_row:
                pos_result['image_path'] = caption_row['image_path']
        
        # Save positive results
        with open(args.positives_output, "w") as f:
            json.dump(positive_results, f, indent=2)
        
        print(f"Generated {len(positive_results)} positive captions and saved to {args.positives_output}")
        
        # --- Component-based negative generation ---
        if args.use_component_negatives:
            print("\n" + "="*80)
            print("COMPONENT-BASED NEGATIVE GENERATION")
            print("="*80)
            
            # Extract components from positive results
            components_list = [pr.get('components', []) for pr in positive_results]
            original_captions = [pr['original_caption'] for pr in positive_results]
            
            # Filter out samples with no components
            valid_indices = [i for i, comps in enumerate(components_list) if comps]
            if not valid_indices:
                print("Warning: No valid components found for negative generation")
            else:
                filtered_components = [components_list[i] for i in valid_indices]
                filtered_captions = [original_captions[i] for i in valid_indices]
                filtered_results = [positive_results[i] for i in valid_indices]
                
                print(f"Generating negative variants for {len(filtered_components)} samples "
                      f"with {sum(len(c) for c in filtered_components)} total components...")
                
                # Generate negative variants for each component
                component_negatives = generate_component_negatives_batched(
                    llm,
                    components_list=filtered_components,
                    original_captions=filtered_captions,
                    batch_size=args.llm_batch,
                    n_neg_per_component=args.n_neg_per_component,
                    hardness=args.difficulty
                )
                
                # Generate mixed negative captions
                print(f"Generating {args.n_mixed_negatives} mixed negative captions per sample...")
                component_based_results = []
                
                for i, comp_neg in enumerate(component_negatives):
                    positive_components = comp_neg['components']
                    negative_variants = comp_neg['negative_variants']
                    
                    # Generate mixed negatives
                    mixed_negatives = generate_mixed_negatives(
                        positive_components=positive_components,
                        negative_variants=negative_variants,
                        n_negatives=args.n_mixed_negatives,
                        strategy=args.component_mixing_strategy
                    )
                    
                    # Add to results with metadata
                    original_idx = valid_indices[i]
                    sample_id = captions[original_idx]['sample_id']
                    original_caption = original_captions[i]
                    
                    for mix_neg in mixed_negatives:
                        result_entry = {
                            "sample_id": sample_id,
                            "caption": original_caption,
                            "negative": mix_neg['negative_caption'],
                            "type": "component_based",
                            "strategy": mix_neg['strategy'],
                            "positive_components": positive_components,
                            "negative_components": mix_neg['components'],
                            "changes": mix_neg['changes'],
                            "num_changes": mix_neg['num_changes'],
                            "generation_method": "component_mixing"
                        }
                        
                        # Add metadata
                        if 'url' in captions[original_idx]:
                            result_entry["image_url"] = captions[original_idx]['url']
                        if '__key__' in captions[original_idx]:
                            result_entry["wds_key"] = captions[original_idx]['__key__']
                        if 'image_path' in captions[original_idx]:
                            result_entry["image_path"] = captions[original_idx]['image_path']
                        
                        component_based_results.append(result_entry)
                
                # Add component-based results to main results
                results.extend(component_based_results)
                print(f"Generated {len(component_based_results)} component-based negatives")

    # --- UNIFIED GENERATION (NEW - FASTEST PATH) ---
    if args.use_unified_generation:
        if llm is None:
            print("Warning: --use_unified_generation requires LLM. Initializing LLM...")
            llm = VLLMWrapper(
                model_name=args.llm_name,
                batch_size=args.llm_batch
            )
        
        print("\n" + "="*80)
        print("⚡ UNIFIED GENERATION - ALL NEGATIVES IN ONE CALL")
        print("="*80)
        print("This replaces the 3-step pipeline with a SINGLE LLM call.")
        print("Generates: extraction + component negatives + binding pairs + relational negatives")
        print("="*80 + "\n")
        
        caption_texts = [c['caption'] for c in captions]
        
        # Single unified call - does EVERYTHING
        unified_results = generate_unified_negatives_batched(
            llm,
            captions=caption_texts,
            batch_size=args.llm_batch,
            n_component_neg=args.unified_n_component_neg,
            n_binding_pairs=args.unified_n_binding_pairs,
            n_relational_neg=args.unified_n_relational_neg,
        )
        
        # Add metadata (sample_id, image_path, etc.) to each result
        for i, result in enumerate(unified_results):
            caption_row = captions[i]
            result['sample_id'] = caption_row['sample_id']
            
            # Add image-related metadata
            if 'url' in caption_row:
                result['image_url'] = caption_row['url']
            if '__key__' in caption_row:
                result['wds_key'] = caption_row['__key__']
            if 'image_path' in caption_row:
                result['image_path'] = caption_row['image_path']
            
            # Add COCO-specific metadata if present
            if 'imgid' in caption_row:
                result['imgid'] = caption_row['imgid']
            if 'sentid' in caption_row:
                result['sentid'] = caption_row['sentid']
            if 'split' in caption_row:
                result['split'] = caption_row['split']
        
        # Save unified results
        print(f"\n💾 Saving unified results to: {args.output}")
        with open(args.output, 'w') as f:
            json.dump(unified_results, f, indent=2)
        print(f"✓ Saved {len(unified_results)} unified results")
        
        # Print statistics
        total_components = sum(len(r['positive_components']) for r in unified_results)
        total_relations = sum(len(r['relations']) for r in unified_results)
        total_comp_negs = sum(
            sum(len(negs) for negs in r['negative_components'].values())
            for r in unified_results
        )
        total_binding_pairs = sum(len(r['binding_pairs']) for r in unified_results)
        # Count relational negatives (embedded within each relation)
        total_rel_negs = sum(
            len(rel.get('negatives', []))
            for r in unified_results
            for rel in r.get('relations', [])
        )
        
        print("\n📊 Final Statistics:")
        print(f"  - {total_components} components extracted")
        print(f"  - {total_relations} relations extracted")
        print(f"  - {total_comp_negs} component negatives generated")
        print(f"  - {total_binding_pairs} binding pairs generated")
        print(f"  - {total_rel_negs} relational negatives generated")
        
        print("\n✅ Unified generation complete!")
        return  # Exit early - no need for other paths

    # --- UNIFIED COMPONENT & RELATIONAL EXTRACTION ---
    if args.use_relational_extraction:
        if llm is None:
            print("Warning: --use_relational_extraction requires LLM. Initializing LLM...")
            llm = VLLMWrapper(
                model_name=args.llm_name,
                batch_size=args.llm_batch
            )
        
        print("\n" + "="*80)
        print("UNIFIED COMPONENT & RELATIONAL EXTRACTION")
        print("="*80)
        
        caption_texts = [c['caption'] for c in captions]
        
        # Step 1: Extract components and relations
        print(f"Extracting components and relations from {len(caption_texts)} captions...")
        relational_data = extract_relational_components_batched(
            llm,
            captions=caption_texts,
            batch_size=args.llm_batch
        )
        
        print(f"✓ Saved {len(relational_data)} relational extraction results")
        
        # Step 2: Generate negative components (includes binding pairs if enabled)
        print("\nGenerating negative components...")
        components_list = [rd['components'] for rd in relational_data]
        
        # Pass n_binding_pairs parameter if binding negatives are requested
        n_binding_pairs = 2 if args.use_attribute_binding_negatives else 0
        
        component_negatives_data = generate_component_negatives_batched(
            llm,
            components_list=components_list,
            original_captions=caption_texts,
            batch_size=args.llm_batch,
            n_neg_per_component=args.n_neg_per_component,
            n_binding_pairs=n_binding_pairs,  # NEW: Generate binding pairs in same call
        )
        
        #CRITICAL DEBUG: Check returned data structure
        print("\n=== CRITICAL DEBUG: Returned component_negatives_data ===")
        print(f"Length: {len(component_negatives_data)}")
        if len(component_negatives_data) > 0:
            print(f"First item keys: {list(component_negatives_data[0].keys())}")
            print(f"First item['negative_components']: {component_negatives_data[0]['negative_components']}")
        print("======================================================\n")
        
        
        # Debug: Check what component negatives we got
        total_with_negs = sum(1 for r in component_negatives_data if r.get('negative_components'))
        total_neg_count = sum(len(r.get('negative_components', {})) for r in component_negatives_data)
        print(f"DEBUG: Component negatives - {total_with_negs}/{len(component_negatives_data)} have negatives, {total_neg_count} total component mappings")
        if total_neg_count > 0:
            # Show first example
            for r in component_negatives_data:
                if r.get('negative_components'):
                    print(f"DEBUG: Example - Components: {list(r['negative_components'].keys())[:2]}")
                    print(f"DEBUG: Example - Negatives: {list(r['negative_components'].values())[:1]}")
                    break
        
        # Step 3: Generate relational negatives (only if --use_relational_negatives flag is set)
        relational_negatives_data = []
        
        print(f"\n🔍 DEBUG: args.use_relational_negatives = {args.use_relational_negatives}")
        print(f"🔍 DEBUG: Type = {type(args.use_relational_negatives)}")
        
        if args.use_relational_negatives:
            print("\n⚡ Generating relational negatives...")
            print(f"DEBUG: Total relational_data entries: {len(relational_data)}")
            print(f"DEBUG: Entries with relations: {sum(1 for r in relational_data if r.get('relations'))}")
            
            relational_negatives_data = generate_relational_negatives_batched(
                llm,
                relational_data=relational_data,
                batch_size=args.llm_batch,
                n_negatives=args.n_relational_negatives,
            )
            print(f"DEBUG: Got {len(relational_negatives_data)} relational negative entries")
            if relational_negatives_data:
                print(f"DEBUG: First entry keys: {list(relational_negatives_data[0].keys())}")
                print(f"DEBUG: First entry original_index: {relational_negatives_data[0].get('original_index')}")
                print(f"DEBUG: First entry caption: '{relational_negatives_data[0]['original_caption'][:50]}'")
                # Count total negatives across all relations in first entry
                first_entry_relations = relational_negatives_data[0].get('relations', [])
                total_negs = sum(len(r.get('negatives', [])) for r in first_entry_relations)
                print(f"DEBUG: First entry has {len(first_entry_relations)} relations with {total_negs} total negatives")
                print(f"DEBUG: All original_index values: {[rn.get('original_index') for rn in relational_negatives_data[:5]]}")
            else:
                print("⚠ WARNING: No relational negatives were generated!")
            
        
        # Step 4: Extract binding pairs from component_negatives_data (already generated in Step 2)
        # Binding pairs are now part of component_negatives_data under 'binding_pairs' key
        if args.use_attribute_binding_negatives:
            print("\n⚡ Extracting attribute binding pairs from component negatives...")
            total_binding_pairs = sum(len(r.get('binding_pairs', [])) for r in component_negatives_data)
            captions_with_bindings = sum(1 for r in component_negatives_data if r.get('binding_pairs'))
            print(f"✓ Found {total_binding_pairs} binding pairs across {captions_with_bindings} captions")
            
            # Optional: Save binding pairs separately for inspection
            if total_binding_pairs > 0:
                intermediate_binding_pairs = args.positives_output.replace('.json', '_step4_binding_pairs.json')
                binding_pairs_only = [
                    {
                        'original_caption': r['original_caption'],
                        'components': r['positive_components'],
                        'binding_pairs': r['binding_pairs']
                    }
                    for r in component_negatives_data if r.get('binding_pairs')
                ]
                with open(intermediate_binding_pairs, 'w') as f:
                    json.dump(binding_pairs_only, f, indent=2)
                print(f"💾 Saved binding pairs to: {intermediate_binding_pairs}")
        
        # Step 5: Merge all data into unified structure
        # NOTE: relational_data, component_negatives_data, and captions all have same indices
        unified_results = []
        skipped_empty = 0
        skipped_placeholder = 0
        
        for i, caption_row in enumerate(captions):
            components = relational_data[i]['components']
            comp_negs = component_negatives_data[i].get('negative_components', {})
            
            # Debug first few
            if i < 3:
                print(f"DEBUG[{i}]: caption='{caption_texts[i][:50]}', components={components}, neg_components keys={list(comp_negs.keys())}")
            
            # Skip samples with no components or empty/placeholder components
            if not components or len(components) == 0:
                skipped_empty += 1
                print(f"⊗ Skipped (no components): {caption_row['sample_id'][:50]} - '{caption_texts[i][:60]}'")
                continue
            
            # Check for placeholder/invalid content
            has_placeholder = any(
                c in ['...', '.', '', ' '] or (isinstance(c, str) and c.strip() in ['', '...', '.'])
                for c in components
            )
            if has_placeholder:
                skipped_placeholder += 1
                print(f"⊗ Skipped (placeholders): {caption_row['sample_id'][:50]} - components: {components}")
                continue
            
            entry = {
                'sample_id': caption_row['sample_id'],
                'original_caption': caption_texts[i],
                'positive_components': components,
                'negative_components': comp_negs,  # Use the pre-fetched value
            }
            
            # Add relations with embedded negatives - use INDEX-based matching
            # The new format stores negatives WITHIN each relation object
            rel_neg = next(
                (rn for rn in relational_negatives_data if rn.get('original_index', -1) == i),
                None
            )
            if rel_neg and rel_neg.get('relations'):
                # Use the updated relations (which have negatives embedded)
                entry['relations'] = rel_neg['relations']
                # Count total negatives across all relations
                total_negs = sum(len(r.get('negatives', [])) for r in rel_neg['relations'])
                if i < 5:
                    print(f"✓ DEBUG[{i}]: Matched relational data by INDEX={i}, {len(rel_neg['relations'])} relations with {total_negs} total negatives")
            else:
                # Fall back to original relations (without negatives)
                entry['relations'] = relational_data[i].get('relations', [])
                # Add empty negatives list to each relation for consistency
                for rel in entry['relations']:
                    if 'negatives' not in rel:
                        rel['negatives'] = []
                # Debug why no match
                has_relations = bool(relational_data[i].get('relations'))
                if i < 5:
                    if has_relations:
                        print(f"✗ DEBUG[{i}]: No relational negative match (has {len(relational_data[i]['relations'])} relations)")
                        print(f"  Available indices in relational_negatives_data: {[rn.get('original_index') for rn in relational_negatives_data[:10]]}")
                    else:
                        print(f"○ DEBUG[{i}]: No relations in original data (skip expected)")
            
            # Add attribute binding pairs if they exist - already in component_negatives_data
            binding_pairs = component_negatives_data[i].get('binding_pairs', [])
            entry['binding_pairs'] = binding_pairs
            
            if i < 5:
                if binding_pairs:
                    print(f"✓ DEBUG[{i}]: Found {len(binding_pairs)} binding pairs from component negatives")
                elif args.use_attribute_binding_negatives:
                    print(f"○ DEBUG[{i}]: No binding pairs (need 2+ components with attributes)")
            
            # Add metadata
            if 'url' in caption_row:
                entry['image_url'] = caption_row['url']
            if '__key__' in caption_row:
                entry['wds_key'] = caption_row['__key__']
            if 'image_path' in caption_row:
                entry['image_path'] = caption_row['image_path']
            
            unified_results.append(entry)
        
        # Save unified results
        unified_output = args.relational_output or args.positives_output.replace('.json', '_unified.json')
        with open(unified_output, 'w') as f:
            json.dump(unified_results, f, indent=2)
        
        # Count relational negatives (now embedded within each relation)
        total_relations = sum(len(r['relations']) for r in unified_results)
        total_rel_negatives = sum(
            len(rel.get('negatives', []))
            for r in unified_results
            for rel in r.get('relations', [])
        )
        relations_with_negatives = sum(
            1 for r in unified_results
            for rel in r.get('relations', [])
            if rel.get('negatives')
        )
        
        print("\n✓ Generated unified component/relational data:")
        print(f"  • {len(unified_results)} samples processed (skipped {skipped_empty} empty, {skipped_placeholder} with placeholders)")
        print(f"  • {sum(len(r['positive_components']) for r in unified_results)} total positive components")
        print(f"  • {sum(len(r['negative_components']) for r in unified_results)} components with negatives")
        print(f"  • {total_relations} total relations ({relations_with_negatives} with negatives)")
        print(f"  • {total_rel_negatives} relational negatives (embedded in relations)")
        print(f"  • {sum(len(r['binding_pairs']) for r in unified_results)} attribute binding pairs")
        print(f"  • Saved to: {unified_output}")
        
    # --- Save negatives results (after all generation is complete) ---
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {len(results)} negatives and saved to {args.output}")

if __name__ == "__main__":
    main()

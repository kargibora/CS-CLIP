import os
import json
import random
import re
import sys
import requests
import pandas as pd
import spacy
import nltk
from nltk.corpus import wordnet as wn
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import time
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import hashlib

# Ensure WordNet data is available
try:
    wn.synsets('person')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")

# ------------------------
# Attribute Lexicons and Expansion
# ------------------------
# (Define attribute_lexicons as before)
attribute_lexicons = {
    "color": {
        "seed": [
            "red", "blue", "green", "yellow", "black", "white",
            "orange", "purple", "pink", "brown", "gray", "grey",
            "golden", "silver", "bright", "dark",
        ],
        "flips": {
            # primary hues flip among one another
            **{c: [h for h in
                   ["red","blue","green","yellow","black","white",
                    "orange","purple","pink","brown","gray","grey"]
                   if h != c]
               for c in ["red","blue","green","yellow","black","white",
                         "orange","purple","pink","brown","gray","grey"]},
            "golden": ["silver"],
            "silver": ["golden"],
            "bright": ["dark", "pale"],
            "dark": ["bright"]
            "pale": ["bright"]
        }
    },
 "size": {
        "seed": [
            "tiny", "minuscule", "small", "little",
            "big", "large", "oversized", "huge", "gigantic", "massive", "colossal"
        ],
        "flips": {
            "tiny": ["large", "big", "oversized", "huge", "gigantic", "massive", "colossal"],
            "minuscule": ["large", "big", "oversized", "huge", "gigantic", "massive", "colossal"],
            "small": ["large", "big", "oversized", "huge", "gigantic", "massive", "colossal"],
            "little": ["large", "big", "oversized", "huge", "gigantic", "massive", "colossal"],
            "big": ["tiny", "minuscule", "small", "little"],
            "large": ["tiny", "minuscule", "small", "little"],
            "oversized": ["tiny", "minuscule", "small", "little"],
            "huge": ["tiny", "minuscule", "small", "little"],
            "gigantic": ["tiny", "minuscule", "small", "little"],
            "massive": ["tiny", "minuscule", "small", "little"],
            "colossal": ["tiny", "minuscule", "small", "little"],
        }
    },
    "relative_size": {
        "seed": [
            "smaller", "larger", "bigger", "tinier", "shorter", "taller", "wider", "narrower"
        ],
        "flips": {
            "smaller": ["larger", "bigger", "wider", "taller"],
            "larger": ["smaller", "tinier", "narrower", "shorter"],
            "bigger": ["smaller", "tinier"],
            "tinier": ["larger", "bigger"],
            "shorter": ["taller"],
            "taller": ["shorter"],
            "wider": ["narrower"],
            "narrower": ["wider"],
        }
    },

    "shape": {
        "seed": [
            "round", "circular", "oval", "sphere", "cylinder",
            "square", "rectangular", "cube", "triangle", "triangular",
            "cone", "pyramid", "star-shaped", "crescent",
            "heart-shaped", "hexagonal", "octagonal"
        ],
        "flips": {
            **{x: [y for y in [
                "round","circular","oval","sphere","cylinder",
                "square","rectangular","cube","triangle","triangular",
                "cone","pyramid","star-shaped","crescent",
                "heart-shaped","hexagonal","octagonal"
            ] if y != x] for x in [
                "round","circular","oval","sphere","cylinder",
                "square","rectangular","cube","triangle","triangular",
                "cone","pyramid","star-shaped","crescent",
                "heart-shaped","hexagonal","octagonal"
            ]}
        }
    },

    "spatial_relation": {
        "seed": [
            "in front of", "behind", "left of", "right of",
            "above", "below", "next to", "near", "far from",
            "between", "among", "atop", "beneath",
            "inside", "outside", "through", "around"
        ],
        "flips": {
            "in front of": ["behind"],
            "behind": ["in front of"],
            "left of": ["right of"],
            "right of": ["left of"],
            "above": ["below", "beneath"],
            "below": ["above", "atop"],
            "atop": ["below", "underneath"],
            "next to": ["far from", "away from"],
            "near": ["far from"],
            "far from": ["near"],
            "between": ["among"],
            "among": ["between"],
            "inside": ["outside"],
            "outside": ["inside"],
            "through": ["around"],
            "around": ["through"]
        }
    },

    "material": {
        "seed": [
            "wooden", "metallic", "plastic", "glass", "ceramic",
            "stone", "leather", "fabric", "paper", "rubber"
        ],
        "flips": {
            # often no simple antonyms, but if you want to “flip” material, pick a random different one
            **{m: [n for n in [
                "wooden","metallic","plastic","glass","ceramic",
                "stone","leather","fabric","paper","rubber"
            ] if n != m] for m in [
                "wooden","metallic","plastic","glass","ceramic",
                "stone","leather","fabric","paper","rubber"
            ]}
        }
    },

    "pattern": {
        "seed": [
            "striped", "spotted", "dotted", "checkered",
            "plaid", "floral", "geometric", "paisley"
        ],
        "flips": {
            **{p: [q for q in [
                "striped","spotted","dotted","checkered",
                "plaid","floral","geometric","paisley"
            ] if q != p] for p in [
                "striped","spotted","dotted","checkered",
                "plaid","floral","geometric","paisley"
            ]}
        }
    }
}

def split_df_by_attribute(df: pd.DataFrame, lexicons: dict) -> dict:
    """Returns dict mapping each attribute to a subset DataFrame containing that attribute."""
    attr_dfs = {}
    docs = list(nlp.pipe(df['caption'].astype(str), disable=["ner"]))
    for attr, data in lexicons.items():
        idxs = []
        for i, doc in enumerate(docs):
            for tok in doc:
                if tok.dep_ == 'amod' and tok.pos_ == 'ADJ' and tok.lemma_.lower() in data['seed']:
                    idxs.append(i)
                    break
        attr_dfs[attr] = df.iloc[idxs].copy().reset_index(drop=True)
    return attr_dfs


def open_tsv(fname, folder):
    df = pd.read_csv(fname, sep='\t', names=["caption", "url"])
    df['folder'] = folder
    return df

def get_num_workers():
    """Returns the number of CPU cores available."""
    try:
        return min(os.cpu_count(), 8)
    except NotImplementedError:
        return 1
    

# ------------------------
# Negative Generators
# ------------------------
def generate_sampled_negative_captions_adj(caption, attribute, lexicons, n_samples=1):
    """
    Generate up to n_samples unique negative captions by flipping each possible attribute occurrence in captions,
    ensuring that only one attribute is changed per negative (no multi-flip).
    """
    doc = nlp(caption)
    flips = lexicons[attribute]['flips']
    all_negatives = set()

    # Find all flippable adjective tokens (could be the same lemma in multiple places!)
    amod_tokens = [tok for tok in doc if tok.dep_ == 'amod' and tok.lemma_.lower() in flips]

    for tok in amod_tokens:
        lemma = tok.lemma_.lower()
        orig = tok.text
        for flip in flips[lemma]:
            rep = flip.title() if orig[0].isupper() else flip

            # Replace ONLY this specific occurrence by its index
            neg_caption = (
                caption[:tok.idx] +
                rep +
                caption[tok.idx+len(orig):]
            )
            if neg_caption != caption:
                all_negatives.add(neg_caption)

    all_negatives = list(all_negatives)
    # Sample without replacement, up to n_samples
    if len(all_negatives) > n_samples:
        sampled = random.sample(all_negatives, n_samples)
    else:
        sampled = all_negatives

    return sampled

def get_wordnet_sibling_nouns(lemma, exclude=None, min_count=1):
    # Get WordNet siblings (hyponyms under a shared hypernym), optionally excluding some
    exclude = set([lemma]) if exclude is None else set(exclude)
    siblings = set()
    for syn in wn.synsets(lemma, pos=wn.NOUN):
        for hyper in syn.hypernyms():
            for hypo in hyper.hyponyms():
                for lem in hypo.lemmas():
                    name = lem.name().lower().replace('_', ' ')
                    if name.isalpha() and name not in exclude:
                        siblings.add(name)
    # Try more general siblings if not enough
    if len(siblings) < min_count:
        for syn in wn.synsets(lemma, pos=wn.NOUN):
            for hyper in syn.hypernyms():
                for hyper2 in hyper.hypernyms():
                    for hypo in hyper2.hyponyms():
                        for lem in hypo.lemmas():
                            name = lem.name().lower().replace('_', ' ')
                            if name.isalpha() and name not in exclude:
                                siblings.add(name)
    siblings = list(siblings)
    if not siblings:
        siblings = []
    return siblings

def generate_sampled_negative_captions_obj(caption, n_samples=1):
    """
    Generate up to n_samples unique negative captions by flipping a single object (noun)
    to a similar but distinct noun (WordNet sibling) per caption.
    """
    doc = nlp(caption)
    all_negatives = set()

    # Find all nouns/objects (allow multiple per caption)
    noun_tokens = [tok for tok in doc if tok.pos_ in {"NOUN", "PROPN"}]

    for tok in noun_tokens:
        lemma = tok.lemma_.lower()
        orig = tok.text
        siblings = get_wordnet_sibling_nouns(lemma, exclude=[lemma])
        # To avoid e.g. "person"->"people", you may want only single-word, or only nouns with freq > X, etc.
        for sib in siblings:
            sib_word = sib.title() if orig[0].isupper() else sib
            neg_caption = re.sub(rf"\b{re.escape(orig)}\b", sib_word, caption, count=1)
            if neg_caption != caption:
                all_negatives.add(neg_caption)

    all_negatives = list(all_negatives)
    # Sample without replacement, up to n_samples
    if len(all_negatives) > n_samples:
        sampled = random.sample(all_negatives, n_samples)
    else:
        sampled = all_negatives

    return sampled


def generate_swapped_attribute_caption(caption, attribute, lexicons):
    doc = nlp(caption)
    seeds = set(lexicons[attribute]['seed'])
    pairs = [(tok, tok.head) for tok in doc if tok.dep_=='amod' and tok.pos_=='ADJ' and tok.lemma_.lower() in seeds and tok.head.pos_ in {"NOUN","PROPN"}]
    if len(pairs) >= 2:
        (tok1,_), (tok2,_) = pairs[:2]
        a1, a2 = tok1.text, tok2.text
        i1, i2 = tok1.idx, tok2.idx
        if i1 < i2:
            return caption[:i1] + a2 + caption[i1+len(a1):i2] + a1 + caption[i2+len(a2):]
        else:
            return caption[:i2] + a1 + caption[i2+len(a2):i1] + a2 + caption[i1+len(a1):]
    return caption

# ------------------------
# TextAugment Class
# ------------------------
class MaskedCaptionsDataset(Dataset):
    def __init__(self, masked):
        self.masked = masked
    def __len__(self):
        return len(self.masked)
    def __getitem__(self, idx):
        return self.masked[idx]

class TextAugment(object):
    def __init__(self, lexicons, attr_neg_n=2, obj_neg_n=2, swap_neg_n=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unmasker = pipeline(
            'fill-mask',
            model='distilroberta-base',
            device=0 if torch.cuda.is_available() else -1,
            top_k=5
        )
        self.nlp = spacy.load("en_core_web_sm", exclude=['ner','parser'])
        self.lexicons = lexicons
        self.attr_neg_n = attr_neg_n
        self.obj_neg_n = obj_neg_n
        self.swap_neg_n = swap_neg_n

    def mask_captions(self, data):
        doc, item = data
        text = doc.text

        # relation swap of first two NOUNs
        nouns = [t.text for t in doc if t.pos_ == 'NOUN']
        if len(nouns) > 1:
            t0, t1 = nouns[0], nouns[1]
            swapped = text.replace(t0, '__tmp__') \
                          .replace(t1, t0) \
                          .replace('__tmp__', t1)
            item['relation_aug_caption'] = swapped
        else:
            item['relation_aug_caption'] = '###'

        # mask one ADJ, NOUN, VERB each, and store original token
        for pos, key in [('ADJ','adj_aug_caption'),
                         ('NOUN','noun_aug_caption'),
                         ('VERB','verb_aug_caption')]:
            toks = [t.text for t in doc if t.pos_ == pos]
            if toks:
                to_mask = random.choice(toks)
                item[f'{key}_orig_token'] = to_mask
                item[key] = text.replace(to_mask, '<mask>', 1)
            else:
                item[key] = "###"
                item[f'{key}_orig_token'] = None

        return item

    def select_unmasked_captions(self, results):
        filled = []
        for result in results:
            if not result:
                filled.append("###")
            else:
                try:
                    filled.append(result[0]['sequence'])
                except (IndexError, KeyError):
                    filled.append("###")
        return filled

    def generate(self, dataset, save_name=None, batch_size=128):
        # 1) POS & mask
        docs = list(self.nlp.pipe([d['caption'] for d in dataset], n_process=8))
        dataset = list(map(self.mask_captions, zip(docs, dataset)))

        # 2) Build masked lists and maps
        adj_mask_map, noun_mask_map, verb_mask_map = {}, {}, {}
        adj_masked, noun_masked, verb_masked = [], [], []
        for i, item in enumerate(dataset):
            if item['adj_aug_caption'] != "###" and '<mask>' in item['adj_aug_caption']:
                adj_mask_map[len(adj_masked)] = i
                adj_masked.append(item['adj_aug_caption'])
            if item['noun_aug_caption'] != "###" and '<mask>' in item['noun_aug_caption']:
                noun_mask_map[len(noun_masked)] = i
                noun_masked.append(item['noun_aug_caption'])
            if item['verb_aug_caption'] != "###" and '<mask>' in item['verb_aug_caption']:
                verb_mask_map[len(verb_masked)] = i
                verb_masked.append(item['verb_aug_caption'])

        # 3) Fill masks
        with torch.no_grad():
            # ADJ
            adj_results = []
            for i in tqdm(range(0, len(adj_masked), batch_size), desc="Filling ADJ masks"):
                batch = [t for t in adj_masked[i:i+batch_size] if t.count('<mask>') == 1]
                if batch:
                    try:
                        results = self.unmasker(batch)
                        adj_results.extend(results)
                    except Exception as e:
                        print(f"Error processing ADJ batch: {e}")
                        # Add empty results for this batch to maintain indices
                        adj_results.extend([[] for _ in batch])
                else:
                    adj_results.extend([])
                    
            # NOUN
            noun_results = []
            for i in tqdm(range(0, len(noun_masked), batch_size), desc="Filling NOUN masks"):
                batch = [t for t in noun_masked[i:i+batch_size] if t.count('<mask>') == 1]
                if batch:
                    try:
                        results = self.unmasker(batch)
                        noun_results.extend(results)
                    except Exception as e:
                        print(f"Error processing NOUN batch: {e}")
                        noun_results.extend([[] for _ in batch])
                else:
                    noun_results.extend([])
                    
            # VERB
            verb_results = []
            for i in tqdm(range(0, len(verb_masked), batch_size), desc="Filling VERB masks"):
                batch = [t for t in verb_masked[i:i+batch_size] if t.count('<mask>') == 1]
                if batch:
                    try:
                        results = self.unmasker(batch)
                        verb_results.extend(results)
                    except Exception as e:
                        print(f"Error processing VERB batch: {e}")
                        verb_results.extend([[] for _ in batch])
                else:
                    verb_results.extend([])

        # 4) Process filled sequences, avoiding original token
        adj_filled_dict, noun_filled_dict, verb_filled_dict = {}, {}, {}

        # Process ADJ results
        for i, result in enumerate(adj_results):
            if i in adj_mask_map:
                idx = adj_mask_map[i]
                orig = dataset[idx].get('adj_aug_caption_orig_token')
                chosen = "###"
                
                # Check if result is a list of dictionaries (expected format)
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    for cand in result:
                        tok = cand.get('token_str', '').strip()
                        if not orig or tok.lower() != orig.lower():
                            chosen = cand['sequence']
                            break
                # Handle case where result might be a string or other unexpected format
                elif isinstance(result, dict):
                    # Single result as dict
                    tok = result.get('token_str', '').strip()
                    if not orig or tok.lower() != orig.lower():
                        chosen = result.get('sequence', "###")
                
                adj_filled_dict[idx] = chosen

        # Process NOUN results
        for i, result in enumerate(noun_results):
            if i in noun_mask_map:
                idx = noun_mask_map[i]
                orig = dataset[idx].get('noun_aug_caption_orig_token')
                chosen = "###"
                
                # Check if result is a list of dictionaries (expected format)
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    for cand in result:
                        tok = cand.get('token_str', '').strip()
                        if not orig or tok.lower() != orig.lower():
                            chosen = cand['sequence']
                            break
                # Handle case where result might be a string or other unexpected format
                elif isinstance(result, dict):
                    # Single result as dict
                    tok = result.get('token_str', '').strip()
                    if not orig or tok.lower() != orig.lower():
                        chosen = result.get('sequence', "###")
                
                noun_filled_dict[idx] = chosen

        # Process VERB results
        for i, result in enumerate(verb_results):
            if i in verb_mask_map:
                idx = verb_mask_map[i]
                orig = dataset[idx].get('verb_aug_caption_orig_token')
                chosen = "###"
                
                # Check if result is a list of dictionaries (expected format)
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    for cand in result:
                        tok = cand.get('token_str', '').strip()
                        if not orig or tok.lower() != orig.lower():
                            chosen = cand['sequence']
                            break
                # Handle case where result might be a string or other unexpected format
                elif isinstance(result, dict):
                    # Single result as dict
                    tok = result.get('token_str', '').strip()
                    if not orig or tok.lower() != orig.lower():
                        chosen = result.get('sequence', "###")
                
                verb_filled_dict[idx] = chosen

        # 5) Assign back to items
        for i, item in enumerate(dataset):
            if i in adj_filled_dict:
                item['adj_aug_caption'] = adj_filled_dict[i]
            elif item['adj_aug_caption'] != "###":
                item['adj_aug_caption'] = "###"

            if i in noun_filled_dict:
                item['noun_aug_caption'] = noun_filled_dict[i]
            elif item['noun_aug_caption'] != "###":
                item['noun_aug_caption'] = "###"

            if i in verb_filled_dict:
                item['verb_aug_caption'] = verb_filled_dict[i]
            elif item['verb_aug_caption'] != "###":
                item['verb_aug_caption'] = "###"

            # 6) Negative gens unchanged
            orig = item['caption']
            attr = item.get('attribute')
            # Uncomment these if needed
            # item['neg_attr_caption'] = [
            #     generate_negative_caption_adj(orig, attr, self.lexicons)
            #     for _ in range(self.attr_neg_n)
            # ]
            # item['neg_obj_caption'] = [
            #     generate_negative_caption_obj(orig)
            #     for _ in range(self.obj_neg_n)
            # ]
            # item['neg_swap_caption'] = [
            #     generate_swapped_attribute_caption(orig, attr, self.lexicons)
            #     for _ in range(self.swap_neg_n)
            # ]

        # 7) Optional save
        if save_name:
            with open(save_name, 'wb') as f:
                np.save(f, dataset)

        return dataset


def filter_augmented_results(dataset):
    """
    Filter the augmented dataset to keep only valid augmentations.
    Returns a new dataset with only valid entries.
    
    Valid entries are those where:
    1. The augmentation is not "###"
    2. The augmentation is different from the original caption
    """
    filtered_dataset = []
    
    for item in dataset:
        # Create a copy of the item to modify
        filtered_item = item.copy()
        
        # Check and filter for each augmentation type
        valid_types = []
        
        # Check ADJ augmentation
        if (item['adj_aug_caption'] != "###" and 
            item['adj_aug_caption'] != item['caption']):
            valid_types.append('adj')
        else:
            filtered_item['adj_aug_caption'] = "###"
        
        # Check NOUN augmentation
        if (item['noun_aug_caption'] != "###" and 
            item['noun_aug_caption'] != item['caption']):
            valid_types.append('noun')
        else:
            filtered_item['noun_aug_caption'] = "###"
        
        # Check VERB augmentation
        if (item['verb_aug_caption'] != "###" and 
            item['verb_aug_caption'] != item['caption']):
            valid_types.append('verb')
        else:
            filtered_item['verb_aug_caption'] = "###"
        
        # Check relation augmentation
        if (item['relation_aug_caption'] != "###" and 
            item['relation_aug_caption'] != item['caption']):
            valid_types.append('relation')
        else:
            filtered_item['relation_aug_caption'] = "###"
        
        # Add item if it has at least one valid augmentation type
        if valid_types:
            filtered_item['valid_aug_types'] = valid_types
            filtered_dataset.append(filtered_item)
    
    return filtered_dataset


def sanitize_filename(caption, max_length=100):
    """Create a safe filename from caption."""
    # Remove invalid characters and replace spaces with underscores
    safe_caption = re.sub(r'[^\w\s-]', '', caption).strip().replace(' ', '_')
    # Limit filename length
    if len(safe_caption) > max_length:
        safe_caption = safe_caption[:max_length]
    # If empty after sanitization, use a hash of the original
    if not safe_caption:
        safe_caption = hashlib.md5(caption.encode()).hexdigest()
    return safe_caption

def download_image(item):
    """Download a single image and return success status."""
    url = item['url']
    caption = item['caption']
    output_dir = item['output_dir']
    timeout = item.get('timeout', 3)
    
    try:
        # Use a session for connection pooling
        with requests.Session() as session:
            # Set timeout for both connection and read
            safe_filename = sanitize_filename(caption)

            # Get the directory name (last part of the path)
            base_dir = os.path.basename(output_dir)
            if os.path.exists(os.path.join(output_dir, f"{safe_filename}.jpg")):
                return {'success': True, 'url': url, 'filename': os.path.join(base_dir,f"{safe_filename}.jpg")}

            response = session.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            img_data = BytesIO(response.content)
            img = Image.open(img_data)
            
            # Verify image is valid
            img.verify()
            img_data.seek(0)  # Reset file pointer
            
            # Reopen and convert to RGB to handle PNG, GIF, etc.
            img = Image.open(img_data).convert('RGB')
            
            # Create a safe filename
            img_path = os.path.join(output_dir, os.path.join(base_dir,f"{safe_filename}.jpg"))
            
            # Save with optimized settings
            img.save(img_path, 'JPEG', quality=85, optimize=True)
            
            return {'success': True, 'url': url, 'filename': os.path.join(base_dir,f"{safe_filename}.jpg")}
            
    except (UnidentifiedImageError, requests.exceptions.RequestException, 
            OSError, ValueError, AttributeError) as e:
        return {'success': False, 'url': url, 'error': str(e)}

def download_images(images_data, output_dir, num_workers=8, batch_size=100, return_metadata=True):
    """
    Download images in parallel using multiple processes.
    
    Args:
        images_data: List of dicts with 'url' and 'caption' keys
        output_dir: Directory to save images
        num_workers: Number of parallel workers
        batch_size: Process images in batches to show progress
        return_metadata: Whether to return updated metadata with filepaths
        
    Returns:
        If return_metadata is True, returns a list of dictionaries with updated metadata
        including downloaded file paths.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare download items
    download_items = [
        {'url': item['url'], 
         'caption': item['caption'], 
         'output_dir': output_dir,
         'original_index': idx,  # Store original index to maintain order
         'original_item': item   # Store original item for metadata return
        } 
        for idx, item in enumerate(images_data)
    ]
    
    # Track statistics
    total_images = len(download_items)
    successful = 0
    failed = 0
    start_time = time.time()
    results = []  # To store results with filepath metadata
    
    print(f"Starting download of {total_images} images using {num_workers} workers")
    
    # Process in batches to show progress
    for i in range(0, len(download_items), batch_size):
        batch = download_items[i:i+batch_size]
        batch_results = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(download_image, item) for item in batch]
            
            # Show progress for this batch
            for future in tqdm(as_completed(futures), total=len(batch), 
                              desc=f"Batch {i//batch_size + 1}/{(total_images-1)//batch_size + 1}"):
                result = future.result()
                
                # Find the original item in the batch
                original_item = None
                for item in batch:
                    if item['url'] == result['url']:
                        original_item = item
                        break
                
                if result['success']:
                    successful += 1
                    # Create updated metadata with filepath
                    updated_metadata = original_item['original_item'].copy()
                    updated_metadata['filename'] = result['filename']
                    batch_results.append(updated_metadata)
                else:
                    failed += 1
                    # Add failed item with error info but no filepath
                    updated_metadata = original_item['original_item'].copy()
                    updated_metadata['error'] = result['error']
                    updated_metadata['download_success'] = False
                    batch_results.append(updated_metadata)
                    print(f"Failed: {result['url']} - {result['error']}")
        
        # Add batch results to overall results
        results.extend(batch_results)
    

    # Show summary
    elapsed = time.time() - start_time
    print(f"\nDownload complete in {elapsed:.2f} seconds")
    print(f"Total: {total_images}, Successful: {successful}, Failed: {failed}")
    print(f"Average speed: {successful/elapsed:.2f} images/second")
    
    if return_metadata:
        # Sort results by original index to maintain original order
        sorted_results = sorted(results, key=lambda x: next((i for i, item in enumerate(download_items) 
                                                            if item['original_item'] == x), 0))
        
        # Drop the failed items from the sorted results
        sorted_results = [item for item in sorted_results if 'error' not in item]

        return sorted_results

# ------------------------
# Main Processing
# ------------------------
def main():
    CC3M_PATH = '/mnt/qb/work/oh/owl336/thesis/CLIP-not-BoW-unimodally/datasets/CC3M'
    # Load a small sample for testing
    df = open_tsv(os.path.join(CC3M_PATH, 'GCC-training.tsv'), 'train').iloc[:10000]
    print(f"Loaded {len(df)} samples from {CC3M_PATH}")
    
    # splits = split_df_by_attribute(df, attribute_lexicons)
    attr = 'all'
    df_sub = df

    # for 'attr', df_sub in splits.items():
    print(f"Processing {attr} with {len(df_sub)} samples")

    # if df_sub.empty: 
    #     print(f"Skipping empty attribute: {attr}")
    #     continue
        
    items = df_sub.to_dict('records')
    for item in items:
        item['attribute'] = attr
        
    augmenter = TextAugment(attribute_lexicons)
    augmented = augmenter.generate(items, save_name=f'{attr}_processed.npy', batch_size=32)
    
    # augmented = filter_augmented_results(augmented)

    # Print some samples to verify the results
    print(f"\n===== {attr} SAMPLES =====")
    for i, sample in enumerate(augmented[:3]):  # Show first 3 samples
        print(f"Sample {i+1}:")
        print(f"Original: {sample['caption']}")
        print(f"ADJ  filled: {sample['adj_aug_caption']}")
        print(f"NOUN filled: {sample['noun_aug_caption']}")
        print(f"VERB filled: {sample['verb_aug_caption']}")
        print(f"Relation swap: {sample['relation_aug_caption']}")
        # print(f"Negative attr: {sample['neg_attr_caption'][0]}")
        print("------------------------------")
    
    # Save metadata as JSON
    meta = []
    print(f"Starting filtering augmented results...")
    filtered_augmented = filter_augmented_results(augmented)
    print(f"Filtered {len(filtered_augmented)} samples with valid augmentations.")
    
    images_output_dir = os.path.join(CC3M_PATH, 'images_neg')
    os.makedirs(images_output_dir, exist_ok=True)

    updated_metadata = download_images(
        images_data=augmented,
        output_dir=images_output_dir,
        num_workers=16,  # Adjust based on your CPU and network
        batch_size=100,  # Process images in batches
        return_metadata=True
    )
    
    # Now you can access the files using the updated metadata with filepaths
    print(f"Downloaded {len(updated_metadata)} images")
    
    # Example: Save the updated metadata to a JSON file for future reference
    import json
    with open(os.path.join(CC3M_PATH, 'download_metadata.json'), 'w') as f:
        json.dump(updated_metadata, f, indent=2)
    
    # Example: Access the first few successfully downloaded images
    print("\nSample of successfully downloaded images:")
    for i, item in enumerate(updated_metadata):
        if 'filename' in item and not item.get('error'):
            print(f"{i+1}. Caption: {item['caption']}")
            print(f"   filename: {item['filename']}")
        if i >= 4:  # Show just first 5 examples
            break

if __name__ == '__main__':
    main()
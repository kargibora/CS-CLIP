#!/usr/bin/env python3
"""
Dataset Visualization Script for Thesis

This script samples random examples from vision-language benchmarks and creates 
modern visualizations showcasing images with their captions (positive and negative).
Perfect for understanding dataset characteristics and creating thesis figures.

Usage:
    python generate_sampling_plot.py
    python generate_sampling_plot.py Winoground=5 VALSE=3 SugarCrepe=2
"""

import argparse
import os
import random
import sys
import textwrap
import warnings
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image

warnings.filterwarnings("ignore")

# Set up path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from data_loading import get_dataset_class
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the correct directory and have all dependencies installed")
    sys.exit(1)


class BenchmarkSampler:
    """Handles sampling and data extraction from different benchmark datasets."""
    
    def __init__(self, data_root_base: str = "./datasets"):
        self.data_root_base = data_root_base
        
        # Dataset configurations - 'all' means sample from all available subsets
        self.dataset_configs = {
            'Winoground': {'subset': 'all', 'path': 'Winoground'},
            'VG_Attribution': {'subset': 'all', 'path': 'VG_Attr'},
            'VG_Relation': {'subset': 'all', 'path': 'VG_Rel'},
            'COCO_Order': {'subset': 'all', 'path': 'COCO_Order'},
            'Flickr30k_Order': {'subset': 'all', 'path': 'Flickr30k_Order'},
            'VALSE': {'subset': 'all', 'path': 'VALSE'},
            'VL_CheckList': {'subset': 'all', 'path': 'VL-CheckList'},
            'SugarCrepe': {'subset': 'all', 'path': 'SugarCrepe'},  # Will sample from all subsets
            'SugarCrepe_PP': {'subset': 'all', 'path': 'SugarCrepe_PP'},  # Will sample from all subsets
            'ColorSwap': {'subset': 'all', 'path': 'ColorSwap'},
            'ColorFoil': {'subset': 'all', 'path': 'ColorFoil'},
            'COCO_Counterfactuals': {'subset': 'all', 'path': 'COCO-Counterfactuals'},
            'ControlledImages': {'subset': 'all', 'path': 'WhatsUp'},  # Will sample from A, B subsets
            'VisMin': {'subset': 'all', 'path': 'VisMin'},
            'BLA': {'subset': 'all', 'path': 'BLA_Benchmark'},  # Will sample from ap, co, rc subsets
            'SPEC_I2T': {'subset': 'all', 'path': 'SPEC'},  # Will sample from all reasoning types
            'COLA': {'subset': 'all', 'path': 'cola'},  # Will sample from all COLA subsets
            'NegBench': {'subset': 'all', 'path': 'negbench'}  # Will sample from all NegBench subsets
        }
    
    def create_mock_args(self, dataset_name: str, subset_name: str, data_path: str):
        """Create mock arguments object for dataset initialization."""
        class MockArgs:
            def __init__(self):
                self.dataset = dataset_name
                self.subset_name = subset_name
                self.data_path = data_path
                self.cache_folder = 'cache_temp_v2'
                
        return MockArgs()
    
    def load_dataset(self, dataset_name: str, config: Dict) -> Optional[Any]:
        """Load a dataset with error handling."""
        try:
            dataset_class = get_dataset_class(dataset_name)
            if dataset_class is None:
                print(f"❌ Dataset class not found: {dataset_name}")
                return None
                
            data_path = os.path.join(self.data_root_base, config['path'])
            
            # Special handling for different datasets - no preprocessing needed for visualization
            if dataset_name == 'Winoground':
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=config['subset'],
                    image_preprocess=None,
                    download=True,
                    use_auth_token=None  # Add your token if needed
                )
            elif dataset_name in ['VALSE']:
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=config['subset'],
                    image_preprocess=None,
                    download=False  # Let it use cached data automatically
                )
            elif dataset_name == 'SugarCrepe':
                # SugarCrepe needs both JSON files (in data_path) and COCO images (in coco_root)
                # Check if COCO images are in the same directory or need separate path
                coco_path = data_path  # Try same directory first
                if not os.path.exists(os.path.join(coco_path, 'val2017')):
                    # Try common alternative paths
                    coco_alternatives = [
                        os.path.join(data_path, 'coco'),
                        os.path.join(os.path.dirname(data_path), 'COCO', 'val2017'),
                        './datasets/COCO'
                    ]
                    for alt_path in coco_alternatives:
                        if os.path.exists(os.path.join(alt_path, 'val2017')):
                            coco_path = alt_path
                            break
                
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=config['subset'],
                    coco_root=coco_path,
                    image_preprocess=None
                )
            elif dataset_name in ['VG_Attribution', 'VG_Relation']:
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=config['subset'],
                    image_preprocess=None,
                    download=False  # Set to False to avoid downloading during sampling
                )
            elif dataset_name in ['COCO_Order', 'Flickr30k_Order']:
                dataset = dataset_class(
                    root=data_path,
                    split=config['subset'],
                    image_preprocess=None,
                    download=True
                )
            elif dataset_name == 'BLA':
                dataset = dataset_class(
                    data_root=data_path,
                    subset=config['subset'],  # 'ap', 'co', or 'rc'
                    split='test',
                    image_preprocess=None,
                    download=False
                )
            elif dataset_name == 'SPEC_I2T':
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=config['subset'],  # 'count', 'relative_spatial', etc.
                    image_preprocess=None
                )
            elif dataset_name == 'ColorSwap':
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=config['subset'],
                    image_preprocess=None,
                    download=True,  # Allow download if needed
                    verbose=False
                )
            elif dataset_name == 'SugarCrepe_PP':
                # SugarCrepe_PP needs COCO root path
                coco_path = "./datasets/SugarCrepe"  # Use SugarCrepe directory as COCO root
                dataset = dataset_class(
                    subset_name=config['subset'],  # 'swap_object', 'swap_atribute', etc.
                    coco_root=coco_path,
                    image_preprocess=None
                )
            elif dataset_name == 'ControlledImages':
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=config['subset'],  # 'A' for controlled_images
                    image_preprocess=None,
                    download=False
                )
            else:
                # Generic initialization
                dataset = dataset_class(
                    data_root=data_path,
                    subset_name=config['subset'],
                    image_preprocess=None
                )
                
            print(f"✓ Loaded {dataset_name}: {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {str(e)}")
            return None
    
    def extract_sample_data(self, dataset: Any, dataset_name: str, idx: int) -> Optional[Dict]:
        """Extract image and caption information from a dataset sample."""
        try:
            sample = dataset[idx]
            
            # Handle different dataset formats
            if dataset_name == 'Winoground':
                image_options = sample['image_options']
                caption_options = sample['caption_options']
                return {
                    'image': image_options[0],  # First image
                    'positive_caption': caption_options[0],  # First caption
                    'negative_captions': [caption_options[1]],  # Second caption
                    'metadata': {
                        'tag': sample.get('tag', 'unknown'),
                        'pair_id': sample.get('pair_id', idx)
                    }
                }
                
            elif dataset_name == 'VALSE':
                # VALSE uses 'caption' and 'foil' from the __getitem__ method
                pos_caption = sample.get('caption', '')
                neg_caption = sample.get('foil', '')
                
                # Fallback to the raw format if needed
                if not pos_caption and sample.get('positive_caption'):
                    pos_caption = sample['positive_caption'][0] if sample['positive_caption'] else ''
                if not neg_caption and sample.get('negative_caption'):
                    neg_caption = sample['negative_caption'][0] if sample['negative_caption'] else ''
                
                return {
                    'image': sample['image'],
                    'positive_caption': pos_caption,
                    'negative_captions': [neg_caption] if neg_caption else [],
                    'metadata': {
                        'phenomena': sample.get('linguistic_phenomena', 'unknown'),
                        'dataset_source': sample.get('dataset', 'unknown')
                    }
                }
                
            elif dataset_name == 'SugarCrepe':
                # SugarCrepe uses 'image_options' and 'caption_options' format
                image = sample.get('image_options')
                caption_options = sample.get('caption_options', [])
                
                pos_caption = caption_options[0] if len(caption_options) > 0 else ''
                neg_captions = caption_options[1:] if len(caption_options) > 1 else []
                
                return {
                    'image': image,
                    'positive_caption': pos_caption,
                    'negative_captions': neg_captions,
                    'metadata': {
                        'id': sample.get('id', idx),
                        'filename': f'sugarcrepe_{idx}'
                    }
                }
                
            elif dataset_name == 'VL_CheckList':
                # VL_CheckList specific handling - uses 'image_options' field
                image = sample.get('image_options', sample.get('image'))
                pos_captions = sample.get('pos_captions', [])
                neg_captions = sample.get('neg_captions', [])
                
                return {
                    'image': image,
                    'positive_caption': pos_captions[0] if pos_captions else 'No caption available',
                    'negative_captions': neg_captions,
                    'metadata': {
                        'relation_type': sample.get('relation', 'unknown'),
                        'image_path': sample.get('image_path', 'unknown')
                    },
                    'subset': sample.get('relation', 'all')
                }
                
            elif dataset_name == 'COCO_Counterfactuals':
                # COCO_Counterfactuals specific handling - uses 'image_options' list with 2 images
                image_options = sample.get('image_options', [])
                caption_options = sample.get('caption_options', [])
                
                image = image_options[0] if image_options else sample.get('image')
                pos_caption = caption_options[0] if len(caption_options) > 0 else 'No caption available'
                neg_captions = caption_options[1:] if len(caption_options) > 1 else []
                
                return {
                    'image': image,
                    'positive_caption': pos_caption,
                    'negative_captions': neg_captions,
                    'metadata': {
                        'pair_id': sample.get('pair_id', 'unknown'),
                        'label': sample.get('label', 0)
                    },
                    'subset': 'all'
                }
                
            elif dataset_name == 'ColorFoil':
                # ColorFoil specific handling - uses 'image_options' as list
                image_options = sample.get('image_options', [])
                caption_options = sample.get('caption_options', [])
                
                image = image_options[0] if image_options else sample.get('image')
                pos_caption = caption_options[0] if len(caption_options) > 0 else 'No caption available'
                neg_captions = caption_options[1:] if len(caption_options) > 1 else []
                
                return {
                    'image': image,
                    'positive_caption': pos_caption,
                    'negative_captions': neg_captions,
                    'metadata': {
                        'image_id': sample.get('image_id', 'unknown'),
                        'image_url': sample.get('image_url', 'unknown')
                    },
                    'subset': 'all'
                }
                
            elif dataset_name == 'BLA':
                # BLA specific handling
                return {
                    'image': sample.get('image'),
                    'positive_caption': sample.get('correct_caption', 'No correct caption'),
                    'negative_captions': [sample.get('foil_caption', 'No foil caption')],
                    'metadata': {
                        'phenomenon': sample.get('phenomenon', 'unknown'),
                        'image_id': sample.get('image_id', 'unknown')
                    },
                    'subset': sample.get('phenomenon', 'ap')
                }
                
            elif dataset_name == 'SPEC_I2T':
                # SPEC specific handling
                return {
                    'image': sample.get('image'),
                    'positive_caption': sample.get('caption', 'No caption available'),
                    'negative_captions': [sample.get('foil_caption', 'No foil caption')],
                    'metadata': {
                        'object_id': sample.get('object_id', 'unknown'),
                        'reasoning_type': sample.get('reasoning_type', 'unknown')
                    },
                    'subset': sample.get('reasoning_type', 'count')
                }
                
            elif dataset_name == 'ColorSwap':
                # ColorSwap specific handling - like Winoground with image pairs
                image_options = sample.get('image_options', [])
                caption_options = sample.get('caption_options', [])
                
                return {
                    'image': image_options[0] if image_options else sample.get('image'),
                    'positive_caption': caption_options[0] if caption_options else 'No caption available',
                    'negative_captions': caption_options[1:] if len(caption_options) > 1 else [],
                    'metadata': {
                        'pair_id': sample.get('pair_id', 'unknown'),
                        'color_swap_type': sample.get('swap_type', 'unknown')
                    },
                    'subset': 'all'
                }
                
            elif dataset_name == 'ControlledImages':
                # WhatsUp/ControlledImages specific handling
                return {
                    'image': sample.get('image_options', sample.get('image')),
                    'positive_caption': sample.get('caption_options', [sample.get('caption', 'No caption')])[0],
                    'negative_captions': sample.get('caption_options', [])[1:] if len(sample.get('caption_options', [])) > 1 else [],
                    'metadata': {
                        'image_id': sample.get('image_id', 'unknown'),
                        'subset': sample.get('subset', 'A')
                    },
                    'subset': 'A'
                }
                
            elif dataset_name == 'SugarCrepe_PP':
                # SugarCrepe_PP specific handling - similar to regular SugarCrepe but different format
                return {
                    'image': sample.get('image'),
                    'positive_caption': sample.get('caption', 'No caption available'),
                    'negative_captions': [sample.get('negative_caption', 'No negative caption')] if sample.get('negative_caption') else [],
                    'metadata': {
                        'id': sample.get('id', idx),
                        'filename': sample.get('filename', f'sugarcrepe_pp_{idx}')
                    },
                    'subset': 'swap_object'  # or extract from sample if available
                }
                
            elif hasattr(sample, 'keys') and 'image' in sample:
                # Generic case for dict-like samples
                image = sample.get('image')
                pos_caption = sample.get('caption', sample.get('positive_caption', ''))
                neg_captions = sample.get('negative_captions', sample.get('negative_caption', []))
                
                if isinstance(neg_captions, str):
                    neg_captions = [neg_captions]
                    
                return {
                    'image': image,
                    'positive_caption': pos_caption,
                    'negative_captions': neg_captions,
                    'metadata': {k: v for k, v in sample.items() 
                               if k not in ['image', 'caption', 'positive_caption', 'negative_caption', 'negative_captions']}
                }
            else:
                # Fallback for different sample formats
                print(f"⚠️  {dataset_name}: Using fallback extraction for sample type {type(sample)}")
                
                # Try to handle as dict/EasyDict
                if hasattr(sample, 'get') or hasattr(sample, '__getitem__'):
                    # Handle both dict.get() and EasyDict.field access
                    try:
                        # Try various image field names
                        image = None
                        for img_field in ['image_options', 'image', 'img']:
                            if hasattr(sample, img_field):
                                image = getattr(sample, img_field)
                                break
                            elif hasattr(sample, 'get') and sample.get(img_field) is not None:
                                image = sample.get(img_field)
                                break
                        
                        # Try various caption field names
                        caption_options = []
                        if hasattr(sample, 'caption_options'):
                            caption_options = getattr(sample, 'caption_options', [])
                        elif hasattr(sample, 'get') and sample.get('caption_options'):
                            caption_options = sample.get('caption_options', [])
                        
                        pos_caption = caption_options[0] if caption_options else 'No caption available'
                        neg_captions = caption_options[1:] if len(caption_options) > 1 else []
                        
                        return {
                            'image': image,
                            'positive_caption': pos_caption,
                            'negative_captions': neg_captions,
                            'metadata': {'type': str(type(sample))},
                            'subset': 'all'
                        }
                    except Exception as e:
                        print(f"❌ Error in fallback extraction for {dataset_name}: {e}")
                        return None
                else:
                    return None
                
        except Exception as e:
            print(f"❌ Failed to extract sample {idx} from {dataset_name}: {e}")
            return None
    
    def _get_subset_info(self, dataset, dataset_name):
        """Get subset information for datasets that have multiple subsets."""
        if dataset_name == 'VALSE':
            # Get all unique linguistic phenomena
            phenomena = set()
            for i in range(min(len(dataset), 100)):  # Sample first 100 to get phenomena types
                try:
                    sample = dataset[i]
                    if 'linguistic_phenomena' in sample:
                        phenomena.add(sample['linguistic_phenomena'])
                except Exception:
                    continue
            return list(phenomena)
        
        elif dataset_name == 'VL_CheckList':
            # VL_CheckList has 14 predefined attribute and relation subsets
            vl_checklist_subsets = [
                # Attribute types (10 subsets)
                'vaw_action', 'vg_action', 'vaw_color', 'vg_color', 'vaw_material', 
                'vg_material', 'vaw_size', 'vg_size', 'vaw_state', 'vg_state',
                # Relation types (4 subsets)
                'hake_action', 'swig_action', 'vg_action_relation', 'vg_spatial'
            ]
            
            # Verify which subsets actually exist in the dataset
            available_subsets = set()
            for i in range(min(len(dataset), 500)):  # Check more samples to find all subsets
                try:
                    sample = dataset[i]
                    if 'relation' in sample:
                        available_subsets.add(sample['relation'])
                except Exception:
                    continue
            
            # Return intersection of predefined and available subsets
            final_subsets = [s for s in vl_checklist_subsets if s in available_subsets]
            if final_subsets:
                print(f"  🔍 Found {len(final_subsets)} VL_CheckList subsets: {final_subsets}")
                return final_subsets
            else:
                print("  ⚠️  No VL_CheckList subsets found in dataset")
                return list(available_subsets) if available_subsets else None
        
        elif dataset_name == 'BLA':
            # BLA has 3 predefined subsets based on the dataset class
            return ['ap', 'co', 'rc']  # active_passive, coordination, relative_clause
            
        elif dataset_name == 'ControlledImages':
            # ControlledImages has subsets for different image types
            return ['A', 'B']  # controlled_images and controlled_clevr
        
        # For SugarCrepe, we need to load each subset separately since they're in different files
        # So we return None here to use regular sampling, but handle multi-subset loading at a higher level
        return None  # No subsets detectable within single dataset instance
    
    def _sample_from_subsets(self, dataset, dataset_name, n_samples, subsets):
        """Sample n_samples from each subset, ensuring diversity."""
        samples = []
        samples_per_subset = max(1, n_samples // len(subsets))
        remaining_samples = n_samples - (samples_per_subset * len(subsets))
        
        print(f"  🔍 Found {len(subsets)} subsets, sampling {samples_per_subset} from each")
        
        for i, subset in enumerate(subsets):
            subset_samples = []
            current_target = samples_per_subset + (1 if i < remaining_samples else 0)
            
            # Find indices for this subset
            subset_indices = []
            for idx in range(len(dataset)):
                try:
                    sample = dataset[idx]
                    if dataset_name == 'VALSE' and sample.get('linguistic_phenomena') == subset:
                        subset_indices.append(idx)
                    elif dataset_name == 'VL_CheckList' and sample.get('relation') == subset:
                        subset_indices.append(idx)
                    elif dataset_name == 'BLA' and sample.get('phenomenon') == subset:
                        subset_indices.append(idx)
                    elif dataset_name == 'ControlledImages' and sample.get('subset_name') == subset:
                        subset_indices.append(idx)
                    
                    if len(subset_indices) >= current_target * 2:  # Stop early if we have enough
                        break
                except Exception:
                    continue
            
            # Sample from this subset
            if subset_indices:
                selected_indices = random.sample(subset_indices, min(current_target, len(subset_indices)))
                for idx in selected_indices:
                    sample_data = self.extract_sample_data(dataset, dataset_name, idx)
                    if sample_data is not None:
                        sample_data['original_idx'] = idx
                        sample_data['subset'] = subset
                        subset_samples.append(sample_data)
                
                print(f"    ✓ {subset}: {len(subset_samples)} samples")
            else:
                print(f"    ⚠️  {subset}: No samples found")
            
            samples.extend(subset_samples)
        
        return samples
    
    def _sample_from_separate_subsets(self, dataset_name, config, n_samples):
        """Sample from datasets that need separate loading for each subset."""
        # Define subsets for each dataset
        if dataset_name == 'BLA':
            subsets = ['ap', 'co', 'rc']
        elif dataset_name == 'ControlledImages':
            subsets = ['A', 'B']
        elif dataset_name == 'SPEC_I2T':
            subsets = ['count', 'relative_spatial', 'relative_size', 'absolute_size', 'absolute_spatial', 'existence']
        elif dataset_name == 'COLA':
            subsets = ['multi_objects']  # Can add 'single_GQA', 'single_CLEVR', 'single_PACO' if needed
        elif dataset_name == 'NegBench':
            subsets = ['msr_vtt_mcq_rephrased_llama', 'COCO_val_mcq_llama3.1_rephrased']
        else:
            print(f"⚠️  No subset configuration for {dataset_name}")
            return []
        
        samples_per_subset = max(1, n_samples // len(subsets))
        remaining_samples = n_samples - (samples_per_subset * len(subsets))
        
        print(f"  🔍 Loading from {len(subsets)} {dataset_name} subsets, sampling {samples_per_subset} from each")
        
        all_samples = []
        
        for i, subset in enumerate(subsets):
            current_target = samples_per_subset + (1 if i < remaining_samples else 0)
            
            try:
                # Load this specific subset
                subset_config = config.copy()
                subset_config['subset'] = subset  # Update the subset name
                subset_dataset = self.load_dataset(dataset_name, subset_config)
                
                if subset_dataset is None or len(subset_dataset) == 0:
                    print(f"    ⚠️  {subset}: No samples found or failed to load")
                    continue
                
                # Sample from this subset
                n_available = len(subset_dataset)
                n_to_sample = min(current_target, n_available)
                indices = random.sample(range(n_available), n_to_sample) if n_to_sample > 0 else []
                
                subset_samples = []
                for idx in indices:
                    sample_data = self.extract_sample_data(subset_dataset, dataset_name, idx)
                    if sample_data is not None:
                        sample_data['subset'] = subset
                        sample_data['original_idx'] = idx
                        subset_samples.append(sample_data)
                
                all_samples.extend(subset_samples)
                print(f"    ✓ {subset}: {len(subset_samples)} samples")
                
            except Exception as e:
                print(f"    ❌ {subset}: Error loading - {e}")
                continue
        
        return all_samples
    
    def _sample_from_multi_file_subsets(self, dataset_name, config, n_samples):
        """Handle datasets like SugarCrepe where each subset is a separate file."""
        if dataset_name == 'SugarCrepe':
            subsets = ['add_att', 'add_obj', 'replace_att', 'replace_obj', 'replace_rel', 'swap_att', 'swap_obj']
        elif dataset_name == 'SugarCrepe_PP':
            subsets = ['swap_att', 'swap_obj', 'replace_att', 'replace_obj', 'add_att', 'add_obj']
        else:
            return []
        
        samples_per_subset = max(1, n_samples // len(subsets))
        remaining_samples = n_samples - (samples_per_subset * len(subsets))
        
        print(f"  🔍 Loading from {len(subsets)} {dataset_name} subsets, sampling ~{samples_per_subset} from each")
        
        all_samples = []
        data_path = os.path.join(self.data_root_base, config['path'])
        
        for i, subset in enumerate(subsets):
            current_target = samples_per_subset + (1 if i < remaining_samples else 0)
            
            # Check if JSON file exists for this subset
            json_path = os.path.join(data_path, f'{subset}.json')
            if not os.path.exists(json_path):
                print(f"    ⚠️  {subset}: JSON file not found at {json_path}")
                continue
            
            try:
                # Load this specific subset
                subset_config = config.copy()
                subset_config['subset'] = subset  # Update the subset name
                subset_dataset = self.load_dataset(dataset_name, subset_config)
                
                if subset_dataset is None or len(subset_dataset) == 0:
                    print(f"    ⚠️  {subset}: No samples found or failed to load")
                    continue
                
                # Sample from this subset
                n_available = len(subset_dataset)
                n_to_sample = min(current_target, n_available)
                indices = random.sample(range(n_available), n_to_sample) if n_to_sample > 0 else []
                
                subset_samples = []
                for idx in indices:
                    sample_data = self.extract_sample_data(subset_dataset, dataset_name, idx)
                    if sample_data is not None:
                        sample_data['original_idx'] = idx
                        sample_data['subset'] = subset
                        subset_samples.append(sample_data)
                
                all_samples.extend(subset_samples)
                print(f"    ✓ {subset}: {len(subset_samples)} samples")
                
            except Exception as e:
                print(f"    ❌ {subset}: Failed to load - {e}")
                continue
        
        return all_samples

    def sample_from_datasets(self, dataset_configs: Dict[str, int], 
                           random_seed: int = 42) -> Dict[str, List[Dict]]:
        """Sample specified number of examples from each dataset, with subset-aware sampling."""
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        sampled_data = {}
        
        for dataset_name, n_samples in dataset_configs.items():
            print(f"\n📊 Sampling {n_samples} examples from {dataset_name}...")
            
            if dataset_name not in self.dataset_configs:
                print(f"❌ Unknown dataset: {dataset_name}")
                continue
                
            config = self.dataset_configs[dataset_name]
            
            # Special handling for multi-file subset datasets like SugarCrepe and SugarCrepe_PP
            if dataset_name in ['SugarCrepe', 'SugarCrepe_PP']:
                samples = self._sample_from_multi_file_subsets(dataset_name, config, n_samples)
                sampled_data[dataset_name] = samples
                print(f"✅ Successfully sampled {len(samples)} examples")
                continue
            
            # Special handling for datasets that need separate loading per subset
            elif dataset_name in ['BLA', 'ControlledImages', 'SPEC_I2T', 'COLA', 'NegBench']:
                samples = self._sample_from_separate_subsets(dataset_name, config, n_samples)
                sampled_data[dataset_name] = samples
                print(f"✅ Successfully sampled {len(samples)} examples")
                continue
            
            # Regular dataset loading
            dataset = self.load_dataset(dataset_name, config)
            
            if dataset is None:
                sampled_data[dataset_name] = []
                continue
            
            # Check if this dataset has subsets
            subsets = self._get_subset_info(dataset, dataset_name)
            
            if subsets and len(subsets) > 1:
                # Sample from multiple subsets within single dataset
                samples = self._sample_from_subsets(dataset, dataset_name, n_samples, subsets)
            else:
                # Regular sampling for datasets without subsets
                if len(dataset) < n_samples:
                    print(f"⚠️  Dataset has only {len(dataset)} samples, sampling all")
                    indices = list(range(len(dataset)))
                else:
                    indices = random.sample(range(len(dataset)), n_samples)
                
                samples = []
                for idx in indices:
                    sample_data = self.extract_sample_data(dataset, dataset_name, idx)
                    if sample_data is not None:
                        sample_data['original_idx'] = idx
                        samples.append(sample_data)
            
            sampled_data[dataset_name] = samples
            print(f"✅ Successfully sampled {len(samples)} examples")
        
        return sampled_data


class ModernBenchmarkVisualizer:
    """Creates modern, publication-quality visualizations of benchmark samples."""
    
    def __init__(self, style: str = 'modern'):
        self.style = style
        self._setup_plotting_style()
        
    def _setup_plotting_style(self):
        """Configure modern plotting aesthetics."""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Modern color scheme
        self.colors = {
            'positive': '#2E8B57',      # Sea Green
            'negative': '#DC143C',       # Crimson
            'background': '#FAFAFA',     # Light Gray
            'text': '#2F2F2F',          # Dark Gray
            'border': '#E0E0E0',        # Light Border
            'accent': '#4A90E2'          # Blue Accent
        }
        
        # Typography
        self.fonts = {
            'title': {'family': 'sans-serif', 'weight': 'bold', 'size': 16},
            'subtitle': {'family': 'sans-serif', 'weight': 'normal', 'size': 12},
            'caption': {'family': 'sans-serif', 'weight': 'normal', 'size': 10},
            'metadata': {'family': 'monospace', 'weight': 'normal', 'size': 8}
        }
        
    def wrap_text(self, text: str, width: int = 40) -> str:
        """Wrap long text for better display."""
        if not text:
            return "No caption available"
        return '\n'.join(textwrap.wrap(text, width=width))
    
    def create_single_benchmark_plot(self, dataset_name: str, samples: List[Dict], 
                                   save_path: str = None) -> None:
        """Create a visualization for a single benchmark dataset."""
        if not samples:
            print(f"⚠️  No samples available for {dataset_name}")
            return
            
        n_samples = len(samples)
        
        # Calculate grid layout
        cols = min(3, n_samples)  # Max 3 columns
        rows = (n_samples + cols - 1) // cols
        
        # Create figure with proper spacing
        fig = plt.figure(figsize=(6 * cols, 8 * rows))
        fig.patch.set_facecolor(self.colors['background'])
        
        # Main title
        fig.suptitle(f'{dataset_name} Benchmark Samples', 
                    fontsize=20, fontweight='bold', 
                    color=self.colors['text'], y=0.98)
        
        # Create subplots for each sample
        for i, sample in enumerate(samples):
            # Create subplot grid for this sample
            ax_main = plt.subplot(rows, cols, i + 1)
            
            # Display image with robust conversion
            image = sample['image']
            try:
                if isinstance(image, torch.Tensor):
                    # Convert tensor to PIL Image
                    image = torch.clamp(image, 0, 1)
                    if image.dim() == 4:  # Batch dimension
                        image = image.squeeze(0)
                    if image.shape[0] == 3:  # CHW format
                        image = image.permute(1, 2, 0)
                    image = (image.numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(image)
                elif isinstance(image, np.ndarray):
                    # Handle numpy arrays
                    if image.dtype != np.uint8:
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                    if len(image.shape) == 3 and image.shape[0] == 3:
                        image = np.transpose(image, (1, 2, 0))
                    image = Image.fromarray(image)
                elif not isinstance(image, Image.Image):
                    print(f"⚠️  Skipping sample {i} in {dataset_name}: unsupported image format {type(image)}")
                    continue
                
                ax_main.imshow(image)
            except Exception as e:
                print(f"⚠️  Failed to display sample {i} from {dataset_name}: {e}")
                # Show a placeholder
                ax_main.text(0.5, 0.5, f"Sample {i+1}\n(Image Error)", 
                           transform=ax_main.transAxes, ha='center', va='center',
                           fontsize=12, color=self.colors['text'])
                ax_main.set_facecolor(self.colors['border'])
            ax_main.set_xticks([])
            ax_main.set_yticks([])
            
            # Add border around image
            for spine in ax_main.spines.values():
                spine.set_edgecolor(self.colors['border'])
                spine.set_linewidth(2)
            
            # Prepare captions
            pos_caption = self.wrap_text(sample['positive_caption'], width=35)
            neg_captions = [self.wrap_text(cap, width=35) 
                           for cap in sample.get('negative_captions', [])]
            
            # Create text box below image
            caption_text = f"✓ Correct: {pos_caption}\n\n"
            for j, neg_cap in enumerate(neg_captions[:2]):  # Show max 2 negatives
                caption_text += f"✗ Wrong {j+1}: {neg_cap}\n"
                if j < len(neg_captions) - 1:
                    caption_text += "\n"
            
            # Add caption box
            ax_main.text(0.5, -0.15, caption_text, 
                        transform=ax_main.transAxes,
                        ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.5", 
                                facecolor='white', 
                                edgecolor=self.colors['border'],
                                alpha=0.9),
                        fontsize=9, linespacing=1.2)
            
            # Add sample metadata as title
            metadata = sample.get('metadata', {})
            title_parts = []
            
            # Add subset information if available
            if 'subset' in sample:
                title_parts.append(f"Subset: {sample['subset']}")
            
            # Add metadata information
            if 'tag' in metadata:
                title_parts.append(f"Tag: {metadata['tag']}")
            if 'phenomena' in metadata:
                title_parts.append(f"Phenomena: {metadata['phenomena']}")
            if 'id' in metadata:
                title_parts.append(f"ID: {metadata['id']}")
                
            if title_parts:
                ax_main.set_title(' | '.join(title_parts), 
                                fontsize=10, fontweight='bold',
                                color=self.colors['accent'], pad=10)
            else:
                ax_main.set_title(f'Sample {sample.get("original_idx", i)}', 
                                fontsize=10, fontweight='bold',
                                color=self.colors['accent'], pad=10)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.2, hspace=0.4, wspace=0.3)
        
        # Save plot as both PNG and PDF
        if save_path:
            # Save as PNG
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.colors['background'], 
                       edgecolor='none')
            print(f"💾 Saved {dataset_name} plot to: {save_path}")
            
            # Save as PDF for publications
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight', 
                       facecolor=self.colors['background'])
            print(f"📄 Saved {dataset_name} PDF to: {pdf_path}")
        
        plt.show()
        plt.close()
    
    def create_overview_plot(self, all_samples: Dict[str, List[Dict]], 
                           save_path: str = None) -> None:
        """Create an overview plot showing positive and negative captions for each benchmark."""
        
        # Group VG datasets under ARO and filter datasets that have samples
        grouped_datasets = {}
        for k, v in all_samples.items():
            if v:  # Only include datasets with samples
                if k in ['VG_Attribution', 'VG_Relation']:
                    if 'ARO' not in grouped_datasets:
                        grouped_datasets['ARO'] = []
                    # Add subset info to samples
                    for sample in v:
                        sample['aro_subset'] = k.replace('VG_', '')
                    grouped_datasets['ARO'].extend(v)
                else:
                    grouped_datasets[k] = v
        
        if not grouped_datasets:
            print("⚠️  No samples available for overview plot")
            return
            
        n_datasets = len(grouped_datasets)
        
        # Calculate grid layout (prefer more rows for caption space)
        cols = min(3, n_datasets)  # Reduce columns for better caption space
        rows = (n_datasets + cols - 1) // cols
        
        # Create more compact figure
        fig = plt.figure(figsize=(5 * cols, 7 * rows))
        fig.patch.set_facecolor(self.colors['background'])
        
        # Main title
        fig.suptitle('Vision-Language Benchmarks Overview', 
                    fontsize=20, fontweight='bold', 
                    color=self.colors['text'], y=0.97)
        
        # Add subtitle
        fig.text(0.5, 0.93, 'Compositional Reasoning Benchmarks', 
                ha='center', fontsize=12, color=self.colors['accent'])
        
        # Plot samples from each dataset with positive and negative captions
        for i, (dataset_name, samples) in enumerate(grouped_datasets.items()):
            # Create subplot with more space
            ax = plt.subplot(rows, cols, i + 1)
            
            # Use first sample
            sample = samples[0]
            image = sample['image']
            
            # Convert image to displayable format (same as before)
            image_displayed = False
            try:
                if isinstance(image, torch.Tensor):
                    image = torch.clamp(image, 0, 1)
                    if image.dim() == 4:
                        image = image.squeeze(0)
                    if image.shape[0] == 3:
                        image = image.permute(1, 2, 0)
                    image = (image.numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(image)
                elif isinstance(image, np.ndarray):
                    if image.dtype != np.uint8:
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                    if len(image.shape) == 3 and image.shape[0] == 3:
                        image = np.transpose(image, (1, 2, 0))
                    image = Image.fromarray(image)
                elif isinstance(image, Image.Image):
                    pass  # Already in correct format
                elif image is None:
                    print(f"⚠️  Skipping {dataset_name}: no image data")
                    image_displayed = False
                else:
                    print(f"⚠️  Skipping {dataset_name}: unsupported image format {type(image)}")
                    image_displayed = False
                
                if image is not None and isinstance(image, Image.Image):
                    ax.imshow(image)
                    image_displayed = True
                    
            except Exception as e:
                print(f"⚠️  Failed to display image from {dataset_name}: {e}")
                image_displayed = False
            
            if not image_displayed:
                # Show a placeholder
                ax.text(0.5, 0.5, f"{dataset_name}\n(Image Error)", 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color=self.colors['text'])
                ax.set_facecolor(self.colors['border'])
                
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Style the border
            for spine in ax.spines.values():
                spine.set_edgecolor(self.colors['accent'])
                spine.set_linewidth(2)
            
            # Add dataset name as title with special handling for ARO
            title = dataset_name
            if dataset_name == 'ARO':
                subset_info = sample.get('aro_subset', 'Attribution')
                title = f"ARO ({subset_info})"
            
            ax.set_title(title, fontsize=14, fontweight='bold',
                        color=self.colors['text'], pad=15)
            
            # Add positive and negative captions below the image with larger text
            pos_caption = self.wrap_text(sample.get('positive_caption', 'No positive caption'), 30)
            neg_captions = sample.get('negative_captions', [])
            neg_caption = self.wrap_text(neg_captions[0] if neg_captions else 'No negative caption', 30)
            
            # Position captions below image with colored text (no symbols)
            ax.text(0.02, -0.06, pos_caption, transform=ax.transAxes, 
                   fontsize=11, color=self.colors['positive'], ha='left', va='top',
                   fontweight='bold')
            
            ax.text(0.02, -0.20, neg_caption, transform=ax.transAxes, 
                   fontsize=11, color=self.colors['negative'], ha='left', va='top',
                   fontweight='bold')
        
        # Remove empty subplots
        for i in range(n_datasets, rows * cols):
            if i < len(fig.axes):
                fig.delaxes(fig.axes[i])
        
        # Adjust layout for more compact design
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.02, hspace=0.45, wspace=0.25)
        
        # Save plot as both PNG and PDF
        if save_path:
            # Save as PNG
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.colors['background'])
            print(f"💾 Saved overview plot to: {save_path}")
            
            # Save as PDF for publications
            pdf_path = save_path.replace('.png', '.pdf')
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight', 
                       facecolor=self.colors['background'])
            print(f"📄 Saved PDF version to: {pdf_path}")
        
        plt.show()
        plt.close()
    
    def create_all_plots(self, all_samples: Dict[str, List[Dict]], 
                        output_dir: str = "./benchmark_plots") -> None:
        """Create all visualization plots and save them."""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🎨 Creating visualizations in: {output_dir}")
        
        # Create overview plot
        overview_path = os.path.join(output_dir, "00_benchmark_overview.png")
        self.create_overview_plot(all_samples, overview_path)
        
        # Create individual benchmark plots
        for dataset_name, samples in all_samples.items():
            if samples:  # Only create plot if samples exist
                plot_path = os.path.join(output_dir, f"{dataset_name.lower()}_samples.png")
                self.create_single_benchmark_plot(dataset_name, samples, plot_path)
        
        print(f"✅ All plots created in: {output_dir}")


def create_default_config() -> Dict:
    """Create default configuration for sampling."""
    return {
        "datasets": {
            "Winoground": 3,
            "VALSE": 6,              # Will sample from multiple phenomena (e.g., 6 phenomena = 1 each)
            "VG_Attribution": 2,
            "VG_Relation": 2,
            "COCO_Order": 2,
            "Flickr30k_Order": 2,
            "SugarCrepe": 7,         # Will sample from 7 subset types (1 each)
            "ColorSwap": 2,
            "ColorFoil": 2,
            "COCO_Counterfactuals": 2,
            "ControlledImages": 2,
            "VisMin": 2,
            "VL_CheckList": 4        # Will sample from different relation types
        },
        "random_seed": 42,
        "output_dir": "./benchmark_plots",
        "data_root": "./datasets"
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize examples from vision-language benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python generate_sampling_plot.py
  python generate_sampling_plot.py Winoground=5 VALSE=3 SugarCrepe=2
  python generate_sampling_plot.py --output-dir ./thesis_figures --seed 123
        """
    )
    
    parser.add_argument('datasets', nargs='*', 
                       help='Dataset sampling config (format: dataset_name=n_samples). If empty, uses defaults.')
    parser.add_argument('--output-dir', type=str, default='./benchmark_plots',
                       help='Output directory for plots')
    parser.add_argument('--data-root', type=str, default='./datasets',
                       help='Root directory containing dataset folders')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    
    print("🎯 Vision-Language Benchmark Visualization")
    print("=" * 50)
    
    # Parse dataset configuration from command line
    if args.datasets:
        dataset_config = {}
        for item in args.datasets:
            if '=' in item:
                name, count = item.split('=', 1)
                try:
                    dataset_config[name] = int(count)
                except ValueError:
                    print(f"⚠️  Invalid count for {name}: {count}")
        if not dataset_config:
            print("⚠️  No valid datasets specified, using defaults")
            dataset_config = create_default_config()['datasets']
    else:
        print("ℹ️  No datasets specified, using default selection")
        dataset_config = create_default_config()['datasets']
    
    print(f"📁 Data root: {args.data_root}")
    print(f"📊 Sampling {sum(dataset_config.values())} total examples from {len(dataset_config)} benchmarks")
    print(f"🎲 Random seed: {args.seed}")
    
    # Initialize sampler and visualizer
    sampler = BenchmarkSampler(data_root_base=args.data_root)
    visualizer = ModernBenchmarkVisualizer()
    
    # Sample data from benchmarks
    print("\n" + "="*50)
    print("🔄 SAMPLING PHASE")
    print("="*50)
    
    sampled_data = sampler.sample_from_datasets(
        dataset_config, 
        random_seed=args.seed
    )
    
    # Create visualizations
    print("\n" + "="*50)
    print("🎨 VISUALIZATION PHASE")
    print("="*50)
    
    visualizer.create_all_plots(sampled_data, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("📋 SUMMARY")
    print("="*50)
    
    total_sampled = sum(len(samples) for samples in sampled_data.values())
    successful_datasets = sum(1 for samples in sampled_data.values() if samples)
    
    print(f"✅ Successfully sampled from {successful_datasets}/{len(dataset_config)} datasets")
    print(f"📊 Total samples collected: {total_sampled}")
    print(f"💾 Plots saved to: {args.output_dir}")
    
    # Show per-dataset summary
    print("\n📈 Per-dataset results:")
    for dataset_name, samples in sampled_data.items():
        status = f"{len(samples)} samples" if samples else "❌ Failed"
        print(f"  {dataset_name:<20}: {status}")
    
    print(f"\n🎉 All done! Check {args.output_dir} for your thesis figures.")


if __name__ == "__main__":
    main()

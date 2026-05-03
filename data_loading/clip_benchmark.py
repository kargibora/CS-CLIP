"""
CLIP Benchmark Dataset Loader

This module provides a data loading interface for CLIP benchmark datasets from:
https://huggingface.co/clip-benchmark

Supports AUTOMATIC downloading and evaluation of:
- Zero-shot classification tasks (ImageNet, CIFAR, VOC, etc.)
- Zero-shot retrieval tasks (COCO, Flickr8k/30k, etc.)
- Multiple datasets across various domains

Two loading methods:
1. **Direct WebDataset from HuggingFace** (Recommended):
   - Loads pre-converted webdataset format directly from HuggingFace
   - Faster download and setup
   - Dataset names: 'wds_mscoco_captions', 'wds_vtab-flowers', etc.
   
2. **Via clip_benchmark library**:
   - Uses clip_benchmark to download and convert datasets
   - More flexible but slower initial setup
   - Dataset names: 'mscoco_captions', 'vtab/flowers', etc.

Features:
✅ Automatic dataset download from HuggingFace
✅ Automatic task detection (classification vs retrieval)
✅ Unified evaluation interface
✅ No manual downloads required

Usage Example (HuggingFace WebDataset):
    ```python
    from data_loading.clip_benchmark import CLIPBenchmarkDataset
    import open_clip
    
    # Load directly from HuggingFace (recommended)
    dataset = CLIPBenchmarkDataset(
        dataset_name='wds_mscoco_captions',  # Webdataset format
        image_preprocess=preprocess
    )
    
    # Or use clip_benchmark library
    dataset = CLIPBenchmarkDataset(
        dataset_name='mscoco_captions',  # clip_benchmark format
        image_preprocess=preprocess
    )
    
    # Evaluate - task type is automatically detected
    model, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    results, embeddings = dataset.evaluate(
        embedding_model=model,
        device='cuda',
        batch_size=64
    )
    
    print(f"Accuracy: {results['acc1']:.2%}")
    ```

Available on HuggingFace: https://huggingface.co/clip-benchmark
"""

import json
import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# Try new clip_benchmark library (with HuggingFace support)
try:
    from clip_benchmark.datasets.builder import build_dataset, get_dataset_default_task
    CLIP_BENCHMARK_AVAILABLE = True
except ImportError:
    CLIP_BENCHMARK_AVAILABLE = False
    build_dataset = None
    get_dataset_default_task = None
    print("⚠️  clip_benchmark not installed. Install with: pip install clip-benchmark")

# Try datasets library for direct HuggingFace access
try:
    from datasets import load_dataset as hf_load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    hf_load_dataset = None
    print("⚠️  datasets library not installed. Install with: pip install datasets")

try:
    from utils.align import (
        compute_caption_embeddings_intermediate_batch,
        compute_image_embeddings_intermediate_batch,
    )
except ImportError:
    print("Warning: utils.align not found. Evaluation functions may not work.")


def load_webdataset_from_huggingface(
    dataset_name: str,
    split: str = 'test',
    cache_dir: str = None,
    verbose: bool = True
):
    """
    Load a webdataset directly from HuggingFace clip-benchmark organization.
    
    Args:
        dataset_name: Name like 'wds_mscoco_captions', 'wds_vtab-flowers', etc.
        split: 'train', 'test', or 'val'
        cache_dir: Optional cache directory
        verbose: Print progress
        
    Returns:
        HuggingFace Dataset object
    """
    if not DATASETS_AVAILABLE:
        raise ImportError(
            "❌ datasets library required but not installed.\n"
            "Install with: pip install datasets"
        )
    
    # Construct HuggingFace dataset path
    hf_dataset_path = f"clip-benchmark/{dataset_name}"
    
    if verbose:
        print(f"Loading from HuggingFace: {hf_dataset_path}")
        print(f"Split: {split}")
    
    try:
        dataset = hf_load_dataset(
            hf_dataset_path,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True  # May be needed for some datasets
        )
        
        if verbose:
            print(f"✓ Loaded {len(dataset)} samples from HuggingFace")
        
        return dataset
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load '{hf_dataset_path}' from HuggingFace.\n"
            f"Error: {str(e)}\n\n"
            f"Available datasets: https://huggingface.co/clip-benchmark\n"
            f"Make sure the dataset name is correct (e.g., 'wds_mscoco_captions')"
        ) from e


class CLIPBenchmarkDataset(Dataset):
    """
    Wrapper for CLIP benchmark datasets that provides a consistent interface
    for zero-shot classification and retrieval tasks.
    
    This dataset loader uses the clip_benchmark library to access various datasets
    and adapts them to work with the existing evaluation pipeline.
    
    Attributes:
        dataset_name: str
            Name of the dataset (e.g., 'cifar10', 'imagenet1k', 'mscoco_captions')
        task: str
            Task type: 'zeroshot_classification' or 'zeroshot_retrieval'
        data_root: str
            Root directory for dataset storage
        dataset: Dataset
            The underlying clip_benchmark dataset
        captions: List[str]
            List of unique captions (for retrieval tasks)
        classes: List[str]
            List of class names (for classification tasks)
    """
    
    def __init__(
        self,
        dataset_name: str,
        task: str = 'auto',
        data_root: str = 'datasets/clip_benchmark',
        split: str = 'test',
        image_preprocess=None,
        annotation_file: Optional[str] = None,
        language: str = 'en',
        download: bool = True,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize CLIP benchmark dataset with AUTOMATIC downloading and task detection.
        
        Args:
            dataset_name: Name of dataset from HuggingFace clip-benchmark
                Two formats supported:
                1. WebDataset format (recommended): 'wds_mscoco_captions', 'wds_vtab-flowers'
                   - Loads directly from HuggingFace clip-benchmark organization
                   - Faster and more reliable
                2. clip_benchmark format: 'mscoco_captions', 'vtab/flowers'
                   - Uses clip_benchmark library
                See: https://huggingface.co/clip-benchmark
                
            task: Task type (default: 'auto' for automatic detection)
                - 'auto': Automatically detect from dataset
                - 'zeroshot_classification': For classification tasks
                - 'zeroshot_retrieval': For image-text retrieval tasks
                
            data_root: Root directory for dataset cache (default: 'datasets/clip_benchmark')
                Datasets are automatically downloaded here on first use.
                
            split: Dataset split (default: 'test')
                Options: 'train', 'test', 'val', 'validation'
                
            image_preprocess: Image preprocessing function (torchvision transform)
                If None, will use basic PIL image loading
                
            annotation_file: Path to custom annotation file (optional)
                Only needed for custom datasets, most benchmarks don't need this
                
            language: Language for templates and classnames (default: 'en')
                
            download: Automatically download if not cached (default: True)
                Set to False only if you've manually downloaded the dataset
                
            verbose: Print progress and info messages (default: True)
                
        Example:
            ```python
            # WebDataset from HuggingFace (recommended)
            dataset = CLIPBenchmarkDataset(
                dataset_name='wds_mscoco_captions',  # Note: wds_ prefix
                image_preprocess=preprocess
            )
            
            # Or via clip_benchmark library
            dataset = CLIPBenchmarkDataset(
                dataset_name='mscoco_captions',
                image_preprocess=preprocess
            )
            ```
        """
        # Detect if this is a webdataset from HuggingFace
        is_huggingface_wds = dataset_name.startswith('wds_')
        
        if is_huggingface_wds:
            # Load directly from HuggingFace
            if not DATASETS_AVAILABLE:
                raise ImportError(
                    "❌ datasets library is required for HuggingFace webdatasets.\n"
                    "Install with: pip install datasets"
                )
            
            self.dataset_name = dataset_name
            self.data_root = data_root
            self.split = split
            self.image_preprocess = image_preprocess
            self.annotation_file = annotation_file
            self.language = language
            self.verbose = verbose
            self.is_huggingface_wds = True
            
            # Load from HuggingFace
            if verbose:
                print(f"\n{'='*70}")
                print(f"📦 Loading from HuggingFace: clip-benchmark/{dataset_name}")
                print(f"{'='*70}")
                print(f"  Split: {split}")
                print(f"  Cache: {data_root}")
                print(f"{'='*70}\n")
            
            try:
                self.dataset = load_webdataset_from_huggingface(
                    dataset_name=dataset_name,
                    split=split,
                    cache_dir=data_root,
                    verbose=verbose
                )
            except Exception as e:
                raise RuntimeError(
                    f"\n❌ Failed to load webdataset '{dataset_name}' from HuggingFace\n"
                    f"   Error: {str(e)}\n\n"
                    f"Available datasets: https://huggingface.co/clip-benchmark\n"
                    f"Example names: wds_mscoco_captions, wds_vtab-flowers, wds_imagenet1k\n"
                ) from e
            
            # Extract metadata from HuggingFace dataset
            # Auto-detect task from name
            if 'caption' in dataset_name.lower() or 'flickr' in dataset_name.lower() or 'coco' in dataset_name.lower():
                self.task = 'zeroshot_retrieval'
                self.classes = []
                self.templates = []
                self.captions = []  # Will be built on demand
                
                if verbose:
                    print(f"\n✓ Retrieval dataset ready:")
                    print(f"   • {len(self)} samples")
                    print()
            else:
                self.task = 'zeroshot_classification'
                
                # Extract class names and templates from dataset
                self.classes = self._extract_classnames_from_hf_dataset()
                self.templates = self._extract_templates_from_hf_dataset()
                self.captions = []
                
                if verbose:
                    print(f"\n✓ Classification dataset ready:")
                    print(f"   • {len(self)} samples")
                    print(f"   • {len(self.classes)} classes")
                    print(f"   • {len(self.templates)} templates")
                    if len(self.classes) > 0 and len(self.classes) <= 10:
                        print(f"   • Classes: {', '.join(self.classes)}")
                    elif len(self.classes) > 10:
                        print(f"   • Sample classes: {', '.join(self.classes[:5])}, ...")
                    print()
            
            return
        
        # Original clip_benchmark library path
        if not CLIP_BENCHMARK_AVAILABLE:
            raise ImportError(
                "❌ clip_benchmark is required but not installed.\n"
                "Install with: pip install clip-benchmark\n"
                "Then datasets will be automatically downloaded from HuggingFace.\n\n"
                "Or use webdataset format with dataset names like 'wds_mscoco_captions'"
            )
        
        self.is_huggingface_wds = False
        
        # Handle dataset name aliases/mappings
        dataset_name_mapping = {
            'stanford_cars': 'cars',
            'stanford-cars': 'cars',
            'oxford_pets': 'pets',
            'oxford-pets': 'pets',
            'oxford_flowers': 'flowers',
            'oxford-flowers': 'flowers',
            'flowers102': 'flowers',
        }
        
        # Apply mapping if dataset name has an alias
        original_name = dataset_name
        dataset_name = dataset_name_mapping.get(dataset_name, dataset_name)
        
        if verbose and dataset_name != original_name:
            print(f"✓ Mapped '{original_name}' → '{dataset_name}'")
        
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.split = split
        self.image_preprocess = image_preprocess
        self.annotation_file = annotation_file
        self.language = language
        self.verbose = verbose
        
        # Auto-detect task if not specified
        if task == 'auto':
            try:
                task = get_dataset_default_task(dataset_name)
                if verbose:
                    print(f"✓ Auto-detected task: {task}")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Could not auto-detect task: {e}")
                    print("   Defaulting to 'zeroshot_classification'")
                task = 'zeroshot_classification'
        
        self.task = task
        
        # Create the dataset using clip_benchmark builder
        if verbose:
            print(f"\n{'='*70}")
            print(f"📦 Loading CLIP Benchmark: {dataset_name}")
            print(f"{'='*70}")
            print(f"  Task: {task}")
            print(f"  Split: {split}")
            print(f"  Data root: {data_root}")
            if download:
                print(f"  Auto-download: ✓ (will download if not cached)")
            else:
                print(f"  Auto-download: ✗ (manual download required)")
            print(f"{'='*70}\n")
        
        try:
            self.dataset = build_dataset(
                dataset_name=dataset_name,
                root=data_root,
                transform=image_preprocess,
                split=split,
                download=download,
                annotation_file=annotation_file,
                language=language,
                task=task,
                **kwargs
            )
            
            if verbose:
                print(f"✓ Dataset loaded successfully!")
                
        except Exception as e:
            error_msg = (
                f"\n❌ Failed to load dataset '{dataset_name}'\n"
                f"   Error: {str(e)}\n\n"
                f"Troubleshooting:\n"
                f"  1. Check dataset name is correct: https://huggingface.co/clip-benchmark\n"
                f"  2. Ensure clip_benchmark is up to date: pip install --upgrade clip-benchmark\n"
                f"  3. Check internet connection for automatic download\n"
                f"  4. Try with download=True to force re-download\n"
            )
            raise RuntimeError(error_msg) from e
        
        # Extract metadata based on task
        if self.task == 'zeroshot_classification':
            self.classes = self.dataset.classes if hasattr(self.dataset, 'classes') else []
            self.templates = self.dataset.templates if hasattr(self.dataset, 'templates') else []
            self.captions = []
            
            if verbose:
                print(f"\n✓ Classification dataset ready:")
                print(f"   • {len(self.classes)} classes")
                print(f"   • {len(self.templates)} prompt templates")
                print(f"   • {len(self)} test samples")
                if len(self.classes) > 0 and len(self.classes) <= 10:
                    print(f"   • Classes: {', '.join(self.classes)}")
                elif len(self.classes) > 10:
                    print(f"   • Sample classes: {', '.join(self.classes[:5])}, ...")
                print()
                
        elif self.task == 'zeroshot_retrieval':
            self.classes = []
            self.templates = []
            # For retrieval, build caption list
            if verbose:
                print("Building caption index for retrieval...")
            self.captions = self._build_caption_list()
            
            if verbose:
                print(f"\n✓ Retrieval dataset ready:")
                print(f"   • {len(self)} image samples")
                print(f"   • {len(self.captions)} unique captions")
                print()
        else:
            self.classes = []
            self.templates = []
            self.captions = []
            
            if verbose:
                print(f"\n✓ Dataset ready: {len(self)} samples\n")
    
    def _build_caption_list(self) -> List[str]:
        """Build list of unique captions for retrieval tasks."""
        caption_set = set()
        
        # Iterate through dataset to collect all unique captions
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                # Handle different caption formats
                if isinstance(sample, tuple):
                    if len(sample) >= 2:
                        captions = sample[1]
                        if isinstance(captions, str):
                            caption_set.add(captions)
                        elif isinstance(captions, list):
                            caption_set.update(captions)
                elif isinstance(sample, dict) and 'captions' in sample:
                    captions = sample['captions']
                    if isinstance(captions, str):
                        caption_set.add(captions)
                    elif isinstance(captions, list):
                        caption_set.update(captions)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not extract captions from sample {idx}: {e}")
                continue
        
        return sorted(caption_set)
    
    def _extract_templates_from_hf_dataset(self) -> List[str]:
        """
        Extract zero-shot classification templates from HuggingFace dataset.
        
        Tries to load zeroshot_classification_templates.txt from the dataset repo.
        Raises error if not found - no fallback templates.
        
        Returns:
            List of template strings with {c} placeholder
        """
        import requests
        
        # Use the full dataset name (including wds_ prefix)
        templates_url = f"https://huggingface.co/datasets/clip-benchmark/{self.dataset_name}/raw/main/zeroshot_classification_templates.txt"
        
        try:
            if self.verbose:
                print(f"  📥 Downloading templates from {templates_url}")
            response = requests.get(templates_url, timeout=10)
            response.raise_for_status()
            
            # Parse templates (one per line, strip whitespace)
            templates_text = response.text
            templates = [line.strip() for line in templates_text.strip().split('\n') if line.strip()]
            
            if len(templates) > 0:
                if self.verbose:
                    print(f"  ✓ Loaded {len(templates)} templates from dataset repo")
                return templates
            else:
                raise ValueError("Templates file is empty")
                
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load templates for dataset {self.dataset_name}.\n"
                f"Tried URL: {templates_url}\n"
                f"Error: {e}\n"
                f"Classification datasets must have zeroshot_classification_templates.txt file."
            )
    
    def _extract_classnames_from_hf_dataset(self) -> List[str]:
        """
        Extract class names from HuggingFace dataset.
        
        Tries to load classnames.txt from the dataset repo first.
        Raises error if not found - no fallback.
        
        Returns:
            List of class names
        """
        import requests
        
        # Use the full dataset name (including wds_ prefix)
        classnames_url = f"https://huggingface.co/datasets/clip-benchmark/{self.dataset_name}/raw/main/classnames.txt"
        
        try:
            if self.verbose:
                print(f"  📥 Downloading classnames from {classnames_url}")
            response = requests.get(classnames_url, timeout=10)
            response.raise_for_status()
            
            # Parse classnames (one per line, strip whitespace)
            classnames_text = response.text
            classes = [line.strip() for line in classnames_text.strip().split('\n') if line.strip()]
            
            if len(classes) > 0:
                if self.verbose:
                    print(f"  ✓ Loaded {len(classes)} class names from dataset repo")
                return classes
            else:
                raise ValueError("Classnames file is empty")
                
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load class names for dataset {self.dataset_name}.\n"
                f"Tried URL: {classnames_url}\n"
                f"Error: {e}\n"
                f"Classification datasets must have classnames.txt file."
            )
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        """
        Get a single sample from the dataset.
        
        Returns format depends on task:
        - Classification: (image_tensor, label)
        - Retrieval: (image_tensor, captions) where captions is a list of strings
        """
        sample = self.dataset[idx]
        
        # Handle different dataset formats
        if isinstance(sample, dict):
            # HuggingFace dataset format - try common image field names
            image = None
            for img_key in ['jpg', 'png', 'webp', 'jpeg', 'image', 'img']:
                if img_key in sample and sample[img_key] is not None:
                    image = sample[img_key]
                    break
            
            if image is None:
                raise ValueError(f"Could not find image in sample. Available keys: {list(sample.keys())}")
            
            # Apply preprocessing if available
            if self.image_preprocess is not None:
                image = self.image_preprocess(image)
            
            # Return based on task type
            if self.task == 'zeroshot_retrieval':
                # Get captions (txt field or captions list)
                captions = sample.get('txt', sample.get('captions', sample.get('caption', [])))
                if isinstance(captions, str):
                    # Check if it's a multi-line string (webdataset format from HuggingFace)
                    # Split by newlines and filter out empty lines
                    caption_lines = [line.strip() for line in captions.split('\n') if line.strip()]
                    if len(caption_lines) > 1:
                        # Multiple captions separated by newlines
                        captions = caption_lines
                    else:
                        # Single caption
                        captions = [captions]
                elif not isinstance(captions, list):
                    # Fallback to empty list if unexpected type
                    captions = []
                return image, captions
            else:
                # Classification task
                label = sample.get('cls', sample.get('label', 0))
                return image, label
        
        elif isinstance(sample, (tuple, list)):
            # Tuple format: (image, label/captions)
            image, target = sample
            
            # Apply preprocessing if available
            if self.image_preprocess is not None:
                image = self.image_preprocess(image)
            
            return image, target
        
        else:
            raise ValueError(f"Unexpected sample format: {type(sample)}")
    
    def get_captions(self) -> List[str]:
        """Return list of unique captions (for retrieval tasks)."""
        return self.captions
    
    def get_classes(self) -> List[str]:
        """Return list of class names (for classification tasks)."""
        return self.classes
    
    def get_templates(self) -> List[str]:
        """Return list of text templates (for classification tasks)."""
        return self.templates
    
    def evaluate(
        self,
        embedding_model,
        aligning_model=None,
        device='cuda',
        batch_size: int = 64,
        indices: Optional[List[int]] = None,
        intermediate_text_layer_names: List[str] = ["final"],
        intermediate_image_layer_names: List[str] = ["final"],
        return_embeddings: bool = True,  # NEW: Control embedding return
        max_samples_for_embeddings: int = 5000,  # NEW: Don't return embeddings if dataset is larger
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model on this dataset.
        
        This method provides a unified interface for evaluating both classification
        and retrieval tasks, compatible with the existing evaluation pipeline.
        
        Args:
            embedding_model: CLIP-like model with encode_image and encode_text methods
            aligning_model: Optional alignment model for adjusting embeddings
            device: Device to run evaluation on ('cuda' or 'cpu')
            batch_size: Batch size for evaluation
            indices: Optional list of indices to evaluate on (for subset evaluation)
            intermediate_text_layer_names: Layer names for text embeddings
            intermediate_image_layer_names: Layer names for image embeddings
            return_embeddings: If True, return embeddings. If False, only return metrics (saves memory).
            max_samples_for_embeddings: Maximum dataset size to return embeddings. If dataset is larger,
                                        embeddings won't be returned regardless of return_embeddings flag.
            
        Returns:
            tuple: (results_dict, embeddings_dict)
                - results_dict: Dictionary of evaluation metrics
                - embeddings_dict: Dictionary of computed embeddings (or empty if disabled)
        """
        if self.task == 'zeroshot_classification':
            return self._evaluate_classification(
                embedding_model=embedding_model,
                aligning_model=aligning_model,
                device=device,
                batch_size=batch_size,
                indices=indices,
                intermediate_text_layer_names=intermediate_text_layer_names,
                intermediate_image_layer_names=intermediate_image_layer_names,
                return_embeddings=return_embeddings,
                max_samples_for_embeddings=max_samples_for_embeddings,
            )
        elif self.task == 'zeroshot_retrieval':
            return self._evaluate_retrieval(
                embedding_model=embedding_model,
                aligning_model=aligning_model,
                device=device,
                batch_size=batch_size,
                indices=indices,
                intermediate_text_layer_names=intermediate_text_layer_names,
                intermediate_image_layer_names=intermediate_image_layer_names,
                return_embeddings=return_embeddings,
                max_samples_for_embeddings=max_samples_for_embeddings,
            )
        else:
            raise ValueError(f"Unsupported task: {self.task}")
    
    def _evaluate_classification(
        self,
        embedding_model,
        aligning_model=None,
        device='cuda',
        batch_size=64,
        indices=None,
        intermediate_text_layer_names=["final"],
        intermediate_image_layer_names=["final"],
        return_embeddings=True,
        max_samples_for_embeddings=5000,
    ):
        """Evaluate zero-shot classification task."""
        # Check if we have classes and templates for classification
        if not self.classes or len(self.classes) == 0:
            raise ValueError(
                f"❌ Classification not supported for HuggingFace webdataset '{self.dataset_name}'.\n"
                f"   Webdatasets (wds_*) currently only support retrieval tasks.\n"
                f"   For classification, use the clip_benchmark library format (without wds_ prefix).\n"
                f"   Example: Use 'vtab-caltech101' instead of 'wds_vtab-caltech101'"
            )
        
        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")
        
        # Create subset if indices provided
        # Use 'self' (CLIPBenchmarkDataset) not 'self.dataset' (raw HuggingFace dataset)
        # This ensures __getitem__ preprocessing is applied
        if indices is not None:
            eval_dataset = Subset(self, indices)
        else:
            eval_dataset = self
        
        # Determine if we should return embeddings based on dataset size
        dataset_size = len(eval_dataset)
        should_return_embeddings = return_embeddings and (dataset_size <= max_samples_for_embeddings)
        
        if not should_return_embeddings and dataset_size > max_samples_for_embeddings:
            if self.verbose:
                print(f"[CLIPBenchmarkDataset] Dataset size ({dataset_size}) exceeds max_samples_for_embeddings ({max_samples_for_embeddings})")
                print("[CLIPBenchmarkDataset] Embeddings will NOT be returned to save memory")
        
        # Create dataloader
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        # Build zero-shot classifier using templates and class names
        with EmbeddingCache(
            dataset_name=f"CLIPBench_cache_{self.dataset_name}",
            subset_name=self.split,
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            # Compute text embeddings for all classes
            if self.verbose:
                print(f"[CLIPBenchmarkDataset] Building zero-shot classifier...")
            
            class_embeddings_list = []
            for classname in tqdm(self.classes, desc="Computing class embeddings", disable=not self.verbose):
                # Generate prompts from templates
                if isinstance(self.templates, dict):
                    # Class-specific templates (e.g., CuPL)
                    texts = self.templates[classname]
                elif isinstance(self.templates, list):
                    # Generic templates
                    texts = [template.format(c=classname) for template in self.templates]
                else:
                    texts = [classname]
                
                # Compute embeddings for all templates of this class
                text_embs = cache.get_or_compute_embeddings(
                    texts,
                    f"class_{classname}",
                    compute_caption_embeddings_intermediate_batch,
                    intermediate_text_layer_names,
                )
                
                # Average over templates
                class_emb = F.normalize(text_embs, dim=-1).mean(dim=0)
                class_emb = class_emb / class_emb.norm()
                class_embeddings_list.append(class_emb)
            
            # Stack into classifier matrix [D, num_classes]
            classifier = torch.stack(class_embeddings_list, dim=1).to(device)
            
            # Evaluate on test images
            if self.verbose:
                print(f"[CLIPBenchmarkDataset] Running classification evaluation...")
            
            all_image_embs = [] if should_return_embeddings else None
            all_logits = []
            all_labels = []
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", disable=not self.verbose)):
                # Unpack batch
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    images, labels = batch
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}")
                
                images = images.to(device)
                labels = labels.to(device)
                
                with torch.no_grad():
                    # Compute image embeddings
                    img_embs = cache.get_or_compute_embeddings(
                        images,
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size
                    )
                    
                    # Compute logits
                    img_embs_norm = F.normalize(img_embs, dim=-1)
                    logits = 100.0 * img_embs_norm @ classifier
                    
                    # Only collect embeddings if needed
                    if should_return_embeddings:
                        all_image_embs.append(img_embs.cpu())
                    all_logits.append(logits.cpu())
                    all_labels.append(labels.cpu())
            
            # Concatenate results
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            if should_return_embeddings:
                all_image_embs = torch.cat(all_image_embs, dim=0)
            
            # Compute metrics
            pred = all_logits.argmax(dim=1)
            acc1 = (pred == all_labels).float().mean().item()
            
            # Top-5 accuracy if applicable
            if len(self.classes) >= 5:
                _, top5_pred = all_logits.topk(5, dim=1)
                acc5 = top5_pred.eq(all_labels.view(-1, 1).expand_as(top5_pred)).any(dim=1).float().mean().item()
            else:
                acc5 = float('nan')
            
            # Per-class recall
            from sklearn.metrics import balanced_accuracy_score
            mean_per_class_recall = balanced_accuracy_score(
                all_labels.numpy(),
                pred.numpy()
            )
            
            results = {
                "acc1": float(acc1),  # Ensure float type
                "acc5": float(acc5),
                "mean_per_class_recall": float(mean_per_class_recall),
                "num_classes": len(self.classes),
                "num_samples": len(all_labels),
            }
            
            # Return embeddings only if requested and dataset size allows
            if should_return_embeddings:
                embeddings = {
                    "image_embeddings": all_image_embs.numpy(),
                    "classifier": classifier.cpu().numpy(),
                    "logits": all_logits.numpy(),
                    "labels": all_labels.numpy(),
                }
            else:
                embeddings = {
                    "image_embeddings": np.array([]),
                    "classifier": np.array([]),
                    "logits": np.array([]),
                    "labels": np.array([]),
                }
            
            if self.verbose:
                print(f"[CLIPBenchmarkDataset] Results: acc1={acc1:.4f}, acc5={acc5:.4f}, "
                      f"mean_per_class_recall={mean_per_class_recall:.4f}")
            
            return results, embeddings
    
    def _evaluate_retrieval(
        self,
        embedding_model,
        aligning_model=None,
        device='cuda',
        batch_size=64,
        indices=None,
        intermediate_text_layer_names=["final"],
        intermediate_image_layer_names=["final"],
        recall_k_list=[1, 5, 10],
        return_embeddings=True,
        max_samples_for_embeddings=5000,
    ):
        """Evaluate zero-shot retrieval task.
        
        Args:
            return_embeddings: If True, return embeddings along with results.
            max_samples_for_embeddings: Maximum dataset size for which embeddings are returned.
                For larger datasets, embeddings are not returned to save memory.
        """
        # Import alignment functions from utils.align
        from utils.align import (
            compute_caption_embeddings_intermediate_batch,
            compute_image_embeddings_intermediate_batch,
        )
        
        # Check dataset size and decide whether to return embeddings
        dataset_size = len(self)
        should_return_embeddings = return_embeddings and (dataset_size <= max_samples_for_embeddings)
        
        if not should_return_embeddings and dataset_size > max_samples_for_embeddings:
            print(f"[CLIPBenchmarkDataset] Dataset size ({dataset_size}) exceeds max_samples_for_embeddings "
                  f"({max_samples_for_embeddings}). Embeddings will not be returned to save memory.")
        
        # NOTE: Disable EmbeddingCache for CLIP Benchmark to avoid stale cached embeddings.
        # For these benchmark sizes, recomputing embeddings is cheap enough.
        
        # Create subset if indices provided
        # Use 'self' (CLIPBenchmarkDataset) not 'self.dataset' (raw HuggingFace dataset)
        # This ensures __getitem__ preprocessing is applied
        if indices is not None:
            eval_dataset = Subset(self, indices)
        else:
            eval_dataset = self
        
        # Create dataloader with custom collate function to handle variable-length captions
        # Default collate fails on MSCOCO because different images can have different numbers of captions
        def custom_collate_fn(batch):
            """Custom collate function that preserves variable-length caption lists.
            
            Args:
                batch: List of (image, captions) tuples from __getitem__
                
            Returns:
                images: Stacked tensor of images [batch_size, C, H, W]
                captions: List of caption lists [[cap1, cap2, ...], [cap1, cap2, cap3], ...]
            """
            images = torch.stack([item[0] for item in batch])
            captions = [item[1] for item in batch]  # Keep as list of lists
            return images, captions
        
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )
        
        # NOTE: Disable EmbeddingCache for CLIP Benchmark to avoid stale cached embeddings.
        # For these benchmark sizes, recomputing embeddings is cheap enough.
        
        if self.verbose:
            print(f"[CLIPBenchmarkDataset] Running retrieval evaluation...")
        
        batch_images_emb_list = []
        batch_texts_emb_list = []
        texts_image_index = []  # Maps text index to image index
        
        # Track cumulative counts for proper cache indexing
        cumulative_image_count = 0
        cumulative_text_count = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing embeddings", disable=not self.verbose)):
            # Unpack batch - format depends on dataset
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, captions = batch
            else:
                raise ValueError(f"Unexpected batch format for retrieval: {type(batch)}")
            
            images = images.to(device)
            current_batch_size = len(images)
            
            # Custom collate_fn preserves caption lists, so no need to transpose
            # captions is already in format: [[cap1_img0, cap2_img0, ...], [cap1_img1, cap2_img1, ...], ...]
            
            # Flatten captions if nested
            flat_captions = []
            caption_image_indices = []
            for i, caps in enumerate(captions):
                if isinstance(caps, list):
                    flat_captions.extend(caps)
                    caption_image_indices.extend([i] * len(caps))
                else:
                    flat_captions.append(caps)
                    caption_image_indices.append(i)
            
            current_text_count = len(flat_captions)

            
            with torch.no_grad():
                # Compute image embeddings directly (no cache)
                # Returns dict: {layer_name: tensor}
                img_embs_dict = compute_image_embeddings_intermediate_batch(
                    images,
                    embedding_model,
                    device,
                    intermediate_image_layer_names,
                    dtype=torch.float32
                )
                
                # Compute text embeddings directly (no cache)
                # Returns dict: {layer_name: tensor}
                text_embs_dict = compute_caption_embeddings_intermediate_batch(
                    flat_captions,
                    embedding_model,
                    device,
                    intermediate_text_layer_names,
                    dtype=torch.float32
                )
                
                # Extract "final" layer embeddings (or first available layer)
                img_layer = "final" if "final" in img_embs_dict else intermediate_image_layer_names[0]
                text_layer = "final" if "final" in text_embs_dict else intermediate_text_layer_names[0]
                
                # Store embeddings temporarily for metric computation
                # Note: For retrieval, we need embeddings to compute metrics, but can discard after
                batch_images_emb_list.append(img_embs_dict[img_layer].cpu())
                batch_texts_emb_list.append(text_embs_dict[text_layer].cpu())
                
                # Track which image each caption belongs to (global indices)
                texts_image_index.extend([cumulative_image_count + i for i in caption_image_indices])
                
                # Update cumulative counts for next batch
                cumulative_image_count += current_batch_size
                cumulative_text_count += current_text_count
        
        # Concatenate all embeddings (needed for metric computation)
        images_emb = torch.cat(batch_images_emb_list, dim=0)  # [num_images, D]
        texts_emb = torch.cat(batch_texts_emb_list, dim=0)    # [num_texts, D]
        
        # Normalize embeddings
        images_emb = F.normalize(images_emb, dim=-1)
        texts_emb = F.normalize(texts_emb, dim=-1)
        
        # Compute similarity scores (needed for metrics)
        scores = texts_emb @ images_emb.t()  # [num_texts, num_images]
        
        # Create positive pairs matrix
        # Each caption should match with its corresponding image
        positive_pairs = torch.zeros_like(scores, dtype=bool)
        texts_image_index_tensor = torch.tensor(texts_image_index)
        positive_pairs[torch.arange(len(scores)), texts_image_index_tensor] = True
        
        if self.verbose:
            num_captions_per_image = len(texts_image_index) / len(images_emb)
            print(f"  Avg captions per image: {num_captions_per_image:.1f}")
        
        # Compute retrieval metrics
        metrics = {}
        for k in recall_k_list:
            # Image retrieval: for each text, retrieve top-k images
            image_recall = self._recall_at_k(scores, positive_pairs, k)
            metrics[f"image_retrieval_recall@{k}"] = float((image_recall > 0).float().mean().item())
            
            # Text retrieval: for each image, retrieve top-k texts
            text_recall = self._recall_at_k(scores.t(), positive_pairs.t(), k)
            metrics[f"text_retrieval_recall@{k}"] = float((text_recall > 0).float().mean().item())
        
        # Add dataset size info
        metrics["num_images"] = len(images_emb)
        metrics["num_texts"] = len(texts_emb)
        
        # Return embeddings only if requested and dataset size allows
        if should_return_embeddings:
            embeddings = {
                "image_embeddings": images_emb.numpy(),
                "text_embeddings": texts_emb.numpy(),
                "similarity_scores": scores.numpy(),
            }
        else:
            embeddings = {
                "image_embeddings": np.array([]),
                "text_embeddings": np.array([]),
                "similarity_scores": np.array([]),
            }
        
        if self.verbose:
            print(f"[CLIPBenchmarkDataset] Retrieval results:")
            for k in recall_k_list:
                print(f"  Image R@{k}: {metrics[f'image_retrieval_recall@{k}']:.4f}")
                print(f"  Text R@{k}: {metrics[f'text_retrieval_recall@{k}']:.4f}")
        
        return metrics, embeddings
    
    @staticmethod
    def _recall_at_k(scores, positive_pairs, k):
        """
        Compute recall@k for retrieval.
        
        Args:
            scores: [N, M] similarity scores
            positive_pairs: [N, M] boolean matrix of positive pairs
            k: number of top results to consider
            
        Returns:
            [N] recall@k for each query
        """
        # Get top-k indices
        topk_indices = torch.topk(scores, k, dim=1)[1]
        
        # Create one-hot encoding of top-k
        topk_onehot = torch.zeros_like(scores, dtype=bool)
        topk_onehot.scatter_(1, topk_indices, True)
        
        # Count true positives in top-k
        true_positives = (topk_onehot & positive_pairs).sum(dim=1)
        
        # Count total positives
        num_positives = positive_pairs.sum(dim=1)
        
        # Compute recall
        recall = true_positives.float() / num_positives.float()
        
        return recall
    
    def split_dataset(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        split_type: str = 'random'
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            seed: Random seed for reproducibility
            split_type: 'random' or 'stratified'
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys, each containing 'indices' array
        """
        rng = random.Random(seed)
        np.random.seed(seed)
        
        n = len(self)
        indices = list(range(n))
        
        if split_type == 'random':
            rng.shuffle(indices)
            
            n_test = int(n * test_ratio)
            n_val = int((n - n_test) * val_ratio)
            
            test_idx = np.array(indices[:n_test], dtype=np.int64)
            val_idx = np.array(indices[n_test:n_test + n_val], dtype=np.int64)
            train_idx = np.array(indices[n_test + n_val:], dtype=np.int64)
            
        elif split_type == 'stratified' and self.task == 'zeroshot_classification':
            # Stratified split for classification tasks
            from sklearn.model_selection import train_test_split
            
            # Get labels
            labels = []
            for idx in range(n):
                _, label = self.dataset[idx]
                labels.append(label if isinstance(label, int) else label.item())
            
            # First split: test
            train_val_idx, test_idx = train_test_split(
                indices, test_size=test_ratio, stratify=labels, random_state=seed
            )
            
            # Second split: train/val
            train_labels = [labels[i] for i in train_val_idx]
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio / (1 - test_ratio),
                stratify=train_labels,
                random_state=seed
            )
            
            train_idx = np.array(train_idx, dtype=np.int64)
            val_idx = np.array(val_idx, dtype=np.int64)
            test_idx = np.array(test_idx, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported split_type: {split_type}")
        
        return {
            "train": {"indices": train_idx},
            "val": {"indices": val_idx},
            "test": {"indices": test_idx},
        }


# Convenience function for quick dataset creation
def load_clip_benchmark_dataset(
    dataset_name: str,
    data_root: str = 'datasets/clip_benchmark',
    task: str = 'auto',
    image_preprocess=None,
    **kwargs
) -> CLIPBenchmarkDataset:
    """
    Convenience function to load a CLIP benchmark dataset with automatic everything.
    
    Args:
        dataset_name: Name of dataset (e.g., 'cifar10', 'imagenet1k', 'mscoco_captions')
            See https://huggingface.co/clip-benchmark for available datasets
        data_root: Root directory for data cache (default: 'datasets/clip_benchmark')
        task: Task type ('auto', 'zeroshot_classification', 'zeroshot_retrieval')
        image_preprocess: Image preprocessing function
        **kwargs: Additional arguments passed to CLIPBenchmarkDataset
        
    Returns:
        CLIPBenchmarkDataset instance ready to use
        
    Example:
        ```python
        import open_clip
        
        model, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        
        # One line to load and auto-download!
        dataset = load_clip_benchmark_dataset('cifar10', image_preprocess=preprocess)
        
        # Evaluate
        results, _ = dataset.evaluate(model, device='cuda')
        print(f"Accuracy: {results['acc1']:.2%}")
        ```
    """
    return CLIPBenchmarkDataset(
        dataset_name=dataset_name,
        task=task,
        data_root=data_root,
        image_preprocess=image_preprocess,
        **kwargs
    )


def list_available_benchmarks(verbose: bool = True) -> Dict[str, List[str]]:
    """
    List all available CLIP benchmark datasets from HuggingFace.
    
    Args:
        verbose: If True, print formatted list of datasets
        
    Returns:
        Dictionary with 'classification' and 'retrieval' keys containing dataset names
        
    Example:
        ```python
        benchmarks = list_available_benchmarks()
        print(f"Available classification: {len(benchmarks['classification'])}")
        print(f"Available retrieval: {len(benchmarks['retrieval'])}")
        ```
    """
    benchmarks = {
        'classification': SUPPORTED_CLASSIFICATION_DATASETS,
        'retrieval': SUPPORTED_RETRIEVAL_DATASETS,
    }
    
    if verbose:
        print("\n" + "="*70)
        print("📊 Available CLIP Benchmark Datasets")
        print("="*70)
        print(f"\nSource: https://huggingface.co/clip-benchmark")
        print(f"\nAll datasets are automatically downloaded on first use.\n")
        
        print("🎯 Classification Datasets ({} available):".format(len(benchmarks['classification'])))
        print("-" * 70)
        for i, name in enumerate(benchmarks['classification'], 1):
            print(f"  {i:2d}. {name}")
        
        print(f"\n🔍 Retrieval Datasets ({len(benchmarks['retrieval'])} available):")
        print("-" * 70)
        for i, name in enumerate(benchmarks['retrieval'], 1):
            print(f"  {i:2d}. {name}")
        
        print("\n" + "="*70)
        print("\nUsage:")
        print("  dataset = load_clip_benchmark_dataset('cifar10', image_preprocess=preprocess)")
        print("  results, _ = dataset.evaluate(model, device='cuda')")
        print("="*70 + "\n")
    
    return benchmarks


# List of supported datasets for reference
SUPPORTED_CLASSIFICATION_DATASETS = [
    'cifar10', 'cifar100', 'cifar20',
    'imagenet1k', 'imagenet-a', 'imagenet-r', 'imagenet-sketch', 'imagenetv2',
    'mnist', 'fashion-mnist',
    'stl10', 'gtsrb', 'country211',
    'fgvc_aircraft', 'stanford_cars', 'dtd', 'oxford_pets', 'oxford_flowers',
    'food101', 'sun397', 'caltech101', 'caltech256',
    'voc2007', 'voc2007_multilabel',
    'eurosat', 'resisc45',
    'renderedsst2',
    'pcam',
]

SUPPORTED_RETRIEVAL_DATASETS = [
    'mscoco_captions',
    'flickr8k',
    'flickr30k',
]

# Usage example in __main__
if __name__ == "__main__":
    import open_clip
    
    print("\n" + "="*70)
    print("CLIP Benchmark - Automatic Dataset Loading Demo")
    print("="*70 + "\n")
    
    # Show available benchmarks
    list_available_benchmarks()
    
    # Example 1: Classification with automatic everything
    print("\n" + "="*70)
    print("Example 1: Zero-shot Classification on CIFAR-10")
    print("="*70 + "\n")
    
    # Load model
    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='openai'
    )
    model = model.eval()
    print("✓ Model loaded\n")
    
    # Load dataset - everything is automatic!
    dataset = load_clip_benchmark_dataset(
        dataset_name='cifar10',  # Will auto-download if needed
        image_preprocess=preprocess,
    )
    
    # Evaluate (small subset for demo)
    print("\nEvaluating on first 100 samples...")
    results, embeddings = dataset.evaluate(
        embedding_model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=64,
        indices=list(range(100))  # Just first 100 samples for demo
    )
    
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    print(f"  Top-1 Accuracy: {results['acc1']:.2%}")
    print(f"  Top-5 Accuracy: {results['acc5']:.2%}")
    print(f"  Mean Per-Class Recall: {results['mean_per_class_recall']:.2%}")
    print("="*70 + "\n")
    
    # Example 2: Try another dataset
    print("="*70)
    print("Example 2: Try loading different benchmarks")
    print("="*70 + "\n")
    
    try_datasets = ['mnist', 'fashion-mnist', 'stl10']
    for ds_name in try_datasets:
        try:
            print(f"Loading {ds_name}...")
            ds = load_clip_benchmark_dataset(
                dataset_name=ds_name,
                image_preprocess=preprocess,
                verbose=False
            )
            print(f"  ✓ {ds_name}: {len(ds)} samples, {len(ds.classes)} classes")
        except Exception as e:
            print(f"  ✗ {ds_name}: {str(e)[:50]}...")
    
    print("\n" + "="*70)
    print("✓ Demo complete!")
    print("="*70)

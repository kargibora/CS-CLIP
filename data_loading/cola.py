"""
COLA (Compose Objects Localized with Attributes) Dataset

Based on: https://github.com/arijitray1993/COLA
Paper: https://arxiv.org/abs/2305.03689

COLA evaluates vision-language models' ability to understand compositional 
attributes of objects in images. It contains two settings:

1. Multi-object setting: Tests if models can match images to captions with 
   swapped object attributes (contrastive matching).
   
2. Single-object setting: Tests if models can recognize 320 multi-attribute 
   object classes (e.g., "square white plate") in images using MAP metric.
"""

import json
import logging
import os
import random
import urllib.request
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset

# Optional: only used in _evaluate
try:
    from utils.align import (
        compute_caption_embeddings_intermediate_batch,
        compute_image_embeddings_intermediate_batch,
    )
except ImportError:
    print("Warning: utils.align not found. Evaluation functions may not work.")

logger = logging.getLogger(__name__)


class COLAMultiObjectDataset(Dataset):
    """
    COLA Multi-Object Setting Dataset
    
    Tests whether models can distinguish between correct and incorrect 
    image-caption pairings when object attributes are swapped.
    
    Format: Each sample contains two images and two captions.
    - Image 1 matches Caption 1
    - Image 2 matches Caption 2
    - The captions have swapped attributes/objects
    
    Returns samples in format:
        {
          'image_options': [image1, image2],
          'caption_options': [caption1, caption2],
          'label': 0,  # Index of correct caption for image1
        }
    """
    
    def __init__(
        self,
        data_root: str,
        subset_name: str = "multi_objects",
        image_preprocess=None,
        cache_dir: Optional[str] = None,
        download: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        self.data_root = data_root
        self.subset_name = subset_name
        self.image_preprocess = image_preprocess
        self.cache_dir = cache_dir or os.path.join(data_root, "cache")
        self.verbose = verbose
        
        # COLA data paths
        self.cola_dir = os.path.join(data_root, "cola")
        self.images_dir = os.path.join(self.cola_dir, "images")
        self.annotations_file = os.path.join(
            self.cola_dir, "data", "COLA_multiobjects_matching_benchmark.json"
        )
        
        # Ensure directories exist
        os.makedirs(self.cola_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if download and not os.path.exists(self.annotations_file):
            self.download()
        
        # Load the dataset
        self._load_data()
        
        if verbose:
            print(f"Loaded COLA Multi-Object dataset with {len(self.examples)} samples")
    
    def download(self):
        """Download COLA dataset files from GitHub."""
        if self.verbose:
            print("Downloading COLA Multi-Object dataset...")
        
        os.makedirs(os.path.join(self.cola_dir, "data"), exist_ok=True)
        
        # Download the JSON file
        url = "https://raw.githubusercontent.com/arijitray1993/COLA/main/data/COLA_multiobjects_matching_benchmark.json"
        try:
            urllib.request.urlretrieve(url, self.annotations_file)
            if self.verbose:
                print(f"Downloaded annotations to {self.annotations_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to download COLA dataset: {e}")
    
    def _load_data(self):
        """Load COLA Multi-Object annotations."""
        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(
                f"COLA annotations not found at {self.annotations_file}. "
                "Set download=True to download the dataset."
            )
        
        with open(self.annotations_file, 'r') as f:
            raw_data = json.load(f)
        
        # Parse data: [image1_url, caption1, image2_url, caption2]
        self.examples = []
        for entry in raw_data:
            if len(entry) != 4:
                logger.warning(f"Skipping malformed entry: {entry}")
                continue
            
            image1_url, caption1, image2_url, caption2 = entry
            
            # Extract Visual Genome image IDs from URLs
            # Format: https://cs-people.bu.edu/array/data/vg_gqa_images/2414605.jpg
            image1_id = os.path.splitext(os.path.basename(image1_url))[0]
            image2_id = os.path.splitext(os.path.basename(image2_url))[0]
            
            self.examples.append({
                'image1_id': image1_id,
                'image1_url': image1_url,
                'caption1': caption1,
                'image2_id': image2_id,
                'image2_url': image2_url,
                'caption2': caption2,
            })
        
        # Build caption vocabulary (all unique captions)
        caption_set = set()
        for ex in self.examples:
            caption_set.add(ex['caption1'])
            caption_set.add(ex['caption2'])
        
        self.captions = sorted(caption_set)
        self.caption_to_idx = {cap: i for i, cap in enumerate(self.captions)}
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def _download_image(self, image_url: str, image_path: str) -> bool:
        """Download image from URL if it doesn't exist."""
        if os.path.exists(image_path):
            return True
        
        try:
            urllib.request.urlretrieve(image_url, image_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to download {image_url}: {e}")
            return False
    
    def __getitem__(self, idx: int) -> Dict:
        idx = int(idx)
        example = self.examples[idx]
        
        # Download images if needed
        image1_path = os.path.join(self.images_dir, f"{example['image1_id']}.jpg")
        image2_path = os.path.join(self.images_dir, f"{example['image2_id']}.jpg")
        
        if not os.path.exists(image1_path):
            self._download_image(example['image1_url'], image1_path)
        if not os.path.exists(image2_path):
            self._download_image(example['image2_url'], image2_path)
        
        # Load images
        try:
            image1 = Image.open(image1_path).convert('RGB')
            image2 = Image.open(image2_path).convert('RGB')
            
            if self.image_preprocess is not None:
                image1 = self.image_preprocess(image1)
                image2 = self.image_preprocess(image2)
        except Exception as e:
            logger.error(f"Failed to load images for idx {idx}: {e}")
            # Return dummy data on error
            raise RuntimeError(f"Failed to load images: {e}")
        
        return {
            "image_options": [image1, image2],
            "caption_options": [example['caption1'], example['caption2']],
            "label": 0,  # Caption1 matches Image1
            "image1_id": example['image1_id'],
            "image2_id": example['image2_id'],
        }
    
    def get_captions(self) -> List[str]:
        """Return the unique caption vocabulary."""
        return self.captions
    
    def get_image_paths(self) -> List[str]:
        """Return list of all image paths."""
        paths = []
        for ex in self.examples:
            paths.append(os.path.join(self.images_dir, f"{ex['image1_id']}.jpg"))
            paths.append(os.path.join(self.images_dir, f"{ex['image2_id']}.jpg"))
        return paths
    
    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        
        Since each sample has 2 images and 2 captions, we need to stack them properly.
        """
        batch_images1 = []
        batch_images2 = []
        batch_captions1 = []
        batch_captions2 = []
        
        for sample in batch:
            batch_images1.append(sample['image_options'][0])
            batch_images2.append(sample['image_options'][1])
            batch_captions1.append(sample['caption_options'][0])
            batch_captions2.append(sample['caption_options'][1])
        
        # Stack images
        batch_images1 = torch.stack(batch_images1)
        batch_images2 = torch.stack(batch_images2)
        
        return {
            'images1': batch_images1,
            'images2': batch_images2,
            'captions1': batch_captions1,
            'captions2': batch_captions2,
        }
    
    def evaluate(
        self,
        embedding_model,
        aligning_model=None,
        device="cuda",
        batch_size: int = 16,
        indices: Optional[List[int]] = None,
        intermediate_text_layer_names: List[str] = ["final"],
        intermediate_image_layer_names: List[str] = ["final"],
    ):
        """
        Evaluate COLA Multi-Object dataset.
        
        For each sample, compute a 2x2 similarity matrix:
                  caption1  caption2
        image1    s[0,0]    s[0,1]
        image2    s[1,0]    s[1,1]
        
        Metrics:
        - text_contrastive_accuracy: Both texts prefer their correct images
          (text1 prefers image1 over image2: sim(t1,i1) > sim(t1,i2) AND 
           text2 prefers image2 over image1: sim(t2,i2) > sim(t2,i1))
        - image_contrastive_accuracy: Both images prefer their correct texts
          (image1 prefers text1 over text2: sim(i1,t1) > sim(i2,t1) AND 
           image2 prefers text2 over text1: sim(i2,t2) > sim(i1,t2))
        - group_contrastive_accuracy: Both text and image contrastive conditions are met
        
        Returns:
            results: dict with accuracy metrics
            embeddings: dict with image and caption embeddings
        """
        if 'compute_caption_embeddings_intermediate_batch' not in globals():
            raise ImportError("utils.align module not available for evaluation")
        
        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")
        
        # Create subset dataset if indices provided
        if indices is not None:
            eval_dataset = Subset(self, indices)
        else:
            eval_dataset = self
        
        # Use DataLoader for efficient batch loading
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        # Accuracy tracking
        text_correct_list = []
        image_correct_list = []
        group_correct_list = []
        
        # Lists to store embeddings for return
        all_img1_embs = []
        all_img2_embs = []
        all_cap1_embs = []
        all_cap2_embs = []
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="COLA",
            subset_name="multi_object",
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Evaluating COLA Multi-Object")):
                batch_images1 = batch['images1'].to(device)
                batch_images2 = batch['images2'].to(device)
                batch_captions1 = batch['captions1']
                batch_captions2 = batch['captions2']
                B = len(batch_captions1)
                
                if B == 0:
                    continue
                
                with torch.no_grad():
                    # Compute embeddings for image1
                    img1_embs = cache.get_or_compute_embeddings(
                        batch_images1,
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size * 2  # *2 because we have 2 images per sample
                    )
                    
                    # Compute embeddings for image2
                    img2_embs = cache.get_or_compute_embeddings(
                        batch_images2,
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size * 2 + B
                    )
                    
                    # Compute embeddings for caption1
                    cap1_embs = cache.get_or_compute_embeddings(
                        batch_captions1,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * batch_size * 2
                    )
                    
                    # Compute embeddings for caption2
                    cap2_embs = cache.get_or_compute_embeddings(
                        batch_captions2,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * batch_size * 2 + B
                    )
                    
                    # Convert to CPU for computing similarities
                    img1_embs = img1_embs.cpu().float()
                    img2_embs = img2_embs.cpu().float()
                    cap1_embs = cap1_embs.cpu().float()
                    cap2_embs = cap2_embs.cpu().float()
                    
                    # Store embeddings
                    all_img1_embs.append(img1_embs)
                    all_img2_embs.append(img2_embs)
                    all_cap1_embs.append(cap1_embs)
                    all_cap2_embs.append(cap2_embs)
                    
                    # Compute similarities for all samples in batch
                    # sim(image1, caption1), sim(image1, caption2)
                    sim_i1_c1 = (img1_embs * cap1_embs).sum(dim=1)  # [B]
                    sim_i1_c2 = (img1_embs * cap2_embs).sum(dim=1)  # [B]
                    # sim(image2, caption1), sim(image2, caption2)
                    sim_i2_c1 = (img2_embs * cap1_embs).sum(dim=1)  # [B]
                    sim_i2_c2 = (img2_embs * cap2_embs).sum(dim=1)  # [B]
                    
                    # Text contrastive accuracy: 
                    # - text1 prefers image1 over image2: sim(t1,i1) > sim(t1,i2)
                    # - text2 prefers image2 over image1: sim(t2,i2) > sim(t2,i1)
                    text_cond_1 = sim_i1_c1 > sim_i1_c2  # text1 prefers image1
                    text_cond_2 = sim_i2_c2 > sim_i2_c1  # text2 prefers image2
                    batch_text_correct = text_cond_1 & text_cond_2
                    
                    # Image contrastive accuracy:
                    # - image1 prefers text1 over text2: sim(i1,t1) > sim(i2,t1)
                    # - image2 prefers text2 over text1: sim(i2,t2) > sim(i1,t2)
                    image_cond_1 = sim_i1_c1 > sim_i2_c1  # image1 prefers text1
                    image_cond_2 = sim_i2_c2 > sim_i1_c2  # image2 prefers text2
                    batch_image_correct = image_cond_1 & image_cond_2
                    
                    # Group contrastive accuracy: both text and image correct
                    batch_group_correct = batch_text_correct & batch_image_correct
                    
                    # Store results
                    text_correct_list.extend(batch_text_correct.cpu().tolist())
                    image_correct_list.extend(batch_image_correct.cpu().tolist())
                    group_correct_list.extend(batch_group_correct.cpu().tolist())
        
        # Compute accuracies
        text_accuracy = float(np.mean(text_correct_list)) if text_correct_list else 0.0
        image_accuracy = float(np.mean(image_correct_list)) if image_correct_list else 0.0
        group_accuracy = float(np.mean(group_correct_list)) if group_correct_list else 0.0
        
        results = {
            "text_contrastive_accuracy": text_accuracy,
            "image_contrastive_accuracy": image_accuracy,
            "group_contrastive_accuracy": group_accuracy,
            "multi_object_accuracy": group_accuracy,  # Backward compatibility
        }
        
        # Prepare embeddings for return
        # Stack all embeddings
        all_img1_embs = torch.cat(all_img1_embs, dim=0).numpy()
        all_img2_embs = torch.cat(all_img2_embs, dim=0).numpy()
        all_cap1_embs = torch.cat(all_cap1_embs, dim=0).numpy()
        all_cap2_embs = torch.cat(all_cap2_embs, dim=0).numpy()
        
        # Interleave images: [img1_0, img2_0, img1_1, img2_1, ...]
        N = all_img1_embs.shape[0]
        D = all_img1_embs.shape[1]
        image_embeddings = np.empty((N * 2, D), dtype=all_img1_embs.dtype)
        image_embeddings[0::2] = all_img1_embs
        image_embeddings[1::2] = all_img2_embs
        
        # Interleave captions similarly
        caption_embeddings = np.empty((N * 2, D), dtype=all_cap1_embs.dtype)
        caption_embeddings[0::2] = all_cap1_embs
        caption_embeddings[1::2] = all_cap2_embs
        
        embeddings = {
            "image_embeddings": image_embeddings,
            "caption_embeddings": caption_embeddings,
        }
        
        return results, embeddings
    
    def split_dataset(
        self,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
        split_type: Literal["random"] = "random",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Split dataset into train/val/test."""
        np.random.seed(seed)
        random.seed(seed)
        
        n = len(self)
        all_idx = list(range(n))
        random.shuffle(all_idx)
        
        n_test = int(n * test_ratio)
        test_idx = all_idx[:n_test]
        rem_idx = all_idx[n_test:]
        
        adj_val = val_ratio / (1 - test_ratio) if test_ratio < 1 else val_ratio
        train_idx, val_idx = train_test_split(rem_idx, test_size=adj_val, random_state=seed)
        
        return {
            'train': {'indices': train_idx},
            'val': {'indices': val_idx},
            'test': {'indices': test_idx},
        }


class COLASingleObjectDataset(Dataset):
    """
    COLA Single-Object Setting Dataset
    
    Tests whether models can recognize 320 multi-attribute object classes
    (e.g., "square white plate") in images.
    
    Format: Each sample contains:
    - image_file: Path to Visual Genome image
    - objects_attributes_annotation: Dict of objects with attributes (optional)
    - label: Binary label (0/1) for each of 320 classes indicating presence
    - hard_list: Binary label (0/1) for each of 320 classes indicating if
                 the image is a "hard" distractor for that class
    
    Evaluation uses Mean Average Precision (MAP) on the hard subset.
    """
    
    def __init__(
        self,
        data_root: str,
        subset_name: str = "GQA",  # Options: "GQA", "CLEVR", "PACO"
        image_preprocess=None,
        cache_dir: Optional[str] = None,
        download: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        assert subset_name in ["GQA", "CLEVR", "PACO"], \
            f"subset_name must be one of ['GQA', 'CLEVR', 'PACO'], got {subset_name}"
        
        self.data_root = data_root
        self.subset_name = subset_name
        self.image_preprocess = image_preprocess
        self.cache_dir = cache_dir or os.path.join(data_root, "cache")
        self.verbose = verbose
        
        # COLA data paths
        self.cola_dir = os.path.join(data_root, "cola")
        self.images_dir = os.path.join(self.cola_dir, "images")
        self.annotations_file = os.path.join(
            self.cola_dir, "data", f"COLA_singleobjects_benchmark_{subset_name}.json"
        )
        
        # Ensure directories exist
        os.makedirs(self.cola_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if download and not os.path.exists(self.annotations_file):
            self.download()
        
        # Load the dataset
        self._load_data()
        
        if verbose:
            print(f"Loaded COLA Single-Object ({subset_name}) dataset:")
            print(f"  - {len(self.examples)} images")
            print(f"  - {len(self.class_labels)} classes")
    
    def download(self):
        """Download COLA dataset files from GitHub."""
        if self.verbose:
            print(f"Downloading COLA Single-Object ({self.subset_name}) dataset...")
        
        os.makedirs(os.path.join(self.cola_dir, "data"), exist_ok=True)
        
        # Download the JSON file
        url = f"https://raw.githubusercontent.com/arijitray1993/COLA/main/data/COLA_singleobjects_benchmark_{self.subset_name}.json"
        try:
            urllib.request.urlretrieve(url, self.annotations_file)
            if self.verbose:
                print(f"Downloaded annotations to {self.annotations_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to download COLA dataset: {e}")
    
    def _load_data(self):
        """Load COLA Single-Object annotations."""
        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(
                f"COLA annotations not found at {self.annotations_file}. "
                "Set download=True to download the dataset."
            )
        
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)
        
        # Parse structure: {"labels": [...], "data": [[image_file, annotations, label, hard_list], ...]}
        # Ensure labels is a flat list of strings
        labels_data = data["labels"]
        if isinstance(labels_data, list) and len(labels_data) > 0:
            # Check if it's a nested structure
            if isinstance(labels_data[0], list):
                # Flatten if nested
                self.class_labels = [str(item) for sublist in labels_data for item in sublist]
            else:
                # Already flat
                self.class_labels = [str(item) for item in labels_data]
        else:
            raise ValueError(f"Invalid labels structure in {self.annotations_file}")
        
        raw_data = data["data"]
        
        self.examples = []
        for entry in raw_data:
            if len(entry) != 4:
                logger.warning(f"Skipping malformed entry: {entry}")
                continue
            
            image_file, annotations, label, hard_list = entry
            
            # Extract image ID from path
            # Format example: "visual_genome/2414605.jpg"
            image_id = os.path.splitext(os.path.basename(image_file))[0]
            
            self.examples.append({
                'image_id': image_id,
                'image_file': image_file,
                'annotations': annotations,  # Dict of objects with attributes (may be None)
                'label': np.array(label, dtype=np.int64),  # Shape: (320,)
                'hard_list': np.array(hard_list, dtype=np.int64),  # Shape: (320,)
            })
        
        # Build caption vocabulary (all class labels)
        self.captions = self.class_labels
        self.caption_to_idx = {cap: i for i, cap in enumerate(self.captions)}
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def _download_image(self, image_url: str, image_path: str) -> bool:
        """Download image from URL if it doesn't exist."""
        if os.path.exists(image_path):
            return True
        
        try:
            urllib.request.urlretrieve(image_url, image_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to download {image_url}: {e}")
            return False
    
    def __getitem__(self, idx: int) -> Dict:
        idx = int(idx)
        example = self.examples[idx]
        
        # Construct image path
        image_path = os.path.join(self.images_dir, f"{example['image_id']}.jpg")
        
        # Download image if needed (from Visual Genome)
        if not os.path.exists(image_path):
            image_url = f"https://cs-people.bu.edu/array/data/vg_gqa_images/{example['image_id']}.jpg"
            self._download_image(image_url, image_path)
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.image_preprocess is not None:
                image = self.image_preprocess(image)
        except Exception as e:
            logger.error(f"Failed to load image for idx {idx}: {e}")
            raise RuntimeError(f"Failed to load image: {e}")
        
        return {
            "image": image,
            "label": example['label'],  # Shape: (320,)
            "hard_list": example['hard_list'],  # Shape: (320,)
            "image_id": example['image_id'],
        }
    
    def get_captions(self) -> List[str]:
        """Return the 320 class labels."""
        return self.captions
    
    def get_image_paths(self) -> List[str]:
        """Return list of image paths."""
        paths = []
        for ex in self.examples:
            paths.append(os.path.join(self.images_dir, f"{ex['image_id']}.jpg"))
        return paths
    
    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        """
        batch_images = []
        batch_labels = []
        batch_hard_lists = []
        batch_image_ids = []
        
        for sample in batch:
            batch_images.append(sample['image'])
            batch_labels.append(torch.from_numpy(sample['label']))
            batch_hard_lists.append(torch.from_numpy(sample['hard_list']))
            batch_image_ids.append(sample['image_id'])
        
        # Stack images and labels
        batch_images = torch.stack(batch_images)
        batch_labels = torch.stack(batch_labels)  # Shape: [B, 320]
        batch_hard_lists = torch.stack(batch_hard_lists)  # Shape: [B, 320]
        
        return {
            'images': batch_images,
            'labels': batch_labels,
            'hard_lists': batch_hard_lists,
            'image_ids': batch_image_ids,
        }
    
    def evaluate(
        self,
        embedding_model,
        aligning_model=None,
        device="cuda",
        batch_size: int = 16,
        indices: Optional[List[int]] = None,
        intermediate_text_layer_names: List[str] = ["final"],
        intermediate_image_layer_names: List[str] = ["final"],
    ):
        """
        Evaluate COLA Single-Object dataset.
        
        Computes Mean Average Precision (MAP) for 320 multi-attribute classes.
        Only evaluates on "hard" examples as specified by hard_list.
        """
        if 'compute_caption_embeddings_intermediate_batch' not in globals():
            raise ImportError("utils.align module not available for evaluation")
        
        try:
            from data_loading.embedding_cache import EmbeddingCache
            import torchmetrics
        except ImportError as e:
            raise ImportError(f"Required module not available for evaluation: {e}")
        
        # Create subset dataset if indices provided
        if indices is not None:
            eval_dataset = Subset(self, indices)
        else:
            eval_dataset = self
        
        # Use DataLoader for efficient batch loading
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        # Lists to accumulate predictions and labels
        all_scores = []  # Will be [N, 320]
        all_labels = []  # Will be [N, 320]
        all_hard_lists = []  # Will be [N, 320]
        all_img_embs = []
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="COLA",
            subset_name=f"single_object_{self.subset_name}",
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            # First, compute embeddings for all 320 class labels (captions)
            # We do this once for all images
            with torch.no_grad():
                caption_embs = cache.get_or_compute_embeddings(
                    self.class_labels,
                    "text",
                    compute_caption_embeddings_intermediate_batch,
                    intermediate_text_layer_names,
                    start_idx=0
                )
                caption_embs = caption_embs.cpu().float()  # Shape: [320, D]
            
            # Now process images in batches
            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Evaluating COLA Single-Object ({self.subset_name})")):
                batch_images = batch['images'].to(device)
                batch_labels = batch['labels']  # Shape: [B, 320]
                batch_hard_lists = batch['hard_lists']  # Shape: [B, 320]
                B = batch_images.shape[0]
                
                if B == 0:
                    continue
                
                with torch.no_grad():
                    # Compute image embeddings
                    img_embs = cache.get_or_compute_embeddings(
                        batch_images,
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size
                    )
                    
                    img_embs = img_embs.cpu().float()  # Shape: [B, D]
                    
                    # Compute similarity scores: [B, D] @ [D, 320] -> [B, 320]
                    scores = torch.matmul(img_embs, caption_embs.T)
                    
                    # Store results
                    all_scores.append(scores)
                    all_labels.append(batch_labels)
                    all_hard_lists.append(batch_hard_lists)
                    all_img_embs.append(img_embs)
        
        # Concatenate all batches
        all_scores = torch.cat(all_scores, dim=0)  # [N, 320]
        all_labels = torch.cat(all_labels, dim=0)  # [N, 320]
        all_hard_lists = torch.cat(all_hard_lists, dim=0)  # [N, 320]
        
        # Transpose for per-class evaluation: [320, N]
        scores_T = all_scores.T
        labels_T = all_labels.T
        hard_lists_T = all_hard_lists.T
        
        # Compute per-class Average Precision on hard examples
        class_aps = []
        for class_idx in range(len(self.class_labels)):
            class_scores = scores_T[class_idx]
            class_labels = labels_T[class_idx]
            class_hard = hard_lists_T[class_idx]
            
            # Skip if no positive examples
            if torch.sum(class_labels) == 0:
                continue
            
            # Get indices of hard examples for this class
            hard_indices = torch.where(class_hard == 1)[0]
            
            # Skip if no hard examples
            if len(hard_indices) == 0:
                continue
            
            # Filter to hard examples only
            hard_scores = class_scores[hard_indices]
            hard_labels = class_labels[hard_indices]
            
            # Compute Average Precision for this class
            ap = torchmetrics.functional.average_precision(
                hard_scores, hard_labels, task="binary"
            )
            
            class_aps.append(ap.item())
        
        # Compute Mean Average Precision
        map_score = np.mean(class_aps) if class_aps else 0.0
        
        results = {
            "single_object_map": map_score,
            "num_classes_evaluated": len(class_aps),
        }
        
        # Prepare embeddings for return
        all_img_embs = torch.cat(all_img_embs, dim=0).numpy()
        caption_embs_np = caption_embs.numpy()
        
        embeddings = {
            "image_embeddings": all_img_embs,  # [N, D]
            "caption_embeddings": caption_embs_np,  # [320, D]
        }
        
        return results, embeddings
    
    def split_dataset(
        self,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
        split_type: Literal["random"] = "random",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Split dataset into train/val/test."""
        np.random.seed(seed)
        random.seed(seed)
        
        n = len(self)
        all_idx = list(range(n))
        random.shuffle(all_idx)
        
        n_test = int(n * test_ratio)
        test_idx = all_idx[:n_test]
        rem_idx = all_idx[n_test:]
        
        adj_val = val_ratio / (1 - test_ratio) if test_ratio < 1 else val_ratio
        train_idx, val_idx = train_test_split(rem_idx, test_size=adj_val, random_state=seed)
        
        return {
            'train': {'indices': train_idx},
            'val': {'indices': val_idx},
            'test': {'indices': test_idx},
        }


if __name__ == "__main__":
    # Simple test to verify dataset loading
    from torchvision import transforms
    
    simple_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    print("=" * 60)
    print("Testing COLA Multi-Object Dataset")
    print("=" * 60)
    
    try:
        multi_ds = COLAMultiObjectDataset(
            "./datasets",
            subset_name="multi_objects",
            image_preprocess=simple_preprocess,
            download=True,
            verbose=True
        )
        print(f"✓ Loaded {len(multi_ds)} samples")
        
        if len(multi_ds) > 0:
            sample = multi_ds[0]
            print("✓ Sample structure:")
            print(f"  - image_options: {len(sample['image_options'])} images")
            print(f"  - caption_options: {len(sample['caption_options'])} captions")
            print(f"  - Captions: {sample['caption_options']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Testing COLA Single-Object Dataset (GQA)")
    print("=" * 60)
    
    try:
        single_ds = COLASingleObjectDataset(
            "./datasets",
            subset_name="GQA",
            image_preprocess=simple_preprocess,
            download=True,
            verbose=True
        )
        print(f"✓ Loaded {len(single_ds)} samples")
        
        if len(single_ds) > 0:
            sample = single_ds[0]
            print("✓ Sample structure:")
            print(f"  - image: {sample['image'].shape if hasattr(sample['image'], 'shape') else 'tensor'}")
            print(f"  - label: {sample['label'].shape}")
            print(f"  - hard_list: {sample['hard_list'].shape}")
            print(f"  - First 5 class labels: {single_ds.class_labels[:5]}")
    except Exception as e:
        print(f"✗ Error: {e}")

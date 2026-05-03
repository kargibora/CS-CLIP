import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from datasets import load_dataset, DownloadMode
except ImportError:
    print("Warning: datasets library not found. Please install with: pip install datasets")
    load_dataset = None
    DownloadMode = None

# Optional: only used in _evaluate
try:
    from utils.align import (
        compute_caption_embeddings_intermediate_batch,
        compute_image_embeddings_intermediate_batch,
    )
except ImportError:
    print("Warning: utils.align not found. Evaluation functions may not work.")


class COCOCounterfactualsDataset(Dataset):
    """
    COCO-Counterfactuals dataset for evaluating multimodal subject binding and counterfactual reasoning.
    
    Each sample has: image_0, image_1, caption_0, caption_1
    - Goal: image_0 should match caption_0, image_1 should match caption_1
    - Captions differ only by a noun subject; images differ only by the altered subject
    
    Returns samples in format compatible with alignment pipeline:
        {
          'image_options': [image_0, image_1],
          'caption_options': [caption_0, caption_1],
          'label': 0,
          'pair_id': int,
        }
    """

    def __init__(
        self,
        data_root: str,
        subset_name: str = "all",
        image_preprocess=None,
        cache_dir: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        download: bool = True,
        verbose: bool = True,
        **kwargs
    ):
        self.data_root = data_root
        self.subset_name = subset_name
        self.image_preprocess = image_preprocess
        self.cache_dir = cache_dir or os.path.join(data_root, "cache")
        self.verbose = verbose
        
        # Ensure data directory exists
        os.makedirs(data_root, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if local data exists first (auto-detection like other datasets)
        local_data_dir = os.path.join(data_root)
        metadata_file = None
        
        # Look for metadata file (examples.jsonl or similar)
        possible_metadata_files = [
            "examples.jsonl",
            "metadata.jsonl", 
            "data.jsonl",
            "coco_counterfactuals.jsonl"
        ]
        
        for filename in possible_metadata_files:
            potential_path = os.path.join(local_data_dir, filename)
            if os.path.exists(potential_path):
                metadata_file = potential_path
                break
        
        if metadata_file:
            # Load from local files (prioritized like other datasets)
            if verbose:
                print(f"[COCOCounterfactualsDataset] Loading from local files: {metadata_file}")
            self.examples = self._load_local_data(metadata_file, local_data_dir)
            if verbose:
                print(f"[COCOCounterfactualsDataset] Loaded {len(self.examples)} examples from local files")
        else:
            # Try to load from local cache
            cache_path = os.path.join(self.cache_dir, "coco_counterfactuals_examples.json")
            if os.path.exists(cache_path):
                if verbose:
                    print(f"[COCOCounterfactualsDataset] Loading from cache: {cache_path}")
                with open(cache_path, 'r') as f:
                    self.examples = json.load(f)
            else:
                raise FileNotFoundError(
                    f"No data found. Tried:\n"
                    f"1. Local data: {local_data_dir}/ (looking for examples.jsonl)\n"
                    f"2. Cache: {cache_path}\n"
                    f"Please either:\n"
                    f"- Put your data in {local_data_dir}/ (with examples.jsonl and images)\n"
                    f"- Set download=True to download from Hugging Face\n"
                    f"- Manually download the dataset"
                )
        
        # No subset filtering for now (future: could filter by subject type)
        
        # Build caption vocabulary (all unique captions)
        caption_set = set()
        for ex in self.examples:
            caption_set.add(ex['caption_0'])
            caption_set.add(ex['caption_1'])
        
        self.captions = sorted(caption_set)
        self.caption_to_idx = {cap: i for i, cap in enumerate(self.captions)}
        
        if verbose:
            print(f"[COCOCounterfactualsDataset] Dataset ready with {len(self.examples)} examples")
            print(f"[COCOCounterfactualsDataset] Unique captions: {len(self.captions)}")

    def _load_local_data(self, metadata_file: str, data_dir: str):
        """Load COCO-Counterfactuals data from local JSONL file and images."""
        examples = []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Validate required fields
                    required_fields = ['id', 'caption_0', 'caption_1', 'image_0', 'image_1']
                    if not all(field in data for field in required_fields):
                        if self.verbose:
                            print(f"[COCOCounterfactualsDataset] Warning: Line {line_num + 1} missing required fields")
                        continue
                    
                    # Check if image files exist
                    img_0_path = os.path.join(data_dir, f"{data['image_0']}.jpg")
                    img_1_path = os.path.join(data_dir, f"{data['image_1']}.jpg")
                    
                    if not os.path.exists(img_0_path):
                        if self.verbose:
                            print(f"[COCOCounterfactualsDataset] Warning: Image not found: {img_0_path}")
                        continue
                    if not os.path.exists(img_1_path):
                        if self.verbose:
                            print(f"[COCOCounterfactualsDataset] Warning: Image not found: {img_1_path}")
                        continue
                    
                    # Store example with local image paths
                    example = {
                        'id': data['id'],
                        'caption_0': data['caption_0'],
                        'caption_1': data['caption_1'],
                        'image_0_path': img_0_path,
                        'image_1_path': img_1_path,
                        'image_0': data['image_0'],  # Keep original names for compatibility
                        'image_1': data['image_1'],
                    }
                    examples.append(example)
                    
                except json.JSONDecodeError as e:
                    if self.verbose:
                        print(f"[COCOCounterfactualsDataset] Warning: Invalid JSON on line {line_num + 1}: {e}")
                    continue
        
        if self.verbose:
            print(f"[COCOCounterfactualsDataset] Successfully loaded {len(examples)} examples from {metadata_file}")
            
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        idx = int(idx)
        example = self.examples[idx]
        
        # Load images from local paths
        try:
            image_0 = Image.open(example['image_0_path']).convert('RGB')
            image_1 = Image.open(example['image_1_path']).convert('RGB')
        except Exception as e:
            if self.verbose:
                print(f"[COCOCounterfactualsDataset] Error loading images for example {idx}: {e}")
        
        if self.image_preprocess is not None:
            if isinstance(image_0, Image.Image):
                image_0 = self.image_preprocess(image_0)
            if isinstance(image_1, Image.Image):
                image_1 = self.image_preprocess(image_1)
        caption_0 = example['caption_0']
        caption_1 = example['caption_1']
        return {
            "image_options": [image_0, image_1],
            "caption_options": [caption_0, caption_1],
            "label": 0,
            "pair_id": example.get('id', idx),
        }

    def get_captions(self) -> List[str]:
        return self.captions

    def get_image_paths(self) -> List[str]:
        paths = []
        for i, ex in enumerate(self.examples):
            paths.append(f"coco_counterfactuals_{ex.get('id', i)}_image_0")
            paths.append(f"coco_counterfactuals_{ex.get('id', i)}_image_1")
        return paths

    def get_idx_to_ptr(self, idx: int) -> int:
        example = self.examples[idx]
        caption = example['caption_0']
        return self.caption_to_idx.get(caption, -1)

    def get_idx_to_candidates_ptr(self, idx: int) -> List[int]:
        example = self.examples[idx]
        caption_1 = example['caption_1']
        ptr = self.caption_to_idx.get(caption_1)
        return [ptr] if ptr is not None else []

    def _collate_fn(self, batch):
        """Custom collate function for DataLoader to handle COCO-Counterfactuals batch format."""
        # batch is a list of samples from __getitem__
        images_0 = []
        images_1 = []
        captions_0 = []
        captions_1 = []
        
        for sample in batch:
            images_0.append(sample['image_options'][0])
            images_1.append(sample['image_options'][1])
            captions_0.append(sample['caption_options'][0])
            captions_1.append(sample['caption_options'][1])
        
        return {
            'images_0': torch.stack(images_0),  # [B, C, H, W]
            'images_1': torch.stack(images_1),  # [B, C, H, W]
            'captions_0': captions_0,  # List[str]
            'captions_1': captions_1,  # List[str]
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
        return self._evaluate(
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device,
            batch_size=batch_size,
            indices=indices,
            intermediate_text_layer_names=intermediate_text_layer_names,
            intermediate_image_layer_names=intermediate_image_layer_names,
        )

    def _evaluate(
        self,
        embedding_model,
        aligning_model=None,
        device="cuda",
        batch_size: int = 16,
        indices: Optional[List[int]] = None,
        intermediate_text_layer_names: List[str] = ["final"],
        intermediate_image_layer_names: List[str] = ["final"],
    ):
        if 'compute_caption_embeddings_intermediate_batch' not in globals():
            raise ImportError("utils.align functions not available for evaluation")
            
        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")
            
        from torch.utils.data import DataLoader, Subset
        
        # Create subset dataset if indices provided
        if indices is not None:
            eval_dataset = Subset(self, indices)
        else:
            eval_dataset = self
            
        # Use DataLoader for efficient multi-threaded batch loading
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Multi-threaded loading
            pin_memory=True,  # Faster GPU transfer
            collate_fn=self._collate_fn  # Custom collate function
        )
        
        text_scores = []
        image_scores = []
        group_scores = []
        image_emb_list_0 = []
        image_emb_list_1 = []
        caption_emb_list_0 = []
        caption_emb_list_1 = []
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="COCO_Counterfactuals",
            subset_name="all",
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating COCO-Counterfactuals")):
                # Batch is already properly formatted by collate_fn
                batch_images_0 = batch['images_0']  # [B, C, H, W]
                batch_images_1 = batch['images_1']  # [B, C, H, W]
                batch_captions_0 = batch['captions_0']  # List[str]
                batch_captions_1 = batch['captions_1']  # List[str]
                B = len(batch_captions_0)
                
                if B == 0:
                    continue
                    
                with torch.no_grad():
                    # Standard CLIP path
                    # Get embeddings for images_0 (with caching)
                    img_embs_0 = cache.get_or_compute_embeddings(
                        batch_images_0.to(device),
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size * 2  # Account for 2 images per sample
                    )
                    
                    # Get embeddings for images_1 (with caching) 
                    img_embs_1 = cache.get_or_compute_embeddings(
                        batch_images_1.to(device),
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size * 2 + B  # Offset for second set of images
                    )
                    
                    # Get caption embeddings (with caching)
                    all_captions = batch_captions_0 + batch_captions_1
                    cap_embs = cache.get_or_compute_embeddings(
                        all_captions,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * batch_size * 2  # Account for 2 captions per sample
                    )
                    
                    cap_embs_0 = cap_embs[:B]
                    cap_embs_1 = cap_embs[B:]
                    
                    # Vectorized similarity computation
                    sim_c0_i0 = (cap_embs_0 * img_embs_0).sum(dim=1)
                    sim_c0_i1 = (cap_embs_0 * img_embs_1).sum(dim=1)
                    sim_c1_i0 = (cap_embs_1 * img_embs_0).sum(dim=1)
                    sim_c1_i1 = (cap_embs_1 * img_embs_1).sum(dim=1)
                    
                    # Vectorized accuracy computation
                    text_correct = (sim_c0_i0 > sim_c1_i0) & (sim_c1_i1 > sim_c0_i1)
                    image_correct = (sim_c0_i0 > sim_c0_i1) & (sim_c1_i1 > sim_c1_i0)
                    group_correct = text_correct & image_correct
                    
                    text_scores.extend(text_correct.cpu().tolist())
                    image_scores.extend(image_correct.cpu().tolist())
                    group_scores.extend(group_correct.cpu().tolist())
                    
                    # Store embeddings
                    image_emb_list_0.extend([emb.cpu().numpy() for emb in img_embs_0])
                    image_emb_list_1.extend([emb.cpu().numpy() for emb in img_embs_1])
                    caption_emb_list_0.extend([emb.cpu().numpy() for emb in cap_embs_0])
                    caption_emb_list_1.extend([emb.cpu().numpy() for emb in cap_embs_1])
        text_acc = float(np.mean(text_scores)) if text_scores else 0.0
        image_acc = float(np.mean(image_scores)) if image_scores else 0.0
        group_acc = float(np.mean(group_scores)) if group_scores else 0.0
        results = {
            "text_contrastive_accuracy": text_acc,
            "image_contrastive_accuracy": image_acc,
            "group_contrastive_accuracy": group_acc,
        }
        if len(image_emb_list_0) > 0:
            img_embs_all = np.concatenate([
                np.stack(image_emb_list_0, axis=0),
                np.stack(image_emb_list_1, axis=0)
            ], axis=0)
            cap_embs_all = np.concatenate([
                np.stack(caption_emb_list_0, axis=0),
                np.stack(caption_emb_list_1, axis=0)
            ], axis=0)
            neg_cap_embs = np.concatenate([
                np.stack(caption_emb_list_1, axis=0),
                np.stack(caption_emb_list_0, axis=0)
            ], axis=0)
            neg_cap_embs = neg_cap_embs[:, np.newaxis, :]
            embeddings = {
                "image_embeddings": img_embs_all,
                "caption_embeddings": cap_embs_all,
                "negative_caption_embeddings": neg_cap_embs,
            }
        else:
            raise ValueError("No valid examples found for evaluation. Check dataset loading.")
        return results, embeddings

    def split_dataset(
        self,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
        split_type: Literal["random", "object"] = "random",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        rng = random.Random(seed)
        if split_type == "random":
            indices = list(range(len(self)))
            rng.shuffle(indices)
            n = len(indices)
            n_test = int(n * test_ratio)
            n_val = int((n - n_test) * val_ratio)
            test_idx = np.array(indices[:n_test], dtype=np.int64)
            val_idx = np.array(indices[n_test:n_test + n_val], dtype=np.int64)
            train_idx = np.array(indices[n_test + n_val:], dtype=np.int64)
            return {
                "train": {"indices": train_idx},
                "val": {"indices": val_idx},
                "test": {"indices": test_idx},
            }
        elif split_type == "object":
            return self.split_dataset(val_ratio, test_ratio, seed, "random")
        else:
            raise ValueError(f"Unknown split_type={split_type}")
    def __getstate__(self):
        state = self.__dict__.copy()
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)

class COCOCounterfactualsNeg(Dataset):
    """
    Wrapper for COCOCounterfactualsDataset to provide tokenized positives and negatives.
    Compatible with other *Neg dataset classes.
    For COCO-Counterfactuals, the "negative" is the other caption in the pair (counterfactual version).
    """
    def __init__(
        self,
        coco_counterfactuals_dataset: COCOCounterfactualsDataset,
        indices: List[int],
        num_negatives: int = 1,
    ):
        super().__init__()
        self.dataset = coco_counterfactuals_dataset
        self.num_negatives = num_negatives
        self.idx_to_dataset_idx = {i: idx for i, idx in enumerate(indices)}
        print(f"COCOCounterfactualsNeg initialized with {self.num_negatives} negatives per sample")
        print("COCO-Counterfactuals dataset for subject binding evaluation")
    def __len__(self) -> int:
        return len(self.idx_to_dataset_idx)
    def __getitem__(self, idx: int):
        idx = int(idx)
        sample = self.dataset[self.idx_to_dataset_idx[idx]]
        pos_image = sample['image_options'][0]
        pos_text = sample['caption_options'][0]
        neg_text = sample['caption_options'][1]
        import clip
        pos_tok = clip.tokenize(pos_text, truncate=True).squeeze(0)
        neg_tok = clip.tokenize(neg_text, truncate=True).squeeze(0)
        all_neg_toks = neg_tok.unsqueeze(0)
        return pos_image, pos_tok, neg_tok, all_neg_toks
    def collate_fn(self, batch: List[tuple]) -> dict:
        images, pos_toks, neg_toks, all_neg_toks = zip(*batch)
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images, dim=0)
        pos_toks = torch.stack(pos_toks, dim=0)
        neg_toks = torch.stack(neg_toks, dim=0)
        all_neg_toks = torch.stack(all_neg_toks, dim=0)
        return {
            'images': images,
            'pos_tokens': pos_toks,
            'neg_token': neg_toks,
            'all_neg_tokens': all_neg_toks,
        }


if __name__ == "__main__":
    dataset = COCOCounterfactualsDataset(
            "./datasets/coco_counterfactuals",
            use_auth_token=os.environ.get("HF_TOKEN"),
            cache_dir='datasets/coco_counterfactuals'
        )
    print(f"Loaded dataset with {len(dataset)} examples")

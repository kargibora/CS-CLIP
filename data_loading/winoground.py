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
    from datasets import load_dataset
except ImportError:
    print("Warning: datasets library not found. Please install with: pip install datasets")
    load_dataset = None

# Optional: only used in _evaluate
try:
    from utils.align import (
        compute_caption_embeddings_intermediate_batch,
        compute_image_embeddings_intermediate_batch,
    )
except ImportError:
    print("Warning: utils.align not found. Evaluation functions may not work.")


class WinogroundDataset(Dataset):
    """
    Winoground dataset for evaluating visio-linguistic compositional reasoning.
    
    The task is to match two images with two captions correctly, where both captions
    contain the same words but in different order. This tests compositional understanding.
    
    Dataset structure:
    - Each sample has: image_0, image_1, caption_0, caption_1
    - Goal: image_0 should match caption_0, image_1 should match caption_1
    - But captions contain identical words, just reordered
    
    Returns samples in format compatible with alignment pipeline:
        {
          'image_options': [image_0, image_1],  # Both images for the pair
          'caption_options': [caption_0, caption_1],  # Both captions for the pair
          'label': 0,  # Always 0 (positive pair index)
          'pair_id': int,  # Unique identifier for this image-caption pair
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
        
        if download and load_dataset is not None:
            if verbose:
                print("[WinogroundDataset] Loading dataset from Hugging Face...")
            try:
                # Load the dataset from Hugging Face
                dataset_hf = load_dataset(
                    'facebook/winoground', 
                    cache_dir=self.cache_dir
                )
                self.examples = dataset_hf['test']  # Winoground only has test split
                if verbose:
                    print(f"[WinogroundDataset] Loaded {len(self.examples)} examples from Hugging Face")
            except Exception as e:
                if verbose:
                    print(f"[WinogroundDataset] Failed to load from Hugging Face: {e}")
                    print("[WinogroundDataset] Please provide use_auth_token or download data manually")
                raise
        else:
            # Try to load from local cache
            cache_path = os.path.join(self.cache_dir, "winoground_examples.json")
            if os.path.exists(cache_path):
                if verbose:
                    print(f"[WinogroundDataset] Loading from cache: {cache_path}")
                with open(cache_path, 'r') as f:
                    self.examples = json.load(f)
            else:
                raise FileNotFoundError(
                    f"No cached data found at {cache_path}. "
                    "Please set download=True and provide use_auth_token, "
                    "or manually download the dataset."
                )
        
        # Filter by subset if specified
        if subset_name != "all":
            original_count = len(self.examples)
            if subset_name == "tag":
                # Don't filter, keep all (for compatibility)
                pass
            else:
                # Filter by specific tag values
                self.examples = [
                    ex for ex in self.examples 
                    if ex.get('tag') == subset_name or ex.get('secondary_tag') == subset_name
                ]
            if verbose:
                print(f"[WinogroundDataset] Filtered from {original_count} to {len(self.examples)} examples for subset '{subset_name}'")
        
        # Build caption vocabulary (all unique captions)
        caption_set = set()
        for ex in self.examples:
            caption_set.add(ex['caption_0'])
            caption_set.add(ex['caption_1'])
        
        self.captions = sorted(caption_set)
        self.caption_to_idx = {cap: i for i, cap in enumerate(self.captions)}
        
        if verbose:
            print(f"[WinogroundDataset] Dataset ready with {len(self.examples)} examples")
            print(f"[WinogroundDataset] Unique captions: {len(self.captions)}")
            if len(self.examples) > 0:
                tags = [ex.get('tag', 'unknown') for ex in self.examples]
                tag_counts = {}
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        # Convert idx to regular Python int to avoid HuggingFace datasets TypeError
        idx = int(idx)
        example = self.examples[idx]
        
        # Load both images
        image_0 = example['image_0']
        image_1 = example['image_1']
        
        # Convert PIL images if needed and apply preprocessing
        if self.image_preprocess is not None:
            if isinstance(image_0, Image.Image):
                image_0 = self.image_preprocess(image_0)
            if isinstance(image_1, Image.Image):
                image_1 = self.image_preprocess(image_1)
        
        # Get captions
        caption_0 = example['caption_0']
        caption_1 = example['caption_1']
        
        return {
            "image_options": [image_0, image_1],  # Both images in the pair
            "caption_options": [caption_0, caption_1],  # Both captions in the pair
            "label": 0,  # Always 0 (first image-caption pair is the "positive")
            "pair_id": example.get('id', idx),  # Unique identifier
            "tag": example.get('tag', 'unknown'),
            "secondary_tag": example.get('secondary_tag', 'unknown'),
        }

    def get_captions(self) -> List[str]:
        """Return the unique caption vocabulary."""
        return self.captions

    def get_image_paths(self) -> List[str]:
        """Return list of image identifiers (for compatibility)."""
        # Since images are loaded from HuggingFace, return identifiers
        paths = []
        for i, ex in enumerate(self.examples):
            paths.append(f"winoground_{ex.get('id', i)}_image_0")
            paths.append(f"winoground_{ex.get('id', i)}_image_1")
        return paths

    def get_idx_to_ptr(self, idx: int) -> int:
        """Map dataset index -> caption pointer for caption_0 (positive caption)."""
        example = self.examples[idx]
        caption = example['caption_0']
        return self.caption_to_idx.get(caption, -1)

    def get_idx_to_candidates_ptr(self, idx: int) -> List[int]:
        """Map dataset index -> list of negative caption indices (caption_1)."""
        example = self.examples[idx]
        caption_1 = example['caption_1']
        ptr = self.caption_to_idx.get(caption_1)
        return [ptr] if ptr is not None else []

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
        """Public wrapper for evaluation."""
        return self._evaluate(
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device,
            batch_size=batch_size,
            indices=indices,
            intermediate_text_layer_names=intermediate_text_layer_names,
            intermediate_image_layer_names=intermediate_image_layer_names,
        )

    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader to handle Winoground samples.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and caption lists
        """
        batch_images_0 = []
        batch_images_1 = []
        batch_captions_0 = []
        batch_captions_1 = []
        batch_tags = []
        
        for sample in batch:
            # Extract components from each sample
            image_options = sample['image_options']  # [image_0, image_1]
            caption_options = sample['caption_options']  # [caption_0, caption_1]
            
            batch_images_0.append(image_options[0])
            batch_images_1.append(image_options[1])
            batch_captions_0.append(caption_options[0])
            batch_captions_1.append(caption_options[1])
            batch_tags.append(sample.get('tag', 'unknown'))
        
        # Stack images into batch tensors
        batch_images_0 = torch.stack(batch_images_0)  # [B, C, H, W]
        batch_images_1 = torch.stack(batch_images_1)  # [B, C, H, W]
        
        return {
            'images_0': batch_images_0,
            'images_1': batch_images_1,
            'captions_0': batch_captions_0,  # List[str]
            'captions_1': batch_captions_1,  # List[str]
            'tags': batch_tags               # List[str]
        }

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
        """
        Evaluate Winoground dataset with DataLoader optimization and caching.
        
        For each example, we have:
        - image_0, image_1, caption_0, caption_1
        - Correct matching: image_0 <-> caption_0, image_1 <-> caption_1
        
        We compute:
        - Text score: Does model prefer (caption_0, image_0) over (caption_0, image_1)?
        - Image score: Does model prefer (image_0, caption_0) over (image_0, caption_1)?
        - Group score: Both text and image scores are correct
        """
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
            iterate = indices
        else:
            eval_dataset = self
            iterate = list(range(len(self)))
            
        # Use DataLoader for efficient multi-threaded batch loading
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Multi-threaded loading
            pin_memory=True,  # Faster GPU transfer
            collate_fn=self._collate_fn  # Custom collate function
        )
        
        text_scores = []  # caption_0 prefers image_0 over image_1
        image_scores = []  # image_0 prefers caption_0 over caption_1
        group_scores = []  # both text and image scores correct
        
        tag_results = defaultdict(list)
        
        image_emb_list_0 = []
        image_emb_list_1 = []
        caption_emb_list_0 = []
        caption_emb_list_1 = []
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="Winoground",
            subset_name=self.subset_name,
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
        
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating Winoground")):
                batch_images_0 = batch['images_0']  # [B, C, H, W]
                batch_images_1 = batch['images_1']  # [B, C, H, W]
                batch_captions_0 = batch['captions_0']  # List[str]
                batch_captions_1 = batch['captions_1']  # List[str]
                batch_tags = batch['tags']  # List[str]
                B = len(batch_captions_0)
                
                with torch.no_grad():
                    # Get image embeddings (with caching) - use standard "image" type
                    # Combine both image sets and compute them together
                    all_images = torch.cat([batch_images_0, batch_images_1], dim=0)  # [2B, C, H, W]
                    all_img_embs = cache.get_or_compute_embeddings(
                        all_images.to(device),
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size * 2  # offset for both image sets
                    )
                    
                    # Split image embeddings back into two sets
                    img_embs_0 = all_img_embs[:B]   # [B, D]
                    img_embs_1 = all_img_embs[B:]   # [B, D]
                    
                    # Get caption embeddings (with caching) - use standard "text" type
                    # Combine both caption sets and compute them together
                    all_captions = batch_captions_0 + batch_captions_1
                    all_cap_embs = cache.get_or_compute_embeddings(
                        all_captions,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * batch_size * 2  # offset for both caption sets
                    )
                    
                    # Split caption embeddings back into two sets
                    cap_embs_0 = all_cap_embs[:B]   # [B, D]
                    cap_embs_1 = all_cap_embs[B:]   # [B, D]

                # Winoground evaluation logic:
                # Text contrastive: sim(t1,i1) > sim(t2,i1) AND sim(t2,i2) > sim(t1,i2)
                # Image contrastive: sim(i1,t1) > sim(i1,t2) AND sim(i2,t2) > sim(i2,t1)
                
                # Text contrastive accuracy: both captions prefer their corresponding images
                text_sim_t0_i0 = (cap_embs_0 * img_embs_0).sum(dim=1)  # sim(t0,i0)
                text_sim_t0_i1 = (cap_embs_0 * img_embs_1).sum(dim=1)  # sim(t0,i1)
                text_sim_t1_i0 = (cap_embs_1 * img_embs_0).sum(dim=1)  # sim(t1,i0)
                text_sim_t1_i1 = (cap_embs_1 * img_embs_1).sum(dim=1)  # sim(t1,i1)
                
                # Text score: t0 prefers i0 over i1 AND t1 prefers i1 over i0
                text_cond_1 = text_sim_t0_i0 > text_sim_t0_i1  # caption_0 prefers image_0
                text_cond_2 = text_sim_t1_i1 > text_sim_t1_i0  # caption_1 prefers image_1
                batch_image_scores = text_cond_1 & text_cond_2
                
                # Image contrastive accuracy: both images prefer their corresponding captions
                image_sim_i0_t0 = (img_embs_0 * cap_embs_0).sum(dim=1)  # sim(i0,t0)
                image_sim_i0_t1 = (img_embs_0 * cap_embs_1).sum(dim=1)  # sim(i0,t1)
                image_sim_i1_t0 = (img_embs_1 * cap_embs_0).sum(dim=1)  # sim(i1,t0)
                image_sim_i1_t1 = (img_embs_1 * cap_embs_1).sum(dim=1)  # sim(i1,t1)
                
                # Image score: i0 prefers t0 over t1 AND i1 prefers t1 over t0
                image_cond_1 = image_sim_i0_t0 > image_sim_i0_t1  # image_0 prefers caption_0
                image_cond_2 = image_sim_i1_t1 > image_sim_i1_t0  # image_1 prefers caption_1
                batch_text_scores = image_cond_1 & image_cond_2
                
                # Group scores: both text and image scores correct
                batch_group_scores = batch_text_scores & batch_image_scores
                
                # Store results
                text_scores.extend(batch_text_scores.cpu().tolist())
                image_scores.extend(batch_image_scores.cpu().tolist())
                group_scores.extend(batch_group_scores.cpu().tolist())
                
                # Store tag-specific results
                for i, tag in enumerate(batch_tags):
                    tag_results[tag].append({
                        'text': batch_text_scores[i].item(),
                        'image': batch_image_scores[i].item(),
                        'group': batch_group_scores[i].item()
                    })
                
                # Store embeddings
                image_emb_list_0.append(img_embs_0.cpu())
                image_emb_list_1.append(img_embs_1.cpu())
                caption_emb_list_0.append(cap_embs_0.cpu())
                caption_emb_list_1.append(cap_embs_1.cpu())

        # Compute overall accuracies
        text_acc = float(np.mean(text_scores)) if text_scores else 0.0
        image_acc = float(np.mean(image_scores)) if image_scores else 0.0
        group_acc = float(np.mean(group_scores)) if group_scores else 0.0
        
        # Compute tag-wise accuracies
        tag_accuracies = {}
        tag_counts = {}
        for tag, scores_list in tag_results.items():
            text_scores_tag = [s['text'] for s in scores_list]
            image_scores_tag = [s['image'] for s in scores_list]
            group_scores_tag = [s['group'] for s in scores_list]
            
            tag_accuracies[tag] = {
                'text_contrastive_accuracy': float(np.mean(text_scores_tag)),
                'image_contrastive_accuracy': float(np.mean(image_scores_tag)),
                'group_contrastive_accuracy': float(np.mean(group_scores_tag))
            }
            tag_counts[tag] = len(scores_list)
        
        # Prepare results
        results = {
            "text_contrastive_accuracy": text_acc,
            "image_contrastive_accuracy": image_acc,
            "group_contrastive_accuracy": group_acc,
        }

        # Macro average across tags
        if tag_accuracies:
            results['macro_contrastive_accuracy'] = float(np.mean(
                [v['group_contrastive_accuracy'] for v in tag_accuracies.values()]
            ))
        else:
            results['macro_contrastive_accuracy'] = group_acc

        # # Add tag-specific results
        # for tag, accs in tag_accuracies.items():
        #     results[f'text_contrastive_accuracy_{tag}'] = accs['text_contrastive_accuracy']
        #     results[f'image_contrastive_accuracy_{tag}'] = accs['image_contrastive_accuracy']
        #     results[f'group_contrastive_accuracy_{tag}'] = accs['group_contrastive_accuracy']
        #     results[f'count_{tag}'] = tag_counts[tag]

        # Prepare embeddings
        embeddings = {
            "image_0_embeddings": torch.cat(image_emb_list_0, dim=0).numpy(),
            "image_1_embeddings": torch.cat(image_emb_list_1, dim=0).numpy(),
            "caption_0_embeddings": torch.cat(caption_emb_list_0, dim=0).numpy(),
            "caption_1_embeddings": torch.cat(caption_emb_list_1, dim=0).numpy(),
        }
    
        return results, embeddings
        
    def split_dataset(
        self,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
        split_type: Literal["random", "object"] = "random",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split dataset into train/val/test.
        
        For Winoground, we typically use the full dataset for evaluation,
        but this method provides compatibility with training pipelines.
        """
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
            # For Winoground, each example is unique, so object split = random split
            return self.split_dataset(val_ratio, test_ratio, seed, "random")
        
        else:
            raise ValueError(f"Unknown split_type={split_type}")

    def __getstate__(self):
        """Ensure the dataset can be pickled for multiprocessing compatibility."""
        state = self.__dict__.copy()
        # Winoground dataset should be safe to pickle as is, but just in case
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)


class WinogroundNeg(Dataset):
    """
    Wrapper for WinogroundDataset to provide tokenized positives and negatives.
    Compatible with other *Neg dataset classes.
    
    For Winoground, the "negative" is the other caption in the pair.
    """
    
    def __init__(
        self,
        winoground_dataset: WinogroundDataset,
        indices: List[int],
        num_negatives: int = 1,
    ):
        super().__init__()
        self.dataset = winoground_dataset
        self.num_negatives = num_negatives
        self.idx_to_dataset_idx = {i: idx for i, idx in enumerate(indices)}
        
        print(f"WinogroundNeg initialized with {self.num_negatives} negatives per sample")
        print("Winoground dataset for compositional visio-linguistic reasoning")

    def __len__(self) -> int:
        return len(self.idx_to_dataset_idx)
    
    def __getitem__(self, idx: int):
        # Convert idx to regular Python int to avoid HuggingFace datasets TypeError
        idx = int(idx)
        sample = self.dataset[self.idx_to_dataset_idx[idx]]
        
        # For Winoground, we use image_0 as positive image and caption_0 as positive caption
        pos_image = sample['image_options'][0]  # image_0
        pos_text = sample['caption_options'][0]  # caption_0
        neg_text = sample['caption_options'][1]  # caption_1 is the negative
        
        # Tokenize using CLIP
        import clip
        pos_tok = clip.tokenize(pos_text, truncate=True).squeeze(0)
        neg_tok = clip.tokenize(neg_text, truncate=True).squeeze(0)
        
        # For compatibility, create all_neg_tokens
        all_neg_toks = neg_tok.unsqueeze(0)  # Shape: (1, seq_len)
        
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
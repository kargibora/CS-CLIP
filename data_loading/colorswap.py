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


class ColorSwapDataset(Dataset):
    """
    ColorSwap dataset for evaluating multimodal color and word order understanding.
    
    The task is to match two images with two captions correctly, where both captions
    contain the same words but the color words are swapped between different objects.
    This tests color-object binding understanding.
    
    Dataset structure:
    - Each sample has: image_1, image_2, caption_1, caption_2
    - Goal: image_1 should match caption_1, image_2 should match caption_2
    - But captions contain identical words, just color assignments swapped
    
    Returns samples in format compatible with alignment pipeline:
        {
          'image_options': [image_1, image_2],  # Both images for the pair
          'caption_options': [caption_1, caption_2],  # Both captions for the pair
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
                print("[ColorSwapDataset] Loading dataset from Hugging Face...")
            try:
                # Load the dataset from Hugging Face
                load_kwargs = {
                    'cache_dir': self.cache_dir,
                }
                
                # Try multiple token sources
                token = use_auth_token
                if not token:
                    # Try environment variable for ColorSwap specifically
                    token = os.environ.get('HF_COLORSWAP_TOKEN')
                if not token:
                    # Try general HF token
                    token = os.environ.get('HF_TOKEN')
                if not token:
                    # Try legacy HF token name
                    token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
                
                if token:
                    load_kwargs['token'] = token
                    if verbose:
                        print("[ColorSwapDataset] Using authentication token")
                
                dataset_hf = load_dataset(
                    'stanfordnlp/colorswap',
                    **load_kwargs
                )
                # ColorSwap has a 'test' split
                self.examples = dataset_hf['test'] if 'test' in dataset_hf else dataset_hf['train']
                if verbose:
                    print(f"[ColorSwapDataset] Loaded {len(self.examples)} examples from Hugging Face")
            except Exception as e:
                if verbose:
                    print(f"[ColorSwapDataset] Failed to load from Hugging Face: {e}")
                    print("[ColorSwapDataset] This dataset requires authentication or permission.")
                    print("[ColorSwapDataset] Try:")
                    print("  1. Request access at https://huggingface.co/datasets/stanfordnlp/colorswap")
                    print("  2. Use huggingface-cli login to authenticate")
                    print("  3. Or provide use_auth_token parameter")
                raise
        else:
            # Try to load from local cache
            cache_path = os.path.join(self.cache_dir, "colorswap_examples.json")
            if os.path.exists(cache_path):
                if verbose:
                    print(f"[ColorSwapDataset] Loading from cache: {cache_path}")
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
            if subset_name == "midjourney":
                # Filter by image source
                self.examples = [
                    ex for ex in self.examples 
                    if ex.get('image_source') == 'midjourney'
                ]
            elif subset_name == "human":
                # Filter by caption source
                self.examples = [
                    ex for ex in self.examples 
                    if ex.get('caption_source') == 'human'
                ]
            else:
                # Keep all for unknown subset names
                pass
            if verbose:
                print(f"[ColorSwapDataset] Filtered from {original_count} to {len(self.examples)} examples for subset '{subset_name}'")
        
        # Build caption vocabulary (all unique captions)
        caption_set = set()
        for ex in self.examples:
            caption_set.add(ex['caption_1'])
            caption_set.add(ex['caption_2'])
        
        self.captions = sorted(caption_set)
        self.caption_to_idx = {cap: i for i, cap in enumerate(self.captions)}
        
        if verbose:
            print(f"[ColorSwapDataset] Dataset ready with {len(self.examples)} examples")
            print(f"[ColorSwapDataset] Unique captions: {len(self.captions)}")
            if len(self.examples) > 0:
                # Show dataset statistics
                image_sources = [ex.get('image_source', 'unknown') for ex in self.examples]
                caption_sources = [ex.get('caption_source', 'unknown') for ex in self.examples]
                
                img_source_counts = {}
                cap_source_counts = {}
                
                for source in image_sources:
                    img_source_counts[source] = img_source_counts.get(source, 0) + 1
                for source in caption_sources:
                    cap_source_counts[source] = cap_source_counts.get(source, 0) + 1
                
                if verbose:
                    print(f"[ColorSwapDataset] Image sources: {img_source_counts}")
                    print(f"[ColorSwapDataset] Caption sources: {cap_source_counts}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        # Convert idx to regular Python int to avoid HuggingFace datasets TypeError
        idx = int(idx)
        example = self.examples[idx]
        
        # Load both images
        image_1 = example['image_1']
        image_2 = example['image_2']
        
        # Convert PIL images if needed and apply preprocessing
        if self.image_preprocess is not None:
            if isinstance(image_1, Image.Image):
                image_1 = self.image_preprocess(image_1)
            if isinstance(image_2, Image.Image):
                image_2 = self.image_preprocess(image_2)
        
        # Get captions
        caption_1 = example['caption_1']
        caption_2 = example['caption_2']
        
        return {
            "image_options": [image_1, image_2],  # Both images in the pair
            "caption_options": [caption_1, caption_2],  # Both captions in the pair
            "label": 0,  # Always 0 (first image-caption pair is the "positive")
            "pair_id": example.get('id', idx),  # Unique identifier
            "image_source": example.get('image_source', 'unknown'),
            "caption_source": example.get('caption_source', 'unknown'),
        }

    def get_captions(self) -> List[str]:
        """Return the unique caption vocabulary."""
        return self.captions

    def get_image_paths(self) -> List[str]:
        """Return list of image identifiers (for compatibility)."""
        # Since images are loaded from HuggingFace, return identifiers
        paths = []
        for i, ex in enumerate(self.examples):
            paths.append(f"colorswap_{ex.get('id', i)}_image_1")
            paths.append(f"colorswap_{ex.get('id', i)}_image_2")
        return paths

    def get_idx_to_ptr(self, idx: int) -> int:
        """Map dataset index -> caption pointer for caption_1 (positive caption)."""
        example = self.examples[idx]
        caption = example['caption_1']
        return self.caption_to_idx.get(caption, -1)

    def get_idx_to_candidates_ptr(self, idx: int) -> List[int]:
        """Map dataset index -> list of negative caption indices (caption_2)."""
        example = self.examples[idx]
        caption_2 = example['caption_2']
        ptr = self.caption_to_idx.get(caption_2)
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
        Custom collate function for DataLoader to handle ColorSwap samples.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and caption lists
        """
        batch_images_1 = []
        batch_images_2 = []
        batch_captions_1 = []
        batch_captions_2 = []
        batch_sources = []
        
        for sample in batch:
            # Extract components from each sample
            image_options = sample['image_options']  # [image_1, image_2]
            caption_options = sample['caption_options']  # [caption_1, caption_2]
            
            batch_images_1.append(image_options[0])
            batch_images_2.append(image_options[1])
            batch_captions_1.append(caption_options[0])
            batch_captions_2.append(caption_options[1])
            batch_sources.append(sample.get('image_source', 'unknown'))
        
        # Stack images into batch tensors
        batch_images_1 = torch.stack(batch_images_1)  # [B, C, H, W]
        batch_images_2 = torch.stack(batch_images_2)  # [B, C, H, W]
        
        return {
            'images_1': batch_images_1,
            'images_2': batch_images_2,
            'captions_1': batch_captions_1,  # List[str]
            'captions_2': batch_captions_2,  # List[str]
            'sources': batch_sources         # List[str]
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
        Evaluate ColorSwap dataset with DataLoader optimization and caching.
        
        For each example, we have:
        - image_1, image_2, caption_1, caption_2
        - Correct matching: image_1 <-> caption_1, image_2 <-> caption_2
        
        We compute:
        - Text score: Does model prefer (caption_1, image_1) over (caption_1, image_2)?
        - Image score: Does model prefer (image_1, caption_1) over (image_1, caption_2)?
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
        
        text_scores = []  # caption_1 prefers image_1 over image_2
        image_scores = []  # image_1 prefers caption_1 over caption_2
        group_scores = []  # both text and image scores correct
        
        source_results = defaultdict(list)
        
        image_emb_list_1 = []
        image_emb_list_2 = []
        caption_emb_list_1 = []
        caption_emb_list_2 = []
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="ColorSwap",
            subset_name=self.subset_name,
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
        
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating ColorSwap")):
                batch_images_1 = batch['images_1']  # [B, C, H, W]
                batch_images_2 = batch['images_2']  # [B, C, H, W]
                batch_captions_1 = batch['captions_1']  # List[str]
                batch_captions_2 = batch['captions_2']  # List[str]
                batch_sources = batch['sources']  # List[str]
                # Batch size
                
                with torch.no_grad():
                    # Use the EmbeddingCache helper to compute pairwise similarities.
                    (text_correct, text_incorrect, image_correct, image_incorrect,
                     img_embs_1, img_embs_2, cap_embs_1, cap_embs_2) = cache.compute_tqa_colorswap_similarity(
                        batch_images_1.to(device),
                        batch_images_2.to(device),
                        batch_captions_1,
                        batch_captions_2,
                        compute_image_embeddings_intermediate_batch,
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        intermediate_text_layer_names,
                    )

                    # The similarity tensors (text_correct, ...) may be on the model device;
                    # move them to CPU for consistent downstream handling.
                    text_correct = text_correct.cpu()
                    text_incorrect = text_incorrect.cpu()
                    image_correct = image_correct.cpu()
                    image_incorrect = image_incorrect.cpu()

                # Use the similarity scores returned by the cache helper
                batch_text_scores = (text_correct > text_incorrect)
                batch_image_scores = (image_correct > image_incorrect)
                
                # Group scores: both text and image scores correct
                batch_group_scores = batch_text_scores & batch_image_scores
                
                # Store results
                text_scores.extend(batch_text_scores.cpu().tolist())
                image_scores.extend(batch_image_scores.cpu().tolist())
                group_scores.extend(batch_group_scores.cpu().tolist())
                
                # Store source-specific results
                for i, source in enumerate(batch_sources):
                    source_results[source].append({
                        'text': batch_text_scores[i].item(),
                        'image': batch_image_scores[i].item(),
                        'group': batch_group_scores[i].item()
                    })
                
                # Store embeddings
                image_emb_list_1.append(img_embs_1.cpu())
                image_emb_list_2.append(img_embs_2.cpu())
                caption_emb_list_1.append(cap_embs_1.cpu())
                caption_emb_list_2.append(cap_embs_2.cpu())

        # Compute overall accuracies
        text_acc = float(np.mean(text_scores)) if text_scores else 0.0
        image_acc = float(np.mean(image_scores)) if image_scores else 0.0
        group_acc = float(np.mean(group_scores)) if group_scores else 0.0
        
        # Compute source-wise accuracies
        source_accuracies = {}
        source_counts = {}
        for source, scores_list in source_results.items():
            text_scores_src = [s['text'] for s in scores_list]
            image_scores_src = [s['image'] for s in scores_list]
            group_scores_src = [s['group'] for s in scores_list]
            
            source_accuracies[source] = {
                'text_contrastive_accuracy': float(np.mean(text_scores_src)),
                'image_contrastive_accuracy': float(np.mean(image_scores_src)),
                'group_contrastive_accuracy': float(np.mean(group_scores_src))
            }
            source_counts[source] = len(scores_list)
        
        # Prepare results
        results = {
            "text_contrastive_accuracy": text_acc,
            "image_contrastive_accuracy": image_acc,
            "group_contrastive_accuracy": group_acc,
        }

        # Macro average across sources
        if source_accuracies:
            results['macro_contrastive_accuracy'] = float(np.mean(
                [v['group_contrastive_accuracy'] for v in source_accuracies.values()]
            ))
        else:
            results['macro_contrastive_accuracy'] = group_acc

        # Add source-specific results
        for source, accs in source_accuracies.items():
            results[f'text_contrastive_accuracy_{source}'] = accs['text_contrastive_accuracy']
            results[f'image_contrastive_accuracy_{source}'] = accs['image_contrastive_accuracy']
            results[f'group_contrastive_accuracy_{source}'] = accs['group_contrastive_accuracy']
            results[f'count_{source}'] = source_counts[source]

        # Handle empty embedding lists safely
        embeddings = {}
        if image_emb_list_1:
            embeddings["image_embeddings"] = torch.cat(image_emb_list_1, dim=0).numpy()
        if image_emb_list_2:
            embeddings["image_2_embeddings"] = torch.cat(image_emb_list_2, dim=0).numpy()
        if caption_emb_list_1:
            embeddings["caption_embeddings"] = torch.cat(caption_emb_list_1, dim=0).numpy()
        if caption_emb_list_2:
            embeddings["caption_2_embeddings"] = torch.cat(caption_emb_list_2, dim=0).numpy()
        
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
        
        For ColorSwap, we typically use the full dataset for evaluation,
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
            # For ColorSwap, each example is unique, so object split = random split
            return self.split_dataset(val_ratio, test_ratio, seed, "random")
        
        else:
            raise ValueError(f"Unknown split_type={split_type}")

    def __getstate__(self):
        """Ensure the dataset can be pickled for multiprocessing compatibility."""
        state = self.__dict__.copy()
        # ColorSwap dataset should be safe to pickle as is, but just in case
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)


class ColorSwapNeg(Dataset):
    """
    Wrapper for ColorSwapDataset to provide tokenized positives and negatives.
    Compatible with other *Neg dataset classes.
    
    For ColorSwap, the "negative" is the other caption in the pair (color-swapped version).
    """
    
    def __init__(
        self,
        colorswap_dataset: ColorSwapDataset,
        indices: List[int],
        num_negatives: int = 1,
    ):
        super().__init__()
        self.dataset = colorswap_dataset
        self.num_negatives = num_negatives
        self.idx_to_dataset_idx = {i: idx for i, idx in enumerate(indices)}
        
        print(f"ColorSwapNeg initialized with {self.num_negatives} negatives per sample")
        print("ColorSwap dataset for color and word order evaluation")

    def __len__(self) -> int:
        return len(self.idx_to_dataset_idx)
    
    def __getitem__(self, idx: int):
        # Convert idx to regular Python int to avoid HuggingFace datasets TypeError
        idx = int(idx)
        sample = self.dataset[self.idx_to_dataset_idx[idx]]
        
        # For ColorSwap, we use image_1 as positive image and caption_1 as positive caption
        pos_image = sample['image_options'][0]  # image_1
        pos_text = sample['caption_options'][0]  # caption_1
        neg_text = sample['caption_options'][1]  # caption_2 is the color-swapped negative
        
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

if __name__ == "__main__":
    # Simple test to verify dataset loading
    from torchvision import transforms

    simple_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    ds = ColorSwapDataset("./datasets",
                          subset_name="all", 
                          image_preprocess=simple_preprocess, 
                          download=True, 
                          use_auth_token=os.environ.get("HF_TOKEN"),
                          verbose=True)
    print(f"Loaded {len(ds)} samples.")
    sample = ds[0]
    print("Sample keys:", sample.keys())
    img1 = sample["image_options"][0]
    img2 = sample["image_options"][1]
    assert isinstance(img1, torch.Tensor)
    assert isinstance(img2, torch.Tensor)
    print("Image 1 shape:", img1.shape)
    print("Image 2 shape:", img2.shape)
    print("Caption 1:", sample["caption_options"][0])
    print("Caption 2:", sample["caption_options"][1])

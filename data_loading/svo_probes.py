"""
SVO Probes Dataset for Verb Understanding

Based on: https://github.com/google-deepmind/svo_probes

The SVO-Probes dataset tests understanding of verbs in vision-language models.
Each sample contains:
- A sentence with a Subject-Verb-Object triplet (e.g., "Girl is standing in the grass")
- A positive image matching the sentence (girl, stand, grass)
- A negative image differing in subject, verb, or object (dog, stand, grass)

The task is to match the sentence with the correct (positive) image over the foil (negative).
"""

import csv
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

# Optional: only used in _evaluate
try:
    from utils.align import (
        compute_caption_embeddings_intermediate_batch,
        compute_image_embeddings_intermediate_batch,
    )
except ImportError:
    print("Warning: utils.align not found. Evaluation functions may not work.")


class SVOProbesDataset(Dataset):
    """
    SVO-Probes dataset for evaluating verb understanding in vision-language models.
    
    Based on: https://github.com/google-deepmind/svo_probes
    
    The task tests whether models can distinguish between correct and foiled triplets
    where the negative differs in subject, verb, or object.
    
    Each sample contains:
    - sentence: The caption describing an image
    - pos_triplet: Subject-Verb-Object triplet for positive image (e.g., "girl,stand,grass")
    - neg_triplet: Triplet for negative image (e.g., "dog,stand,grass")
    - pos_image: Image matching the sentence
    - neg_image: Foil image differing in one component
    - negative_type: One of ['subj_neg', 'verb_neg', 'obj_neg']
    
    Returns samples in format compatible with alignment pipeline:
        {
          'image_options': [pos_image, neg_image],
          'caption_options': [sentence],  # Single caption
          'label': 0,  # Index of correct image (always 0 = positive)
          'image_id': str,
          'negative_type': str,  # Which component is different in negative
        }
    """

    def __init__(
        self,
        data_root: str,
        subset_name: str = "all",
        image_preprocess=None,
        cache_dir: Optional[str] = None,
        download: bool = True,
        verbose: bool = True,
        **kwargs
    ):
        self.data_root = data_root
        self.subset_name = subset_name
        self.image_preprocess = image_preprocess
        self.cache_dir = cache_dir or os.path.join(data_root, "cache")
        self.verbose = verbose
        
        # SVO Probes data directory
        self.svo_dir = data_root
        self.images_dir = os.path.join(self.svo_dir, "images")
        self.csv_file = os.path.join(self.svo_dir, "svo_probes.csv")
        
        # Ensure directories exist
        os.makedirs(self.svo_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Download dataset if needed
        if not os.path.exists(self.csv_file):
            if download:
                self._download_dataset()
            else:
                raise FileNotFoundError(
                    f"SVO Probes CSV not found at {self.csv_file}. "
                    f"Please set download=True to download the dataset."
                )
        
        # Load the dataset
        self._load_data()
        
        # Pre-download images to speed up evaluation
        if len(self.examples) > 0:
            self._predownload_images()
        
        if verbose:
            print(f"[SVOProbesDataset] Dataset ready with {len(self.examples)} examples")

    def _download_dataset(self):
        """Download SVO Probes dataset from GitHub."""
        if self.verbose:
            print("[SVOProbesDataset] Downloading SVO Probes CSV from GitHub...")
        
        import requests
        
        # Download CSV file
        csv_url = "https://raw.githubusercontent.com/google-deepmind/svo_probes/main/svo_probes.csv"
        
        try:
            response = requests.get(csv_url, timeout=30)
            response.raise_for_status()
            
            with open(self.csv_file, 'wb') as f:
                f.write(response.content)
            
            if self.verbose:
                print(f"[SVOProbesDataset] Downloaded CSV to {self.csv_file}")
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to download SVO Probes CSV from {csv_url}: {e}\n"
                "Please download manually from https://github.com/google-deepmind/svo_probes"
            )

    def _load_data(self):
        """Load SVO Probes data from CSV."""
        if self.verbose:
            print(f"[SVOProbesDataset] Loading data from {self.csv_file}")
        
        self.examples = []
        
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Parse negative type
                negative_type = None
                if row.get('subj_neg', '').lower() == 'true':
                    negative_type = 'subj_neg'
                elif row.get('verb_neg', '').lower() == 'true':
                    negative_type = 'verb_neg'
                elif row.get('obj_neg', '').lower() == 'true':
                    negative_type = 'obj_neg'
                
                example = {
                    'sentence': row['sentence'].strip(),
                    'pos_triplet': row['pos_triplet'].strip(),
                    'neg_triplet': row['neg_triplet'].strip(),
                    'pos_url': row['pos_url'].strip(),
                    'neg_url': row['neg_url'].strip(),
                    'pos_image_id': row['pos_image_id'].strip(),
                    'neg_image_id': row['neg_image_id'].strip(),
                    'negative_type': negative_type,
                }
                
                self.examples.append(example)
        
        if self.verbose:
            print(f"[SVOProbesDataset] Loaded {len(self.examples)} examples")
        
        # Filter by subset if specified
        if self.subset_name != "all":
            original_count = len(self.examples)
            
            if self.subset_name in ['subj_neg', 'verb_neg', 'obj_neg']:
                # Filter by negative type
                self.examples = [
                    ex for ex in self.examples 
                    if ex['negative_type'] == self.subset_name
                ]
            
            if self.verbose:
                print(f"[SVOProbesDataset] Filtered from {original_count} to {len(self.examples)} examples for subset '{self.subset_name}'")
        
        # Build caption vocabulary (all unique sentences)
        caption_set = set()
        for ex in self.examples:
            caption_set.add(ex['sentence'])
        
        self.captions = sorted(caption_set)
        self.caption_to_idx = {cap: i for i, cap in enumerate(self.captions)}
        
        # Print statistics
        if self.verbose:
            neg_types = defaultdict(int)
            for ex in self.examples:
                neg_types[ex['negative_type']] += 1
            
            print("[SVOProbesDataset] Negative type distribution:")
            for neg_type, count in sorted(neg_types.items()):
                print(f"  - {neg_type}: {count} ({100 * count / len(self.examples):.1f}%)")

    def __len__(self) -> int:
        return len(self.examples)

    def _download_image(self, image_url: str, image_path: str) -> bool:
        """Download image from URL if it doesn't exist."""
        if os.path.exists(image_path):
            return True
        
        try:
            import requests
            response = requests.get(image_url, timeout=15)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                return True
        except Exception as e:
            if self.verbose:
                print(f"[SVOProbesDataset] Failed to download {image_url}: {e}")
        
        return False

    def _predownload_images(self, max_workers: int = 4):
        """Pre-download all images to avoid slow evaluation."""
        import concurrent.futures
        import requests
        
        missing_images = []
        for example in self.examples:
            # Check positive image
            pos_image_path = os.path.join(self.images_dir, f"pos_{example['pos_image_id']}.jpg")
            if not os.path.exists(pos_image_path):
                missing_images.append((example['pos_url'], pos_image_path))
            
            # Check negative image
            neg_image_path = os.path.join(self.images_dir, f"neg_{example['neg_image_id']}.jpg")
            if not os.path.exists(neg_image_path):
                missing_images.append((example['neg_url'], neg_image_path))
        
        if not missing_images:
            if self.verbose:
                print("[SVOProbesDataset] All images already downloaded")
            return
        
        if self.verbose:
            print(f"[SVOProbesDataset] Pre-downloading {len(missing_images)} images...")
        
        def download_single(url_path_pair):
            url, path = url_path_pair
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    with open(path, 'wb') as f:
                        f.write(response.content)
                    return True
            except Exception:
                pass
            return False
        
        # Download in parallel
        success_count = 0
        failed_images = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_single, pair): pair for pair in missing_images}
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(missing_images), 
                             desc="Downloading images",
                             disable=not self.verbose):
                if future.result():
                    success_count += 1
                else:
                    failed_images.append(futures[future])
        
        if self.verbose:
            print(f"[SVOProbesDataset] Successfully downloaded {success_count}/{len(missing_images)} images")
            if failed_images:
                print(f"[SVOProbesDataset] WARNING: Failed to download {len(failed_images)} images")
                print(f"[SVOProbesDataset] First few failed URLs: {[url for url, _ in failed_images[:5]]}")
                print("[SVOProbesDataset] These samples may cause errors during evaluation")

    def __getitem__(self, idx: int) -> Dict:
        idx = int(idx)
        example = self.examples[idx]
        
        # Determine image paths
        pos_image_path = os.path.join(self.images_dir, f"pos_{example['pos_image_id']}.jpg")
        neg_image_path = os.path.join(self.images_dir, f"neg_{example['neg_image_id']}.jpg")
        
        # Load and preprocess images directly (they should already be downloaded)
        # Removed file existence checks to speed up data loading
        try:
            pos_image = Image.open(pos_image_path).convert("RGB")
            neg_image = Image.open(neg_image_path).convert("RGB")
            
            if self.image_preprocess is not None:
                pos_image = self.image_preprocess(pos_image)
                neg_image = self.image_preprocess(neg_image)
                
        except FileNotFoundError:
            # Only if file is actually missing, try to download once
            print(f"[SVOProbesDataset] Missing image, attempting download for sample {idx}")
            if not os.path.exists(pos_image_path):
                self._download_image(example['pos_url'], pos_image_path)
            if not os.path.exists(neg_image_path):
                self._download_image(example['neg_url'], neg_image_path)
            
            # Try loading again
            try:
                pos_image = Image.open(pos_image_path).convert("RGB")
                neg_image = Image.open(neg_image_path).convert("RGB")
                if self.image_preprocess is not None:
                    pos_image = self.image_preprocess(pos_image)
                    neg_image = self.image_preprocess(neg_image)
            except Exception as retry_error:
                print(f"[SVOProbesDataset] Failed to load sample {idx} after retry: {retry_error}")
                # Return first valid sample as fallback
                if idx > 0:
                    return self.__getitem__(0)
                raise
                
        except Exception as e:
            print(f"[SVOProbesDataset] Error loading images for idx {idx}: {e}")
            # Try to return a different sample
            if idx > 0:
                return self.__getitem__((idx + 1) % len(self))
            raise
        
        return {
            "image_options": [pos_image, neg_image],  # [correct, foil]
            "caption_options": [example['sentence']],  # Single caption
            "label": 0,  # Index of correct image (always first = positive)
            "image_id": f"svo_{idx}",
            "pos_triplet": example['pos_triplet'],
            "neg_triplet": example['neg_triplet'],
            "negative_type": example['negative_type'],
        }

    def get_captions(self) -> List[str]:
        """Return the unique caption vocabulary."""
        return self.captions

    def get_image_paths(self) -> List[str]:
        """Return list of image paths."""
        paths = []
        for ex in self.examples:
            paths.append(os.path.join(self.images_dir, f"pos_{ex['pos_image_id']}.jpg"))
            paths.append(os.path.join(self.images_dir, f"neg_{ex['neg_image_id']}.jpg"))
        return paths

    def get_idx_to_ptr(self, idx: int) -> int:
        """Map dataset index -> caption pointer for sentence."""
        example = self.examples[idx]
        sentence = example['sentence']
        return self.caption_to_idx.get(sentence, -1)

    def get_idx_to_candidates_ptr(self, idx: int) -> List[int]:
        """Map dataset index -> list of candidate caption indices (same sentence)."""
        example = self.examples[idx]
        sentence = example['sentence']
        ptr = self.caption_to_idx.get(sentence)
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
        Custom collate function for DataLoader to handle SVO Probes samples.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and caption lists
        """
        batch_pos_images = []
        batch_neg_images = []
        batch_sentences = []
        batch_neg_types = []
        
        for sample in batch:
            # Extract components from each sample
            image_options = sample['image_options']  # [pos_image, neg_image]
            sentence = sample['caption_options'][0]  # Single sentence
            
            batch_pos_images.append(image_options[0])
            batch_neg_images.append(image_options[1])
            batch_sentences.append(sentence)
            batch_neg_types.append(sample['negative_type'])
        
        # Stack images into batch tensors
        batch_pos_images = torch.stack(batch_pos_images)  # [B, C, H, W]
        batch_neg_images = torch.stack(batch_neg_images)  # [B, C, H, W]
        
        return {
            'pos_images': batch_pos_images,
            'neg_images': batch_neg_images,
            'sentences': batch_sentences,  # List[str]
            'negative_types': batch_neg_types,  # List[str]
        }

    def _evaluate(
        self,
        embedding_model,
        aligning_model=None,
        device="cuda",
        batch_size: int = 64,  # Increased from 16 to 64 for faster evaluation
        indices: Optional[List[int]] = None,
        intermediate_text_layer_names: List[str] = ["final"],
        intermediate_image_layer_names: List[str] = ["final"],
    ):
        """
        Evaluate SVO Probes dataset with DataLoader optimization and caching.
        
        For each example, we test whether the model assigns higher similarity
        to the positive image vs. the negative image for the given sentence.
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
            
        # Use DataLoader for efficient batch loading
        # Using num_workers=2 for faster data loading (reduced from 4 to minimize issues)
        # Images should already be pre-downloaded during __init__
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,  # Use 2 workers for faster loading (compromise between speed and reliability)
            pin_memory=True,  # Faster GPU transfer
            collate_fn=self._collate_fn,  # Custom collate function
            prefetch_factor=2,  # Prefetch 2 batches per worker
            persistent_workers=True  # Keep workers alive between batches
        )
        
        correct_scores = []  # Whether positive image preferred over negative
        all_similarities = []
        
        # Track by negative type
        neg_type_results = defaultdict(list)
        
        pos_image_emb_list = []
        neg_image_emb_list = []
        sentence_emb_list = []
        
        # Use embedding cache context manager with proper cache directory
        with EmbeddingCache(
            dataset_name="SVOProbes",
            subset_name=self.subset_name,
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device,
            cache_dir=self.cache_dir  # Use dataset's cache directory for persistence
        ) as cache:
        
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating SVO Probes")):
                batch_pos_images = batch['pos_images']  # [B, C, H, W]
                batch_neg_images = batch['neg_images']  # [B, C, H, W]
                batch_sentences = batch['sentences']  # List[str]
                batch_neg_types = batch['negative_types']  # List[str]
                B = len(batch_sentences)
                
                with torch.no_grad():
                    # Get image embeddings (with caching)
                    # Combine both image sets and compute them together
                    all_images = torch.cat([batch_pos_images, batch_neg_images], dim=0)  # [2B, C, H, W]
                    all_img_embs = cache.get_or_compute_embeddings(
                        all_images.to(device),
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size * 2  # offset for both image sets
                    )
                    
                    # Split image embeddings back into two sets
                    pos_img_embs = all_img_embs[:B]   # [B, D]
                    neg_img_embs = all_img_embs[B:]   # [B, D]
                    
                    # Get sentence embeddings (with caching)
                    sent_embs = cache.get_or_compute_embeddings(
                        batch_sentences,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * batch_size
                    )

                # Compute similarities
                pos_sims = (sent_embs * pos_img_embs).sum(dim=1)  # similarity with positive image
                neg_sims = (sent_embs * neg_img_embs).sum(dim=1)  # similarity with negative image
                
                # Check if positive > negative for each sample
                batch_correct = pos_sims > neg_sims
                correct_scores.extend(batch_correct.cpu().tolist())
                
                # Store detailed similarity information
                for i in range(B):
                    all_similarities.append({
                        'pos_similarity': pos_sims[i].item(),
                        'neg_similarity': neg_sims[i].item(),
                        'difference': (pos_sims[i] - neg_sims[i]).item(),
                        'negative_type': batch_neg_types[i],
                    })
                    
                    # Track by negative type
                    neg_type_results[batch_neg_types[i]].append(batch_correct[i].item())
                
                # Store embeddings
                pos_image_emb_list.append(pos_img_embs.cpu())
                neg_image_emb_list.append(neg_img_embs.cpu())
                sentence_emb_list.append(sent_embs.cpu())

        # Compute overall accuracy
        accuracy = float(np.mean(correct_scores)) if correct_scores else 0.0
        
        # Compute negative-type-specific accuracies
        neg_type_accuracies = {}
        neg_type_counts = {}
        for neg_type, scores in neg_type_results.items():
            neg_type_accuracies[neg_type] = float(np.mean(scores))
            neg_type_counts[neg_type] = len(scores)
        
        # Compute additional statistics
        sim_differences = [s['difference'] for s in all_similarities]
        mean_diff = float(np.mean(sim_differences)) if sim_differences else 0.0
        std_diff = float(np.std(sim_differences)) if sim_differences else 0.0
        
        # Prepare results
        results = {
            "contrastive_accuracy": accuracy,
            "mean_similarity_difference": mean_diff,
            "std_similarity_difference": std_diff,
        }
        
        # Add negative-type-specific results
        for neg_type, acc in neg_type_accuracies.items():
            results[f'accuracy_{neg_type}'] = acc
            results[f'count_{neg_type}'] = neg_type_counts[neg_type]
        
        # Compute macro average across negative types
        if neg_type_accuracies:
            results['macro_accuracy'] = float(np.mean(list(neg_type_accuracies.values())))
        
        # Prepare embeddings
        pos_image_embeddings = torch.cat(pos_image_emb_list, dim=0).numpy()
        neg_image_embeddings = torch.cat(neg_image_emb_list, dim=0).numpy()
        sentence_embeddings = torch.cat(sentence_emb_list, dim=0).numpy()

        embeddings = {
            "pos_image_embeddings": pos_image_embeddings,
            "neg_image_embeddings": neg_image_embeddings,
            "sentence_embeddings": sentence_embeddings,
        }

        return results, embeddings

    def split_dataset(
        self,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
        split_type: Literal["random", "object"] = "random",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Split dataset into train/val/test."""
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
            # For SVO Probes, object split = random split since each example is unique
            return self.split_dataset(val_ratio, test_ratio, seed, "random")
        
        else:
            raise ValueError(f"Unknown split_type={split_type}")

    def __getstate__(self):
        """Ensure the dataset can be pickled."""
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)


class SVOProbesNeg(Dataset):
    """
    Wrapper for SVOProbesDataset to provide tokenized positives and negatives.
    Compatible with other *Neg dataset classes.
    """
    
    def __init__(
        self,
        svo_dataset: SVOProbesDataset,
        indices: List[int],
        num_negatives: int = 1,
    ):
        super().__init__()
        self.dataset = svo_dataset
        self.num_negatives = num_negatives
        self.idx_to_dataset_idx = {i: idx for i, idx in enumerate(indices)}
        
        print(f"SVOProbesNeg initialized with {self.num_negatives} negatives per sample")
        print("SVO Probes dataset for verb understanding evaluation")

    def __len__(self) -> int:
        return len(self.idx_to_dataset_idx)
    
    def __getitem__(self, idx: int):
        idx = int(idx)
        sample = self.dataset[self.idx_to_dataset_idx[idx]]
        
        pos_image = sample['image_options'][0]  # Positive image
        pos_text = sample['caption_options'][0]  # Sentence
        
        # For SVO Probes, we don't have text negatives, only image negatives
        # So we use the same sentence for both positive and "negative" text
        neg_text = pos_text  # Same sentence
        
        # Tokenize using CLIP
        import clip
        pos_tok = clip.tokenize(pos_text, truncate=True).squeeze(0)
        neg_tok = clip.tokenize(neg_text, truncate=True).squeeze(0)
        
        # For compatibility
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
    
    # Test the dataset
    ds = SVOProbesDataset(
        "./datasets/svo_probes",
        subset_name="all", 
        image_preprocess=simple_preprocess, 
        download=True,
        verbose=True
    )
    print(f"Loaded {len(ds)} samples.")
    
    if len(ds) > 0:
        sample = ds[0]
        print("Sample keys:", sample.keys())
        pos_img = sample["image_options"][0]
        neg_img = sample["image_options"][1]
        print("Positive image shape:", pos_img.shape if hasattr(pos_img, 'shape') else type(pos_img))
        print("Negative image shape:", neg_img.shape if hasattr(neg_img, 'shape') else type(neg_img))
        print("Sentence:", sample["caption_options"][0])
        print("Positive triplet:", sample["pos_triplet"])
        print("Negative triplet:", sample["neg_triplet"])
        print("Negative type:", sample["negative_type"])

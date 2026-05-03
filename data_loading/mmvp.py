"""
MMVP-VLM Benchmark Dataset

The MMVP-VLM (Multimodal Visual Patterns - Visual Language Models) Benchmark evaluates
CLIP-based models' ability to understand visual patterns. It contains text-image pairs
categorized by visual pattern types (Camera Perspective, Color, Orientation, Presence, etc.).

Each sample consists of a pair of images and their corresponding statements, where
the first statement matches the first image. The evaluation measures whether the model
can correctly match images to their descriptions.

IMPORTANT - MMVP Scoring (from paper):
"A pair is deemed correctly answered if the model can accurately match BOTH 
image-text combinations."

This means:
- text_contrastive_accuracy: img1 prefers stmt1 AND img2 prefers stmt2 (pair-level)
- image_contrastive_accuracy: stmt1 prefers img1 AND stmt2 prefers img2 (pair-level)
- group_contrastive_accuracy: ALL four matches correct (pair-level)

Dataset: https://huggingface.co/datasets/MMVP/MMVP_VLM
Paper: https://arxiv.org/abs/2401.06209
"""

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Optional, Literal
import random

try:
    from utils.align import (
        compute_caption_embeddings_intermediate_batch,
        compute_image_embeddings_intermediate_batch,
    )
except ImportError:
    print("Warning: utils.align not found. Evaluation functions may not work.")


# Available visual pattern types in MMVP-VLM
MMVP_PATTERN_TYPES = [
    "Camera Perspective",
    "Color",
    "Orientation", 
    "Presence",
    "Quantity",
    "Spatial",
    "State",
    "Structural Character",
    "Text",
]


class MMVPDataset(Dataset):
    """
    MMVP-VLM (MultiModal Visual Patterns) dataset for evaluating vision-language 
    compositional reasoning on visual patterns.
    
    Loads from MMVP/MMVP_VLM on the Hugging Face Hub.
    
    The dataset is organized as PAIRS:
    - Each pair has 2 images and 2 statements
    - Statement 1 matches Image 1, Statement 2 matches Image 2
    - For evaluation, we compute:
        * text_contrastive: image -> which caption is correct?
        * image_contrastive: caption -> which image is correct?
        * group_contrastive: both correct for a sample
    
    Available pattern types (subset_name):
    - "all": All patterns combined
    - "Camera Perspective", "Color", "Orientation", "Presence", 
      "Quantity", "Shape", "Size", "State", "Texture", "Structural Character", "Text"
    
    Returns samples in format:
        {
            'image': preprocessed image tensor,
            'caption': positive caption (correct description),
            'foil': negative caption (incorrect description),
            'label': 0 or 1 (which caption is correct - always 0 since caption comes first),
            'pattern_type': visual pattern category,
            'pair_id': identifier for the image pair,
        }
    """
    
    def __init__(
        self,
        data_root: Optional[str] = None,
        subset_name: str = "all",
        image_preprocess=None,
        download: bool = True,
        verbose: bool = True,
        **kwargs
    ):
        """
        Args:
            data_root: Optional cache directory for the dataset
            subset_name: Pattern type to filter by ("all" for all patterns)
            image_preprocess: Image preprocessing function (e.g., CLIP preprocess)
            download: Whether to download if not cached (default True)
            verbose: Whether to print loading information
        """
        from datasets import load_dataset
        
        self.data_root = data_root
        self.subset_name = subset_name
        self.image_preprocess = image_preprocess
        self.verbose = verbose
        
        # Load the dataset from HuggingFace
        cache_dir = data_root if data_root else None
        
        if verbose:
            print(f"[MMVP] Loading dataset from HuggingFace (cache_dir: {cache_dir})")
        
        self.ds = load_dataset("MMVP/MMVP_VLM", split="train", cache_dir=cache_dir)
        
        if verbose:
            print(f"[MMVP] Dataset loaded successfully ({len(self.ds)} total rows)")
        
        # Load questions CSV and process into pairs
        self.questions = self._load_questions_csv()
        self.pairs = self._process_pairs()
        
        # Filter by pattern type if specified
        if subset_name != "all":
            self.pairs = [
                p for p in self.pairs 
                if p['pattern_type'].lower() == subset_name.lower()
            ]
            
        if verbose:
            print(f"[MMVP] Processed {len(self.pairs)} pairs (subset: {subset_name})")
            # Print breakdown by pattern type
            pattern_counts = defaultdict(int)
            for p in self.pairs:
                pattern_counts[p['pattern_type']] += 1
            print("[MMVP] Pattern distribution:")
            for pattern, count in sorted(pattern_counts.items()):
                print(f"       - {pattern}: {count}")
    
    def _load_questions_csv(self) -> Dict[int, Dict]:
        """
        Load Questions.csv from HuggingFace to get statements for each image.
        
        Returns:
            Dict mapping Question ID to {'type': str, 'statement': str}
        """
        import requests
        
        csv_url = "https://huggingface.co/datasets/MMVP/MMVP_VLM/raw/main/Questions.csv"
        
        if self.verbose:
            print("[MMVP] Downloading Questions.csv...")
        
        response = requests.get(csv_url)
        response.raise_for_status()
        
        # Parse CSV
        questions = {}
        lines = response.text.strip().split('\n')
        
        # Skip header
        for line in lines[1:]:
            # Handle CSV with potential commas in quoted strings
            # Format: Question ID,Type,Statement
            parts = line.split(',', 2)  # Split into at most 3 parts
            if len(parts) >= 3:
                try:
                    q_id = int(parts[0].strip())
                    q_type = parts[1].strip()
                    statement = parts[2].strip().strip('"')
                    questions[q_id] = {
                        'type': q_type,
                        'statement': statement,
                    }
                except ValueError:
                    continue
        
        if self.verbose:
            print(f"[MMVP] Loaded {len(questions)} questions from CSV")
        
        return questions
    
    def _process_pairs(self) -> List[Dict]:
        """
        Process raw dataset rows into pairs.
        
        Each pair contains:
        - image1, image2: the two images
        - statement1, statement2: the two statements
        - pattern_type: the visual pattern category
        - pair_id: unique identifier
        
        Statement1 matches Image1, Statement2 matches Image2.
        """
        pairs = []
        
        # The dataset has 270 images, paired as (1,2), (3,4), etc.
        # HuggingFace dataset indices are 0-269, corresponding to Question IDs 1-270
        num_pairs = len(self.ds) // 2
        
        for pair_idx in range(num_pairs):
            # HuggingFace indices (0-based)
            hf_idx1 = pair_idx * 2
            hf_idx2 = pair_idx * 2 + 1
            
            # Question IDs (1-based)
            q_id1 = hf_idx1 + 1
            q_id2 = hf_idx2 + 1
            
            # Get images from HuggingFace dataset
            row1 = self.ds[hf_idx1]
            row2 = self.ds[hf_idx2]
            
            image1 = row1.get('image')
            image2 = row2.get('image')
            
            # Get statements and types from CSV
            q1 = self.questions.get(q_id1, {'type': 'unknown', 'statement': ''})
            q2 = self.questions.get(q_id2, {'type': 'unknown', 'statement': ''})
            
            statement1 = q1['statement']
            statement2 = q2['statement']
            pattern_type = q1['type']  # Both should have same type
            
            pairs.append({
                'image1_obj': image1,
                'image2_obj': image2,
                'statement1': statement1,
                'statement2': statement2,
                'pattern_type': pattern_type,
                'pair_id': pair_idx,
            })
        
        return pairs
    
    def __len__(self) -> int:
        # Return number of individual samples (2 per pair)
        return len(self.pairs) * 2
    
    def _load_image(self, img_obj) -> Image.Image:
        """Load image from HuggingFace format."""
        if isinstance(img_obj, Image.Image):
            return img_obj.convert("RGB")
        elif isinstance(img_obj, dict):
            if "bytes" in img_obj:
                from io import BytesIO
                return Image.open(BytesIO(img_obj["bytes"])).convert("RGB")
            elif "path" in img_obj:
                return Image.open(img_obj["path"]).convert("RGB")
            else:
                raise ValueError(f"Unknown image object format: {img_obj.keys()}")
        else:
            return Image.open(img_obj).convert("RGB")
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Even indices (0, 2, 4, ...) -> first image of pair
        Odd indices (1, 3, 5, ...) -> second image of pair
        """
        pair_idx = idx // 2
        is_first = (idx % 2 == 0)
        
        pair = self.pairs[pair_idx]
        
        if is_first:
            img_obj = pair['image1_obj']
            caption = pair['statement1']
            foil = pair['statement2']
        else:
            img_obj = pair['image2_obj']
            caption = pair['statement2']
            foil = pair['statement1']
        
        image = self._load_image(img_obj)
        
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        
        return {
            "image": image,
            "caption": caption,
            "foil": foil,
            "label": 0,  # Caption is always the correct one (at index 0)
            "pattern_type": pair['pattern_type'],
            "pair_id": pair['pair_id'],
            "is_first_in_pair": is_first,
        }
    
    def get_pair(self, pair_idx: int) -> Dict:
        """
        Get a full pair with both images and statements preprocessed.
        """
        pair = self.pairs[pair_idx]
        
        image1 = self._load_image(pair['image1_obj'])
        image2 = self._load_image(pair['image2_obj'])
        
        if self.image_preprocess is not None:
            image1 = self.image_preprocess(image1)
            image2 = self.image_preprocess(image2)
        
        return {
            "image1": image1,
            "image2": image2,
            "statement1": pair['statement1'],
            "statement2": pair['statement2'],
            "pattern_type": pair['pattern_type'],
            "pair_id": pair['pair_id'],
        }
    
    def get_captions(self) -> List[str]:
        """Get all unique captions in the dataset."""
        captions = set()
        for pair in self.pairs:
            if pair['statement1']:
                captions.add(pair['statement1'])
            if pair['statement2']:
                captions.add(pair['statement2'])
        return sorted(list(captions))
    
    def get_image_paths(self) -> List[str]:
        """Get image identifiers (for HuggingFace datasets, return placeholders)."""
        return [f"mmvp_image_{i}" for i in range(len(self))]
    
    def get_idx_to_ptr(self, idx: int) -> int:
        """Map dataset index to positive caption pointer (for compatibility)."""
        return idx
    
    def get_idx_to_candidates_ptr(self, idx: int) -> List[int]:
        """Map dataset index to negative caption pointers (for compatibility)."""
        return [idx]
    
    def _collate_pairs_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function for pair-based DataLoader."""
        images1 = []
        images2 = []
        statements1 = []
        statements2 = []
        pattern_types = []
        pair_ids = []
        
        for pair in batch:
            images1.append(pair['image1'])
            images2.append(pair['image2'])
            statements1.append(pair['statement1'])
            statements2.append(pair['statement2'])
            pattern_types.append(pair['pattern_type'])
            pair_ids.append(pair['pair_id'])
        
        return {
            'images1': torch.stack(images1),  # [B, C, H, W]
            'images2': torch.stack(images2),  # [B, C, H, W]
            'statements1': statements1,  # List[str]
            'statements2': statements2,  # List[str]
            'pattern_types': pattern_types,  # List[str]
            'pair_ids': pair_ids,  # List[int]
        }
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function for sample-based DataLoader."""
        images = []
        captions = []
        foils = []
        pattern_types = []
        
        for sample in batch:
            images.append(sample['image'])
            captions.append(sample['caption'])
            foils.append(sample['foil'])
            pattern_types.append(sample['pattern_type'])
        
        return {
            'images': torch.stack(images),  # [B, C, H, W]
            'captions': captions,  # List[str]
            'foils': foils,  # List[str]
            'pattern_types': pattern_types,  # List[str]
        }
    
    def evaluate(
        self,
        embedding_model,
        aligning_model=None,
        device: str = "cuda",
        batch_size: int = 16,
        indices: Optional[List[int]] = None,
        intermediate_text_layer_names: List[str] = ["final"],
        intermediate_image_layer_names: List[str] = ["final"],
    ):
        """
        Evaluate model on MMVP benchmark using pair-based evaluation.
        
        For each pair (image1, image2, statement1, statement2):
        - text_contrastive: image1 should prefer statement1 over statement2
                           image2 should prefer statement2 over statement1
        - image_contrastive: statement1 should prefer image1 over image2
                            statement2 should prefer image2 over image1
        - group_contrastive: both text and image correct for each sample
        
        Args:
            embedding_model: The CLIP model to evaluate
            aligning_model: Optional alignment model
            device: Device to run evaluation on
            batch_size: Batch size for evaluation (in pairs)
            indices: Optional subset of pair indices to evaluate
            intermediate_text_layer_names: Layer names for text embeddings
            intermediate_image_layer_names: Layer names for image embeddings
            
        Returns:
            tuple: (results_dict, embeddings_dict)
        """
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
        device: str = "cuda",
        batch_size: int = 16,
        indices: Optional[List[int]] = None,
        intermediate_text_layer_names: List[str] = ["final"],
        intermediate_image_layer_names: List[str] = ["final"],
    ):
        """
        Internal evaluation implementation using pairs.
        
        Per MMVP paper scoring: "A pair is deemed correctly answered if the model 
        can accurately match BOTH image-text combinations."
        
        Computes (all at PAIR level - both samples in pair must be correct):
        - text_contrastive_accuracy: img1->stmt1 correct AND img2->stmt2 correct
        - image_contrastive_accuracy: stmt1->img1 correct AND stmt2->img2 correct  
        - group_contrastive_accuracy: All four matches correct
        """
        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")
        
        # Create a simple dataset wrapper for pairs
        class PairDataset(Dataset):
            def __init__(self, mmvp_dataset, pair_indices=None):
                self.mmvp = mmvp_dataset
                self.pair_indices = pair_indices if pair_indices is not None else list(range(len(mmvp_dataset.pairs)))
            
            def __len__(self):
                return len(self.pair_indices)
            
            def __getitem__(self, idx):
                pair_idx = self.pair_indices[idx]
                return self.mmvp.get_pair(pair_idx)
        
        # Determine which pairs to evaluate
        if indices is not None:
            # indices refer to pair indices
            pair_dataset = PairDataset(self, indices)
        else:
            pair_dataset = PairDataset(self)
        
        if len(pair_dataset) == 0:
            raise ValueError(f"No pairs found for pattern type: {self.subset_name}")
        
        # Use DataLoader for efficient batch loading
        dataloader = DataLoader(
            pair_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_pairs_fn,
        )
        
        # Results tracking - per sample (2 per pair)
        all_text_correct = []
        all_image_correct = []
        all_group_correct = []
        pattern_results = defaultdict(list)
        
        # Collect embeddings
        all_img1_embs = []
        all_img2_embs = []
        all_stmt1_embs = []
        all_stmt2_embs = []
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="MMVP",
            subset_name=self.subset_name,
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device,
        ) as cache:
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating MMVP ({self.subset_name})")):
                images1 = batch['images1']  # [B, C, H, W]
                images2 = batch['images2']  # [B, C, H, W]
                statements1 = batch['statements1']  # List[str]
                statements2 = batch['statements2']  # List[str]
                pattern_types = batch['pattern_types']  # List[str]
                B = len(statements1)
                
                if B == 0:
                    continue
                
                with torch.no_grad():
                    # Get image embeddings
                    img1_embs = cache.get_or_compute_embeddings(
                        images1.to(device),
                        "image1",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size,
                    )
                    
                    img2_embs = cache.get_or_compute_embeddings(
                        images2.to(device),
                        "image2",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size,
                    )
                    
                    # Get text embeddings
                    stmt1_embs = cache.get_or_compute_embeddings(
                        statements1,
                        "text1",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * batch_size,
                    )
                    
                    stmt2_embs = cache.get_or_compute_embeddings(
                        statements2,
                        "text2",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * batch_size,
                    )
                    
                    # Store embeddings
                    all_img1_embs.append(img1_embs.cpu())
                    all_img2_embs.append(img2_embs.cpu())
                    all_stmt1_embs.append(stmt1_embs.cpu())
                    all_stmt2_embs.append(stmt2_embs.cpu())
                
                # Compute similarity matrix for each pair
                # For pair b: img1[b], img2[b], stmt1[b], stmt2[b]
                # text_contrastive for sample 1: img1[b] · stmt1[b] > img1[b] · stmt2[b]
                # text_contrastive for sample 2: img2[b] · stmt2[b] > img2[b] · stmt1[b]
                # image_contrastive for sample 1: stmt1[b] · img1[b] > stmt1[b] · img2[b]
                # image_contrastive for sample 2: stmt2[b] · img2[b] > stmt2[b] · img1[b]
                
                # Sample 1: image1 with statement1 as positive
                sim_img1_stmt1 = (img1_embs * stmt1_embs).sum(dim=1)  # [B]
                sim_img1_stmt2 = (img1_embs * stmt2_embs).sum(dim=1)  # [B]
                text_correct_1 = sim_img1_stmt1 > sim_img1_stmt2  # [B]
                image_correct_1 = sim_img1_stmt1 > (img2_embs * stmt1_embs).sum(dim=1)  # [B]
                
                # Sample 2: image2 with statement2 as positive
                sim_img2_stmt2 = (img2_embs * stmt2_embs).sum(dim=1)  # [B]
                sim_img2_stmt1 = (img2_embs * stmt1_embs).sum(dim=1)  # [B]
                text_correct_2 = sim_img2_stmt2 > sim_img2_stmt1  # [B]
                image_correct_2 = sim_img2_stmt2 > (img1_embs * stmt2_embs).sum(dim=1)  # [B]
                
                # MMVP scoring: a pair is correct ONLY if BOTH samples in the pair are correct
                # See paper: "A pair is deemed correctly answered if the model can 
                # accurately match both image-text combinations."
                
                # For text_contrastive: both image1->stmt1 AND image2->stmt2 must be correct
                pair_text_correct = text_correct_1 & text_correct_2  # [B]
                
                # For image_contrastive: both stmt1->img1 AND stmt2->img2 must be correct
                pair_image_correct = image_correct_1 & image_correct_2  # [B]
                
                # For group: all four matches must be correct
                pair_group_correct = pair_text_correct & pair_image_correct  # [B]
                
                # Extend results (one result per pair, not per sample)
                for i in range(B):
                    pattern = pattern_types[i]
                    
                    all_text_correct.append(pair_text_correct[i].item())
                    all_image_correct.append(pair_image_correct[i].item())
                    all_group_correct.append(pair_group_correct[i].item())
                    pattern_results[pattern].append({
                        'text_score': pair_text_correct[i].item(),
                        'image_score': pair_image_correct[i].item(),
                        'group_score': pair_group_correct[i].item(),
                    })
        
        # Compute overall accuracies (pair-level, as per MMVP paper)
        text_acc = float(np.mean(all_text_correct)) if all_text_correct else 0.0
        image_acc = float(np.mean(all_image_correct)) if all_image_correct else 0.0
        group_acc = float(np.mean(all_group_correct)) if all_group_correct else 0.0
        
        results = {
            "text_contrastive_accuracy": text_acc,
            "image_contrastive_accuracy": image_acc,
            "group_contrastive_accuracy": group_acc,
            "num_pairs": len(pair_dataset),
            "num_correct_pairs_text": sum(all_text_correct),
            "num_correct_pairs_image": sum(all_image_correct),
            "num_correct_pairs_group": sum(all_group_correct),
        }
        
        # Add per-pattern results
        for pattern, scores in pattern_results.items():
            pattern_key = pattern.replace(" ", "_").lower()
            results[f"{pattern_key}_text_contrastive_accuracy"] = float(np.mean([s['text_score'] for s in scores]))
            results[f"{pattern_key}_image_contrastive_accuracy"] = float(np.mean([s['image_score'] for s in scores]))
            results[f"{pattern_key}_group_contrastive_accuracy"] = float(np.mean([s['group_score'] for s in scores]))
        
        # Prepare embeddings dict (interleaved: img1, img2 for each pair)
        img1_embs_cat = torch.cat(all_img1_embs, dim=0)  # [N_pairs, D]
        img2_embs_cat = torch.cat(all_img2_embs, dim=0)  # [N_pairs, D]
        stmt1_embs_cat = torch.cat(all_stmt1_embs, dim=0)  # [N_pairs, D]
        stmt2_embs_cat = torch.cat(all_stmt2_embs, dim=0)  # [N_pairs, D]
        
        # Interleave to match sample order
        N_pairs = img1_embs_cat.shape[0]
        D = img1_embs_cat.shape[1]
        
        image_embeddings = torch.zeros(N_pairs * 2, D)
        caption_embeddings = torch.zeros(N_pairs * 2, D)
        negative_caption_embeddings = torch.zeros(N_pairs * 2, 1, D)
        
        image_embeddings[0::2] = img1_embs_cat
        image_embeddings[1::2] = img2_embs_cat
        caption_embeddings[0::2] = stmt1_embs_cat
        caption_embeddings[1::2] = stmt2_embs_cat
        negative_caption_embeddings[0::2, 0] = stmt2_embs_cat
        negative_caption_embeddings[1::2, 0] = stmt1_embs_cat
        
        embeddings = {
            "image_embeddings": image_embeddings.numpy(),
            "caption_embeddings": caption_embeddings.numpy(),
            "negative_caption_embeddings": negative_caption_embeddings.numpy(),
        }
        
        return results, embeddings
    
    def split_dataset(
        self,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
        split_type: Literal["random", "pair"] = "random",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split dataset into train/val/test sets (by pairs).
        
        Args:
            val_ratio: Fraction of pairs for validation
            test_ratio: Fraction of pairs for test
            seed: Random seed
            split_type: "random" or "pair" (keep pairs together, same behavior for this dataset)
            
        Returns:
            Dict with 'train', 'val', 'test' keys, each containing 'indices' (pair indices)
        """
        np.random.seed(seed)
        random.seed(seed)
        n_pairs = len(self.pairs)
        
        pair_indices = list(range(n_pairs))
        random.shuffle(pair_indices)
        
        n_test = int(n_pairs * test_ratio)
        n_val = int(n_pairs * val_ratio)
        
        test_idx = pair_indices[:n_test]
        val_idx = pair_indices[n_test:n_test + n_val]
        train_idx = pair_indices[n_test + n_val:]
        
        return {
            'train': {'indices': np.array(train_idx)},
            'val': {'indices': np.array(val_idx)},
            'test': {'indices': np.array(test_idx)},
        }
    
    def check_dataset_integrity(self) -> Dict:
        """Check dataset integrity and return statistics."""
        stats = {
            'total_pairs': len(self.pairs),
            'total_samples': len(self),
            'pattern_counts': defaultdict(int),
            'missing_images': 0,
            'missing_captions': 0,
        }
        
        for pair in self.pairs:
            stats['pattern_counts'][pair['pattern_type']] += 1
            if pair['image1_obj'] is None or pair['image2_obj'] is None:
                stats['missing_images'] += 1
            if not pair['statement1'] or not pair['statement2']:
                stats['missing_captions'] += 1
        
        stats['pattern_counts'] = dict(stats['pattern_counts'])
        return stats


class MMVPNeg(Dataset):
    """
    Wrapper for MMVPDataset to provide tokenized positives and negatives.
    Compatible with other *Neg dataset classes.
    """
    
    def __init__(
        self,
        mmvp_dataset: MMVPDataset,
        indices: List[int],
        num_negatives: int = 1,
    ):
        import clip
        self.mmvp_dataset = mmvp_dataset
        self.indices = indices
        self.num_negatives = num_negatives
        self.clip = clip
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        sample = self.mmvp_dataset[real_idx]
        
        # Get positive and negative captions
        pos_caption = sample['caption']
        neg_caption = sample['foil']
        
        # Tokenize
        pos_tokens = self.clip.tokenize([pos_caption], truncate=True).squeeze(0)
        neg_tokens = self.clip.tokenize([neg_caption], truncate=True).squeeze(0)
        
        return sample['image'], pos_tokens, neg_tokens
    
    def collate_fn(self, batch: List[tuple]) -> Dict:
        """Collate function for DataLoader."""
        images, pos_tokens, neg_tokens = zip(*batch)
        
        return {
            'image': torch.stack(images),
            'pos_tokens': torch.stack(pos_tokens),
            'neg_tokens': torch.stack(neg_tokens),
        }


if __name__ == "__main__":
    # Quick test
    print("Testing MMVP Dataset...")
    dataset = MMVPDataset(
        data_root="datasets/",
        subset_name="all",
        verbose=True,
    )
    
    print(f"\nDataset: {len(dataset.pairs)} pairs, {len(dataset)} samples")
    
    print("\nFirst pair:")
    pair = dataset.get_pair(0)
    print(f"  Statement 1: {pair['statement1']}")
    print(f"  Statement 2: {pair['statement2']}")
    print(f"  Pattern type: {pair['pattern_type']}")
    
    print("\nFirst sample (from __getitem__):")
    sample = dataset[0]
    for k, v in sample.items():
        if k != 'image':
            print(f"  {k}: {v}")
    
    print("\nIntegrity check:")
    stats = dataset.check_dataset_integrity()
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Pattern counts: {stats['pattern_counts']}")
    print(f"  Missing images: {stats['missing_images']}")
    print(f"  Missing captions: {stats['missing_captions']}")
    
    # Test subset loading
    print("\n\nTesting subset loading (Orientation)...")
    orientation_dataset = MMVPDataset(
        data_root="datasets/",
        subset_name="Orientation",
        verbose=True,
    )
    print(f"Orientation subset: {len(orientation_dataset.pairs)} pairs")

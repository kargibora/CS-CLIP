#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO dataset with structured negatives for training.

- COCODataset: Base dataset with split_dataset(), __getitem__ returns raw captions
- COCONeg: Wrapper that handles tokenization for training
"""

import json
import os
import random
import logging
from glob import glob
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils.sampler import StructuredSampler, OriginalCaptionNegativeSampler

try:
    import clip
except Exception:
    clip = None

logger = logging.getLogger(__name__)


class COCODataset(Dataset):
    """
    Base COCO dataset that loads JSON files and provides split_dataset().
    
    Returns raw data in __getitem__.
    Use COCONeg wrapper for tokenization.
    
    Args:
        json_folder: Path to folder containing JSON annotation files
        image_root: Root path for images
        image_preprocess: Image preprocessing transform
        subset_name: Filter samples by split name in image_path
        num_entity_captions: Number of entity/relation captions to sample per image
        use_structured_sampling: Structured positive-negative sampling must stay enabled
    """
    
    def __init__(
        self,
        json_folder: str,
        image_root: str,
        image_preprocess=None,
        subset_name: str = "all",
        num_entity_captions: int = 3,
        use_structured_sampling: bool = True,
        structured_relation_prob: float = 0.5,
        use_context_in_entity_pairs: bool = True,
        swap_negative_prob: float = 0.5,
        inplace_replacement_prob: float = 0.7,
        max_entities_per_sample: Optional[int] = None,
        max_positive_entities_with_negative: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        
        self.json_folder = json_folder
        self.image_root = image_root
        self.image_preprocess = image_preprocess
        self.subset_name = subset_name.lower().strip()
        self.num_entity_captions = num_entity_captions
        if not use_structured_sampling:
            raise ValueError("COCODataset requires structured sampling.")
        self.use_structured_sampling = True
        self.structured_relation_prob = structured_relation_prob
        self.use_context_in_entity_pairs = use_context_in_entity_pairs
        self.swap_negative_prob = swap_negative_prob
        self.inplace_replacement_prob = inplace_replacement_prob
        self.max_entities_per_sample = max_entities_per_sample
        self.max_positive_entities_with_negative = max_positive_entities_with_negative
        
        all_samples = self._load_all_jsons(json_folder)
        
        if self.subset_name and self.subset_name != "all":
            self.samples = self._filter_by_subset(all_samples, self.subset_name)
            logger.info(f"[COCODataset] Filtered to {len(self.samples)}/{len(all_samples)} samples for subset '{self.subset_name}'")
        else:
            self.samples = all_samples
            logger.info(f"[COCODataset] Loaded {len(self.samples)} samples from {json_folder}")
        
        self.structured_sampler = StructuredSampler(
            structured_relation_prob=structured_relation_prob,
            use_context_in_entity_pairs=use_context_in_entity_pairs,
        )
        
        self.original_caption_neg_sampler = OriginalCaptionNegativeSampler(
            swap_negative_prob=swap_negative_prob,
            inplace_replacement_prob=inplace_replacement_prob,
        )
    
    def _filter_by_subset(self, samples: List[Dict], subset_name: str) -> List[Dict]:
        """
        Filter samples based on subset name appearing in image_path.
        
        Supports:
        - "train" or "train2014" or "train2017" -> matches if "train" in path
        - "val" or "val2014" or "val2017" -> matches if "val" in path  
        - "test" or "test2014" or "test2017" -> matches if "test" in path
        
        Args:
            samples: List of sample dictionaries
            subset_name: Subset to filter for (e.g., "train", "val", "test", "train2014")
        
        Returns:
            Filtered list of samples
        """
        subset_lower = subset_name.lower().strip()
        
        if "train" in subset_lower:
            filter_key = "train"
        elif "val" in subset_lower:
            filter_key = "val"
        elif "test" in subset_lower:
            filter_key = "test"
        else:
            # Use as-is for custom subsets
            filter_key = subset_lower
        
        filtered = []
        for sample in samples:
            image_path = sample.get("image_path", "").lower()
            
            if filter_key in image_path:
                filtered.append(sample)
        
        if not filtered:
            logger.warning(f"[COCODataset] No samples found for subset '{subset_name}'. "
                          f"Check that image_path contains '{filter_key}'.")
        
        return filtered
    
    def _load_all_jsons(self, folder: str) -> List[Dict]:
        """Load all JSON files from folder into memory."""
        samples = []
        json_files = sorted(glob(os.path.join(folder, "*.json")))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {folder}")
        
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    samples.extend(data)
                else:
                    samples.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {json_path}: {e}")
                continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from path, handling relative paths."""
        if not os.path.isabs(image_path):
            full_path = os.path.join(self.image_root, image_path)
        else:
            full_path = image_path
        
        return Image.open(full_path).convert("RGB")
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns raw sample data with captions (not tokenized).
        
        Format matches the FT training pipeline expectations.
        """
        sample = self.samples[idx]
        
        # Load and preprocess image
        image_path = sample.get("image_path", "")
        img = self._load_image(image_path)
        if self.image_preprocess is not None:
            img = self.image_preprocess(img)
        
        original_caption = sample.get("original_caption", sample.get("caption", ""))
        positive_entities = sample.get("entities", [])
        num_entities_available = len(positive_entities) if positive_entities else 0
        
        paraphrased_caption = sample.get("paraphrased_caption", None)
        has_paraphrase = bool(paraphrased_caption and str(paraphrased_caption).strip())
        
        all_positives = []
        all_negatives = []
        entities_per_caption = []
        
        original_neg = self.original_caption_neg_sampler.sample_negative(sample)
        all_positives.append(original_caption)
        all_negatives.append(original_neg if original_neg else "")
        entities_per_caption.append(num_entities_available if num_entities_available > 0 else 1)
        
        valid_pairs = []
        
        for _ in range(self.num_entity_captions):
            pos_caption, neg_caption, metadata = None, None, {}
            
            pos_caption, neg_caption, metadata = (
                self.structured_sampler.sample_structured_positive_and_negative(sample)
            )
            
            if pos_caption and neg_caption:
                pair_type = metadata.get("pair_type", "")
                if pair_type == "entity":
                    num_units = 2 if self.use_context_in_entity_pairs else 1
                elif pair_type == "relation":
                    num_units = 3
                else:
                    num_units = metadata.get("num_sampled", 1)
                
                all_positives.append(pos_caption)
                all_negatives.append(neg_caption)
                entities_per_caption.append(num_units)
                valid_pairs.append((pos_caption, neg_caption, num_units))
            else:
                all_positives.append(None)
                all_negatives.append(None)
                entities_per_caption.append(0)
        
        for i in range(1, len(all_positives)):
            if all_positives[i] is None:
                if valid_pairs:
                    pos, neg, num_units = random.choice(valid_pairs)
                    all_positives[i] = pos
                    all_negatives[i] = neg
                    entities_per_caption[i] = num_units
                else:
                    all_positives[i] = original_caption
                    all_negatives[i] = original_neg if original_neg else ""
                    entities_per_caption[i] = 0
        
        caption_valid_mask = []
        for i in range(len(all_positives)):
            pos_ok = bool(all_positives[i] and str(all_positives[i]).strip())
            neg_ok = bool(all_negatives[i] and str(all_negatives[i]).strip())
            units_ok = entities_per_caption[i] >= 1
            caption_valid_mask.append(bool(pos_ok and neg_ok and units_ok))
        
        num_total_captions = 1 + self.num_entity_captions
        
        return {
            "image_options": img,
            "caption_options": all_positives + all_negatives,
            "num_positives": num_total_captions,
            "entities_per_caption": entities_per_caption,
            "num_entities_available": num_entities_available,
            "caption_valid_mask": caption_valid_mask,
            "sample_is_valid": any(caption_valid_mask),
            "paraphrased_caption": paraphrased_caption if has_paraphrase else None,
            "has_paraphrase": has_paraphrase,
            "label": 0,
            "index": idx,
        }
    
    def get_caption_options(self, idx: int) -> List[str]:
        """Get caption options for a sample."""
        sample = self[idx]
        return sample["caption_options"]
    
    def split_dataset(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        seed: int = 42,
        split_type: Literal["random", "object"] = "random",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for test
            seed: Random seed for reproducibility
            split_type: "random" or "object" (group by sample_id)
        
        Returns:
            Dict with "train", "val", "test" keys, each containing {"indices": np.ndarray}
        """
        rng = random.Random(seed)
        
        if split_type == "random":
            idxs = list(range(len(self)))
            rng.shuffle(idxs)
            n = len(idxs)
            n_test = int(n * test_ratio)
            n_val = int((n - n_test) * val_ratio)
            test_idx = np.array(idxs[:n_test], dtype=np.int64)
            val_idx = np.array(idxs[n_test:n_test + n_val], dtype=np.int64)
            train_idx = np.array(idxs[n_test + n_val:], dtype=np.int64)
            return {
                "train": {"indices": train_idx},
                "val": {"indices": val_idx},
                "test": {"indices": test_idx},
            }
        
        elif split_type == "object":
            # Group by sample_id
            sid_to_indices: Dict[str, List[int]] = defaultdict(list)
            for idx in range(len(self)):
                sample = self.samples[idx]
                sid = sample.get("sample_id", str(idx))
                sid_to_indices[sid].append(idx)
            
            sids = list(sid_to_indices.keys())
            rng.shuffle(sids)
            n = len(sids)
            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)
            test_ids = set(sids[:n_test])
            val_ids = set(sids[n_test:n_test + n_val])
            train_ids = set(sids[n_test + n_val:])
            
            def _make_indices(idset: Set[str]) -> np.ndarray:
                out = []
                for sid in idset:
                    out.extend(sid_to_indices[sid])
                return np.array(out, dtype=np.int64)
            
            return {
                "train": {"indices": _make_indices(train_ids)},
                "val": {"indices": _make_indices(val_ids)},
                "test": {"indices": _make_indices(test_ids)},
            }
        else:
            raise ValueError(f"Unknown split_type={split_type}")


class COCONeg(Dataset):
    """
    Wrapper dataset that tokenizes captions from COCODataset.
    
    Training wrapper around COCODataset.
    
    Returns:
        {
            "image": tensor [C, H, W],
            "pos_tokens": tensor [1+N, 77],
            "neg_tokens": tensor [1+N, 77],
            "entities_per_caption": list [1+N],
            "num_entities_available": int,
            "caption_valid_mask": list [1+N] of bools,
        }
    """
    
    def __init__(
        self,
        coco_dataset: COCODataset,
        indices: List[int],
        num_negatives: int = 1,
        skip_negatives: bool = False,
    ):
        super().__init__()
        self.dataset = coco_dataset
        self.num_negatives = int(num_negatives)
        self.idx_map = {i: int(idx) for i, idx in enumerate(indices)}
        self.skip_negatives = skip_negatives
        
        self.is_multi_caption_mode = getattr(coco_dataset, 'num_entity_captions', 0) > 0
        self.num_entity_captions = getattr(coco_dataset, 'num_entity_captions', 0)
    
    def __len__(self) -> int:
        return len(self.idx_map)
    
    def _ensure_clip(self):
        global clip
        if clip is None:
            import clip as _clip
            clip = _clip
    
    def __getitem__(self, i: int) -> Dict:
        ds_idx = self.idx_map[i]
        sample = self.dataset[ds_idx]
        caption_options = sample["caption_options"]
        
        entities_per_caption = sample.get("entities_per_caption", [])
        num_entities_available = sample.get("num_entities_available", 0)
        caption_valid_mask = sample.get("caption_valid_mask", [])
        
        paraphrased_caption = sample.get("paraphrased_caption", None)
        has_paraphrase = sample.get("has_paraphrase", False)
        
        self._ensure_clip()
        
        num_captions = 1 + self.num_entity_captions
        
        pos_captions = caption_options[:num_captions]
        neg_captions = caption_options[num_captions:num_captions * 2]
        
        pos_toks = clip.tokenize(pos_captions, truncate=True)
        
        if self.skip_negatives:
            neg_toks = torch.zeros(num_captions, 77, dtype=torch.long)
        else:
            neg_toks = clip.tokenize(neg_captions, truncate=True)
        
        if has_paraphrase and paraphrased_caption:
            paraphrase_toks = clip.tokenize([paraphrased_caption], truncate=True).squeeze(0)
        else:
            paraphrase_toks = torch.zeros(77, dtype=torch.long)
        
        if len(caption_valid_mask) != num_captions:
            caption_valid_mask = [
                bool(pos_captions[j] and pos_captions[j].strip() and
                     neg_captions[j] and neg_captions[j].strip())
                for j in range(num_captions)
            ]
        
        return {
            "image": sample["image_options"],
            "pos_tokens": pos_toks,
            "neg_tokens": neg_toks,
            "paraphrase_tokens": paraphrase_toks,
            "has_paraphrase": has_paraphrase,
            "entities_per_caption": entities_per_caption,
            "num_entities_available": num_entities_available,
            "caption_valid_mask": caption_valid_mask,
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function for the FT training pipeline.
        """
        images = [b["image"] for b in batch]
        pos_toks = [b["pos_tokens"] for b in batch]
        neg_toks = [b["neg_tokens"] for b in batch]
        paraphrase_toks = [b["paraphrase_tokens"] for b in batch]
        has_paraphrase = [b["has_paraphrase"] for b in batch]
        entities_per_caption = [b["entities_per_caption"] for b in batch]
        num_available = [b["num_entities_available"] for b in batch]
        caption_valid_mask = [b["caption_valid_mask"] for b in batch]
        
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images, dim=0)
        
        pos_toks = torch.stack(pos_toks, dim=0)
        neg_toks = torch.stack(neg_toks, dim=0)
        paraphrase_toks = torch.stack(paraphrase_toks, dim=0)
        
        has_paraphrase_tensor = torch.tensor(has_paraphrase, dtype=torch.bool)
        
        max_captions = max(len(epc) for epc in entities_per_caption) if entities_per_caption else 1
        entities_padded = []
        for epc in entities_per_caption:
            if isinstance(epc, list):
                padded = epc + [0] * (max_captions - len(epc))
            else:
                padded = [0] * max_captions
            entities_padded.append(padded)
        entities_per_caption_tensor = torch.tensor(entities_padded, dtype=torch.long)
        
        num_available_tensor = torch.tensor(num_available, dtype=torch.long)
        
        valid_mask_padded = []
        for vm in caption_valid_mask:
            if isinstance(vm, list):
                padded = vm + [False] * (max_captions - len(vm))
            else:
                padded = [True] * max_captions
            valid_mask_padded.append(padded)
        caption_valid_mask_tensor = torch.tensor(valid_mask_padded, dtype=torch.bool)
        
        return {
            "images": images,
            "pos_tokens": pos_toks,
            "neg_tokens": neg_toks,
            "paraphrase_tokens": paraphrase_toks,
            "has_paraphrase": has_paraphrase_tensor,
            "entities_per_caption": entities_per_caption_tensor,
            "num_entities_available": num_available_tensor,
            "caption_valid_mask": caption_valid_mask_tensor,
        }


class COCONegDataset(Dataset):
    """
    Convenience class that combines COCODataset + COCONeg in a single interface.
    """
    
    def __init__(
        self,
        json_folder: str,
        image_root: str,
        image_preprocess=None,
        num_entity_captions: int = 3,
        use_structured_sampling: bool = True,
        structured_relation_prob: float = 0.5,
        use_context_in_entity_pairs: bool = True,
        swap_negative_prob: float = 0.5,
        inplace_replacement_prob: float = 0.7,
        num_negatives: int = 1,
        skip_negatives: bool = False,
        subset_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        
        self._base_dataset = COCODataset(
            json_folder=json_folder,
            image_root=image_root,
            image_preprocess=image_preprocess,
            num_entity_captions=num_entity_captions,
            use_structured_sampling=use_structured_sampling,
            structured_relation_prob=structured_relation_prob,
            use_context_in_entity_pairs=use_context_in_entity_pairs,
            swap_negative_prob=swap_negative_prob,
            inplace_replacement_prob=inplace_replacement_prob,
            subset_name=subset_name,
            **kwargs,
        )
        
        all_indices = list(range(len(self._base_dataset)))
        self._neg_wrapper = COCONeg(
            coco_dataset=self._base_dataset,
            indices=all_indices,
            num_negatives=num_negatives,
            skip_negatives=skip_negatives,
        )
    
    def __len__(self) -> int:
        return len(self._neg_wrapper)
    
    def __getitem__(self, idx: int) -> Dict:
        return self._neg_wrapper[idx]
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return self._neg_wrapper.collate_fn(batch)
    
    def split_dataset(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        seed: int = 42,
        split_type: Literal["random", "by_sample_id"] = "random",
    ) -> Dict[str, Dict]:
        """Delegate to base dataset's split_dataset."""
        return self._base_dataset.split_dataset(
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_type=split_type,
        )
    
    @property
    def samples(self):
        """Access base dataset samples."""
        return self._base_dataset.samples


def create_coco_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    distributed: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[torch.utils.data.Sampler]]:
    """
    Create a DataLoader for COCONeg dataset.
    """
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    
    return loader, sampler

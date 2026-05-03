import json
import os
import random
from typing import List, Literal, Optional

import clip
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from collections import defaultdict


ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
from easydict import EasyDict as edict
from tqdm import tqdm
from utils.align import (
    compute_caption_embeddings_intermediate_batch,
    compute_image_embeddings_intermediate_batch,
)

def group_samples_by_caption_id(flat_samples, keep_all_metadata=False):
    grouped = {}
    for item in flat_samples:
        capid = item['caption_id']
        if capid not in grouped:
            grouped[capid] = {
                "caption_id": capid,
                "caption": item["caption"],
                "negatives": [],
                "image_path": item["image_path"],
                "type": item.get("type", "")
            }
            if keep_all_metadata:
                grouped[capid]["concepts"] = []
                grouped[capid]["original_attributes"] = []
                grouped[capid]["original_objects"] = []
                grouped[capid]["generation_methods"] = []
        grouped[capid]["negatives"].append(item["negative"])

        if keep_all_metadata:
            # Flat format: top-level keys
            grouped[capid]["concepts"].append(item.get("concept", ""))
            grouped[capid]["original_attributes"].append(item.get("original_attribute", ""))
            grouped[capid]["original_objects"].append(item.get("original_object", ""))
            grouped[capid]["generation_methods"].append(item.get("generation_method", ""))

    grouped_samples = list(grouped.values())
    return grouped_samples



class CC3MDataset(Dataset):
    """
    Dataset for loading normalized JSON with structure:
      [
        {
            "caption_id": ...,
            "caption": ...,
            "negative": ...,
            "type": ...,
            "image_path": ...,
            "metadata": {
                "concept": ...,  # or "concepts": [...]
                "original_attribute": ...,
                "original_object": ...,
                "generation_method": ...,
                ...
            }
        },
        ...
      ]
    Args:
        data_root (str): Root directory of the dataset.
        subset_name (str): Name of the subset to load (color, size, etc.) or type filter.
        image_preprocess (callable, optional): Preprocessing function for images.
        filter_by (str): What to filter by - 'concept', 'type', or 'all'
    """
    def __init__(
        self,
        data_root: str,
        subset_name: str,
        image_preprocess: Optional[callable] = None,
        combine_by_caption_id=False,
        filter_by: str = 'all'  # 'concept', 'type', or 'all'
    ):
        self.data_root = data_root
        self.subset_name = subset_name
        self.image_preprocess = image_preprocess
        self.filter_by = filter_by

        json_path = 'datasets/CC3M-Neg/cc3m-train-local-full.json'

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Metadata file not found: {json_path}")
        
        # Read the metadata JSON file
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        sample_list = []
        for item in metadata:
            sample = edict(item)

            if subset_name != 'all':
                should_include = False

                if filter_by == 'concept':
                    if sample.get('concept', '') == subset_name:
                        should_include = True
                elif filter_by == 'type':
                    if sample.get('type', '') == subset_name:
                        should_include = True

                if not should_include:
                    continue
            
            sample['id'] = sample['caption_id']
            sample['local_filepath'] = os.path.join(self.data_root, sample['image_path'])

            # For backward compatibility, extract concept and original_value to top level
            metadata_dict = sample.get('metadata', {})
            sample['concept'] = self._extract_concept(metadata_dict)
            sample['original_value'] = self._extract_original_value(metadata_dict)

            sample_list.append(sample)

        if combine_by_caption_id:
            sample_list = group_samples_by_caption_id(sample_list, keep_all_metadata=True)
            sample_list = [edict(sample) for sample in sample_list]
        else:
            pass

        self.sample_list = sample_list

        self.captions = self.get_captions()
        self.image_paths = self.get_image_paths()

        print(f"Loaded {len(self.sample_list)} samples from {self.subset_name} subset (filter_by: {filter_by}).")
        print(f"Found {len(self.captions)} unique captions.")
        print(f"Found {len(self.image_paths)} unique images.")

        # Create a mapping from captions to indices
        self.caption_to_idx = {caption: i for i, caption in enumerate(self.captions)}

    def _extract_concept(self, sample):
        return sample.get('concept', '')

    def _extract_original_value(self, sample):
            # Prefer attribute, fallback to object
            return sample.get('original_attribute', '') or sample.get('original_object', '')

    def get_captions(self):
        """
        Get all captions in the dataset
        """
        captions = []
        for sample in self.sample_list:
            captions.append(sample['caption'])
            if isinstance(sample.get('negative'), str):
                captions.append(sample['negative'])
            elif isinstance(sample.get('negatives'), list):
                captions.extend(sample['negatives'])
        
        return sorted(set(captions))

    def get_image_paths(self):
        """
        Get all image paths in the dataset
        """
        image_paths = []
        for sample in self.sample_list:
            image_paths.append(sample['local_filepath'])
        return image_paths

    def get_idx_to_ptr(self, idx : int):
        """
        Get a mapping from original index of the caption in the dataset to the index in the caption_to_idx dictionary
        """
        sample = self.sample_list[idx]
        caption = sample['caption']

        return self.caption_to_idx[caption]
    
    def get_idx_to_candidates_ptr(self, idx: int):
        """
        Get a mapping from image index to caption indices
        
        Args:
            idx: Index of the image
        
        Returns:
            List of pointers to candidate captions in the caption_to_idx dictionary
        """
        sample = self.sample_list[idx]
        
        # Handle both single negative and list of negatives
        if isinstance(sample.get('negative'), str):
            candidates = [sample['negative']]
        elif isinstance(sample.get('negatives'), list):
            candidates = sample['negatives']
        else:
            candidates = []

        # Map captions to their indices in the caption dictionary
        return [self.caption_to_idx[caption] for caption in candidates if caption in self.caption_to_idx]

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> dict:
        sample = self.sample_list[idx]
        try:
            image = Image.open(sample['local_filepath']).convert('RGB')
        except OSError as e:
            image = cv2.imread(sample['local_filepath'])
            if image is None:
                raise OSError(f"Could not read image: {sample['local_filepath']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        if self.image_preprocess:
            image = self.image_preprocess(image)

        # For grouped (combined) samples, negatives is a list. For flat, it may be a string.
        negatives = sample.get('negatives', None)
        if negatives is not None:
            caption_options = [sample['caption']] + negatives
        elif isinstance(sample.get('negative'), str):
            caption_options = [sample['caption'], sample['negative']]
        else:
            caption_options = [sample['caption']]

        return {
            'image_options': image,
            'caption_options': caption_options,
            'label': 0,
        }


    # def evaluate(
    #     self,
    #     embedding_model,
    #     aligning_model=None,
    #     device='cuda',
    #     batch_size=16,
    #     indices=None,
    #     intermediate_text_layer_names=['final'],
    #     intermediate_image_layer_names=['final']
    #     ):
    #     """
    #     Batched and memory-efficient: loops over batches of unique caption_id,
    #     processes images and all (padded) candidate captions in parallel.

    #     Returns:
    #         results: dict with metrics,
    #         embeddings: dict with arrays:
    #             - image_embeddings: [N, D]
    #             - caption_embeddings: [N, D]
    #             - neg_caption_embeddings: [N, max_num_negs, D] (K = max #neg in whole dataset)
    #     """
    #     from collections import defaultdict
    #     import numpy as np
    #     from tqdm import tqdm

    #     # 1. Group indices by caption_id
    #     caption_id_to_indices = defaultdict(list)
    #     for idx, sample in tqdm(enumerate(self.sample_list), desc="Grouping by caption_id"):
    #         if indices is not None and idx not in indices:
    #             continue
    #         capid = sample['caption_id']
    #         caption_id_to_indices[capid].append(idx)

    #     # Prepare batches of groups
    #     group_keys = list(caption_id_to_indices.keys())
    #     N = len(group_keys)

    #     group_accuracies = {}
    #     type_counts = defaultdict(list)

    #     image_emb_list = []
    #     caption_emb_list = []
    #     neg_caption_emb_list = []
    #     group_caps = []

    #     max_num_negs_global = 0

    #     for batch_start in tqdm(range(0, N, batch_size), desc="Grouped caption_id batched eval"):
    #         batch_capids = group_keys[batch_start: batch_start + batch_size]

    #         batch_imgs = []
    #         batch_pos_caps = []
    #         batch_neg_caps = []
    #         batch_neg_types = []
    #         batch_K = []  # Number of negatives per group
    #         batch_capid_ids = []

    #         for capid in batch_capids:
    #             idxs = caption_id_to_indices[capid]
    #             if not idxs:
    #                 continue

    #             anchor_sample = self[idxs[0]]
    #             image = anchor_sample['image_options']
    #             positive = anchor_sample['caption_options'][0]

    #             negatives = []
    #             neg_types = []
    #             for idx in idxs:
    #                 sample = self[idx]
    #                 if len(sample['caption_options']) > 1:
    #                     negatives.append(sample['caption_options'][1])
    #                     neg_types.append(self.sample_list[idx].get('type', 'unknown'))
    #             K = len(negatives)
    #             batch_K.append(K)
    #             if K > max_num_negs_global:
    #                 max_num_negs_global = K

    #             batch_imgs.append(image)
    #             batch_pos_caps.append(positive)
    #             batch_neg_caps.append(negatives)
    #             batch_neg_types.append(neg_types)
    #             batch_capid_ids.append(capid)

    #         B = len(batch_imgs)
    #         if B == 0:
    #             continue

    #         # Pad negative captions within the batch
    #         K_max = max(batch_K) if batch_K else 0
    #         padded_neg_caps = []
    #         padded_neg_types = []
    #         for negs, neg_types in zip(batch_neg_caps, batch_neg_types):
    #             pad_len = K_max - len(negs)
    #             padded_negs = negs + [batch_pos_caps[0]] * pad_len if pad_len > 0 else negs  # pad with dummy (won't matter if masked)
    #             padded_neg_types_this = neg_types + ['__pad__'] * pad_len if pad_len > 0 else neg_types
    #             padded_neg_caps.append(padded_negs)
    #             padded_neg_types.append(padded_neg_types_this)

    #         # Flatten all captions in batch (positives and negatives)
    #         all_caps_flat = []
    #         for pos, negs in zip(batch_pos_caps, padded_neg_caps):
    #             all_caps_flat.append(pos)
    #             all_caps_flat.extend(negs)
    #         # For indexing: offsets for each group's positive
    #         cap_offsets = [i * (1 + K_max) for i in range(B)]

    #         with torch.no_grad():
    #             # Process images
    #             img_tensors = torch.stack(batch_imgs).to(device)
    #             img_embs = compute_image_embeddings_intermediate_batch(
    #                 img_tensors, embedding_model, device=device,
    #                 intermediate_layer_names=intermediate_image_layer_names
    #             )
    #             img_embs = img_embs['final'] if aligning_model is None else aligning_model.encode_image(img_embs)
    #             img_embs = img_embs.cpu().float()
    #             img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)  # [B, D]

    #             # Process all captions in batch
    #             cap_embs = compute_caption_embeddings_intermediate_batch(
    #                 all_caps_flat, embedding_model, device=device,
    #                 intermediate_layer_names=intermediate_text_layer_names
    #             )
    #             cap_embs = cap_embs['final'] if aligning_model is None else aligning_model.encode_text(cap_embs)
    #             cap_embs = cap_embs.cpu().float()
    #             cap_embs = cap_embs / cap_embs.norm(dim=-1, keepdim=True)  # [B * (1+K_max), D]

    #         D = img_embs.shape[-1]

    #         # For each group in batch, slice positive and negatives
    #         for i in range(B):
    #             pos_offset = i * (1 + K_max)
    #             pos_emb = cap_embs[pos_offset]                # [D]
    #             neg_embs = cap_embs[pos_offset + 1 : pos_offset + 1 + batch_K[i]]  # [K_i, D]

    #             # Compute similarities
    #             sims = torch.cat([img_embs[i].unsqueeze(0), img_embs[i].unsqueeze(0).expand_as(neg_embs)], dim=0)  # [1+K_i, D]
    #             caption_group_embs = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)  # [1+K_i, D]
    #             similarities = (img_embs[i].unsqueeze(0) @ caption_group_embs.T).squeeze(0)  # [1+K_i]
    #             correct = (similarities[0] > similarities[1:]).all().item()
    #             group_accuracies[batch_capid_ids[i]] = correct

    #             # Per-type accuracy (per negative)
    #             for idx, neg_type in enumerate(batch_neg_types[i][:batch_K[i]]):
    #                 is_correct = (similarities[0] > similarities[idx + 1]).item()
    #                 type_counts[neg_type].append(is_correct)

    #             # Store embeddings
    #             image_emb_list.append(img_embs[i].cpu().numpy())           # [D]
    #             caption_emb_list.append(pos_emb.cpu().numpy())             # [D]
    #             neg_arr = neg_embs.cpu().numpy() if batch_K[i] > 0 else np.zeros((0, D))
    #             neg_caption_emb_list.append(neg_arr)
    #             group_caps.append(batch_capid_ids[i])

    #     # --- Padding negatives to global max_num_negs ---
    #     D = image_emb_list[0].shape[-1]
    #     padded_neg_caption_embs = []
    #     for arr in neg_caption_emb_list:
    #         if arr.shape[0] < max_num_negs_global:
    #             pad = np.zeros((max_num_negs_global - arr.shape[0], D), dtype=arr.dtype)
    #             arr = np.concatenate([arr, pad], axis=0)
    #         padded_neg_caption_embs.append(arr)
    #     neg_caption_embeddings = np.stack(padded_neg_caption_embs, axis=0)   # [N, max_num_negs, D]
    #     image_embeddings = np.stack(image_emb_list, axis=0)                  # [N, D]
    #     caption_embeddings = np.stack(caption_emb_list, axis=0)              # [N, D]

    #     type_accuracies = {t: float(np.mean(accs)) for t, accs in type_counts.items()}
    #     type_counts_final = {t: len(accs) for t, accs in type_counts.items()}
    #     overall_accuracy = float(np.mean(list(group_accuracies.values()))) if group_accuracies else 0.0
    #     results = {
    #         "type_accuracies": type_accuracies,
    #         "overall_accuracy": overall_accuracy,
    #         "type_counts": type_counts_final,
    #     }
    #     embeddings = {
    #         "image_embeddings": image_embeddings,             # [N, D]
    #         "caption_embeddings": caption_embeddings,         # [N, D]
    #         "negative_caption_embeddings": neg_caption_embeddings  # [N, max_num_negs, D]
    #     }
    #     return results, embeddings

        

    def split_dataset(self, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42, split_type: str = 'random') -> dict:
        """
        Splits the dataset into a new dataset with only the specified indices.
        """
        return train_val_test_split(
            self,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_type=split_type
        )

def train_val_test_split(
    dataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    split_type: Literal['random', 'object', 'variation'] = 'random'
):
    """
    Generates train/val/test indices for CC3M.
    'random': splits at the sample level.
    'object': splits by caption_id/object, so samples of the same object can't be in multiple splits.
    """
    np.random.seed(seed)
    random.seed(seed)
    n = len(dataset)
    all_idx = list(range(n))

    if split_type == 'random':
        random.shuffle(all_idx)
        n_test = int(n * test_ratio)
        test_idx = all_idx[:n_test]
        rem_idx = all_idx[n_test:]

        adj_val = val_ratio / (1 - test_ratio)
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(rem_idx, test_size=adj_val, random_state=seed)

        return {
            'train': {'indices': train_idx},
            'val': {'indices': val_idx},
            'test': {'indices': test_idx},
        }

    elif split_type == 'object':
        # Group indices by caption_id (object)
        caption_id_to_indices = {}
        for idx, sample in enumerate(dataset.sample_list):
            capid = sample['caption_id']
            if capid not in caption_id_to_indices:
                caption_id_to_indices[capid] = []
            caption_id_to_indices[capid].append(idx)

        unique_caption_ids = list(caption_id_to_indices.keys())
        random.shuffle(unique_caption_ids)
        n_caps = len(unique_caption_ids)
        n_test = int(n_caps * test_ratio)
        n_val = int(n_caps * val_ratio)
        test_ids = unique_caption_ids[:n_test]
        val_ids = unique_caption_ids[n_test:n_test + n_val]
        train_ids = unique_caption_ids[n_test + n_val:]

        train_idx = [idx for cid in train_ids for idx in caption_id_to_indices[cid]]
        val_idx   = [idx for cid in val_ids   for idx in caption_id_to_indices[cid]]
        test_idx  = [idx for cid in test_ids  for idx in caption_id_to_indices[cid]]

        return {
            'train': {'indices': train_idx},
            'val': {'indices': val_idx},
            'test': {'indices': test_idx},
        }

    else:
        raise NotImplementedError(f"Split type '{split_type}' not supported for CC3M.")

class CC3MNeg(Dataset):
    """
    Wraps CC3MDataset to provide tokenized positives and negatives.
    For combined datasets, always returns exactly `num_negatives` negatives per sample.
    Returns (image, pos_token, neg_token, all_neg_tokens).
    """
    def __init__(
        self,
        cc3m_dataset: 'CC3MDataset',
        indices: List[int],
        num_negatives: int = 1,  # Fixed number of negatives per sample
    ):
        super().__init__()
        self.dataset = cc3m_dataset
        self.num_negatives = num_negatives
        
        # Map indices to dataset samples,
        # so that we can access them in __getitem__
        self.idx_to_dataset_idx = {i: idx for i, idx in enumerate(indices)}
        
        # Check if dataset is combined (has multiple negatives per sample)
        sample_check = self.dataset[indices[0]] if indices else None
        caption_options = sample_check['caption_options'] if sample_check else []
        self.is_combined = len(caption_options) > 2  # pos + at least 2 negatives
        
        print(f"CC3MNeg initialized with {self.num_negatives} negatives per sample")
        print(f"Dataset appears to be {'combined' if self.is_combined else 'flat'}")
    
    def __len__(self) -> int:
        return len(self.idx_to_dataset_idx)

    def _sample_negatives(self, neg_texts: List[str]) -> List[str]:
        """
        Sample exactly num_negatives from the available negatives.
        If fewer available: sample with replacement
        If more available: sample without replacement
        """
        if len(neg_texts) == 0:
            raise ValueError("No negative texts available for sampling")
        
        if len(neg_texts) >= self.num_negatives:
            # More negatives than needed: sample without replacement
            return random.sample(neg_texts, self.num_negatives)
        else:
            # Fewer negatives than needed: sample with replacement
            return random.choices(neg_texts, k=self.num_negatives)

    def __getitem__(self, idx: int):
        sample = self.dataset[self.idx_to_dataset_idx[idx]]
        pos_image = sample['image_options']
        caption_options = sample['caption_options']
        
        pos_text = caption_options[0]
        neg_texts = caption_options[1:]

        if not self.is_combined:
            # For flat datasets, we expect only one negative
            if len(neg_texts) != 1:
                print(f"Warning: Expected 1 negative for flat dataset, got {len(neg_texts)}")
            # Just repeat the single negative to match num_negatives
            sampled_neg_texts = neg_texts * self.num_negatives if neg_texts else [""] * self.num_negatives
            sampled_neg_texts = sampled_neg_texts[:self.num_negatives]  # Truncate if too many
        else:
            # For combined datasets, use the sampling strategy
            sampled_neg_texts = self._sample_negatives(neg_texts)

        # Tokenize
        pos_tok = clip.tokenize(pos_text, truncate=True).squeeze(0)
        all_neg_toks = clip.tokenize(sampled_neg_texts, truncate=True)  # Shape: [num_negatives, seq_len]

        # Select one random negative for the single neg_tok (backward compatibility)
        rand_idx = random.randint(0, len(sampled_neg_texts) - 1)
        neg_tok = all_neg_toks[rand_idx]

        return pos_image, pos_tok, neg_tok, all_neg_toks

    def collate_fn(self, batch: List[tuple]) -> dict:
        images, pos_toks, neg_toks, all_neg_toks = zip(*batch)
        
        # Stack images if preprocessed tensors
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images, dim=0)
        
        pos_toks = torch.stack(pos_toks, dim=0)  # [batch_size, seq_len]
        neg_toks = torch.stack(neg_toks, dim=0)  # [batch_size, seq_len]
        all_neg_toks = torch.stack(all_neg_toks, dim=0)  # [batch_size, num_negatives, seq_len]

        return {
            'images': images,
            'pos_tokens': pos_toks,
            'neg_token': neg_toks,  # Single negative for compatibility
            'all_neg_tokens': all_neg_toks,  # All negatives
        }
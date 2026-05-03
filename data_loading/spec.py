import json
import logging
import math
import os
import random
from collections import defaultdict
from typing import Literal, Union

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.align import (
    compute_caption_embeddings_intermediate_batch,
    compute_image_embeddings_intermediate_batch,
)

from torch.utils.data import Sampler

class GroupUniqueBatchSampler(Sampler):
    """
    Batch sampler ensuring each batch contains at most one sample per group.
    For datasets that provide both `indices` and a reference to the original dataset.
    """
    def __init__(self, dataset, batch_size, group_key='object_id', drop_last=False, shuffle=True):
        """
        Args:
            dataset: Must have `indices` and `dataset` (original) if using embedding dataset,
                     or just `sample_list` if it's the raw dataset.
            batch_size: Batch size.
            group_key: Key for grouping.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_key = group_key
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Get the sample list and index mapping:
        if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
            # For MultiLayerNegEmbeddingsDataset (or similar)
            self.indices = dataset.indices  # subset of all indices
            self.sample_list = [dataset.dataset.sample_list[i] for i in self.indices]
            self.idx_map = {idx : i for i, idx in enumerate(self.indices)}

        elif hasattr(dataset, "sample_list"):
            # For SPECNeg or original dataset
            self.indices = list(range(len(dataset)))
            self.sample_list = dataset.sample_list
            self.idx_map = None
            
        else:
            raise ValueError("Dataset must provide either indices+dataset or sample_list.")

        # Build group mapping
        self.group_to_indices = {}
        for i, sample in enumerate(self.sample_list):
            group_id = sample.get(self.group_key)
            if group_id is None:
                group_id = sample.get('query', '').split('/')[0]  # fallback: object_id from filename
            if group_id not in self.group_to_indices:
                self.group_to_indices[group_id] = []
            self.group_to_indices[group_id].append(self.indices[i])
        self.all_group_ids = list(self.group_to_indices.keys())

    def __iter__(self):
        group_ids = self.all_group_ids[:]
        if self.shuffle:
            random.shuffle(group_ids)

        print(f"Total groups: {len(group_ids)}")
        batch = []
        for group_id in group_ids:
            # Randomly select a sample from this group for the batch
            indices = self.group_to_indices[group_id]
            idx_in_sample = random.choice(indices)
            if self.idx_map is not None:
                # Map to index in full dataset (for MultiLayerNeg)
                idx_in_dataset = self.idx_map[idx_in_sample]
            else:
                idx_in_dataset = idx_in_sample
            batch.append(idx_in_dataset)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(self.all_group_ids)
        if self.drop_last:
            return n // self.batch_size
        else:
            return (n + self.batch_size - 1) // self.batch_size
        
class SPECImage2TextDataset(Dataset):
    def __init__(self,
                 data_root : Union[str,os.PathLike],
                 subset_name : str,
                 image_preprocess : callable = None,
                 **kwargs
                 ):
        """
        Args:
            subset_root: the path to the root dir of a subset, (e.g. `absolute_size`)
        """
        self.data_root = data_root
        self.image_preprocess = image_preprocess
        self.subset_root = os.path.join(data_root, subset_name)

        ann = os.path.join(self.subset_root, 'image2text.json')
        
        # Load the annotation file
        with open(ann, 'r') as f:
            self.sample_list = json.load(f)
        f.close()

        self.captions = self.get_captions()  
        self.image_paths = self.get_image_paths()
        self.number_of_candidates = len(self.sample_list[0]['keys'])

        logging.info(f"Number of unique images: {len(self.image_paths)}")
        logging.info(f"Number of unique captions: {len(self.captions)}")

        # Mapping to unique caption index to save some space for large datasets with repeated captions
        self.caption_to_idx = { caption : idx for idx, caption in enumerate(self.captions) }

    def get_captions(self):
        """
        Get all captions in the dataset
        """
        captions = []
        for sample in self.sample_list:
            captions.extend(sample['keys'])
        return sorted(set(captions))  # Remove duplicates and sort

    def get_image_paths(self):
        """
        Get all image paths in the dataset
        """
        image_paths = []
        for sample in self.sample_list:
            image_paths.append(os.path.join(self.subset_root, sample['query']))
        return image_paths

    def get_idx_to_ptr(self, idx : int):
        """
        Get a mapping from original index of the caption in the dataset to the index in the caption_to_idx dictionary
        """
        sample = self.sample_list[idx]
        label = sample['label']
        
        caption = sample['keys'][label]
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
        candidates = sample['keys']
            
        # Map captions to their indices in the caption dictionary
        return [self.caption_to_idx[caption] for i,caption in enumerate(candidates) if i != sample['label']]
    
    def get_idx_to_candidates_indices(self, idx: int):
        """
        Get a mapping from image index to caption indices
        
        Args:
            idx: Index of the image
        
        Returns:
            List of indices of candidate captions in the sample_list
        """
        sample = self.sample_list[idx]
        candidates = sample['keys']

        # Get the candidates. For this dataset negatives appears after positives
        # idx = base + label where base = idx//n_candidates * n_candidates
        base = idx // self.number_of_candidates * self.number_of_candidates 
        negative_indices = [base + i for i in range(len(candidates)) if base + i != idx]

        return negative_indices
    
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_info = self.sample_list[idx]

        # query image
        image_path = os.path.join(self.subset_root, sample_info['query'])
        query_image = Image.open(image_path).convert('RGB')
        if self.image_preprocess is not None:
            query_image = self.image_preprocess(query_image)

        # candidate texts
        candidate_texts = sample_info['keys']
        label = sample_info['label']

        sample = {
            "image_options": query_image,
            "caption_options": candidate_texts,
            "label": label,
        }

        return sample

    def collate_fn(self, batch):
        query_image = []
        candidate_texts = []
        labels = []
        for sample in batch:
            query_image.append(sample['query_image'])
            candidate_texts.append(sample['candidate_texts'])
            labels.append(sample['label'])

        if self.image_preprocess is not None:
            query_image = torch.stack(query_image, dim=0)
        batch = {
            'image_options': query_image,
            'caption_options': candidate_texts,
            'label': labels,
        }
        return batch
    
    def split_dataset(self, val_ratio=0.1, test_ratio=0.1, seed=42, split_type: Literal['random', 'object', 'variation'] = 'random'):
        """
        Split the dataset into train, validation and test sets.
        
        Args:
            val_ratio: portion of validation samples (relative to the remaining data after test split)
            test_ratio: portion of test samples (of the whole dataset)
            seed: random seed for reproducibility
            split_type: strategy to split the remaining data into train and val; options are 'random', 'object', 'variation'
        """
        return train_val_test_split(
            dataset=self,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_type=split_type
        )
        

    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader to handle SPEC samples.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and caption lists
        """
        batch_images = []
        batch_caption_options = []
        batch_labels = []
        
        for sample in batch:
            # Extract components from each sample
            image = sample['image_options']  # Single image tensor
            caption_options = sample['caption_options']  # List of captions
            label = sample['label']  # Integer label
            
            batch_images.append(image)
            batch_caption_options.append(caption_options)
            batch_labels.append(label)
        
        # Stack images into a batch tensor
        batch_images = torch.stack(batch_images)  # [B, C, H, W]
        
        return {
            'images': batch_images,
            'caption_options': batch_caption_options,  # List[List[str]] - preserves structure
            'labels': batch_labels  # List[int]
        }

    def evaluate(
        self,
        embedding_model,
        aligning_model=None,
        device='cuda',
        batch_size=64,
        indices=None,
        intermediate_text_layer_names=['final'],
        intermediate_image_layer_names=['final'],
    ):
        """
        Object-grouped evaluation with caching and batch processing.

        For each object group (object_id = "<object_id>/..."):
        
        GROUP-LEVEL METRICS (harsh - all samples in object group must be correct):
        - Text contrastive accuracy (i→t, group): for each image b, its positive caption
            must score higher than ALL other captions in the SAME OBJECT group (BK pool).
        - Image contrastive accuracy (t→i, group): for each positive caption (b, label[b]),
            it must score higher than ALL other images in the SAME OBJECT group (B pool).
        - Group contrastive accuracy: both of the above true for the (image b, its pos caption).
        - Object contrastive accuracy: all samples in the object group are group-correct.
        
        INDIVIDUAL-LEVEL METRICS (less harsh - each sample vs its own negatives):
        - Individual text contrastive accuracy: for each image, its positive caption beats 
            its own negative captions (not the entire object group).
        - Individual image contrastive accuracy: for each positive caption, it beats other
            images in the same object group.
        - Individual group contrastive accuracy: both individual metrics true for each sample.

        Returns:
        results: {
            # Group-level (harsh)
            text_contrastive_accuracy, image_contrastive_accuracy,
            group_contrastive_accuracy, object_contrastive_accuracy,
            # Individual-level (less harsh)  
            individual_text_contrastive_accuracy, individual_image_contrastive_accuracy,
            individual_group_contrastive_accuracy,
            num_objects, num_samples
        }
        embeddings: {
            image_embeddings:  (N, D)  # order follows iterated object groups then within-group order
            caption_embeddings:(N, D)
            negative_caption_embeddings: (N, K-1, D)
        }
        """
        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")
        
        from collections import defaultdict

        # --- 1) Build object groups (respecting indices filter) ---
        object_to_indices = defaultdict(list)
        if indices is None:
            candidate_range = range(len(self.sample_list))
        else:
            selected = set(indices)
            candidate_range = [i for i in range(len(self.sample_list)) if i in selected]

        for idx in candidate_range:
            image_path = self.sample_list[idx]['query']
            object_id = image_path.split('/')[0]
            object_to_indices[object_id].append(idx)

        # Keep only non-empty groups
        object_ids = [oid for oid in sorted(object_to_indices.keys()) if len(object_to_indices[oid]) > 0]

        all_text_correct = []
        all_image_correct = []
        all_group_correct = []
        all_object_correct = []
        
        # Individual sample-level accuracies (less harsh)
        all_individual_text_correct = []
        all_individual_image_correct = []
        all_individual_group_correct = []

        # Store embeddings (CPU floats) + sample indices in the exact order evaluated
        img_bank = []
        pos_cap_bank = []
        neg_cap_bank = []
        idx_bank = []

        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="SPEC",
            subset_name=os.path.basename(self.subset_root),
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:

            object_sample_count = 0  # Track total samples processed for cache indexing
            
            for object_id in tqdm(object_ids, desc="Evaluate (by object)"):
                group_indices = object_to_indices[object_id]
                if not group_indices:
                    continue

                # --- 2) Build a batch for this object group ---
                images = []
                captions_list = []
                labels = []
                for idx in group_indices:
                    item = self[idx]
                    images.append(item['image_options'])          # tensor (C,H,W) already preprocessed by dataset
                    captions_list.append(item['caption_options']) # list length K
                    labels.append(item['label'])                  # int in [0, K)
                B = len(images)
                K = len(captions_list[0])
                assert all(len(caps) == K for caps in captions_list), "Inconsistent #candidates within the object group."

                images_tensor = torch.stack(images, dim=0)  # (B, C, H, W)
                flat_captions = [cap for caps in captions_list for cap in caps]  # length = B*K

                with torch.no_grad():
                    # ---- 3) Encode & L2-normalize with caching ----
                    # Get image embeddings (with caching)
                    img_embs = cache.get_or_compute_embeddings(
                        images_tensor.to(device),
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=object_sample_count  # Careful indexing for cache
                    )
                    
                    # Get caption embeddings (with caching)
                    cap_embs = cache.get_or_compute_embeddings(
                        flat_captions,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=object_sample_count * K  # B samples * K captions each
                    )
                    
                    # Normalize embeddings
                    img_embs = img_embs.float().to(device)
                    img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)       # (B, D)
                    
                    cap_embs = cap_embs.float().to(device)
                    cap_embs = cap_embs / cap_embs.norm(dim=-1, keepdim=True)       # (B*K, D)

                    D = img_embs.shape[-1]
                    cap_embs_view = cap_embs.view(B, K, D)                               # (B, K, D)

                    labels_tensor = torch.tensor(labels, device=img_embs.device, dtype=torch.long)  # (B,)

                    # Split pos / neg structures for embedding outputs
                    pos_cap_embs = cap_embs_view[torch.arange(B, device=img_embs.device), labels_tensor]  # (B, D)
                    neg_mask = torch.ones_like(cap_embs_view, dtype=torch.bool)
                    neg_mask[torch.arange(B, device=img_embs.device), labels_tensor] = False
                    neg_cap_embs = cap_embs_view[neg_mask].view(B, K - 1, D)             # (B, K-1, D)

                    # ---- 4) Object-level contrastive checks ----
                    # i -> t (group): compare each image b against ALL captions in the object group (BK pool)
                    group_cap_bank = cap_embs                                          # (BK, D)
                    sims_i2t = img_embs @ group_cap_bank.T                             # (B, BK)
                    
                    # t -> i (group): for each positive caption (row in BK), compare against ALL images in the object group (B pool)
                    sims_t2i = group_cap_bank @ img_embs.T                             # (BK, B)
                    
                    pos_flat_idx = torch.arange(B, device=img_embs.device) * K + labels_tensor  # (B,)

                # ---- 4) Object-level contrastive checks ----
                # Mask out each row's own positive to compute max over "others"
                # Build a mask of shape (B, BK) with True everywhere except the positive index
                others_mask = torch.ones_like(sims_i2t, dtype=torch.bool)
                others_mask[torch.arange(B, device=sims_i2t.device), pos_flat_idx] = False
                # Get pos scores and max of others per row
                pos_scores_i2t = sims_i2t[torch.arange(B, device=sims_i2t.device), pos_flat_idx]           # (B,)
                # Set the excluded position to -inf and take max
                sims_i2t_masked = sims_i2t.masked_fill(~others_mask, float('-inf'))
                max_others_i2t, _ = sims_i2t_masked.max(dim=1)                                             # (B,)
                text_correct = (pos_scores_i2t > max_others_i2t).tolist()

                # t -> i (group): for each positive caption (row in BK), compare against ALL images in the object group (B pool)
                # For each sample b, grab the row of its positive caption
                pos_rows = sims_t2i[pos_flat_idx]                                                          # (B, B)
                # Exclude the diagonal (its own image)
                others_mask_t2i = torch.ones_like(pos_rows, dtype=torch.bool)
                others_mask_t2i[torch.arange(B, device=sims_i2t.device), torch.arange(B, device=sims_i2t.device)] = False
                pos_scores_t2i = pos_rows[torch.arange(B, device=sims_i2t.device), torch.arange(B, device=sims_i2t.device)]  # (B,)
                pos_rows_masked = pos_rows.masked_fill(~others_mask_t2i, float('-inf'))
                max_others_t2i, _ = pos_rows_masked.max(dim=1)                                              # (B,)
                image_correct = (pos_scores_t2i > max_others_t2i).tolist()

                group_correct = [t and i for t, i in zip(text_correct, image_correct)]
                object_correct = all(group_correct)

                # ---- Individual sample-level contrastive accuracy (less harsh) ----
                # For each individual sample, check if its positive caption beats its own negatives
                individual_text_correct = []
                individual_image_correct = []
                
                for b in range(B):
                    # Individual text contrastive: image b's positive caption vs image b's negative captions
                    sample_pos_score = sims_i2t[b, b * K + labels[b]]  # similarity with its own positive
                    sample_neg_start = b * K
                    sample_neg_end = (b + 1) * K
                    sample_neg_scores = torch.cat([
                        sims_i2t[b, sample_neg_start:sample_neg_start + labels[b]],  # negatives before positive
                        sims_i2t[b, sample_neg_start + labels[b] + 1:sample_neg_end]  # negatives after positive
                    ])
                    
                    if len(sample_neg_scores) > 0:
                        individual_text_correct.append((sample_pos_score > sample_neg_scores.max()).item())
                    else:
                        individual_text_correct.append(True)  # No negatives means correct by default
                    
                    # Individual image contrastive: positive caption b vs other images in the same object
                    pos_cap_row = b * K + labels[b]  # Row index of positive caption in BK matrix
                    cap_pos_score = sims_t2i[pos_cap_row, b]  # similarity with its own image
                    cap_other_scores = torch.cat([
                        sims_t2i[pos_cap_row, :b],  # other images before this one
                        sims_t2i[pos_cap_row, b + 1:]  # other images after this one
                    ])
                    
                    if len(cap_other_scores) > 0:
                        individual_image_correct.append((cap_pos_score > cap_other_scores.max()).item())
                    else:
                        individual_image_correct.append(True)  # No other images means correct by default

                individual_group_correct = [t and i for t, i in zip(individual_text_correct, individual_image_correct)]

                all_text_correct.extend(text_correct)
                all_image_correct.extend(image_correct)
                all_group_correct.extend(group_correct)
                all_object_correct.append(object_correct)
                
                # Store individual accuracies
                all_individual_text_correct.extend(individual_text_correct)
                all_individual_image_correct.extend(individual_image_correct)
                all_individual_group_correct.extend(individual_group_correct)

                # ---- 5) Collect embeddings for return (move to CPU) ----
                img_bank.append(img_embs.detach().cpu())                       # (B, D)
                pos_cap_bank.append(pos_cap_embs.detach().cpu())               # (B, D)
                neg_cap_bank.append(neg_cap_embs.detach().cpu())               # (B, K-1, D)
                idx_bank.extend(group_indices)
                
                # Update sample count for next object group's cache indexing
                object_sample_count += B

        image_embeddings = torch.cat(img_bank, dim=0)           # (N, D)
        pos_caption_embeddings = torch.cat(pos_cap_bank, dim=0)     # (N, D)
        neg_caption_embeddings = torch.cat(neg_cap_bank, dim=0) # (N, K-1, D)

        results = {
            # Group-level accuracies (harsh - all samples in object group must be correct)
            "group_text_contrastive_accuracy": float(np.mean(all_text_correct)),
            "group_image_contrastive_accuracy": float(np.mean(all_image_correct)),
            "group_group_contrastive_accuracy": float(np.mean(all_group_correct)),
            "object_contrastive_accuracy": float(np.mean(all_object_correct)),
            "text_contrastive_accuracy": float(np.mean(all_individual_text_correct)),
            "image_contrastive_accuracy": float(np.mean(all_individual_image_correct)),
            "group_contrastive_accuracy": float(np.mean(all_individual_group_correct)),
            
        }
        embeddings = {
            "image_embeddings": image_embeddings,                       # (N, D)
            "caption_embeddings": pos_caption_embeddings,               # (N, D)
            "negative_caption_embeddings": neg_caption_embeddings,      # (N, K-1, D)
        }
        return results, embeddings


    
def train_val_test_split(dataset : 'SPECImage2TextDataset', 
                         val_ratio=0.1, 
                         test_ratio=0.1,
                         seed=42, 
                         split_type: Literal['random', 'object', 'variation'] = 'random'):
    """
    Returns a dictionary with index lists and corresponding labels for train, val, and test splits.
    
    The test split is fixed and computed first, ensuring it remains the same regardless 
    of the splitting strategy used for train and val.
    
    Args:
        dataset: a dataset class (e.g., Image2TextDataset or Text2ImageDataset)
        val_ratio: portion of validation samples (relative to the remaining data after test split)
        test_ratio: portion of test samples (of the whole dataset)
        seed: random seed for reproducibility
        split_type: strategy to split the remaining data into train and val; options are 'random', 'object', 'variation'
    
    Returns:
        A dictionary with keys 'train', 'val', and 'test'. Each key maps to another dictionary containing:
            - 'indices': the list of indices for that split,
            - 'labels': corresponding ground truth labels,
            - 'neg_labels': negative labels for contrastive training.
    """

    # Set seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    total_indices = list(range(len(dataset)))
    
    # --- Step 1: Fix the Test Set ---
    # Compute number of test samples based on the total dataset length.
    n_total = len(dataset)
    n_test = int(n_total * test_ratio)
    
    # Shuffle the total indices with the fixed seed
    shuffled_indices = total_indices.copy()
    random.shuffle(shuffled_indices)
    
    # Select fixed test indices (sorted for consistency if needed)
    fixed_test_idx = sorted(shuffled_indices[:n_test])
    
    # --- Step 2: Prepare the Remaining Indices ---
    remaining_idx = [i for i in total_indices if i not in fixed_test_idx]

    # --- Step 3: Split the Remaining Data for Train and Val ---
    if split_type == 'random':
        # For random split, simply use train_test_split on the remaining indices
        adjusted_val_ratio = val_ratio / (1 - test_ratio)  # Adjust the ratio relative to remaining samples.
        train_idx, val_idx = train_test_split(remaining_idx, test_size=adjusted_val_ratio, random_state=seed)

        # Find the indexes where 
    elif split_type == 'object':
        # Group remaining indices by object ID extracted from the image paths
        object_to_indices = defaultdict(list)
        
        for idx in remaining_idx:
            sample = dataset.sample_list[idx]
            # Get image path depending on dataset type  
            if 'query' in sample and sample['query'].endswith(('.jpg', '.png')):
                image_path = sample['query']
            elif 'keys' in sample and isinstance(sample['keys'][0], str) and sample['keys'][0].endswith(('.jpg', '.png')):
                image_path = sample['keys'][sample['label']]
            else:
                raise ValueError("Couldn't find an image path in sample.")
            
            object_id = image_path.split('/')[0]
            object_to_indices[object_id].append(idx)
        
        # Shuffle the object ids
        object_ids = list(object_to_indices.keys())
        random.shuffle(object_ids)
        
        n_objects = len(object_ids)
        n_val_objects = int(n_objects * val_ratio)
        n_train_objects = n_objects - n_val_objects
        
        train_obj_ids = object_ids[:n_train_objects]
        val_obj_ids = object_ids[n_train_objects:]
        
        # Flatten object indices into train and val lists
        train_idx = [i for obj in train_obj_ids for i in object_to_indices[obj]]
        val_idx = [i for obj in val_obj_ids for i in object_to_indices[obj]]
    
    elif split_type == 'variation':
        # Split by labels (assuming labels are from 0 to n_candidates-1) on the remaining indices.
        n_candidates = dataset.number_of_candidates
        # Get unique labels and shuffle them
        unique_labels = list(range(n_candidates))
        random.shuffle(unique_labels)
        
        n_labels = len(unique_labels)
        n_val_labels = max(1, math.ceil(n_labels * val_ratio)) if val_ratio > 0 else 0
        n_train_labels = n_labels - n_val_labels
        
        train_labels_unique = unique_labels[:n_train_labels]
        val_labels_unique = unique_labels[n_train_labels:]
        
        # Partition the remaining indices by matching label values.
        train_idx = [i for i in remaining_idx if dataset.sample_list[i]['label'] in train_labels_unique]
        val_idx = [i for i in remaining_idx if dataset.sample_list[i]['label'] in val_labels_unique]
    
    else:
        raise ValueError("Invalid split type. Choose from ['random', 'object', 'variation']")
        
    return {
        'train': {'indices': train_idx},
        'val':   {'indices': val_idx},
        'test':  {'indices': fixed_test_idx},
    }
    
class SPECNeg(Dataset):
    def __init__(self, 
                 dataset : Dataset, 
                 indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
    
        # index to indices mapping
        self.idx_to_indices = {idx: i for idx, i in enumerate(indices)} 

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_info = self.dataset[self.idx_to_indices[idx]]

        # query image
        label = sample_info['label']

        pos_image = sample_info['image_options']
        pos_text = sample_info['caption_options'][label]
        neg_texts = sample_info['caption_options']
                
        neg_texts = neg_texts.copy()  # Create a copy of the list to avoid modifying the original
        neg_texts.pop(label)  # Remove the positive text from the list
        neg_text = neg_texts[np.random.randint(0, len(neg_texts))]

        # Tokenize both positive and negative captions
        tokenized_caption = clip.tokenize(pos_text).squeeze(0)
        neg_tokenized_caption = clip.tokenize(neg_text).squeeze(0)

        # tokenize all negative captions
        neg_tokenized_captions = clip.tokenize(neg_texts).squeeze(0)

        return pos_image, tokenized_caption, neg_tokenized_caption, neg_tokenized_captions
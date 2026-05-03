import json
import logging
import os
import random
from collections import defaultdict
from typing import List, Optional, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import clip
import numpy as np
import torch
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, load_from_disk
from utils.align import (
    compute_caption_embeddings_intermediate_batch,
    compute_image_embeddings_intermediate_batch,
)
from utils.evaluate import get_results_i2t

from torch.utils.data import Sampler

class SugarCrepeDataset(Dataset):
    """
    - id: Id for data instance in a file.
    - filename: Image filename from MS-COCO validation set.
    - caption: Positive caption that describes the image.
    - negative_caption: Negative caption that does not describe the image.
    """
    def __init__(
            self,
            data_root: Optional[Union[str, os.PathLike]] = None,
            subset_name: str = None,
            coco_root: Optional[Union[str, os.PathLike]] = None,
            image_preprocess: callable = None,
            **kwargs
        ):
        self.image_preprocess = image_preprocess

        assert subset_name in [
            'add_att', 'add_obj', 'replace_att', 'replace_obj', 'replace_rel', 'swap_att', 'swap_obj'
        ], f"Invalid subset_name: {subset_name}."

        self.subset_name = subset_name
        assert data_root is not None, "data_root must be provided to load the dataset."
        json_path = os.path.join(data_root, f'{subset_name}.json')
        assert os.path.exists(json_path), f"Dataset file {subset_name}.json does not exist in {data_root}."

        with open(json_path, 'r') as f:
            data = json.load(f)
            self.sample_list = []
            for k, v in data.items():
                v['id'] = k
                self.sample_list.append(v)

        assert coco_root is not None, "coco_root must be provided to load images from COCO dataset."
        assert os.path.exists(os.path.join(coco_root, 'val2017')), f"COCO root directory {coco_root} does not contain 'val2017' subdirectory."

        for sample in self.sample_list:
            filename = sample['filename']
            image_path = os.path.join(coco_root, 'val2017', f'{filename}')
            sample['image_path'] = image_path
            if not os.path.exists(image_path):
                logging.warning(f"Image path {image_path} does not exist. Skipping sample with id {filename}.")

        self.captions = self.get_captions()
        self.num_negatives = 1
        self.number_of_candidates = self.num_negatives + 1  # +1 for the anchor caption
        logging.info(f"Number of unique captions: {len(self.captions)}")
        self.caption_to_idx = {caption: idx for idx, caption in enumerate(self.captions)}

    def get_captions(self):
        """
        Get all captions in the dataset
        """
        captions = []
        for sample in self.sample_list:
            captions.append(sample['negative_caption'])
            captions.append(sample['caption'])
        return sorted(set(captions))

    def get_image_paths(self):
        """
        Get all image paths in the dataset
        """
        image_paths = []
        for sample in self.sample_list:
            image_path = sample['image_path']
            image_paths.append(image_path)
        return sorted(set(image_paths))
    

    def get_idx_to_ptr(self, idx : int):
        """
        Get a mapping from original index of the caption in the dataset to the index in the caption_to_idx dictionary
        """
        caption = self.sample_list[idx]['caption']
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
        candidates = [sample['negative_caption']]

        if len(candidates) < self.num_negatives:
            # If there are not enough candidates, repeat some to fill the required number
            candidates = candidates + np.random.choice(
                candidates,
                size=self.num_negatives - len(candidates),
                replace=True
            ).tolist()
        elif len(candidates) > self.num_negatives:
            # If there are too many candidates, randomly sample the required number
            candidates = random.sample(candidates, self.num_negatives)

        # Map captions to their indices in the caption dictionary
        return [self.caption_to_idx[caption] for caption in candidates]
    
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_info = self.sample_list[idx]
        query_image = Image.open(sample_info['image_path']).convert('RGB')
        if self.image_preprocess is not None:
            query_image = self.image_preprocess(query_image)
        sample = {
            "image_options": query_image,
            "caption_options": [sample_info['caption'], sample_info['negative_caption']],
            "label": 0,  # Label is not used in evaluation, but can be useful for training
        }
        return sample

    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader to handle SugarCrepe samples.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and grouped captions
        """
        batch_images = []
        batch_pos_captions = []
        batch_neg_captions = []
        
        for sample in batch:
            # Extract components from each sample
            image = sample['image_options']  # Already preprocessed tensor
            pos_caption = sample['caption_options'][0]  # Positive caption
            neg_caption = sample['caption_options'][1]  # Negative caption
            
            batch_images.append(image)
            batch_pos_captions.append(pos_caption)
            batch_neg_captions.append(neg_caption)
        
        # Stack images into a batch tensor
        batch_images = torch.stack(batch_images)  # [B, C, H, W]
        
        return {
            'images': batch_images,
            'pos_captions': batch_pos_captions,  # List[str]
            'neg_captions': batch_neg_captions   # List[str]
        }

    def evaluate(
            self,
            embedding_model,
            aligning_model=None,
            device='cuda',
            batch_size=64,
            indices=None,
            intermediate_text_layer_names=[],
            intermediate_image_layer_names=[]):
        
        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")
        
        from torch.utils.data import Subset
        
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
            
        results_cap = []
        results_neg = []

        # Collect for return
        all_img_embs = []
        all_cap_embs = []
        all_neg_embs = []

        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="SugarCrepe",
            subset_name=self.subset_name,
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:

            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Evaluating SugarCrepe")):
                # Batch is already properly formatted by collate_fn
                batch_images = batch['images']  # [B, C, H, W]
                batch_pos_captions = batch['pos_captions']  # List[str]
                batch_neg_captions = batch['neg_captions']  # List[str]
                B = len(batch_pos_captions)
                
                if B == 0:
                    continue

                with torch.no_grad():
                    # Compute binary similarity (positive vs negative caption)
                    score_cap, score_neg, img_embs, cap_embs, neg_embs = cache.compute_tqa_binary_similarity(
                        batch_images.to(device),
                        batch_pos_captions,
                        batch_neg_captions,
                        compute_image_embeddings_intermediate_batch,
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        intermediate_text_layer_names,
                    )
                    
                    # Ensure on CPU and float
                    score_cap = score_cap.cpu().float()
                    score_neg = score_neg.cpu().float()
                    img_embs = img_embs.cpu().float()
                    cap_embs = cap_embs.cpu().float()
                    neg_embs = neg_embs.cpu().float()

                results_cap.append(score_cap)
                results_neg.append(score_neg)

                all_img_embs.append(img_embs)
                all_cap_embs.append(cap_embs)
                all_neg_embs.append(neg_embs.unsqueeze(1))  # Shape [B, 1, D]

        scores_cap = torch.cat(results_cap).numpy()
        scores_neg = torch.cat(results_neg).numpy()
        correct = scores_cap > scores_neg
        acc = correct.mean()
        results = {
            "text_contrastive_accuracy": float(acc)
        }
        embeddings = {
            "image_embeddings": torch.cat(all_img_embs, dim=0),        # (N, D)
            "caption_embeddings": torch.cat(all_cap_embs, dim=0),       # (N, D)
            "negative_caption_embeddings": torch.cat(all_neg_embs, dim=0),           # (N, 1, D)
        }
        return results, embeddings

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
    np.random.seed(seed)
    random.seed(seed)
    n = len(dataset)

    if split_type == 'object':
        # Group samples by image ID
        id_to_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.sample_list):
            id_to_indices[sample['id']].append(idx)
        
        groups = list(id_to_indices.values())
        random.shuffle(groups)

        n_groups = len(groups)
        n_test_groups = int(n_groups * test_ratio)
        n_val_groups = int(n_groups * val_ratio)

        test_groups = groups[:n_test_groups]
        val_groups = groups[n_test_groups:n_test_groups + n_val_groups]
        train_groups = groups[n_test_groups + n_val_groups:]

        test_idx = [i for group in test_groups for i in group]
        val_idx = [i for group in val_groups for i in group]
        train_idx = [i for group in train_groups for i in group]

        return {
            'train': {'indices': train_idx},
            'val': {'indices': val_idx},
            'test': {'indices': test_idx},
        }

    elif split_type == 'random':
        all_idx = list(range(n))
        random.shuffle(all_idx)
        n_test = int(n * test_ratio)
        test_idx = all_idx[:n_test]
        rem_idx = all_idx[n_test:]
        adj_val = val_ratio / (1 - test_ratio)
        train_idx, val_idx = train_test_split(rem_idx, test_size=adj_val, random_state=seed)

        return {
            'train': {'indices': train_idx},
            'val': {'indices': val_idx},
            'test': {'indices': test_idx},
        }
    
    else:
        raise NotImplementedError(f"Split type '{split_type}' not supported for this dataset.")

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
from torch.utils.data import Dataset

from datasets import load_dataset, load_from_disk
from utils.align import (
    compute_caption_embeddings_intermediate_batch,
    compute_image_embeddings_intermediate_batch,
)
from utils.evaluate import get_results_i2t

from torch.utils.data import Sampler, DataLoader

SUGARCREPE_PP_NAME = "Aman-J/SugarCrepe_pp"
class SugarCrepePPDataset(Dataset):
    """
    - id: Id for data instance in a file.
    - filename: Image filename from MS-COCO validation set.
    - caption: Positive caption that describes the image.
    - caption2: Second positive caption.
    - negative_caption: Negative caption.
    """
    def __init__(
            self,
            subset_name: str = None,
            coco_root: Optional[Union[str, os.PathLike]] = None,
            image_preprocess: callable = None,
            **kwargs
        ):
        self.image_preprocess = image_preprocess
        assert subset_name in ['swap_object', 'swap_atribute', 'replace_object', 'replace_attribute', 'replace_relation'], \
            f"Invalid subset_name: {subset_name}."
        
        dataset = load_dataset(SUGARCREPE_PP_NAME, subset_name, split='train')
        self.sample_list = [dict(sample) for sample in dataset]  # Now mutable list of dicts
        self.subset_name = subset_name

        assert coco_root is not None, "coco_root must be provided to load images from COCO dataset."
        assert os.path.exists(os.path.join(coco_root, 'val2017')), f"COCO root directory {coco_root} does not contain 'val2017' subdirectory."

        for sample in self.sample_list:
            filename = sample['filename']
            image_path = os.path.join(coco_root, 'val2017', f'{filename}')
            sample['image_path'] = image_path

        # Don't expand - keep original structure with caption, caption2, negative_caption per sample
        # Each sample has: id, filename, image_path, caption, caption2, negative_caption


        self.captions = self.get_captions()
        self.num_negatives = 1
        self.number_of_candidates = self.num_negatives + 1  # +1 for the anchor caption
        self.caption_to_idx = {caption: idx for idx, caption in enumerate(self.captions)}


    def get_captions(self):
        """
        Get all unique captions in the dataset (caption, caption2, negative_caption)
        """
        captions = []
        for sample in self.sample_list:
            captions.append(sample['caption'])
            captions.append(sample['caption2'])
            captions.append(sample['negative_caption'])
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
            "caption_options": [sample_info['caption'], sample_info['caption2'], sample_info['negative_caption']],
            "label": 0,  # Label is not used in evaluation, but can be useful for training
        }
        return sample

    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader to handle SugarCrepe PP samples.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and grouped captions
        """
        batch_images = []
        batch_cap1 = []
        batch_cap2 = []
        batch_neg = []
        
        for sample in batch:
            # Extract components from each sample
            image = sample['image_options']  # Already preprocessed tensor
            cap1 = sample['caption_options'][0]  # First positive caption
            cap2 = sample['caption_options'][1]  # Second positive caption  
            neg = sample['caption_options'][2]   # Negative caption
            
            batch_images.append(image)
            batch_cap1.append(cap1)
            batch_cap2.append(cap2)
            batch_neg.append(neg)
        
        # Stack images into a batch tensor
        batch_images = torch.stack(batch_images)  # [B, C, H, W]
        
        return {
            'images': batch_images,
            'cap1': batch_cap1,  # List[str]
            'cap2': batch_cap2,  # List[str]  
            'neg': batch_neg     # List[str]
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

        results_cap1, results_cap2, results_neg = [], [], []

        # --- Collect embeddings as well ---
        img_emb_list = []
        cap1_emb_list = []
        cap2_emb_list = []
        neg_emb_list = []

        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="SugarCrepe_PP",
            subset_name=self.subset_name,
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
        
            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Evaluating SugarCrepe++")):
                # Batch is already properly formatted by collate_fn
                batch_images = batch['images']  # [B, C, H, W]
                batch_cap1 = batch['cap1']      # List[str]
                batch_cap2 = batch['cap2']      # List[str]
                batch_neg = batch['neg']        # List[str]
                B = len(batch_cap1)
                
                if B == 0:
                    continue

                with torch.no_grad():
                    # Build caption options as [[cap1, cap2, neg], ...]
                    caption_options = [[c1, c2, n] for c1, c2, n in zip(batch_cap1, batch_cap2, batch_neg)]
                    
                    # Compute multichoice similarity
                    # Returns: similarity_matrix [B, 3], img_embs [B, D], txt_embs [B, 3, D]
                    similarity_matrix, img_embs, txt_embs = cache.compute_tqa_multichoice_similarity(
                        batch_images.to(device),
                        caption_options,
                        compute_image_embeddings_intermediate_batch,
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        intermediate_text_layer_names
                    )
                    
                    # Extract scores for each caption type
                    score1 = similarity_matrix[:, 0]    # sim(img, caption)
                    score2 = similarity_matrix[:, 1]    # sim(img, caption2)
                    score_neg = similarity_matrix[:, 2]  # sim(img, negative_caption)
                    
                    # Extract embeddings
                    cap1_embs = txt_embs[:, 0, :]  # [B, D]
                    cap2_embs = txt_embs[:, 1, :]  # [B, D]
                    neg_embs = txt_embs[:, 2, :]   # [B, D]
                    
                    # Convert to CPU for consistency
                    img_embs = img_embs.cpu().float()
                    cap1_embs = cap1_embs.cpu().float()
                    cap2_embs = cap2_embs.cpu().float()
                    neg_embs = neg_embs.cpu().float()
                    score1 = score1.cpu()
                    score2 = score2.cpu()
                    score_neg = score_neg.cpu()

                results_cap1.append(score1)
                results_cap2.append(score2)
                results_neg.append(score_neg)

                # Store embeddings
                img_emb_list.append(img_embs.numpy())
                cap1_emb_list.append(cap1_embs.numpy())
                cap2_emb_list.append(cap2_embs.numpy())
                neg_emb_list.append(neg_embs.numpy())

        scores_cap1 = torch.cat(results_cap1).numpy()
        scores_cap2 = torch.cat(results_cap2).numpy()
        scores_neg = torch.cat(results_neg).numpy()

        correct1 = scores_cap1 > scores_neg
        correct2 = scores_cap2 > scores_neg

        acc1 = correct1.mean()
        acc2 = correct2.mean()
        overall_acc = (correct1.sum() + correct2.sum()) / (2 * len(correct1))
        contrastive_acc = np.logical_and(correct1, correct2).mean()

        results = {
            "accuracy_caption1": float(acc1),
            "accuracy_caption2": float(acc2),
            "accuracy_overall": float(overall_acc),
            "text_contrastive_accuracy": float(contrastive_acc)
        }

        # Stack embeddings
        embeddings = {
            "image_embeddings": np.concatenate(img_emb_list, axis=0),                # [N, D]
            "caption_embeddings": np.concatenate(cap1_emb_list, axis=0),            # [N, D]
            "caption2_embeddings": np.concatenate(cap2_emb_list, axis=0),            # [N, D]
            "negative_caption_embeddings": np.concatenate(neg_emb_list, axis=0)[:, None, :]    # [N, 1, D]
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


class SugarCrepeNeg(Dataset):
    def __init__(self, dataset: 'SugarCrepePPDataset', indices: List[int]):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.idx_to_indices = {idx: i for idx, i in enumerate(indices)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_info = self.dataset[self.idx_to_indices[idx]]
        pos_image = sample_info['image_options']
        caption_options = sample_info['caption_options']

        # In SugarCrepe: positive caption at 0, negative caption at 1
        pos_text = caption_options[0]
        neg_text = caption_options[1]

        # Tokenize
        tokenized_caption = clip.tokenize(pos_text, truncate=True).squeeze(0)
        neg_tokenized_caption = clip.tokenize(neg_text, truncate=True).squeeze(0)

        # For compatibility, negative captions as a batch of 1
        neg_tokenized_captions = clip.tokenize([neg_text], truncate=True).squeeze(0)

        return pos_image, tokenized_caption, neg_tokenized_caption, neg_tokenized_captions


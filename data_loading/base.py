"""
Base dataset classes for CLIP fine-tuning.
Simplified version with only FT-relevant classes.
"""

import abc
import os
from torch.utils.data import Dataset 
import numpy as np
import torch
from PIL import Image
import logging
from typing import Optional


def dict_of_tensors_to_dtype(d, dtype):
    """
    Recursively convert all tensors in a (possibly nested) dict to the specified dtype.
    """
    return {k: v.to(dtype) for k, v in d.items()}


def standardize_negatives_to_fixed_count(neg_indices_global, target_count=3):
    """
    Standardize negative indices to always return exactly target_count negatives.
    
    Args:
        neg_indices_global: List of negative indices (can be 1, 2, or 3 elements)
        target_count: Target number of negatives (default: 3)
    
    Returns:
        List of exactly target_count negative indices
    """
    if len(neg_indices_global) == 0:
        raise ValueError("No negative samples available for this item.")
    
    if len(neg_indices_global) >= target_count:
        return list(neg_indices_global[:target_count])
    else:
        if len(neg_indices_global) == 1:
            return [neg_indices_global[0]] * target_count
        elif len(neg_indices_global) == 2:
            result = list(neg_indices_global)
            result.append(neg_indices_global[np.random.randint(0, 2)])
            return result
        else:
            return list(np.random.choice(neg_indices_global, size=target_count, replace=True))


class BaseEmbeddingsDataset(Dataset):
    """Handles pre-computed embeddings for images and text.
    Note: Each image has 5 corresponding captions."""
    def __init__(self, image_embeddings, text_embeddings):
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        return self.image_embeddings[idx], self.text_embeddings[idx]


class BaseNegEmbeddingsDataset(Dataset):
    """Handles pre-computed embeddings for images and text, including negative pairs.
    Note: Each image has 5 corresponding captions."""
    def __init__(self, 
                 image_embeddings, 
                 text_embeddings, 
                 neg_text_embeddings):
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
        self.neg_text_embeddings = neg_text_embeddings

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        sample_neg_text_embeddings = self.neg_text_embeddings[idx]

        if len(sample_neg_text_embeddings.shape) == 1:
            sample_neg_text_embeddings = sample_neg_text_embeddings.unsqueeze(0)

        random_neg_text_embedding = sample_neg_text_embeddings[np.random.randint(0, len(sample_neg_text_embeddings), (1,))]
        random_neg_text_embedding = random_neg_text_embedding.squeeze(0)

        return self.image_embeddings[idx], self.text_embeddings[idx], random_neg_text_embedding, sample_neg_text_embeddings

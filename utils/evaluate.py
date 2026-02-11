from tqdm import tqdm
import torch
import os
from PIL import Image
from typing import List, Tuple, Dict
import numpy as np
import logging

def get_contrastive_accuracy(image_embeddings,
                             caption_embeddings_split,
                             neg_caption_embeddings_split,
                             get_average=True):
    """
    Computes:
      1) overall contrastive accuracy (fraction of images where pos > all negs)
      2) average # of negatives beaten per image
      3) per-negative accuracy: for each negative index k, fraction of images
         where pos_similarity > that neg_similarity[:, k]
      4) logs mismatches: when negative caption similarity ≥ positive caption similarity

    Args:
        image_embeddings (Tensor):         (N, D)
        caption_embeddings_split (Tensor): (N, D)
        neg_caption_embeddings_split (Tensor): (N, K, D)
        get_average (bool):                whether to normalize by N or return sums

    Returns:
        contrastive_accuracy: Tensor scalar
        avg_pos_greater_than_neg:  Tensor scalar
        per_neg_accuracy:         Tensor of shape (K,)
    """

    N, D = image_embeddings.shape
    K = neg_caption_embeddings_split.size(1)

    # 1) positive similarities: (N,)
    pos_sim = (image_embeddings * caption_embeddings_split).sum(dim=1)

    # 2) negative similarities: (N, K)
    neg_sim = torch.bmm(
        image_embeddings.unsqueeze(1),                # (N,1,D)
        neg_caption_embeddings_split.transpose(1, 2)   # (N,D,K)
    ).squeeze(1)

    # mask out any “zero” negatives so we don’t count them
    nonzero_mask = neg_caption_embeddings_split.norm(dim=2) > 0  # (N, K)

    # 3) for each (i, k), did pos>neg?
    wins = pos_sim.unsqueeze(1) > neg_sim               # (N, K)
    wins = wins & nonzero_mask                          # ignore zeroed captions

    # overall contrastive accuracy: fraction of images where pos>all negs
    correct_all = wins.all(dim=1).float()               # (N,)
    contrastive_accuracy = correct_all.mean() if get_average else correct_all.sum()

    # average # of negatives beaten per image
    beaten_counts = wins.sum(dim=1).float()             # (N,)
    avg_beaten = beaten_counts.div(nonzero_mask.sum(dim=1).float())
    avg_beaten = avg_beaten.mean() if get_average else avg_beaten.sum()

    # per-negative accuracy: for each k, fraction of valid images where pos>neg_k
    per_neg_accuracy = []
    for k in range(K):
        valid = nonzero_mask[:, k]
        if valid.any():
            acc_k = wins[valid, k].float().mean() if get_average else wins[valid, k].float().sum()
        else:
            acc_k = torch.tensor(0.0, device=image_embeddings.device)
        per_neg_accuracy.append(acc_k)
    per_neg_accuracy = torch.stack(per_neg_accuracy)     # (K,)

    return contrastive_accuracy, avg_beaten, per_neg_accuracy

def get_negative_similarity(pos_embedding, neg_embeddings, get_average=True):
    """
    Computes mean cosine similarity between positives and their corresponding negatives.
    Returns:
        per_sample_mean: (N,)
        per_neg_mean: (K,)
    """
    pos_embedding = pos_embedding / pos_embedding.norm(dim=1, keepdim=True)
    neg_embeddings = neg_embeddings / neg_embeddings.norm(dim=2, keepdim=True)
    neg_similarity = torch.einsum('nd,nkd->nk', pos_embedding, neg_embeddings)
    per_sample_mean = neg_similarity.mean(dim=1)
    if get_average:
        per_neg_similarity = neg_similarity.mean(dim=0)
        return per_sample_mean.mean(), per_neg_similarity
    else:
        per_neg_similarity = neg_similarity.sum(dim=0)
        return per_sample_mean.sum(), per_neg_similarity

def get_negative_similarity_img(img_embedding, neg_embeddings, get_average=True):
    """
    Computes mean cosine similarity between image embeddings and their corresponding negatives.
    Returns:
        per_sample_mean: (N,)
        per_neg_mean: (K,)
    """
    img_embedding = img_embedding / img_embedding.norm(dim=1, keepdim=True)
    neg_embeddings = neg_embeddings / neg_embeddings.norm(dim=2, keepdim=True)
    neg_similarity = torch.einsum('nd,nkd->nk', img_embedding, neg_embeddings)
    per_sample_mean = neg_similarity.mean(dim=1)
    if get_average:
        per_neg_similarity = neg_similarity.mean(dim=0)
        return per_sample_mean.mean(), per_neg_similarity
    else:
        per_neg_similarity = neg_similarity.sum(dim=0)
        return per_sample_mean.sum(), per_neg_similarity

def get_caption_image_similarity(caption_embeddings, image_embeddings, get_average=True):
    """
    Computes mean cosine similarity between caption and image embeddings.
    """
    caption_embeddings = caption_embeddings / caption_embeddings.norm(dim=1, keepdim=True)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    similarity = torch.einsum('nd,nd->n', caption_embeddings, image_embeddings)
    return similarity.mean() if get_average else similarity.sum()

def get_results_i2t(image_embeddings, text_embeddings, indices=None):
    """
    Computes image-to-text retrieval metrics: top-1, top-5, top-10.
    """
    pred_true_1 = pred_true_5 = pred_true_10 = 0
    similarity = image_embeddings @ text_embeddings.t()
    similarity = similarity.cpu()
    num_images = similarity.shape[0]
    if indices:
        assert len(indices) == num_images
    if image_embeddings.shape[0] != text_embeddings.shape[0] and indices is None:
        raise ValueError("Image and text embeddings must have the same number of samples or provide indices.")
    if indices is None:
        indices = list(range(num_images))
    for img_idx, caption_idx in enumerate(indices):
        pred = similarity[img_idx]
        b = pred.argsort()
        rank = (b.flip(0) == caption_idx).nonzero().item()
        if caption_idx in b[-1:]:  pred_true_1 += 1
        if caption_idx in b[-5:]:  pred_true_5 += 1
        if caption_idx in b[-10:]: pred_true_10 += 1
    return pred_true_1 / num_images, pred_true_5 / num_images, pred_true_10 / num_images

def get_results_i2t_double_batched(
    image_embeddings,
    text_embeddings,
    indices=None,
    img_batch_size=512,
    txt_batch_size=2048,
    device='cuda'
):
    """
    Computes retrieval metrics for huge datasets using double batching.
    """
    pred_true_1 = pred_true_5 = pred_true_10 = 0
    num_images = image_embeddings.shape[0]
    num_texts = text_embeddings.shape[0]
    if indices is None:
        indices = list(range(num_images))
    assert len(indices) == num_images

    image_embeddings = image_embeddings.to(device)
    text_embeddings = text_embeddings.to(device)

    for img_start in tqdm(range(0, num_images, img_batch_size), desc='Computing i2t metrics'):
        img_end = min(img_start + img_batch_size, num_images)
        imgs = image_embeddings[img_start:img_end]  # [img_batch, D]

        # Collect similarity scores for all texts, for each image in batch
        all_similarities = []
        for txt_start in range(0, num_texts, txt_batch_size):
            txt_end = min(txt_start + txt_batch_size, num_texts)
            txts = text_embeddings[txt_start:txt_end]  # [txt_batch, D]
            sim = imgs @ txts.t()  # [img_batch, txt_batch]
            all_similarities.append(sim.cpu())
        # Concatenate all chunks to get full similarity row for each image
        all_similarities = torch.cat(all_similarities, dim=1)  # [img_batch, num_texts]

        for idx_in_batch, caption_idx in enumerate(indices[img_start:img_end], img_start):
            pred = all_similarities[idx_in_batch - img_start]
            b = pred.argsort()
            rank = (b.flip(0) == caption_idx).nonzero()
            if len(rank) > 0:
                rank = rank.item()
                if caption_idx in b[-1:]:  pred_true_1 += 1
                if caption_idx in b[-5:]:  pred_true_5 += 1
                if caption_idx in b[-10:]: pred_true_10 += 1

    return pred_true_1 / num_images, pred_true_5 / num_images, pred_true_10 / num_images


def flatten_dict(d, parent_key='evaluation_dataset', sep='/'):
    """Recursively flattens a nested dict for wandb logging under 'evaluation_dataset/' group."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}"
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep='-').items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten_per_neg_metric(metric, name, prefix='val'):
    """
    Flattens a per-negative metric (1D tensor or list) into a dict of scalars.
    Example: flatten_per_neg_metric([0.8, 0.6], 'per_neg_accuracy') returns
    {
        'val/per_neg_accuracy/0': 0.8,
        'val/per_neg_accuracy/1': 0.6
    }
    """
    if hasattr(metric, 'cpu'):
        metric = metric.cpu().numpy()
    return {f"{prefix}/{name}/{i}": float(val) for i, val in enumerate(metric)}
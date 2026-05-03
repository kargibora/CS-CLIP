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

VISMIN_DATASET_NAME = "mair-lab/vismin"
VISMIN_DATASET_BENCHMARK_NAME = "mair-lab/vismin-bench"

def compute_contrastive_accuracy_from_embs(img_embs_0, img_embs_1, txt_embs_0, txt_embs_1, categories):
    """
    img_embs_0, img_embs_1, txt_embs_0, txt_embs_1: [n, d] tensors
    categories: list of strings, len=n
    Returns: category-to-metrics dict (matching VisMin official)
    """
    split_correct_count = defaultdict(lambda: {"text": 0, "image": 0, "group": 0, "count": 0})

    for i in range(len(categories)):
        sim_00 = (img_embs_0[i] * txt_embs_0[i]).sum().item()
        sim_01 = (img_embs_1[i] * txt_embs_0[i]).sum().item()
        sim_10 = (img_embs_0[i] * txt_embs_1[i]).sum().item()
        sim_11 = (img_embs_1[i] * txt_embs_1[i]).sum().item()
        split = categories[i]

        # Official definitions
        text_correct = (sim_00 > sim_10) and (sim_11 > sim_01)
        image_correct = (sim_00 > sim_01) and (sim_11 > sim_10)
        split_correct_count[split]["text"] += text_correct
        split_correct_count[split]["image"] += image_correct
        split_correct_count[split]["group"] += text_correct and image_correct
        split_correct_count[split]["count"] += 1

    full_correct_count = {}
    for split in split_correct_count:
        denominator = split_correct_count[split]["count"]
        curr_text = round(split_correct_count[split]["text"] / denominator, 2)
        curr_image = round(split_correct_count[split]["image"] / denominator, 2)
        curr_group = round(split_correct_count[split]["group"] / denominator, 2)
        full_correct_count[split] = {
            "text": curr_text,
            "image": curr_image,
            "group": curr_group,
            "count": denominator,
        }
    return full_correct_count

def collect_grouped_edited_samples(
    dataset,
    category_filter=None,
    include_anchor_negatives_as_negative=False,
):
    """
    For each group, returns samples:
    - For original: negatives = [all edits]
    - For each edit: negatives = [original] or [original + other edits] (based on flag)
    """
    from collections import defaultdict

    import tqdm

    # Group samples by their original image
    groups = defaultdict(list)
    for idx, sample in tqdm.tqdm(enumerate(dataset), desc="Building groups"):
        if sample['source_image_id'] == '':
            group_id = sample['image_id']
        else:
            group_id = sample['source_image_id']
        if group_id != sample['image_id']:
            if category_filter and sample.get('category') != category_filter:
                continue
        groups[group_id].append((idx, sample))

    samples = []
    for group_id, group_items in groups.items():
        # Identify original and edits
        original_idx = None
        for i, (_, item) in enumerate(group_items):
            if not item.get('source_image_id'):
                original_idx = i
                break
        if original_idx is None:
            print(f"Warning: No original found in group {group_id}. Skipping this group.")
            continue  # Shouldn't happen, but skip malformed group

        # Build per-anchor negatives
        n = len(group_items)

        if len(group_items) < 2:
            continue

        for anchor_idx in range(n):
            if anchor_idx == original_idx:
                # Anchor is original: negatives are all edits
                neg_indices = [i for i in range(n) if i != anchor_idx]
            else:
                # Anchor is an edit: negative is [original] or [original + other edits]
                if include_anchor_negatives_as_negative:
                    neg_indices = [i for i in range(n) if i != anchor_idx]
                else:
                    neg_indices = [original_idx]
            if not neg_indices:
                continue
            captions = [item['caption'] for _, item in group_items]
            images = [None for _ in group_items]
            ids = [item['image_id'] for _, item in group_items]
            edit_instructions = [item.get('edit_instruction', '') for _, item in group_items]
            categories = [item.get('category', '') for _, item in group_items]
            dataset_indices = [idx for idx, _ in group_items]

            samples.append({
                'image_id': ids[anchor_idx],
                'anchor_caption': captions[anchor_idx],
                'anchor_image': images[anchor_idx],
                'candidate_captions': [captions[i] for i in neg_indices],
                'candidate_images': [images[i] for i in neg_indices],
                'candidate_ids': [ids[i] for i in neg_indices],
                'edit_instructions': [edit_instructions[i] for i in neg_indices],
                'category': categories[anchor_idx],
                'anchor_dataset_index': dataset_indices[anchor_idx],
                'candidate_dataset_indices': [dataset_indices[i] for i in neg_indices],
            })
    return samples

class VisMinDataset(Dataset):
    def __init__(self,
                    data_root : Optional[Union[str,os.PathLike]] = None,
                    subset_name : Optional[Literal['object','attribute','counting']] = None,
                    image_preprocess : callable = None,
                    **kwargs
                    ):
        """
        Args:
            subset_root: the path to the root dir of a subset, (e.g. `absolute_size`)
        """
        self.image_preprocess = image_preprocess
        self.data_root = data_root

        self.dataset = load_dataset(VISMIN_DATASET_NAME, split='train')
        self.evaluate_dataset = load_dataset(VISMIN_DATASET_BENCHMARK_NAME, split='test')

        if subset_name not in ['object', 'attribute', 'counting']:
            subset_name = None

        self.sample_list = collect_grouped_edited_samples(self.dataset.remove_columns(['image']), category_filter=subset_name, include_anchor_negatives_as_negative=True)
        self.captions = self.get_captions()
        self.max_num_negatives = max(
            len(sample['candidate_captions']) for sample in self.sample_list
        )

        # Preset the number of possible negatives
        self.num_negatives = 4 # or 4,
        self.number_of_candidates = self.num_negatives + 1  # +1 for the anchor caption

        logging.info(f"Number of unique captions: {len(self.captions)}")

        self.caption_to_idx = { caption : idx for idx, caption in enumerate(self.captions) }

        self.dataset_idx_to_sample_idx = { sample['anchor_dataset_index'] : idx for idx, sample in enumerate(self.sample_list) }

    def get_captions(self):
        """
        Get all captions in the dataset
        """
        captions = []
        for sample in self.sample_list:
            captions.append(sample['anchor_caption'])
            captions.extend(sample['candidate_captions'])
        return sorted(set(captions))

    def get_image_paths(self):
        """
        Get all image paths in the dataset
        """
        raise NotImplementedError("get_image_paths is not implemented")

    def get_idx_to_ptr(self, idx : int):
        """
        Get a mapping from original index of the caption in the dataset to the index in the caption_to_idx dictionary
        """
        caption = self.sample_list[idx]['anchor_caption']
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
        candidates = sample['candidate_captions']

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

    def get_idx_to_candidates_indices(self, idx: int):
        """
        Get a mapping from image index to negative image indices
        
        Args:
            idx: Index of the image
        
        Returns:
            List of indices of candidate captions in the sample_list
        """
        sample = self.sample_list[idx]
        neg_dataset_indeces = sample['candidate_dataset_indices']

        neg_indices = [self.dataset_idx_to_sample_idx[neg_index] for neg_index in neg_dataset_indeces]
        return neg_indices # Return the index of the negative caption in the sample_list
    
    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader to handle VisMin samples with variable negatives.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and caption lists
        """
        batch_images = []
        batch_captions = []
        
        for sample in batch:
            # Extract components from each sample
            image = sample['image_options']  # Single image tensor
            caption_options = sample['caption_options']  # [anchor, neg1, neg2, ...] - variable length
            
            batch_images.append(image)
            batch_captions.extend(caption_options)  # Flatten all captions
        
        # Stack images into a batch tensor
        batch_images = torch.stack(batch_images)  # [B, C, H, W]
        
        return {
            'images': batch_images,
            'captions': batch_captions,  # List[str] - flattened captions from all samples
            'caption_counts': [len(sample['caption_options']) for sample in batch]  # Track count per sample
        }

    def evaluate(self, 
                 embedding_model, 
                 aligning_model = None, 
                 device='cuda', 
                 batch_size=64,
                 indices : List[int] = None,
                 intermediate_text_layer_names = ['final'],
                 intermediate_image_layer_names = ['final']):
        """
        Evaluates a CLIP model on the VisMin Bench dataset with DataLoader optimization and caching.
        Returns: dict with accuracy, recall@1, group accuracy, etc.
        """
        # No need to use indices for bench evaluation - we use the evaluation dataset directly
        if indices is not None:
            logging.warning("Indices are provided but not used in evaluation. Using the evaluation dataset instead.")

        bench = self.evaluate_dataset
        n = len(bench)

        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")

        from torch.utils.data import DataLoader
        
        # Create a simple wrapper dataset for the bench data
        class BenchDataset(Dataset):
            def __init__(self, bench_data, image_preprocess):
                self.bench_data = bench_data
                self.image_preprocess = image_preprocess
                
            def __len__(self):
                return len(self.bench_data)
                
            def __getitem__(self, idx):
                item = self.bench_data[idx]
                return {
                    'image_0': self.image_preprocess(item['image_0']),
                    'image_1': self.image_preprocess(item['image_1']),  
                    'text_0': item['text_0'],
                    'text_1': item['text_1'],
                    'category': item['category']
                }
        
        def bench_collate_fn(batch):
            images_0 = torch.stack([item['image_0'] for item in batch])
            images_1 = torch.stack([item['image_1'] for item in batch])
            texts_0 = [item['text_0'] for item in batch]
            texts_1 = [item['text_1'] for item in batch]
            categories = [item['category'] for item in batch]
            return {
                'images_0': images_0,
                'images_1': images_1, 
                'texts_0': texts_0,
                'texts_1': texts_1,
                'categories': categories
            }

        bench_dataset = BenchDataset(bench, self.image_preprocess)
        dataloader = DataLoader(
            bench_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=bench_collate_fn
        )

        # Preallocate lists for embeddings
        img_embs_0, img_embs_1, txt_embs_0, txt_embs_1 = [], [], [], []
        all_categories = []

        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="VisMin",
            subset_name="bench", 
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:

            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Embedding VisMin Bench")):
                batch_images_0 = batch['images_0']  # [B, C, H, W]
                batch_images_1 = batch['images_1']  # [B, C, H, W]
                batch_texts_0 = batch['texts_0']    # List[str]
                batch_texts_1 = batch['texts_1']    # List[str]
                B = len(batch_texts_0)

                with torch.no_grad():
                    # Get image embeddings (with caching)
                    # Each batch has 2B images total (B images_0 + B images_1)
                    img_emb_0 = cache.get_or_compute_embeddings(
                        batch_images_0.to(device),
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size * 2  # Account for 2 images per sample
                    )
                    
                    img_emb_1 = cache.get_or_compute_embeddings(
                        batch_images_1.to(device), 
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size * 2 + B  # Offset for second set
                    )

                    # Get text embeddings (with caching)
                    # Each batch has 2B texts total (B texts_0 + B texts_1) 
                    all_texts = batch_texts_0 + batch_texts_1
                    txt_embs = cache.get_or_compute_embeddings(
                        all_texts,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * batch_size * 2  # Account for 2 texts per sample
                    )
                    
                    txt_emb_0 = txt_embs[:B]
                    txt_emb_1 = txt_embs[B:]

                # Convert to CPU and store
                img_embs_0.append(img_emb_0.cpu().float())
                img_embs_1.append(img_emb_1.cpu().float())
                txt_embs_0.append(txt_emb_0.cpu().float())
                txt_embs_1.append(txt_emb_1.cpu().float())
                all_categories.extend(batch['categories'])

        # Concatenate all batches
        img_embs_0 = torch.cat(img_embs_0, dim=0)
        img_embs_1 = torch.cat(img_embs_1, dim=0)
        txt_embs_0 = torch.cat(txt_embs_0, dim=0)
        txt_embs_1 = torch.cat(txt_embs_1, dim=0)

        # Normalize
        img_embs_0 = img_embs_0 / img_embs_0.norm(dim=-1, keepdim=True)
        img_embs_1 = img_embs_1 / img_embs_1.norm(dim=-1, keepdim=True)
        txt_embs_0 = txt_embs_0 / txt_embs_0.norm(dim=-1, keepdim=True)
        txt_embs_1 = txt_embs_1 / txt_embs_1.norm(dim=-1, keepdim=True)

        # Stack all images and texts (concatenate 0 and 1)
        all_img_embs = torch.cat([img_embs_0, img_embs_1], dim=0)  # shape (2n, D)
        all_txt_embs = torch.cat([txt_embs_0, txt_embs_1], dim=0)  # shape (2n, D)

        # For each text, compute similarity to all images
        sims = all_txt_embs @ all_img_embs.T  # shape (2n, 2n)
        sims = sims.numpy()

        # Recall@K for text->image retrieval
        recalls = {}
        K_list = [1, 5, 10]
        correct = np.zeros(len(K_list))
        for i in range(2 * n):
            gt_img_idx = i  # text_i should match image_i
            sorted_indices = np.argsort(-sims[i])
            for k_idx, K in enumerate(K_list):
                if gt_img_idx in sorted_indices[:K]:
                    correct[k_idx] += 1
        for k_idx, K in enumerate(K_list):
            recalls[f"recall@{K}"] = correct[k_idx] / (2 * n)

        # (Optional) Reverse: image->text retrieval
        recalls_image_to_text = {}
        correct = np.zeros(len(K_list))
        for i in range(2 * n):
            gt_txt_idx = i
            sorted_indices = np.argsort(-sims[:,i])
            for k_idx, K in enumerate(K_list):
                if gt_txt_idx in sorted_indices[:K]:
                    correct[k_idx] += 1
        for k_idx, K in enumerate(K_list):
            recalls_image_to_text[f"recall_i2t@{K}"] = correct[k_idx] / (2 * n)

        # Redo per-pair/group accuracy
        # This is for direct per-sample evaluation (as before)
        group_accuracies = []
        text_accuracies = []
        image_accuracies = []
        per_pair_correct = 0
        total = 0
        for i in range(n):
            # text_0 vs [image_0, image_1]
            sim_00 = (img_embs_0[i] * txt_embs_0[i]).sum().item()
            sim_01 = (img_embs_1[i] * txt_embs_0[i]).sum().item()
            sim_10 = (img_embs_0[i] * txt_embs_1[i]).sum().item()
            sim_11 = (img_embs_1[i] * txt_embs_1[i]).sum().item()
            correct_0 = sim_00 > sim_01
            correct_1 = sim_11 > sim_10
            per_pair_correct += correct_0 + correct_1
            total += 2
            group_accuracies.append(correct_0 and correct_1)
            text_accuracies.append(correct_0)
            image_accuracies.append(correct_1)
            
        group_accuracy = np.mean(group_accuracies)
        text_accuracy = np.mean(text_accuracies)
        image_accuracy = np.mean(image_accuracies)

        overall_acc = per_pair_correct / total

        metrics = {
            "accuracy": overall_acc,
            "group_contrastive_accuracy": float(group_accuracy),
            "text_contrastive_accuracy": float(text_accuracy),
            "image_contrastive_accuracy": float(image_accuracy),
        }
        contrastive_results = compute_contrastive_accuracy_from_embs(img_embs_0, img_embs_1, txt_embs_0, txt_embs_1, all_categories)
        metrics['group_contrastive_accuracy'] = contrastive_results

        metrics.update(recalls)
        metrics.update(recalls_image_to_text)

        embeddings = {
            "image_embeddings": img_embs_0,
            "negative_image_embeddings": img_embs_1,
            "caption_embeddings": txt_embs_0,
            "negative_caption_embeddings": txt_embs_1.unsqueeze(1),
        }
        
        return metrics, embeddings

    def evaluate_training_data(self, 
                             embedding_model, 
                             aligning_model = None, 
                             device='cuda', 
                             batch_size=64,
                             indices : List[int] = None,
                             intermediate_text_layer_names = ['final'],
                             intermediate_image_layer_names = ['final']):
        """
        Evaluates the training/validation dataset with variable negative counts and DataLoader caching.
        This handles the complex case where each sample has different numbers of candidate captions.
        """
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

        scores_list = []
        preds_list = []
        
        img_emb_list = []
        pos_cap_emb_list = []
        neg_cap_emb_list = []

        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="VisMin",
            subset_name="training",
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            # Track cumulative caption counts for proper cache indexing
            cumulative_caption_count = 0

            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Evaluating VisMin Training")):
                batch_images = batch['images']  # [B, C, H, W]
                batch_captions = batch['captions']  # List[str] - all captions flattened
                caption_counts = batch['caption_counts']  # List[int] - captions per sample
                B = len(batch_images)
                
                if B == 0:
                    continue

                with torch.no_grad():
                    # Get image embeddings (with caching)
                    img_embs = cache.get_or_compute_embeddings(
                        batch_images.to(device),
                        "image", 
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size  # Simple indexing for images (1 per sample)
                    )
                    
                    # Get caption embeddings with cumulative indexing (handles variable counts)
                    cap_embs = cache.get_or_compute_embeddings(
                        batch_captions,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=cumulative_caption_count  # Use cumulative count for proper indexing
                    )
                    
                    # Update cumulative count for next batch
                    cumulative_caption_count += len(batch_captions)

                # Parse caption embeddings back into per-sample format
                cap_start_idx = 0
                batch_scores = []
                batch_preds = []
                
                batch_pos_caps = []
                batch_neg_caps = []
                
                for i, cap_count in enumerate(caption_counts):
                    # Get embeddings for this sample
                    sample_cap_embs = cap_embs[cap_start_idx:cap_start_idx + cap_count]  # [cap_count, D]
                    sample_img_emb = img_embs[i:i+1]  # [1, D]
                    
                    # Compute similarities: image vs all captions
                    similarities = (sample_img_emb * sample_cap_embs).sum(dim=-1)  # [cap_count]
                    pred = torch.argmax(similarities).item()  # Index of highest similarity
                    
                    batch_scores.append(similarities)
                    batch_preds.append(pred)
                    
                    # Store embeddings (positive = first caption, negatives = rest)
                    batch_pos_caps.append(sample_cap_embs[0])  # First is positive
                    if cap_count > 1:
                        batch_neg_caps.append(sample_cap_embs[1:])  # Rest are negatives
                    else:
                        # Edge case: no negatives, duplicate positive as placeholder
                        batch_neg_caps.append(sample_cap_embs[0:1])
                    
                    cap_start_idx += cap_count

                # Convert to tensors and store
                img_emb_list.append(img_embs.cpu())
                pos_cap_emb_list.append(torch.stack(batch_pos_caps))
                # Handle variable negative counts by padding to max
                max_negs = max(neg_caps.shape[0] for neg_caps in batch_neg_caps)
                padded_neg_caps = []
                for neg_caps in batch_neg_caps:
                    if neg_caps.shape[0] < max_negs:
                        # Pad by repeating the last negative
                        padding = neg_caps[-1:].repeat(max_negs - neg_caps.shape[0], 1)
                        neg_caps = torch.cat([neg_caps, padding], dim=0)
                    padded_neg_caps.append(neg_caps)
                neg_cap_emb_list.append(torch.stack(padded_neg_caps))
                
                scores_list.extend(batch_scores)
                preds_list.extend(batch_preds)

        # Calculate accuracy (correct = pred index 0, since first caption is always positive)
        correct_preds = [pred == 0 for pred in preds_list]
        accuracy = np.mean(correct_preds)

        results = {
            "text_contrastive_accuracy": float(accuracy)
        }

        # Concatenate embeddings
        image_embeddings = torch.cat(img_emb_list, dim=0).numpy()
        caption_embeddings = torch.cat(pos_cap_emb_list, dim=0).numpy() 
        negative_caption_embeddings = torch.cat(neg_cap_emb_list, dim=0).numpy()

        embeddings = {
            "image_embeddings": image_embeddings,                    # [N, D]
            "caption_embeddings": caption_embeddings,               # [N, D] 
            "negative_caption_embeddings": negative_caption_embeddings,  # [N, max_negs, D]
        }

        return results, embeddings


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_info = self.sample_list[idx] 

        dataset_index = sample_info['anchor_dataset_index']
        query_image = self.dataset[dataset_index]['image']
        if self.image_preprocess is not None:
            query_image = self.image_preprocess(query_image)

        candidate_neg_captions = sample_info['candidate_captions']

        # Randomly sample num_negatives negatives from the candidate captions
        # So every epoch, we will have a different set of negatives. Makes sense for dynamic number of negatives.
        # During training, we only choose one negative caption!
        if len(candidate_neg_captions) >= self.num_negatives:
            candidate_neg_captions = random.sample(candidate_neg_captions, self.num_negatives)
        else:
            # Repeat negatives if not enough
            candidate_neg_captions = candidate_neg_captions + np.random.choice(
                candidate_neg_captions,
                size=self.num_negatives - len(candidate_neg_captions),
                replace=True
            ).tolist()
        
        candidate_texts = [sample_info['anchor_caption']] + candidate_neg_captions
        label = 0

        sample = {
            "image_options": query_image,
            "caption_options": candidate_texts,  # [anchor, negatives...]
            "label": label,
        }

        return sample
            
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
    Generates train/val/test indices for the dataset.
    If split_type == 'object', ensure all samples with the same original ('object') are grouped together.
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
        train_idx, val_idx = train_test_split(rem_idx, test_size=adj_val, random_state=seed)
    elif split_type == 'object':
        # 1. Build groups: map each group id (original image_id) to all indices (original + edits)
        object_groups = defaultdict(list)
        for idx, sample in enumerate(dataset.sample_list):
            # Use source_image_id if present, else use own image_id
            group_id = sample.get('source_image_id') or sample['image_id']
            object_groups[group_id].append(idx)
        group_keys = list(object_groups.keys())

        # 2. Shuffle and split group_keys into test/val/train (preserve groups)
        random.shuffle(group_keys)
        n_groups = len(group_keys)
        n_test = int(n_groups * test_ratio)
        n_val = int(n_groups * val_ratio)
        test_keys = group_keys[:n_test]
        val_keys = group_keys[n_test:n_test + n_val]
        train_keys = group_keys[n_test + n_val:]

        # 3. Gather indices for each split
        test_idx = [i for key in test_keys for i in object_groups[key]]
        val_idx = [i for key in val_keys for i in object_groups[key]]
        train_idx = [i for key in train_keys for i in object_groups[key]]
    else:
        raise NotImplementedError(f"Split type '{split_type}' not supported for this dataset.")
    
    return {
        'train': {'indices': train_idx},
        'val': {'indices': val_idx},
        'test': {'indices': test_idx},
    }
    
class VisMinNeg(Dataset):
    def __init__(self, 
                 dataset : 'VisMinDataset', 
                 indices : List[int],
                 num_negatives: Optional[int] = None):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        # Set num_negatives to max if not provided
        self.idx_to_indices = {idx: i for idx, i in enumerate(indices)} 

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_info = self.dataset[self.idx_to_indices[idx]]
        
        pos_image = sample_info['image_options']
        caption_options = sample_info['caption_options']
        pos_text = caption_options[0]
        neg_texts = caption_options[1:]  # list of negatives

        tokenized_caption = clip.tokenize([pos_text], truncate=True).squeeze(0) # shape: [77]
        neg_tokenized_captions = clip.tokenize(neg_texts, truncate=True) # shape: [num_neg, 77]
        
        # Randomly select a negative caption if there are more than one
        neg_tokenized_caption = neg_tokenized_captions[0].squeeze(0)

        return pos_image, tokenized_caption, neg_tokenized_caption, neg_tokenized_captions
import os
import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Union, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import clip
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import subprocess
from easydict import EasyDict as edict

from utils.perturbations import TextShuffler, pre_caption
from torchvision.datasets.utils import download_url
from utils.align import compute_image_embeddings_intermediate_batch, compute_caption_embeddings_intermediate_batch


# Import ARO datasets from dataset_zoo

def train_val_test_split_vg(
    dataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    split_type: Literal['random', 'object', 'variation'] = 'random'
):
    """
    Generates train/val/test indices for a dataset with 'image_id' in each sample.
    For 'object' split, ensures all samples of the same image are in the same split.
    """
    np.random.seed(seed)
    random.seed(seed)

    n = len(dataset)
    all_idx = list(range(n))
    random.shuffle(all_idx)
    n_test = int(n * test_ratio)
    test_idx = all_idx[:n_test]
    rem_idx = all_idx[n_test:]

    if split_type == 'random':
        adj_val = val_ratio / (1 - test_ratio)
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(rem_idx, test_size=adj_val, random_state=seed)
        return {
            'train': {'indices': train_idx},
            'val': {'indices': val_idx},
            'test': {'indices': test_idx},
        }
    elif split_type == 'object':
        # Group remaining indices by image_id
        image_id_to_indices = defaultdict(list)
        for idx in rem_idx:
            sample = dataset.sample_list[idx]
            image_id = sample['image_id']
            image_id_to_indices[image_id].append(idx)
        
        # Shuffle the image_ids to randomize split
        image_ids = list(image_id_to_indices.keys())
        random.shuffle(image_ids)
        
        n_images = len(image_ids)
        n_val_images = int(n_images * val_ratio)
        n_train_images = n_images - n_val_images
        
        train_img_ids = image_ids[:n_train_images]
        val_img_ids = image_ids[n_train_images:]
        
        # Flatten indices
        train_idx = [i for img_id in train_img_ids for i in image_id_to_indices[img_id]]
        val_idx = [i for img_id in val_img_ids for i in image_id_to_indices[img_id]]

        return {
            'train': {'indices': train_idx},
            'val': {'indices': val_idx},
            'test': {'indices': test_idx},
        }
    else:
        raise NotImplementedError(f"Split type '{split_type}' not supported for this dataset.")

class VG_Relation(Dataset):
    def __init__(self, 
                 data_root : Union[str,os.PathLike],
                 subset_name : str, 
                 image_preprocess : callable = None,
                 download : bool = False,
                 **kwargs
                 ):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        text_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        image_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        data_root: Directory for the VG-R dataset.
        download: Whether to download the dataset if it does not exist.
        '''
        self.data_root = data_root
        annotation_file = os.path.join(data_root, "visual_genome_relation.json")
        image_dir = os.path.join(data_root, "images")
        if not os.path.exists(image_dir):
            print("Image Directory for VG_Relation could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")
        
        if not os.path.exists(annotation_file):
            subprocess.call(["gdown", "--id", "1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3", "--output", annotation_file])
        
        with open(annotation_file, "r") as f:
            self.sample_list = json.load(f)
        
        self.all_relations = list()
        for item in self.sample_list:
            item["image_path"] = os.path.join(image_dir, item["image_path"])
            self.all_relations.append(item["relation_name"])


        self.image_preprocess = image_preprocess
        self.captions = self.get_captions()
        self.image_paths = self.get_image_paths()

        self.number_of_candidates = len(self[0]["caption_options"])

        self.caption_to_idx = { caption : idx for idx, caption in enumerate(self.captions) }
        logging.info(f"VG_Relation samples: {len(self.sample_list)}, unique captions: {len(self.captions)}")

    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader to handle VG_Relation samples.
        Efficiently batches samples with proper image stacking and caption grouping.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and grouped captions
        """
        batch_images = []
        batch_false_captions = []
        batch_true_captions = []
        
        for sample in batch:
            batch_images.append(sample['image_options'])
            batch_false_captions.append(sample['caption_options'][0])  # false caption
            batch_true_captions.append(sample['caption_options'][1])   # true caption
        
        # Stack images into a batch tensor
        batch_images = torch.stack(batch_images)  # [B, C, H, W]
        
        return {
            'images': batch_images,
            'false_captions': batch_false_captions,  # List[str]
            'true_captions': batch_true_captions     # List[str]
        }

    def get_captions(self):
        """
        Get all captions in the dataset
        """
        captions = []
        for sample in self.sample_list:
            captions.append(sample["true_caption"])
            captions.append(sample["false_caption"])
        return sorted(set(captions))

    
    def get_image_paths(self):
        """
        Get all image paths in the dataset
        """
        image_paths = []
        for sample in self.sample_list:
            image_paths.append(sample["image_path"])
        return image_paths
    
    def get_idx_to_ptr(self, idx : int):
        """
        Get a mapping from original index of the caption in the dataset to the index in the caption_to_idx dictionary
        """
        sample = self.sample_list[idx]
        
        return self.caption_to_idx[sample['true_caption']]
    
    def get_idx_to_candidates_ptr(self, idx: int):
        """
        Get a mapping from image index to caption indices
        
        Args:
            idx: Index of the image
        
        Returns:
            List of pointers to candidate captions in the caption_to_idx dictionary
        """
        sample = self.sample_list[idx]
        candidates = [sample['false_caption']]
        
        # Map captions to their indices in the caption dictionary
        return [self.caption_to_idx[caption] for caption in candidates]


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        test_case = self.sample_list[index]
        image = Image.open(test_case["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        img = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"], test_case["bbox_y"] + test_case["bbox_h"]))

        if self.image_preprocess is not None:
            img = self.image_preprocess(img)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
        item = edict({"image_options": img, "caption_options": [false_caption, true_caption]})
        return item
    
    def download(self):
        os.makedirs(self.data_root, exist_ok=True)
        image_zip_file = os.path.join(self.data_root, "vgr_vga_images.zip")
        subprocess.call(["gdown", "--no-cookies", "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9", "--output", image_zip_file])
        subprocess.call(["unzip", "vgr_vga_images.zip"], cwd=self.data_root)

    def split_dataset(self, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42, split_type: str = 'random') -> dict:
        """
        Splits the dataset into a new dataset with only the specified indices.
        """
        return train_val_test_split_vg(
            self,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_type=split_type
        )
    
    def evaluate(self, 
                embedding_model, 
                aligning_model=None, 
                device='cuda', 
                batch_size=64,
                indices : Optional[List[int]] = None,
                intermediate_text_layer_names=['final'],
                intermediate_image_layer_names=['final']
                ):
        """
        Evaluates model on VG_Relation dataset with DataLoader optimization and caching.
        Computes per-relation accuracy using efficient batch processing.
        
        Returns:
            result_records: dict with macro and text contrastive accuracy plus per-relation stats
            embeddings: dict with keys:
                - image_embeddings: [N, D]
                - caption_embeddings: [N, D]  
                - negative_caption_embeddings: [N, 1, D]
        """
        if 'compute_caption_embeddings_intermediate_batch' not in globals():
            raise ImportError("utils.align functions not available for evaluation")
        
        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")
        
        from torch.utils.data import DataLoader, Subset
        
        n = len(self)
        if indices is None:
            indices = list(range(n))
            eval_dataset = self
        else:
            eval_dataset = Subset(self, indices)
        
        # Use DataLoader for efficient multi-threaded batch loading
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Multi-threaded loading
            pin_memory=True,  # Faster GPU transfer
            collate_fn=self._collate_fn  # Custom collate function
        )
        
        scores = []
        img_emb_list = []
        false_emb_list = []
        true_emb_list = []
        all_relations = np.array(self.all_relations)[indices]
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="VG_Relation",
            subset_name="default",
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating VG_Relation")):
                batch_images = batch['images']  # [B, C, H, W]
                batch_false_captions = batch['false_captions']  # List[str]
                batch_true_captions = batch['true_captions']    # List[str]
                B = len(batch_false_captions)
                
                if B == 0:
                    continue
                
                with torch.no_grad():
                    # Get image embeddings (with caching)
                    img_embs = cache.get_or_compute_embeddings(
                        batch_images.to(device),
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size
                    )
                    
                    # Get caption embeddings (with caching)
                    all_captions = batch_false_captions + batch_true_captions
                    all_cap_embs = cache.get_or_compute_embeddings(
                        all_captions,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * (batch_size*2)
                    )
                    
                    # Split caption embeddings
                    false_embs = all_cap_embs[:B]   # [B, D]
                    true_embs = all_cap_embs[B:]    # [B, D]
                    
                    # Vectorized similarity computation
                    false_scores = torch.sum(img_embs * false_embs, dim=1)  # [B]
                    true_scores = torch.sum(img_embs * true_embs, dim=1)    # [B]
                    
                    batch_scores = torch.stack([false_scores, true_scores], dim=1)  # [B, 2]
                    scores.append(batch_scores.cpu())
                    
                    # Store embeddings for output (convert to numpy for compatibility)
                    img_emb_list.append(img_embs.cpu().numpy())         # [B, D]
                    false_emb_list.append(false_embs.cpu().numpy())     # [B, D]
                    true_emb_list.append(true_embs.cpu().numpy())       # [B, D]

        # Aggregate results over all batches
        scores = torch.cat(scores, dim=0).numpy()  # [N, 2]
        preds = np.argmax(scores, axis=-1)
        correct_mask = (preds == 1)

        result_records = {}
        relation_records = {}

        for relation in np.unique(all_relations):
            relation_mask = (all_relations == relation)
            if relation_mask.sum() == 0:
                continue
            if relation not in relation_records:
                relation_records[relation] = {
                    "accuracy": float(correct_mask[relation_mask].mean()),
                    "count": int(relation_mask.sum()),
                }
            else:
                relation_records[relation]["accuracy"] += float(correct_mask[relation_mask].mean())
                relation_records[relation]["count"] += int(relation_mask.sum())

        # Macro average
        result_records['macro_contrastive_accuracy'] = float(np.mean([v['accuracy'] for v in relation_records.values()]))

        total_accuracy = float(correct_mask.mean())
        result_records['text_contrastive_accuracy'] = total_accuracy

        # Stack embeddings [N, D]
        image_embeddings = np.concatenate(img_emb_list, axis=0)            # [N, D]
        true_caption_embeddings = np.concatenate(true_emb_list, axis=0)    # [N, D]
        false_caption_embeddings = np.concatenate(false_emb_list, axis=0)  # [N, D]

        # Add a singleton dimension for negatives: [N, 1, D]
        false_caption_embeddings = false_caption_embeddings[:, None, :]

        embeddings = {
            "image_embeddings": image_embeddings,                    # [N, D]
            "caption_embeddings": true_caption_embeddings,           # [N, D]
            "negative_caption_embeddings": false_caption_embeddings, # [N, 1, D]
        }
        return result_records, embeddings
    
class VG_Attribution(Dataset):
    def __init__(self, 
                data_root : Union[str,os.PathLike],
                subset_name : str, 
                image_preprocess : callable = None,
                download : bool = False,
                **kwargs
                ):
        """
        image_preprocess: function to transform PIL image to tensor
        data_root: dataset directory
        download: whether to fetch data if missing
        """
        self.data_root = data_root
        ann_file = os.path.join(data_root, "visual_genome_attribution.json")
        img_dir = os.path.join(data_root, "images")

        if not os.path.exists(img_dir):
            logging.warning("Image directory for VG_Attribution not found")
            if download:
                self.download()
            else:
                raise RuntimeError("Images missing: set download=True or adjust data_root")

        if not os.path.exists(ann_file):
            subprocess.call(["gdown", "--id", "13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw", "--output", ann_file])

        with open(ann_file, 'r') as f:
            self.sample_list = json.load(f)

        # Fix image paths
        for item in self.sample_list:
            item['image_path'] = os.path.join(img_dir, item['image_path'])
        self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.sample_list]
        self.image_preprocess = image_preprocess
        self.captions = self.get_captions()
        self.image_paths = self.get_image_paths()
        self.caption_to_idx = {cap: idx for idx, cap in enumerate(self.captions)}
        self.number_of_candidates = len(self[0]["caption_options"])

        logging.info(f"VG_Attribution samples: {len(self.sample_list)}, unique captions: {len(self.captions)}")

    def get_captions(self):
        captions = []
        for s in self.sample_list:
            captions.append(s['true_caption'])
            captions.append(s['false_caption'])
        return sorted(set(captions))


    def get_image_paths(self) -> List[str]:
        return [s['image_path'] for s in self.sample_list]


    def get_idx_to_ptr(self, idx : int):
        """
        Get a mapping from original index of the caption in the dataset to the index in the caption_to_idx dictionary
        """
        sample = self.sample_list[idx]
        
        return self.caption_to_idx[sample['true_caption']]
    
    def get_idx_to_candidates_ptr(self, idx: int):
        """
        Get a mapping from image index to caption indices
        
        Args:
            idx: Index of the image
        
        Returns:
            List of pointers to candidate captions in the caption_to_idx dictionary
        """
        sample = self.sample_list[idx]
        candidates = [sample['false_caption']]
        
        # Map captions to their indices in the caption dictionary
        return [self.caption_to_idx[caption] for caption in candidates]


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        sample = self.sample_list[index]
        img = Image.open(sample['image_path']).convert('RGB')
        # crop to bbox
        x, y, w, h = sample['bbox_x'], sample['bbox_y'], sample['bbox_w'], sample['bbox_h']
        img = img.crop((x, y, x+w, y+h))

        if self.image_preprocess:
            img = self.image_preprocess(img)

        return edict({
            'image_options': img,
            'caption_options': [sample['false_caption'], sample['true_caption']]
        })

    def _collate_fn(self, batch):
        """
        Custom collate function for efficient batch processing.
        
        Args:
            batch: List of items from __getitem__
            
        Returns:
            dict with:
                - images: [B, C, H, W] stacked tensor
                - false_captions: List[str] of length B  
                - true_captions: List[str] of length B
        """
        images = []
        false_captions = []
        true_captions = []
        
        for item in batch:
            images.append(item['image_options'])
            false_captions.append(item['caption_options'][0])  # false
            true_captions.append(item['caption_options'][1])   # true
        
        # Stack images into batch tensor
        images = torch.stack(images)  # [B, C, H, W]
        
        return {
            'images': images,
            'false_captions': false_captions,
            'true_captions': true_captions
        }

    def download(self):
        os.makedirs(self.data_root, exist_ok=True)
        zipf = os.path.join(self.data_root, 'vgr_vga_images.zip')
        subprocess.call(["gdown", "--no-cookies", "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9", "--output", zipf])
        subprocess.call(["unzip", zipf], cwd=self.data_root)

    def split_dataset(self, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42, split_type: str = 'random') -> dict:
        """
        Splits the dataset into a new dataset with only the specified indices.
        """
        return train_val_test_split_vg(
            self,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_type=split_type
        )
    

    def evaluate(self, 
                embedding_model, 
                aligning_model=None, 
                device='cuda', 
                batch_size=64,
                indices : Optional[List[int]] = None,
                intermediate_text_layer_names=['final'],
                intermediate_image_layer_names=['final']
                ):
        """
        Evaluates model on VG_Attribution dataset with DataLoader optimization and caching.
        Computes per-attribute accuracy using efficient batch processing.
        
        Returns:
            result_records: dict with macro and text contrastive accuracy plus per-attribute stats
            embeddings: dict with keys:
                - image_embeddings: [N, D]
                - caption_embeddings: [N, D]  
                - negative_caption_embeddings: [N, 1, D]
        """
        if 'compute_caption_embeddings_intermediate_batch' not in globals():
            raise ImportError("utils.align functions not available for evaluation")
        
        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")
        
        from torch.utils.data import DataLoader, Subset
        
        n = len(self)
        if indices is None:
            indices = list(range(n))
            eval_dataset = self
        else:
            eval_dataset = Subset(self, indices)
        
        # Use DataLoader for efficient multi-threaded batch loading
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Multi-threaded loading
            pin_memory=True,  # Faster GPU transfer
            collate_fn=self._collate_fn  # Custom collate function
        )
        
        scores = []
        img_emb_list = []
        false_emb_list = []
        true_emb_list = []
        all_attributes = np.array(self.all_attributes)[indices]
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="VG_Attribution",
            subset_name="default",
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating VG_Attribution")):
                batch_images = batch['images']  # [B, C, H, W]
                batch_false_captions = batch['false_captions']  # List[str]
                batch_true_captions = batch['true_captions']    # List[str]
                B = len(batch_false_captions)
                
                if B == 0:
                    continue
                
                with torch.no_grad():
                    # Get image embeddings (with caching)
                    img_embs = cache.get_or_compute_embeddings(
                        batch_images.to(device),
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size
                    )
                    
                    # Get caption embeddings (with caching)
                    all_captions = batch_false_captions + batch_true_captions
                    all_cap_embs = cache.get_or_compute_embeddings(
                        all_captions,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * (batch_size*2)
                    )
                    
                    # Split caption embeddings
                    false_embs = all_cap_embs[:B]   # [B, D]
                    true_embs = all_cap_embs[B:]    # [B, D]
                    
                    # Vectorized similarity computation
                    false_scores = torch.sum(img_embs * false_embs, dim=1)  # [B]
                    true_scores = torch.sum(img_embs * true_embs, dim=1)    # [B]
                    
                    batch_scores = torch.stack([false_scores, true_scores], dim=1)  # [B, 2]
                    scores.append(batch_scores.cpu())
                    
                    # Store embeddings for output (convert to numpy for compatibility)
                    img_emb_list.append(img_embs.cpu().numpy())         # [B, D]
                    false_emb_list.append(false_embs.cpu().numpy())     # [B, D]
                    true_emb_list.append(true_embs.cpu().numpy())       # [B, D]

        # Aggregate results over all batches
        scores = torch.cat(scores, dim=0).numpy()  # [N, 2]
        preds = np.argmax(scores, axis=-1)
        correct_mask = (preds == 1)

        result_records = {}
        attribute_records = {}

        for attribute in np.unique(all_attributes):
            attribute_mask = (all_attributes == attribute)
            if attribute_mask.sum() == 0:
                continue
            if attribute not in attribute_records:
                attribute_records[attribute] = {
                    "accuracy": float(correct_mask[attribute_mask].mean()),
                    "count": int(attribute_mask.sum()),
                }
            else:
                attribute_records[attribute]["accuracy"] += float(correct_mask[attribute_mask].mean())
                attribute_records[attribute]["count"] += int(attribute_mask.sum())

        # Macro average
        result_records['macro_contrastive_accuracy'] = float(np.mean([v['accuracy'] for v in attribute_records.values()]))

        total_accuracy = float(correct_mask.mean())
        result_records['text_contrastive_accuracy'] = total_accuracy

        # Stack embeddings [N, D]
        image_embeddings = np.concatenate(img_emb_list, axis=0)            # [N, D]
        true_caption_embeddings = np.concatenate(true_emb_list, axis=0)    # [N, D]
        false_caption_embeddings = np.concatenate(false_emb_list, axis=0)  # [N, D]

        # Add a singleton dimension for negatives: [N, 1, D]
        false_caption_embeddings = false_caption_embeddings[:, None, :]

        embeddings = {
            "image_embeddings": image_embeddings,                    # [N, D]
            "caption_embeddings": true_caption_embeddings,           # [N, D]
            "negative_caption_embeddings": false_caption_embeddings, # [N, 1, D]
        }
        return result_records, embeddings


    
# --- COCO_Order (pipeline-aligned) ---
class COCO_Order(Dataset):
    def __init__(self, 
                 data_root: Union[str, os.PathLike],
                 subset_name: str,                      # kept for API parity; unused
                 image_preprocess: callable = None,
                 download: bool = False,
                 max_words: int = 30,
                 split: Literal['train', 'val', 'test'] = 'test'):

        self.data_root = data_root
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root, exist_ok=True)
            if not download:
                raise RuntimeError("COCO missing: set download=True or adjust data_root")
            # (optional) self.download()

        urls = {
            'val':  'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
            'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'
        }
        fname = f"coco_karpathy_{split}.json"
        download_url(urls[split], self.data_root)
        ann_path = os.path.join(self.data_root, fname)
        with open(ann_path, 'r') as f:
            annotations = json.load(f)

        shuffler = TextShuffler()
        funcs = [shuffler.shuffle_nouns_and_adj,
                 shuffler.shuffle_allbut_nouns_and_adj,
                 shuffler.shuffle_within_trigrams,
                 shuffler.shuffle_trigrams]

        self.sample_list = []
        for ann in tqdm(annotations):
            for cap in ann['caption']:
                entry = {
                    'image': ann['image'],
                    'caption_options': [pre_caption(cap, max_words)]
                }
                for fn in funcs:
                    entry['caption_options'].append(pre_caption(fn(cap), max_words))
                self.sample_list.append(entry)

        self.image_preprocess = image_preprocess
        self.captions = self.get_captions()                      # deterministic, sorted
        self.image_paths = self.get_image_paths()
        self.caption_to_idx = {cap: idx for idx, cap in enumerate(self.captions)}
        self.number_of_candidates = len(self[0]["caption_options"])

        logging.info(f"COCO_Order samples: {len(self.sample_list)}, unique captions: {len(self.captions)}")

    def get_captions(self):
        caps = []
        for s in self.sample_list:
            caps.extend(s["caption_options"])
        return sorted(set(caps))

    def get_image_paths(self):
        return [s["image"] for s in self.sample_list]

    def get_idx_to_ptr(self, idx: int):
        """Pointer to the positive (index 0) caption in the global caption dict."""
        return self.caption_to_idx[self.sample_list[idx]['caption_options'][0]]

    def get_idx_to_candidates_ptr(self, idx: int):
        """Pointers to the negatives (indices 1:)."""
        return [self.caption_to_idx[c] for c in self.sample_list[idx]['caption_options'][1:]]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        e = self.sample_list[idx]
        img = Image.open(os.path.join(self.data_root, e['image'])).convert('RGB')
        if self.image_preprocess:
            img = self.image_preprocess(img)
        return edict({'image_options': img, 'caption_options': e['caption_options']})

    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader to handle COCO_Order samples.
        Efficiently batches samples with proper image stacking and caption grouping.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and grouped captions
        """
        batch_images = []
        batch_captions = []
        
        for sample in batch:
            batch_images.append(sample['image_options'])
            batch_captions.append(sample['caption_options'])
        
        # Stack images into a batch tensor
        batch_images = torch.stack(batch_images)  # [B, C, H, W]
        
        return {
            'images': batch_images,
            'captions': batch_captions  # List[List[str]], each inner list has candidates
        }

    def split_dataset(self, val_ratio: float = 0.1, test_ratio: float = 0.1,
                      seed: int = 42, split_type: str = 'random') -> dict:
        return train_val_test_split(self, val_ratio, test_ratio, seed, split_type)

    def evaluate(self, 
                 embedding_model,
                 aligning_model=None,
                 device='cuda',
                 batch_size=64,
                 indices: Optional[List[int]] = None,
                 intermediate_text_layer_names=['final'],
                 intermediate_image_layer_names=['final']):
        """
        Evaluates model on COCO_Order dataset with DataLoader optimization and caching.
        Precision@1 over each image's candidate set (index 0 is positive).
        
        Returns:
            result_records: dict with Precision@1 accuracy
            embeddings: dict with keys:
                - image_embeddings: [N, D]
                - caption_embeddings: [N, D]  
                - negative_caption_embeddings: [N, K, D] where K is number of negatives
        """
        if 'compute_caption_embeddings_intermediate_batch' not in globals():
            raise ImportError("utils.align functions not available for evaluation")
        
        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")
        
        from torch.utils.data import DataLoader, Subset
        
        n = len(self)
        if indices is None:
            indices = list(range(n))
            eval_dataset = self
        else:
            eval_dataset = Subset(self, indices)
        
        # Use DataLoader for efficient multi-threaded batch loading
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Multi-threaded loading
            pin_memory=True,  # Faster GPU transfer
            collate_fn=self._collate_fn  # Custom collate function
        )
        
        correct = []
        img_emb_list = []
        pos_cap_emb_list = []
        neg_cap_emb_list = []
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="COCO_Order",
            subset_name="default",
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating COCO_Order")):
                batch_images = batch['images']  # [B, C, H, W]
                batch_captions = batch['captions']  # List[List[str]]
                B = len(batch_captions)
                
                if B == 0:
                    continue
                
                # Extract all captions and positive/negative indices
                all_captions = []
                pos_indices = []
                neg_start_indices = []
                
                for i, caption_options in enumerate(batch_captions):
                    pos_indices.append(len(all_captions))  # Index of positive caption
                    neg_start_indices.append(len(all_captions) + 1)  # Start index of negatives
                    all_captions.extend(caption_options)
                
                n_candidates = len(batch_captions[0])  # Assuming all have same number of candidates
                
                with torch.no_grad():
                    # Get image embeddings (with caching)
                    img_embs = cache.get_or_compute_embeddings(
                        batch_images.to(device),
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size
                    )
                    
                    # Get caption embeddings (with caching)
                    all_cap_embs = cache.get_or_compute_embeddings(
                        all_captions,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * len(all_captions)
                    )
                    
                    # Reshape caption embeddings back to [B, n_candidates, D]
                    cap_embs_reshaped = all_cap_embs.view(B, n_candidates, -1)  # [B, n_candidates, D]
                    
                    # Vectorized similarity computation
                    # img_embs: [B, D], cap_embs_reshaped: [B, n_candidates, D]
                    img_embs_expanded = img_embs.unsqueeze(1)  # [B, 1, D]
                    scores = torch.sum(img_embs_expanded * cap_embs_reshaped, dim=2)  # [B, n_candidates]
                    
                    # Get predictions (index 0 should be highest for correct)
                    preds = torch.argmax(scores, dim=1)  # [B]
                    batch_correct = (preds == 0).cpu().tolist()
                    correct.extend(batch_correct)
                    
                    # Store embeddings for output
                    img_emb_list.append(img_embs.cpu().numpy())                    # [B, D]
                    pos_cap_emb_list.append(cap_embs_reshaped[:, 0].cpu().numpy()) # [B, D] - positive captions
                    neg_cap_emb_list.append(cap_embs_reshaped[:, 1:].cpu().numpy()) # [B, K, D] - negative captions

        # Stack embeddings
        image_embeddings = np.concatenate(img_emb_list, axis=0)         # [N, D]
        caption_embeddings = np.concatenate(pos_cap_emb_list, axis=0)   # [N, D]
        negative_caption_embeddings = np.concatenate(neg_cap_emb_list, axis=0)  # [N, K, D]
        
        # Recompute precision on CPU using all embeddings for verification
        # This is more memory efficient than GPU batch computation
        N, K, D = negative_caption_embeddings.shape
        all_caption_embeddings = np.concatenate([
            caption_embeddings[:, np.newaxis, :],  # [N, 1, D] - positive captions
            negative_caption_embeddings            # [N, K, D] - negative captions
        ], axis=1)  # [N, K+1, D]
        
        # Compute similarities on CPU: [N, K+1]
        # image_embeddings: [N, D], all_caption_embeddings: [N, K+1, D]
        similarities = np.sum(image_embeddings[:, np.newaxis, :] * all_caption_embeddings, axis=2)  # [N, K+1]
        
        # Get predictions (index 0 should be highest for correct)
        cpu_preds = np.argmax(similarities, axis=1)  # [N]
        cpu_precision_at_1 = float(np.mean(cpu_preds == 0))
        
        # Use GPU batch precision (accumulated during iteration) as primary
        gpu_precision_at_1 = float(np.mean(correct))
        
        result_records = {
            "Precision@1": cpu_precision_at_1,  # CPU recomputed precision (more accurate)
            "Precision@1_GPU": gpu_precision_at_1,  # GPU batch precision (for comparison)
            "text_contrastive_accuracy": cpu_precision_at_1  # For consistency with other datasets
        }
        
        embeddings = {
            "image_embeddings": image_embeddings,                     # [N, D]
            "caption_embeddings": caption_embeddings,                 # [N, D]
            "negative_caption_embeddings": negative_caption_embeddings, # [N, K, D]
        }
        
        return result_records, embeddings


# --- Flickr30k_Order (pipeline-aligned) ---
class Flickr30k_Order(Dataset):
    def __init__(self, 
                 data_root: Union[str, os.PathLike],
                 subset_name: str,                         # kept for API parity; unused
                 image_preprocess: callable = None,
                 download: bool = False,
                 max_words: int = 30,
                 split: Literal['val', 'test'] = 'test'):
        """
        data_root must contain the flickr json files and the images directory you use.
        """
        self.data_root = data_root
        if not os.path.exists(self.data_root):
            raise RuntimeError(f"Flickr30k root not found at {self.data_root}")

        urls = {
            'val':  'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
            'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'
        }
        fname = f"flickr30k_{split}.json"
        download_url(urls[split], self.data_root)
        ann_path = os.path.join(self.data_root, fname)
        with open(ann_path, 'r') as f:
            annotations = json.load(f)

        shuffler = TextShuffler()
        funcs = [shuffler.shuffle_nouns_and_adj,
                 shuffler.shuffle_allbut_nouns_and_adj,
                 shuffler.shuffle_within_trigrams,
                 shuffler.shuffle_trigrams]

        self.sample_list = []
        for ann in tqdm(annotations):
            for cap in ann['caption']:
                entry = {'image': ann['image'], 'caption_options': [pre_caption(cap, max_words)]}
                for fn in funcs:
                    entry['caption_options'].append(pre_caption(fn(cap), max_words))
                self.sample_list.append(entry)

        self.image_preprocess = image_preprocess
        # make deterministic caption vocab (not a bare set)
        self.captions = self.get_captions()
        self.image_paths = self.get_image_paths()
        self.caption_to_idx = {cap: idx for idx, cap in enumerate(self.captions)}
        self.number_of_candidates = len(self[0]["caption_options"])

        logging.info(f"Flickr30k_Order samples: {len(self.sample_list)}, unique captions: {len(self.captions)}")

    def get_captions(self):
        caps = []
        for s in self.sample_list:
            caps.extend(s["caption_options"])
        return sorted(set(caps))

    def get_image_paths(self):
        return [os.path.join(self.data_root, s["image"]) for s in self.sample_list]

    def get_idx_to_ptr(self, idx: int):
        return self.caption_to_idx[self.sample_list[idx]['caption_options'][0]]

    def get_idx_to_candidates_ptr(self, idx: int):
        return [self.caption_to_idx[c] for c in self.sample_list[idx]['caption_options'][1:]]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        e = self.sample_list[idx]
        img = Image.open(os.path.join(self.data_root, e['image'])).convert('RGB')
        if self.image_preprocess:
            img = self.image_preprocess(img)
        return edict({'image_options': img, 'caption_options': e['caption_options']})

    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader to handle Flickr30k_Order samples.
        Efficiently batches samples with proper image stacking and caption grouping.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and grouped captions
        """
        batch_images = []
        batch_captions = []
        
        for sample in batch:
            batch_images.append(sample['image_options'])
            batch_captions.append(sample['caption_options'])
        
        # Stack images into a batch tensor
        batch_images = torch.stack(batch_images)  # [B, C, H, W]
        
        return {
            'images': batch_images,
            'captions': batch_captions  # List[List[str]], each inner list has candidates
        }

    def split_dataset(self, val_ratio: float = 0.1, test_ratio: float = 0.1,
                      seed: int = 42, split_type: str = 'random') -> dict:
        return train_val_test_split(self, val_ratio, test_ratio, seed, split_type)

    def evaluate(self, 
                 embedding_model,
                 aligning_model=None,
                 device='cuda',
                 batch_size=64,
                 indices: Optional[List[int]] = None,
                 intermediate_text_layer_names=['final'],
                 intermediate_image_layer_names=['final']):
        """
        Evaluates model on Flickr30k_Order dataset with DataLoader optimization and caching.
        Precision@1 over each image's candidate set (index 0 is positive).
        
        Returns:
            result_records: dict with Precision@1 accuracy
            embeddings: dict with keys:
                - image_embeddings: [N, D]
                - caption_embeddings: [N, D]  
                - negative_caption_embeddings: [N, K, D] where K is number of negatives
        """
        if 'compute_caption_embeddings_intermediate_batch' not in globals():
            raise ImportError("utils.align functions not available for evaluation")
        
        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")
        
        from torch.utils.data import DataLoader, Subset
        
        n = len(self)
        if indices is None:
            indices = list(range(n))
            eval_dataset = self
        else:
            eval_dataset = Subset(self, indices)
        
        # Use DataLoader for efficient multi-threaded batch loading
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Multi-threaded loading
            pin_memory=True,  # Faster GPU transfer
            collate_fn=self._collate_fn  # Custom collate function
        )
        
        correct = []
        img_emb_list = []
        pos_cap_emb_list = []
        neg_cap_emb_list = []
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="Flickr30k_Order",
            subset_name="default",
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating Flickr30k_Order")):
                batch_images = batch['images']  # [B, C, H, W]
                batch_captions = batch['captions']  # List[List[str]]
                B = len(batch_captions)
                
                if B == 0:
                    continue
                
                # Extract all captions and positive/negative indices
                all_captions = []
                pos_indices = []
                neg_start_indices = []
                
                for i, caption_options in enumerate(batch_captions):
                    pos_indices.append(len(all_captions))  # Index of positive caption
                    neg_start_indices.append(len(all_captions) + 1)  # Start index of negatives
                    all_captions.extend(caption_options)
                
                n_candidates = len(batch_captions[0])  # Assuming all have same number of candidates
                
                with torch.no_grad():
                    # Get image embeddings (with caching)
                    img_embs = cache.get_or_compute_embeddings(
                        batch_images.to(device),
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size
                    )
                    
                    # Get caption embeddings (with caching)
                    all_cap_embs = cache.get_or_compute_embeddings(
                        all_captions,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * len(all_captions)
                    )
                    
                    # Reshape caption embeddings back to [B, n_candidates, D]
                    cap_embs_reshaped = all_cap_embs.view(B, n_candidates, -1)  # [B, n_candidates, D]
                    
                    # Vectorized similarity computation
                    # img_embs: [B, D], cap_embs_reshaped: [B, n_candidates, D]
                    img_embs_expanded = img_embs.unsqueeze(1)  # [B, 1, D]
                    scores = torch.sum(img_embs_expanded * cap_embs_reshaped, dim=2)  # [B, n_candidates]
                    
                    # Get predictions (index 0 should be highest for correct)
                    preds = torch.argmax(scores, dim=1)  # [B]
                    batch_correct = (preds == 0).cpu().tolist()
                    correct.extend(batch_correct)
                    
                    # Store embeddings for output
                    img_emb_list.append(img_embs.cpu().numpy())                    # [B, D]
                    pos_cap_emb_list.append(cap_embs_reshaped[:, 0].cpu().numpy()) # [B, D] - positive captions
                    neg_cap_emb_list.append(cap_embs_reshaped[:, 1:].cpu().numpy()) # [B, K, D] - negative captions

        # Stack embeddings
        image_embeddings = np.concatenate(img_emb_list, axis=0)         # [N, D]
        caption_embeddings = np.concatenate(pos_cap_emb_list, axis=0)   # [N, D]
        negative_caption_embeddings = np.concatenate(neg_cap_emb_list, axis=0)  # [N, K, D]
        
        # Recompute precision on CPU using all embeddings for verification
        # This is more memory efficient than GPU batch computation
        N, K, D = negative_caption_embeddings.shape
        all_caption_embeddings = np.concatenate([
            caption_embeddings[:, np.newaxis, :],  # [N, 1, D] - positive captions
            negative_caption_embeddings            # [N, K, D] - negative captions
        ], axis=1)  # [N, K+1, D]
        
        # Compute similarities on CPU: [N, K+1]
        # image_embeddings: [N, D], all_caption_embeddings: [N, K+1, D]
        similarities = np.sum(image_embeddings[:, np.newaxis, :] * all_caption_embeddings, axis=2)  # [N, K+1]
        
        # Get predictions (index 0 should be highest for correct)
        cpu_preds = np.argmax(similarities, axis=1)  # [N]
        cpu_precision_at_1 = float(np.mean(cpu_preds == 0))
        
        # Use GPU batch precision (accumulated during iteration) as primary
        gpu_precision_at_1 = float(np.mean(correct))
        
        result_records = {
            "Precision@1": cpu_precision_at_1,  # CPU recomputed precision (more accurate)
            "Precision@1_GPU": gpu_precision_at_1,  # GPU batch precision (for comparison)
            "text_contrastive_accuracy": cpu_precision_at_1  # For consistency with other datasets
        }
        
        embeddings = {
            "image_embeddings": image_embeddings,                     # [N, D]
            "caption_embeddings": caption_embeddings,                 # [N, D]
            "negative_caption_embeddings": negative_caption_embeddings, # [N, K, D]
        }
        
        return result_records, embeddings

    

def train_val_test_split(
    dataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    split_type: Literal['random', 'object', 'variation'] = 'random'
):
    """
    Generates train/val/test indices for CC3M.
    Only 'random' split is implemented; others raise errors.
    """
    np.random.seed(seed)
    random.seed(seed)

    n = len(dataset)
    all_idx = list(range(n))
    random.shuffle(all_idx)
    n_test = int(n * test_ratio)
    test_idx = all_idx[:n_test]
    rem_idx = all_idx[n_test:]

    if split_type not in ['random', 'object']:
        raise NotImplementedError(f"Split type '{split_type}' not supported for ARO datasets.")

    adj_val = val_ratio / (1 - test_ratio)
    train_idx, val_idx = train_test_split(rem_idx, test_size=adj_val, random_state=seed)

    return {
        'train': {'indices': train_idx},
        'val': {'indices': val_idx},
        'test': {'indices': test_idx},
    }


class ARONeg(Dataset):
    def __init__(self, 
                 dataset : Union[VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order],
                 indices : List[int],):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

        # index to indices mapping
        self.idx_to_indices = {idx: i for idx, i in enumerate(indices)} 

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_info = self.dataset[self.idx_to_indices[idx]]
        
        # Get the image
        pos_image = sample_info['image_options']
        caption_options = sample_info['caption_options']

        # Get the positive caption and negative texts based on dataset type
        if self.dataset.__class__.__name__ in ['VG_Relation', 'VG_Attribution']:
            pos_text = caption_options[1]  # True caption is at index 1
            neg_texts = [caption_options[0]]  # False caption is at index 0
        elif self.dataset.__class__.__name__ in ['COCO_Order', 'Flickr30k_Order']:
            pos_text = caption_options[0]  # Original caption is at index 0
            neg_texts = caption_options[1:]  # Shuffled captions are at indices 1+
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset.__class__.__name__}")

        # Randomly select one negative text for the single negative
        neg_text = neg_texts[np.random.randint(0, len(neg_texts))]
    
        # Tokenize both positive and negative captions
        tokenized_caption = clip.tokenize(pos_text, truncate=True).squeeze(0)
        neg_tokenized_caption = clip.tokenize(neg_text, truncate=True).squeeze(0)

        # Tokenize all negative captions
        if len(neg_texts) == 1:
            # For VG datasets with single negative
            neg_tokenized_captions = clip.tokenize(neg_texts, truncate=True).squeeze(0)
        else:
            # For COCO/Flickr with multiple negatives, keep batch dimension
            neg_tokenized_captions = clip.tokenize(neg_texts, truncate=True)  # [K, seq_len]

        return pos_image, tokenized_caption, neg_tokenized_caption, neg_tokenized_captions
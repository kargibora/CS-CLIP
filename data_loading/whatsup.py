import json
import logging
import math
import os
import random
import subprocess
from collections import defaultdict
from typing import List, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import clip
import numpy as np
import torch
from easydict import EasyDict as edict
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
# download_url not needed here
from tqdm import tqdm

from utils.align import (
    compute_caption_embeddings_intermediate_batch,
    compute_image_embeddings_intermediate_batch
)

# TODO: Set these paths to relative
ARO_ROOT = "/mnt/lustre/work/oh/owl336/thesis/CLIP-not-BoW-unimodally/datasets/WhatsUp"
COCO_ROOT = "/mnt/lustre/work/oh/owl336/thesis/CLIP-not-BoW-unimodally/datasets/WhatsUp"
FLICKR_ROOT = "/mnt/lustre/work/oh/owl336/thesis/CLIP-not-BoW-unimodally/datasets/WhatsUp"

class Controlled_Images(Dataset):
    def __init__(self, 
                 data_root : Union[str,os.PathLike],
                 subset_name : str, 
                 image_preprocess : callable = None,
                 download : bool = False,
                 **kwargs
                 ):
        self.data_root = data_root
        # Normalize subset name for matching
        self.subset_name = subset_name

        # Support COCO/VG QA subsets: COCO-One, COCO-Two, VG-One, VG-Two
        if subset_name in ('COCO-One', 'COCO-Two', 'VG-One', 'VG-Two'):
            # Determine dataset root and json filename
            if subset_name.startswith('COCO'):
                qa_json = os.path.join(data_root, 'coco_qa_one_obj.json' if subset_name.endswith('One') else 'coco_qa_two_obj.json')
                image_root = os.path.join(data_root, 'val2014')
            else:
                qa_json = os.path.join(data_root, 'vg_qa_one_obj.json' if subset_name.endswith('One') else 'vg_qa_two_obj.json')
                image_root = os.path.join(data_root, 'images')

            if not os.path.exists(qa_json):
                raise FileNotFoundError(f"QA JSON file not found: {qa_json}. Place the COCO/VG QA JSON in the data_root")

            # Load QA pairs: list of [image_id, correct_caption, wrong_caption]
            qa_pairs = json.load(open(qa_json))

            # Build sample_list with expected keys: 'image_path' and 'caption_options'
            self.sample_list = []
            for entry in qa_pairs:
                if len(entry) < 3:
                    continue
                img_id = str(entry[0])
                correct = entry[1]
                wrong = entry[2]
                
                # Handle COCO filename format: COCO_val2014_000000000042.jpg
                # Extract numeric ID and construct the expected filename
                if subset_name.startswith('COCO'):
                    # Remove any existing extension
                    img_id_numeric = img_id.replace('.jpg', '').replace('.png', '')
                    # Zero-pad to 12 digits if needed
                    img_id_numeric = img_id_numeric.zfill(12)
                    img_name = f"COCO_val2014_{img_id_numeric}.jpg"
                else:
                    # VG images: use ID as-is with .jpg extension if needed
                    img_name = img_id if img_id.lower().endswith(('.jpg', '.png')) else f"{img_id}.jpg"
                
                image_path = os.path.join(image_root, img_name)
                if not os.path.exists(image_path):
                    # Try alternate common subfolder structure
                    alt_path = os.path.join(image_root, img_id, img_name)
                    if os.path.exists(alt_path):
                        image_path = alt_path
                if not os.path.exists(image_path):
                    # If image missing, warn and skip
                    logging.warning(f"Image not found for QA entry: {image_path} (img_id: {img_id}) - skipping")
                    continue

                self.sample_list.append({
                    'image_path': image_path,
                    'caption_options': [correct, wrong]
                })

            self.image_preprocess = image_preprocess
            self.caption_to_label = {c: i for i, c in enumerate(self.get_captions())}
            self.number_of_candidates = 2
            self.captions = self.get_captions()
            self.image_paths = self.get_image_paths()
            logging.info(f"Loaded {len(self.sample_list)} QA samples for subset {subset_name}")
            return

        if subset_name == 'A':
            annotation_file = os.path.join(data_root, "controlled_images_dataset.json")
            image_dir = os.path.join(data_root, 'data', 'controlled_images')

            if not os.path.exists(image_dir):
                print("Image directory for Controlled Images A could not be found!")
                if download:
                    self.download()
                else:
                    raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")

            if not os.path.exists(annotation_file):
                subprocess.call(["gdown", "--id", "1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm", "--output", annotation_file])

        else:
            annotation_file = os.path.join(data_root, "controlled_clevr_dataset.json")
            image_dir = os.path.join(data_root, 'data', 'controlled_clevr')
            if not os.path.exists(image_dir):
                print("Image directory for Controlled Images B could not be found!")
                if download:
                    self.download()
                else:
                    raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")

            if not os.path.exists(annotation_file):
                subprocess.call(["gdown", "--id", "1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z", "--output", annotation_file])


        self.sample_list = json.load(open(annotation_file))
        self.subset_name = subset_name
        self.image_preprocess = image_preprocess
        
        self.caption_to_label = {}
        self.all_prepositions = []
        self.number_of_candidates = len(self.sample_list[0]['caption_options'])

        if self.subset_name == 'A':
            for d in self.sample_list:
                # Get the caption
                caption_options = d['caption_options']
                caption = caption_options[0]

                image_name = d['image_path'].split('/')[-1]
                image_path = os.path.join(image_dir, image_name)
                d['image_path'] = image_path

                if 'left_of' in d['image_path']:
                    self.caption_to_label[caption] = 0
                    self.all_prepositions.append('left_of')
                elif 'right_of' in d['image_path']:
                    self.caption_to_label[caption] = 1
                    self.all_prepositions.append('right_of')
                elif '_on_' in d['image_path']:
                    self.caption_to_label[caption] = 2
                    self.all_prepositions.append('on')
                else:
                    self.caption_to_label[caption] = 3
                    self.all_prepositions.append('under')

            self.eval_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': 0, 'right': 0, \
                                'on': 0, 'under': 0} for d in self.sample_list}
            self.pred_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': '', 'right': '', \
                                'on': '', 'under': ''} for d in self.sample_list}
        else:
            for d in self.sample_list:
                caption_options = d['caption_options']
                caption = caption_options[0]
                image_name = d['image_path'].split('/')[-1]
                image_path = os.path.join(image_dir, image_name)
                d['image_path'] = image_path

                if 'left_of' in d['image_path']:
                    self.caption_to_label[caption] = 0
                    self.all_prepositions.append('left_of')
                elif 'right_of' in d['image_path']:
                    self.caption_to_label[caption] = 1
                    self.all_prepositions.append('right_of')
                elif '_in-front_of_' in d['image_path']:
                    self.caption_to_label[caption] = 2
                    self.all_prepositions.append('in-front_of')
                else:
                    self.caption_to_label[caption] = 3
                    self.all_prepositions.append('behind')

            self.eval_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': 0, 'right': 0, \
                                'in-front': 0, 'behind': 0} for d in self.sample_list}
            self.pred_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': '', 'right': '', \
                                'in-front': '', 'behind': ''} for d in self.sample_list}

        self.captions = self.get_captions()
        self.image_paths = self.get_image_paths()

        logging.info(f"Number of unique images: {len(self.image_paths)}")
        logging.info(f"Number of unique captions: {len(self.captions)}")

        self.caption_to_idx = { caption : idx for idx, caption in enumerate(self.captions) }

    def get_captions(self):
        """
        Get all captions in the dataset
        """
        captions = []
        for sample in self.sample_list:
            caption_options = sample['caption_options']
            captions.extend(caption_options)
        return sorted(set(captions))

    def get_image_paths(self):
        """
        Get all image paths in the dataset
        """
        image_paths = []
        for sample in self.sample_list:
            image_paths.append(sample['image_path'])
        return image_paths
    
    def get_idx_to_ptr(self, idx : int):
        """
        Get a mapping from original index of the caption in the dataset to the index in the caption_to_idx dictionary
        """
        sample = self.sample_list[idx]
        label = 0
        
        caption = sample['caption_options'][label]
        return self.caption_to_idx[caption]
    
    def get_idx_to_candidates_ptr(self, idx: int):
        """
        Get a mapping from image index to caption indices
        
        Args:
            idx: Index of the image
            only_negative: If True, return only negative candidates (excluding the correct caption)
                           If False, return all candidate captions
        
        Returns:
            List of pointers to candidate captions in the caption_to_idx dictionary
        """
        sample = self.sample_list[idx]
        candidates = sample['caption_options']  # Exclude the correct caption
            
        # Map captions to their indices in the caption dictionary
        return [self.caption_to_idx[caption] for i,caption in enumerate(candidates) if i != 0]
    
    def get_idx_to_candidates_indices(self, idx: int):
        """
        Get a mapping from image index to caption indices
        
        Args:
            idx: Index of the image
            
        Returns:
            List of indices of candidate captions in the sample_list
        """
        sample = self.sample_list[idx]
        candidates = sample['caption_options']

        # Get the modular of the index
        unique_object_count = len(self.sample_list) // self.number_of_candidates
        
        mod_idx = idx % unique_object_count # If 2000 unique object and idx is 2003, this returns 3
        candidate_indices = [mod_idx + i * unique_object_count for i in range(len(candidates))]
    
        # Return all negatives but the original one
        print(f"Candidate indices for {idx}: {candidate_indices}")
        print(len(self.sample_list), len(candidate_indices))
        exit(1)
        return [i for i in candidate_indices if i != idx]


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        test_case = self.sample_list[index]
        image = Image.open(test_case["image_path"]).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        
        item = edict({"image_options": image, "caption_options": test_case['caption_options']})
        return item

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "controlled_images.tar.gz")
        subprocess.call(["gdown", "--no-cookies",  "19KGYVQjrV3syb00GgcavB2nZTW5NXX0H", "--output", image_zip_file])
        subprocess.call(["tar", "-xvf", "controlled_images.tar.gz"], cwd=self.root_dir)
        image_zip_file = os.path.join(self.root_dir, "controlled_clevr.tar.gz")
        subprocess.call(["gdown", "--no-cookies",  "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG", "--output", image_zip_file])
        subprocess.call(["tar", "-xvf", "controlled_clevr.tar.gz"], cwd=self.root_dir)


    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader to handle WhatsUp samples.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and caption lists
        """
        batch_images = []
        batch_caption_options = []
        
        for sample in batch:
            # Extract components from each sample
            image = sample['image_options']  # Single image tensor
            caption_options = sample['caption_options']  # List of captions (positive + negatives)
            
            batch_images.append(image)
            batch_caption_options.append(caption_options)
        
        # Stack images into a batch tensor
        batch_images = torch.stack(batch_images)  # [B, C, H, W]
        
        return {
            'images': batch_images,
            'caption_options': batch_caption_options  # List[List[str]] - preserves structure
        }

    def evaluate(self, 
             embedding_model, 
             aligning_model=None, 
             device='cuda', 
             batch_size=64,
             indices=None,
             intermediate_text_layer_names=['final'],
             intermediate_image_layer_names=['final']):
        """
        Evaluates model on WhatsUp dataset with DataLoader optimization and caching.
        
        Returns:
            result_records: dict of metrics,
            embeddings: dict with:
                - image_embeddings: [N, D]
                - caption_embeddings: [N, D]         # correct captions
                - negative_caption_embeddings: [N, C-1, D] # negatives
        """
        import numpy as np
        import torch

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
            
        # Handle prepositions: some subsets (COCO/VG QA) don't have preposition annotations
        if hasattr(self, 'all_prepositions') and len(getattr(self, 'all_prepositions', [])) > 0:
            all_prepositions = np.array(self.all_prepositions)[indices]
        else:
            # Use placeholder 'none' so np.unique still works
            all_prepositions = np.array(['none'] * len(indices))

        img_emb_list, pos_emb_list, neg_emb_list = [], [], []
        preds_list = []

        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="WhatsUp",
            subset_name=self.subset_name,
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:

            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Embedding WhatsUp")):
                batch_images = batch['images']  # [B, C, H, W]
                batch_caption_options = batch['caption_options']  # List[List[str]]
                B = len(batch_caption_options)

                if B == 0:
                    continue

                # Number of caption candidates for this dataset (may be 2 or 4)
                C = len(batch_caption_options[0])

                with torch.no_grad():
                    # Standard CLIP path
                    img_embs = cache.get_or_compute_embeddings(
                        batch_images.to(device),
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size
                    )

                    # Flatten all captions for batch processing
                    flat_captions = []
                    for caption_options in batch_caption_options:
                        flat_captions.extend(caption_options)

                    # Get all caption embeddings at once
                    txt_embs_flat = cache.get_or_compute_embeddings(
                        flat_captions,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * batch_size * C
                    )

                    # Reshape back to [B, C, D]
                    txt_embs = txt_embs_flat.view(B, C, -1)

                    # Vectorized similarity computation
                    img_embs_expand = img_embs.unsqueeze(1)  # [B, 1, D]
                    scores = (img_embs_expand * txt_embs).sum(dim=-1)  # [B, C]
                    preds = torch.argmax(scores, dim=-1)  # [B]

                    # Store results
                    img_emb_list.append(img_embs.cpu())
                    pos_emb_list.append(txt_embs[:, 0, :].cpu())  # Positive captions
                    neg_emb_list.append(txt_embs[:, 1:, :].cpu())
                    preds_list.append(preds.cpu())

            image_embeddings = torch.cat(img_emb_list, dim=0).numpy()
            caption_embeddings = torch.cat(pos_emb_list, dim=0).numpy()
            negative_caption_embeddings = torch.cat(neg_emb_list, dim=0).numpy()
            all_preds = torch.cat(preds_list, dim=0).numpy()
            correct_mask = (all_preds == 0)

        result_records = {}

        # Here is the new format for preposition-wise accuracy
        prep_acc_records = {}

        for prep in np.unique(all_prepositions):
            mask = (all_prepositions == prep)
            if mask.sum() == 0:
                continue
            acc = float(correct_mask[mask].mean())
            count = int(mask.sum())
            prep_acc_records[prep] = {"accuracy": acc, "count": count}

        result_records['text_contrastive_accuracy'] = float(correct_mask.mean())
        # Macro accuracy across all prepositions
        if len(prep_acc_records) > 0:
            macro_acc = np.mean([v['accuracy'] for v in prep_acc_records.values()])
            result_records['macro_contrastive_accuracy'] = macro_acc
        
        embeddings = {
            "image_embeddings": image_embeddings,                    
            "caption_embeddings": caption_embeddings,                
            "negative_caption_embeddings": negative_caption_embeddings, 
        }
        return result_records, embeddings

        
    def split_dataset(self, val_ratio=0.1, test_ratio=0.1, seed=42, split_type='random'):
        """
        Splits the dataset into train, validation, and test sets.
        
        Args:
            dataset: The dataset to split.
            val_ratio: The ratio of the validation set.
            test_ratio: The ratio of the test set.
            seed: Random seed for reproducibility.
            split_type: The type of split to perform ('random', 'object', 'variation').
        
        Returns:
            A dictionary containing the train, validation, and test splits.
        """
        return train_val_test_split(self, 
                                    val_ratio=val_ratio, 
                                    test_ratio=test_ratio,
                                    seed=seed, 
                                    split_type=split_type)
    

def train_val_test_split(dataset, val_ratio=0.1, test_ratio=0.1,  
                         seed=42, split_type: Literal['random', 'object', 'variation'] = 'random'):
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
        
    elif split_type == 'object':
        # Group remaining indices by object ID extracted from the image paths
        object_to_indices = defaultdict(list)
        
        for idx in remaining_idx:
            sample = dataset.sample_list[idx]
            # Get image path depending on dataset type
            image_path = sample['image_path']
            image_base = os.path.basename(image_path)
            if '_right_of_' in image_base:
                object_id = "-".join(image_base.split('_right_of_'))
            elif '_left_of_' in image_base:
                object_id = "-".join(image_base.split('_left_of_'))
            elif '_under_' in image_base:
                object_id = "-".join(image_base.split('_under_'))
            elif '_on_' in image_base:
                object_id = "-".join(image_base.split('_on_'))
            object_to_indices[object_id].append(idx)
        
        # Shuffle the object ids
        object_ids = list(object_to_indices.keys())
        random.shuffle(object_ids)
        
        n_objects = len(object_ids)
        n_val_objects = int(n_objects * val_ratio)
        n_train_objects = n_objects - n_val_objects
        
        train_obj_ids = object_ids[:n_train_objects]
        val_obj_ids = object_ids[n_train_objects:]
        
        print(f"Train Obj ids : {train_obj_ids}")
        print(f"Val Obj ids : {val_obj_ids}")
        
        # Flatten object indices into train and val lists
        train_idx = [i for obj in train_obj_ids for i in object_to_indices[obj]]
        val_idx = [i for obj in val_obj_ids for i in object_to_indices[obj]]
    
    elif split_type == 'variation':

        # Split by labels (assuming labels are from 0 to n_candidates-1) on the remaining indices.
        n_candidates = len(dataset.sample_list[0]['caption_options'])

        # Get unique labels and shuffle them
        unique_labels = list(range(n_candidates))
        random.shuffle(unique_labels)
        
        n_labels = len(unique_labels)
        n_val_labels = max(1, math.ceil(n_labels * val_ratio)) if val_ratio > 0 else 0
        n_train_labels = n_labels - n_val_labels
        
        train_labels_unique = unique_labels[:n_train_labels]
        val_labels_unique = unique_labels[n_train_labels:]
        
        # Partition the remaining indices by matching label values.
        dataset_labels = []
        for i in len(dataset):
            caption = dataset.sample_list[i]['caption_options'][0]
            dataset_labels.append(dataset.caption_to_label[caption])
            
        train_idx = [i for i in remaining_idx if dataset_labels[i] in train_labels_unique]
        val_idx = [i for i in remaining_idx if dataset_labels[i] in val_labels_unique]
    
    else:
        raise ValueError("Invalid split type. Choose from ['random', 'object', 'variation']")
    
    def get_labels_and_negatives(indices):
        neg_labels = [
            random.choice(
                [j for j in range(1,len(dataset.sample_list[i]['caption_options']))]
            )
            for i in indices
        ]
        labels = [0 for i in indices]
        return labels, neg_labels

    train_labels, train_neg_labels = get_labels_and_negatives(train_idx)
    val_labels, val_neg_labels = get_labels_and_negatives(val_idx)
    test_labels, test_neg_labels = get_labels_and_negatives(fixed_test_idx)
    
    return {
        'train': {'indices': train_idx, 'labels': train_labels, 'neg_labels': train_neg_labels},
        'val':   {'indices': val_idx, 'labels': val_labels, 'neg_labels': val_neg_labels},
        'test':  {'indices': fixed_test_idx, 'labels': test_labels, 'neg_labels': test_neg_labels},
    }

class ControlledImagesNeg(Dataset):
    def __init__(self, 
                 dataset : 'Controlled_Images', 
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

        label = 0

        # Get the positive caption
        pos_text = caption_options[label]
        neg_texts = caption_options[1:]
        neg_text = neg_texts[np.random.randint(0, len(neg_texts))]
    
        # Tokenize both positive and negative captions
        tokenized_caption = clip.tokenize(pos_text, truncate=True).squeeze(0)
        neg_tokenized_caption = clip.tokenize(neg_text, truncate=True).squeeze(0)

        # tokenize all negative captions
        neg_tokenized_captions = clip.tokenize(neg_texts, truncate=True).squeeze(0)

        return pos_image, tokenized_caption, neg_tokenized_caption, neg_tokenized_captions
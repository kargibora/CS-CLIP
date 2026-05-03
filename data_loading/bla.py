import os
import json
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

# Optional: only used in _evaluate
try:
    from utils.align import (
        compute_caption_embeddings_intermediate_batch,
        compute_image_embeddings_intermediate_batch,
    )
except ImportError:
    print("Warning: utils.align not found. Evaluation functions may not work.")

class BLADataset(Dataset):
    """
    BLA: Basic Language Abilities Benchmark
    Each sample contains an image, a correct caption, and a foil (incorrect caption).
    """
    def __init__(
        self,
        data_root: str,
        subset: str = "ap",
        split: str = "test",
        image_preprocess=None,
        download: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        self.data_root = data_root
        self.phenomenon = subset
        self.split = split
        self.image_preprocess = image_preprocess
        self.verbose = verbose
        self.samples = []
        self._load_data()

    @staticmethod
    def print_download_instructions():
        print("\n=== BLA Benchmark Download Instructions ===")
        print("1. Download the dataset:")
        print("   wget 'https://www.dropbox.com/scl/fi/kocjr2gk3hf667r53p10a/BLA_benchmark.zip?rlkey=dxi37fvx3bxhgrgr5ps9pp851&dl=1' -O BLA_benchmark.zip")
        print("2. Unzip:")
        print("   unzip BLA_benchmark.zip -d ./datasets/")
        print("3. You should now have ./datasets/BLA_Benchmark/annotations/ and ./datasets/BLA_Benchmark/images/")
        print("===========================================\n")

    def _load_data(self):
        ann_dir = os.path.join(self.data_root, "annotations")
        
        # Map phenomenon codes to actual file names
        file_mapping = {
            "ap": "active_passive_captions.json",
            "co": "coordination_captions.json", 
            "rc": "relative_clause_captions.json"
        }
        
        if self.phenomenon in file_mapping:
            ann_file = os.path.join(ann_dir, file_mapping[self.phenomenon])
        else:
            # Fallback to original naming pattern
            ann_file = os.path.join(ann_dir, f"{self.phenomenon}_{self.split}.json")
        
        if not os.path.exists(ann_file):
            if self.verbose:
                print(f"Tried to load: {ann_file}")
                print(f"Available files in {ann_dir}:")
                if os.path.exists(ann_dir):
                    for f in os.listdir(ann_dir):
                        if f.endswith('.json'):
                            print(f"  - {f}")
            raise FileNotFoundError(f"BLA annotation file not found: {ann_file}\nRun BLADataset.print_download_instructions() for help.")
        
        with open(ann_file, "r") as f:
            data = json.load(f)
        
        # Detect group-based format (e.g., relative_clause_captions.json)
        if data and isinstance(data[0], dict) and "caption_group" in data[0]:
            valid_entries = 0
            skipped_entries = 0
            
            for entry in data:
                # Skip entries without image_id or with empty caption_group
                if "image_id" not in entry:
                    skipped_entries += 1
                    continue
                    
                caption_groups = entry.get("caption_group", [])
                if not caption_groups:
                    skipped_entries += 1
                    continue
                
                image_id = entry["image_id"]
                
                image_file = f"{image_id}.jpg"
                for group in caption_groups:
                    # Validate that group has required fields
                    required_fields = ["True1", "False1", "True2", "False2"]
                    if not all(field in group for field in required_fields):
                        continue
                    
                    group_index = group.get("group_index", 0)
                    predicate = group.get("predicate", "")
                    
                    # True1 vs False1
                    self.samples.append({
                        "image": image_file,
                        "caption": group["True1"],
                        "foil": group["False1"],
                        "label": 1,
                        "phenomenon": self.phenomenon,
                        "group_index": group_index,
                        "predicate": predicate,
                        "caption_type": "True1_vs_False1",
                    })
                    # True2 vs False2
                    self.samples.append({
                        "image": image_file,
                        "caption": group["True2"],
                        "foil": group["False2"],
                        "label": 1,
                        "phenomenon": self.phenomenon,
                        "group_index": group_index,
                        "predicate": predicate,
                        "caption_type": "True2_vs_False2",
                    })
                    valid_entries += 1
                    
            if self.verbose:
                print(f"[BLA] Processed {len(data)} entries, found {valid_entries} valid caption groups")
                if skipped_entries > 0:
                    print(f"[BLA] Skipped {skipped_entries} entries with missing data")
        else:
            # Handle simple format where each entry is already a complete sample
            for sample in data:
                # Each sample should have: image, caption, foil, label, phenomenon
                if all(key in sample for key in ["image", "caption", "foil"]):
                    self.samples.append(sample)
                    
        if self.verbose:
            print(f"[BLA] Loaded {len(self.samples)} samples from {ann_file}")

    def __len__(self):
        return len(self.samples)

    def _find_image_path(self, sample):
        img_dir = os.path.join(self.data_root, "images")
        img_path = os.path.join(img_dir, sample["image"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        return img_path

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img_path = self._find_image_path(sample)
        image = Image.open(img_path).convert("RGB")
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        
        # Return format compatible with evaluation pipeline
        return {
            "image_options": [image],  # List format for consistency
            "caption_options": [sample["caption"], sample["foil"]],  # [correct, foil]
            "label": 0,  # Index of correct caption (first one)
            "image": image,  # For backward compatibility
            "caption": sample["caption"],
            "foil": sample["foil"],
            "phenomenon": sample.get("phenomenon", self.phenomenon),
            "image_file": sample["image"],
        }

    def _collate_fn(self, batch):
        """Custom collate function for DataLoader to handle BLA batch format."""
        # batch is a list of samples from __getitem__
        images = []
        captions = []
        foils = []
        phenomena = []
        
        for sample in batch:
            images.append(sample['image'])
            captions.append(sample['caption'])
            foils.append(sample['foil'])
            phenomena.append(sample['phenomenon'])
        
        return {
            'images': torch.stack(images),  # [B, C, H, W]
            'captions': captions,  # List[str]
            'foils': foils,  # List[str]
            'phenomena': phenomena,  # List[str]
        }

    def evaluate(
        self,
        embedding_model,
        aligning_model=None,
        device="cuda",
        batch_size: int = 16,
        indices: Optional[List[int]] = None,
        intermediate_text_layer_names: List[str] = ["final"],
        intermediate_image_layer_names: List[str] = ["final"],
    ):
        return self._evaluate(
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device,
            batch_size=batch_size,
            indices=indices,
            intermediate_text_layer_names=intermediate_text_layer_names,
            intermediate_image_layer_names=intermediate_image_layer_names,
        )

    def _evaluate(
        self,
        embedding_model,
        aligning_model=None,
        device="cuda",
        batch_size: int = 16,
        indices: Optional[List[int]] = None,
        intermediate_text_layer_names: List[str] = ["final"],
        intermediate_image_layer_names: List[str] = ["final"],
    ):
        if 'compute_caption_embeddings_intermediate_batch' not in globals():
            raise ImportError("utils.align functions not available for evaluation")
        
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
        
        text_scores = []
        phenomena_results = defaultdict(list)
        
        # Collect embeddings for return (like other datasets)
        all_img_embs = []
        all_cap_embs = []
        all_foil_embs = []
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="BLA",
            subset_name=self.phenomenon,
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating BLA")):
                # Batch is already properly formatted by collate_fn
                batch_images = batch['images']  # [B, C, H, W]
                batch_captions = batch['captions']  # List[str]
                batch_foils = batch['foils']  # List[str]
                batch_phenomena = batch['phenomena']  # List[str]
                B = len(batch_captions)
                
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
                    cap_embs = cache.get_or_compute_embeddings(
                        batch_captions,
                        "text",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * batch_size
                    )
                    
                    # Get foil embeddings (with caching)
                    foil_embs = cache.get_or_compute_embeddings(
                        batch_foils,
                        "foil",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * batch_size
                    )
                    
                    # Store embeddings (convert to CPU numpy for consistency with other datasets)
                    all_img_embs.append(img_embs.cpu().numpy())
                    all_cap_embs.append(cap_embs.cpu().numpy())
                    all_foil_embs.append(foil_embs.cpu().numpy())
                    
                # Vectorized similarity computation - much faster!
                sim_caption = (cap_embs * img_embs).sum(dim=1)  # [B] - caption similarities
                sim_foil = (foil_embs * img_embs).sum(dim=1)    # [B] - foil similarities
                text_correct = sim_caption > sim_foil  # [B] boolean tensor
                
                # Convert to lists and extend results
                text_scores.extend(text_correct.cpu().tolist())
                
                # Per-phenomena tracking (still need to iterate for this)
                for i, phenomenon in enumerate(batch_phenomena):
                    phenomena_results[phenomenon].append({
                        'text_score': text_correct[i].item(),
                    })
        
        text_acc = float(np.mean(text_scores)) if text_scores else 0.0
        results = {
            "text_contrastive_accuracy": text_acc,
            # For BLA, we only report text contrastive accuracy since there are no negative images
        }
        
        # Macro average across phenomena
        if phenomena_results:
            results['macro_contrastive_accuracy'] = np.mean([
                np.mean([s['text_score'] for s in scores])
                for scores in phenomena_results.values()
            ])
        else:
            results['macro_contrastive_accuracy'] = text_acc
        
        # Add per-phenomena results
        for ph, scores in phenomena_results.items():
            arr = np.array([s['text_score'] for s in scores])
            results[f"{ph}_group_contrastive_accuracy"] = arr.mean() if len(arr) > 0 else 0.0
        
        # Prepare embeddings dict (like other datasets)
        if all_img_embs:
            embeddings = {
                "image_embeddings": np.concatenate(all_img_embs, axis=0),
                "caption_embeddings": np.concatenate(all_cap_embs, axis=0),
                "negative_caption_embeddings": np.concatenate(all_foil_embs, axis=0)[:, None, :],  # Add dimension for consistency
            }
        else:
            embeddings = {
                "image_embeddings": np.array([]),
                "caption_embeddings": np.array([]),
                "negative_caption_embeddings": np.array([]),
            }
        
        return results, embeddings

if __name__ == '__main__':
    BLADataset(
        data_root="./datasets",
    )

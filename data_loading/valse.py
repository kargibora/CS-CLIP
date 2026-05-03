
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm

class VALSEDataset(Dataset):
    """
    VALSE: Vision And Language Structured Evaluation (Hugging Face version)
    Loads from Mayfull/valse_vlms on the Hugging Face Hub.
    Each sample contains an image, positive and negative captions, and metadata.
    """
    def __init__(self, data_root=None, subset_name="all", image_preprocess=None, download=False, verbose=True, **kwargs):
        from datasets import load_dataset
        self.subset_name = subset_name
        self.image_preprocess = image_preprocess
        self.verbose = verbose
        
        # Load the dataset (test split only) - will automatically cache and reuse
        # Setting cache_dir to use data_root if provided for better cache management
        cache_dir = data_root if data_root else None
        
        if verbose:
            print(f"[VALSE] Loading dataset from Hugging Face (cache_dir: {cache_dir})")
        
        self.ds = load_dataset("Mayfull/valse_vlms", split="test", cache_dir=cache_dir)
        
        if verbose:
            print(f"[VALSE] Dataset loaded successfully ({len(self.ds)} total samples)")
        
        # Pre-filter dataset at initialization using select_columns for efficient filtering
        if subset_name != "all":
            if self.verbose:
                print(f"[VALSE] Filtering dataset for linguistic phenomena: {subset_name}")
            
            # Step 1: Create metadata-only dataset (no images loaded)
            metadata_ds = self.ds.select_columns(["linguistic_phenomena"])
            
            # Step 2: Find indices that match the criteria (fast, no image loading)
            filtered_indices = []
            for i in range(len(metadata_ds)):
                if metadata_ds[i]["linguistic_phenomena"] == subset_name:
                    filtered_indices.append(i)
            
            if self.verbose:
                print(f"[VALSE] Found {len(filtered_indices)} samples matching '{subset_name}' from {len(self.ds)} total")
            
            # Step 3: Create filtered dataset using select (efficient)
            self.samples = self.ds.select(filtered_indices)
            
        else:
            self.samples = self.ds
            
        if self.verbose:
            print(f"[VALSE] Loaded {len(self.samples)} samples from Hugging Face (subset: {subset_name})")

    def __len__(self):
        return len(self.samples)

    def get_captions(self):
        """Get all unique captions in the dataset"""
        captions = set()
        # Sample a subset for efficiency since we don't filter upfront anymore
        sample_size = min(1000, len(self.samples))
        for i in range(sample_size):
            sample = self.samples[i]
            if sample["positive_caption"]:
                captions.add(sample["positive_caption"][0])
            if sample["negative_caption"]:
                captions.add(sample["negative_caption"][0])
        return sorted(list(captions))

    def get_image_paths(self):
        """Get all unique image paths in the dataset"""
        # For HuggingFace datasets, we don't have file paths, so return a placeholder
        return [f"huggingface_image_{i}" for i in range(len(self.samples))]

    def get_idx_to_ptr(self, idx):
        """Map dataset index to positive caption pointer"""
        # For compatibility, return idx (not used in VALSE evaluation)
        return idx

    def get_idx_to_candidates_ptr(self, idx):
        """Map dataset index to negative caption pointers"""
        # For compatibility, return list with idx (not used in VALSE evaluation)
        return [idx]

    def __getitem__(self, idx):
        sample = self.samples[int(idx)]  # Convert to regular int for HuggingFace dataset
        # The dataset provides a list of images, but only one per sample
        img_obj = sample["images"][0]
        # Handle different image formats from Hugging Face Datasets
        if isinstance(img_obj, Image.Image):
            image = img_obj
        elif isinstance(img_obj, dict):
            if "bytes" in img_obj:
                from io import BytesIO
                image = Image.open(BytesIO(img_obj["bytes"])).convert("RGB")
            elif "path" in img_obj:
                image = Image.open(img_obj["path"]).convert("RGB")
            else:
                raise ValueError("Unknown image object format in VALSE dataset.")
        else:
            # Fallback: try to open as path
            image = Image.open(img_obj).convert("RGB")
        
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        # Use the first positive/negative caption for evaluation
        caption = sample["positive_caption"][0] if sample["positive_caption"] else ""
        foil = sample["negative_caption"][0] if sample["negative_caption"] else ""
        return {
            "image": image,
            "caption": caption,
            "foil": foil,
            "label": 1,
            "dataset": sample.get("dataset", "unknown"),
            "image_file": sample.get("original_file_name", ""),
            "linguistic_phenomena": sample.get("linguistic_phenomena", ""),
        }

    def _collate_fn(self, batch):
        """Custom collate function for DataLoader to handle VALSE batch format."""
        # batch is a list of samples from __getitem__
        images = []
        captions = []
        foils = []
        phenomena = []
        
        for sample in batch:
            images.append(sample['image'])
            captions.append(sample['caption'])
            foils.append(sample['foil'])
            phenomena.append(sample['linguistic_phenomena'])
        
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
        indices=None,
        intermediate_text_layer_names=["final"],
        intermediate_image_layer_names=["final"],
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
        indices=None,
        intermediate_text_layer_names=["final"],
        intermediate_image_layer_names=["final"],
    ):
        try:
            from utils.align import compute_caption_embeddings_intermediate_batch, compute_image_embeddings_intermediate_batch
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("utils.align functions or embedding_cache not available for evaluation")
        
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
        
        if len(eval_dataset) == 0:
            raise ValueError(f"No samples found for linguistic phenomena: {self.subset_name}")
        
        text_scores = []
        phenomena_results = defaultdict(list)
        
        # Collect embeddings for return (like other datasets)
        all_img_embs = []
        all_cap_embs = []
        all_foil_embs = []
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="VALSE",
            subset_name=self.subset_name,
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating VALSE")):
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
            # For VALSE, we only report text contrastive accuracy since there are no negative images
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
        embeddings = {
            "image_embeddings": np.concatenate(all_img_embs, axis=0),
            "caption_embeddings": np.concatenate(all_cap_embs, axis=0),
            "negative_caption_embeddings": np.concatenate(all_foil_embs, axis=0)[:, None, :],  # Add dimension for consistency
        }

        return results, embeddings
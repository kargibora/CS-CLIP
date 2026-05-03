import json
import os
import random
import webcolors
from typing import Dict, List, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

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


class ColorFoilDataset(Dataset):
    """
    ColorFoil dataset for evaluating color understanding in vision-language models.
    
    Based on: https://github.com/samin9796/ColorFoil/tree/main
    
    The task tests whether models can distinguish between correct color descriptions
    and "foiled" versions where colors are randomly replaced with other colors.
    
    Each sample contains:
    - An image with objects of specific colors
    - A correct caption describing the image with color words
    - A foil caption where color words are randomly replaced
    
    Returns samples in format compatible with alignment pipeline:
        {
          'image_options': [image],
          'caption_options': [correct_caption, foil_caption],
          'label': 0,  # Index of correct caption (always 0)
          'image_id': str,
        }
    """

    def __init__(
        self,
        data_root: str,
        subset_name: str = "all",
        image_preprocess=None,
        cache_dir: Optional[str] = None,
        download: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        self.data_root = data_root
        self.subset_name = subset_name
        self.image_preprocess = image_preprocess
        self.cache_dir = cache_dir or os.path.join(data_root, "cache")
        self.verbose = verbose
        
        # ColorFoil data directory
        self.colorfoil_dir = data_root
        self.images_dir = os.path.join(self.colorfoil_dir, "images")
        self.annotations_file = os.path.join(self.colorfoil_dir, 'annotations', "captions_val2017.json")
        
        # Ensure directories exist
        os.makedirs(self.colorfoil_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if not os.path.exists(self.annotations_file):
            if download:
                self._create_colorfoil_data()
            else:
                raise FileNotFoundError(
                    f"ColorFoil annotations not found at {self.annotations_file}. "
                    f"Please ensure the ColorFoil dataset is prepared or set download=True."
                )
        
        # Load the dataset
        self._load_data()
        
        # Pre-download images to speed up evaluation
        if len(self.examples) > 0:
            self._predownload_images()
        
        if verbose:
            print(f"[ColorFoilDataset] Dataset ready with {len(self.examples)} examples")

    def _create_colorfoil_data(self):
        """
        Create ColorFoil dataset following the original methodology.
        This assumes you have COCO val2017 annotations available.
        """
        if self.verbose:
            print("[ColorFoilDataset] Creating ColorFoil data from COCO annotations...")
        
        # Look for COCO annotations
        coco_annotations_path = os.path.join(self.data_root, "coco", "annotations", "captions_val2017.json")
        if not os.path.exists(coco_annotations_path):
            # Try alternative paths
            alt_paths = [
                os.path.join(self.data_root, "annotations", "captions_val2017.json"),
                os.path.join(self.data_root, "captions_val2017.json"),
                "/annotations/captions_val2017.json"  # Original path from code
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    coco_annotations_path = path
                    break
            else:
                raise FileNotFoundError(
                    "COCO captions_val2017.json not found. Please download COCO 2017 validation annotations."
                )
        
        img_list, cap_list, foil_list = self._prepare_data(coco_annotations_path)
        
        # Save the processed data
        colorfoil_data = []
        for i, (img_url, caption, foil) in enumerate(zip(img_list, cap_list, foil_list)):
            colorfoil_data.append({
                "image_id": f"colorfoil_{i:06d}",
                "image_url": img_url,
                "caption": caption,
                "foil": foil
            })
        
        with open(self.annotations_file, 'w') as f:
            json.dump(colorfoil_data, f, indent=2)
        
        if self.verbose:
            print(f"[ColorFoilDataset] Created {len(colorfoil_data)} ColorFoil examples")

    def _create_foil(self, caption: str) -> str:
        """Create a foil caption by replacing color words with random colors."""
        # Most commonly used colors
        colors = ["blue", "black", "red", "pink", "yellow", "grey", "orange", "white", "green", "brown"]
        
        # Get CSS3 color names
        try:
            css3_colors = set(webcolors.names('css3'))
        except Exception:
            # Fallback to common colors if webcolors fails
            css3_colors = set(colors + ["purple", "cyan", "magenta", "lime", "navy", "teal"])
        
        # Split caption into words and process
        words = caption.split(' ')
        foil_words = []
        
        for word in words:
            # Remove punctuation for color checking
            clean_word = word.strip('.,!?;:"()[]').lower()
            
            if clean_word in css3_colors:
                # Replace with a different random color
                num = random.randint(0, 9)
                if colors[num] == clean_word:
                    foiling_color = colors[num-1]
                else:
                    foiling_color = colors[num]
                
                # Preserve original case and punctuation
                if word[0].isupper():
                    foiling_color = foiling_color.capitalize()
                
                # Add back punctuation
                punctuation = ""
                for char in word:
                    if char in '.,!?;:"()[]':
                        punctuation += char
                
                foil_words.append(foiling_color + punctuation)
            else:
                foil_words.append(word)
        
        return ' '.join(foil_words)

    def _prepare_data(self, coco_annotations_path: str):
        """Prepare ColorFoil data from COCO annotations."""
        img_list = []  # list of image urls
        cap_list = []  # list of captions
        foil_list = []  # list of foiled captions

        # Get CSS3 color names
        try:
            css3_colors = set(webcolors.names('css3'))
        except Exception:
            # Fallback to common colors if webcolors fails
            css3_colors = set(["blue", "black", "red", "pink", "yellow", "grey", "orange", "white", "green", "brown", "purple", "cyan", "magenta", "lime", "navy", "teal"])

        with open(coco_annotations_path) as f:
            d = json.load(f)
            
            if self.verbose:
                print(f"[ColorFoilDataset] Processing {len(d['images'])} COCO images...")
            
            # Create three lists of images, captions and foils
            for img_info in tqdm(d["images"], desc="Processing COCO images"):
                img_id = img_info["id"]
                flag = False
                
                # Find captions for this image
                for annotation in d["annotations"]:
                    if annotation["image_id"] == img_id:
                        caption = annotation["caption"]
                        
                        # Check if caption contains color words
                        for word in caption.split(' '):
                            clean_word = word.strip('.,!?;:"()[]').lower()
                            if clean_word in css3_colors:
                                flag = True
                        
                        if flag:
                            foil = self._create_foil(caption)
                            img_list.append(img_info["coco_url"])
                            cap_list.append(caption)
                            foil_list.append(foil)
               
                            break

        return img_list, cap_list, foil_list

    def _load_data(self):
        """Load ColorFoil annotations."""
        if self.verbose:
            print(f"[ColorFoilDataset] Loading annotations from {self.annotations_file}")
        
        # Check if the file contains ColorFoil format or COCO format
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)
        
        # If it's COCO format (has 'annotations' key), we need to create ColorFoil data
        if isinstance(data, dict) and 'annotations' in data:
            if self.verbose:
                print("[ColorFoilDataset] Found COCO format, creating ColorFoil data...")
            img_list, cap_list, foil_list = self._prepare_data(self.annotations_file)
            
            # Create ColorFoil format
            self.examples = []
            for i, (img_url, caption, foil) in enumerate(zip(img_list, cap_list, foil_list)):
                self.examples.append({
                    "image_id": f"colorfoil_{i:06d}",
                    "image_url": img_url,
                    "caption": caption,
                    "foil": foil
                })
            
            # Save the processed data for future use
            colorfoil_path = os.path.join(os.path.dirname(self.annotations_file), "colorfoil_annotations.json")
            with open(colorfoil_path, 'w') as f:
                json.dump(self.examples, f, indent=2)
            
            if self.verbose:
                print(f"[ColorFoilDataset] Created and saved {len(self.examples)} ColorFoil examples to {colorfoil_path}")
        else:
            # Already in ColorFoil format
            self.examples = data
        
        # Filter by subset if specified
        if self.subset_name != "all":
            original_count = len(self.examples)
            # Add subset filtering logic here if needed
            # For now, keep all examples
            if self.verbose:
                print(f"[ColorFoilDataset] Subset '{self.subset_name}' filtering: {original_count} -> {len(self.examples)} examples")
        
        # Build caption vocabulary
        caption_set = set()
        for ex in self.examples:
            caption_set.add(ex['caption'])
            caption_set.add(ex['foil'])
        
        self.captions = sorted(caption_set)
        self.caption_to_idx = {cap: i for i, cap in enumerate(self.captions)}

    def __len__(self) -> int:
        return len(self.examples)

    def _download_image(self, image_url: str, image_path: str) -> bool:
        """Download image from URL if it doesn't exist."""
        if os.path.exists(image_path):
            return True
        
        try:
            import requests
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                return True
        except Exception as e:
            if self.verbose:
                print(f"[ColorFoilDataset] Failed to download {image_url}: {e}")
        
        return False

    def _predownload_images(self, max_workers: int = 4):
        """Pre-download all images to avoid slow evaluation."""
        import concurrent.futures
        import requests
        
        missing_images = []
        for example in self.examples:
            image_id = example['image_id']
            image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                missing_images.append((example['image_url'], image_path))
        
        if not missing_images:
            if self.verbose:
                print("[ColorFoilDataset] All images already downloaded")
            return
        
        if self.verbose:
            print(f"[ColorFoilDataset] Pre-downloading {len(missing_images)} images...")
        
        os.makedirs(self.images_dir, exist_ok=True)
        
        def download_single(url_path_pair):
            url, path = url_path_pair
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    with open(path, 'wb') as f:
                        f.write(response.content)
                    return True
            except Exception:
                pass
            return False
        
        # Download in parallel
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_single, pair) for pair in missing_images]
            
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), 
                                          total=len(missing_images), 
                                          desc="Downloading images")):
                if future.result():
                    success_count += 1
        
        if self.verbose:
            print(f"[ColorFoilDataset] Successfully downloaded {success_count}/{len(missing_images)} images")

    def __getitem__(self, idx: int) -> Dict:
        idx = int(idx)
        example = self.examples[idx]
        
        # Determine image path
        image_id = example['image_id']
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        
        # Download image if needed and not exists
        if not os.path.exists(image_path):
            os.makedirs(self.images_dir, exist_ok=True)
            if not self._download_image(example['image_url'], image_path):
                # If download fails, create a placeholder or skip
                raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
            if self.image_preprocess is not None:
                image = self.image_preprocess(image)
        except Exception as e:
            if self.verbose:
                print(f"[ColorFoilDataset] Error loading image {image_path}: {e}")
            raise
        
        return {
            "image_options": [image],  # Single image
            "caption_options": [example['caption'], example['foil']],  # [correct, foil]
            "label": 0,  # Index of correct caption (always first)
            "image_id": example['image_id'],
            "image_url": example.get('image_url', ''),
        }

    def get_captions(self) -> List[str]:
        """Return the unique caption vocabulary."""
        return self.captions

    def get_image_paths(self) -> List[str]:
        """Return list of image paths."""
        paths = []
        for ex in self.examples:
            image_path = os.path.join(self.images_dir, f"{ex['image_id']}.jpg")
            paths.append(image_path)
        return paths

    def get_idx_to_ptr(self, idx: int) -> int:
        """Map dataset index -> caption pointer for correct caption."""
        example = self.examples[idx]
        caption = example['caption']
        return self.caption_to_idx.get(caption, -1)

    def get_idx_to_candidates_ptr(self, idx: int) -> List[int]:
        """Map dataset index -> list of foil caption indices."""
        example = self.examples[idx]
        foil = example['foil']
        ptr = self.caption_to_idx.get(foil)
        return [ptr] if ptr is not None else []

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
        """Public wrapper for evaluation."""
        return self._evaluate(
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device,
            batch_size=batch_size,
            indices=indices,
            intermediate_text_layer_names=intermediate_text_layer_names,
            intermediate_image_layer_names=intermediate_image_layer_names,
        )

    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader to handle ColorFoil samples.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and caption lists
        """
        batch_images = []
        batch_captions = []
        batch_foils = []
        batch_image_ids = []
        
        for sample in batch:
            # Extract components from each sample
            image = sample['image_options'][0]  # Single image
            caption_options = sample['caption_options']  # [correct_caption, foil_caption]
            
            batch_images.append(image)
            batch_captions.append(caption_options[0])  # correct caption
            batch_foils.append(caption_options[1])     # foil caption
            batch_image_ids.append(sample.get('image_id', ''))
        
        # Stack images into a batch tensor
        batch_images = torch.stack(batch_images)  # [B, C, H, W]
        
        return {
            'images': batch_images,
            'captions': batch_captions,    # List[str] - correct captions
            'foils': batch_foils,          # List[str] - foil captions
            'image_ids': batch_image_ids   # List[str] - image identifiers
        }

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
        """
        Evaluate ColorFoil dataset with DataLoader optimization and caching.
        
        For each example, we test whether the model assigns higher similarity
        to the correct caption vs. the foil caption.
        """
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
            iterate = indices
        else:
            eval_dataset = self
            iterate = list(range(len(self)))
            
        # Use DataLoader for efficient multi-threaded batch loading
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Multi-threaded loading
            pin_memory=True,  # Faster GPU transfer
            collate_fn=self._collate_fn  # Custom collate function
        )
        
        correct_scores = []
        all_similarities = []
        
        image_emb_list = []
        caption_emb_list = []
        foil_emb_list = []
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="ColorFoil",
            subset_name=self.subset_name,
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
        
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating ColorFoil")):
                batch_images = batch['images']  # [B, C, H, W]
                batch_captions = batch['captions']  # List[str]
                batch_foils = batch['foils']  # List[str]
                B = len(batch_captions)
                
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
                    
                    foil_embs = cache.get_or_compute_embeddings(
                        batch_foils,
                        "foil",
                        compute_caption_embeddings_intermediate_batch,
                        intermediate_text_layer_names,
                        start_idx=batch_idx * batch_size
                    )

                # Vectorized similarity computation for the entire batch
                correct_sims = (img_embs * cap_embs).sum(dim=1)   # similarity with correct caption
                foil_sims = (img_embs * foil_embs).sum(dim=1)     # similarity with foil caption
                
                # Check if correct > foil for each sample
                batch_correct = correct_sims > foil_sims
                correct_scores.extend(batch_correct.cpu().tolist())
                
                # Store detailed similarity information
                for i in range(B):
                    all_similarities.append({
                        'correct_similarity': correct_sims[i].item(),
                        'foil_similarity': foil_sims[i].item(),
                        'difference': (correct_sims[i] - foil_sims[i]).item()
                    })
                
                # Store embeddings
                image_emb_list.append(img_embs.cpu())
                caption_emb_list.append(cap_embs.cpu())
                foil_emb_list.append(foil_embs.cpu())

        # Compute accuracy
        accuracy = float(np.mean(correct_scores)) if correct_scores else 0.0
        
        # Compute additional statistics
        sim_differences = [s['difference'] for s in all_similarities]
        mean_diff = float(np.mean(sim_differences)) if sim_differences else 0.0
        std_diff = float(np.std(sim_differences)) if sim_differences else 0.0
        
        # Prepare results
        results = {
            "contrastive_accuracy": accuracy,
            "mean_similarity_difference": mean_diff,
            "std_similarity_difference": std_diff,
        }
        
        # Prepare embeddings
        image_embeddings = torch.cat(image_emb_list, dim=0).numpy()
        caption_embeddings = torch.cat(caption_emb_list, dim=0).numpy()
        negative_caption_embeddings = torch.cat(foil_emb_list, dim=0).numpy()
        negative_caption_embeddings = np.expand_dims(negative_caption_embeddings, axis=1)

        embeddings = {
            "image_embeddings": image_embeddings,
            "caption_embeddings": caption_embeddings,
            "negative_caption_embeddings": negative_caption_embeddings,
        }


        return results, embeddings


    def split_dataset(
        self,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
        split_type: Literal["random", "object"] = "random",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Split dataset into train/val/test."""
        rng = random.Random(seed)
        
        if split_type == "random":
            indices = list(range(len(self)))
            rng.shuffle(indices)
            n = len(indices)
            n_test = int(n * test_ratio)
            n_val = int((n - n_test) * val_ratio)
            
            test_idx = np.array(indices[:n_test], dtype=np.int64)
            val_idx = np.array(indices[n_test:n_test + n_val], dtype=np.int64)
            train_idx = np.array(indices[n_test + n_val:], dtype=np.int64)
            
            return {
                "train": {"indices": train_idx},
                "val": {"indices": val_idx},
                "test": {"indices": test_idx},
            }
        
        elif split_type == "object":
            # For ColorFoil, object split = random split since each image is unique
            return self.split_dataset(val_ratio, test_ratio, seed, "random")
        
        else:
            raise ValueError(f"Unknown split_type={split_type}")

    def __getstate__(self):
        """Ensure the dataset can be pickled."""
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)


class ColorFoilNeg(Dataset):
    """
    Wrapper for ColorFoilDataset to provide tokenized positives and negatives.
    Compatible with other *Neg dataset classes.
    """
    
    def __init__(
        self,
        colorfoil_dataset: ColorFoilDataset,
        indices: List[int],
        num_negatives: int = 1,
    ):
        super().__init__()
        self.dataset = colorfoil_dataset
        self.num_negatives = num_negatives
        self.idx_to_dataset_idx = {i: idx for i, idx in enumerate(indices)}
        
        print(f"ColorFoilNeg initialized with {self.num_negatives} negatives per sample")
        print("ColorFoil dataset for color understanding evaluation")

    def __len__(self) -> int:
        return len(self.idx_to_dataset_idx)
    
    def __getitem__(self, idx: int):
        idx = int(idx)
        sample = self.dataset[self.idx_to_dataset_idx[idx]]
        
        pos_image = sample['image_options'][0]  # Single image
        pos_text = sample['caption_options'][0]  # Correct caption
        neg_text = sample['caption_options'][1]  # Foil caption
        
        # Tokenize using CLIP
        import clip
        pos_tok = clip.tokenize(pos_text, truncate=True).squeeze(0)
        neg_tok = clip.tokenize(neg_text, truncate=True).squeeze(0)
        
        # For compatibility
        all_neg_toks = neg_tok.unsqueeze(0)  # Shape: (1, seq_len)
        
        return pos_image, pos_tok, neg_tok, all_neg_toks
    
    def collate_fn(self, batch: List[tuple]) -> dict:
        images, pos_toks, neg_toks, all_neg_toks = zip(*batch)
        
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images, dim=0)
        
        pos_toks = torch.stack(pos_toks, dim=0)
        neg_toks = torch.stack(neg_toks, dim=0)
        all_neg_toks = torch.stack(all_neg_toks, dim=0)
        
        return {
            'images': images,
            'pos_tokens': pos_toks,
            'neg_token': neg_toks,
            'all_neg_tokens': all_neg_toks,
        }


if __name__ == "__main__":
    # Simple test to verify dataset loading
    from torchvision import transforms

    simple_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Test the dataset
    ds = ColorFoilDataset(
        "./datasets",
        subset_name="all", 
        image_preprocess=simple_preprocess, 
        download=False,  # Set to True to create from COCO
        verbose=True
    )
    print(f"Loaded {len(ds)} samples.")
    
    if len(ds) > 0:
        sample = ds[0]
        print("Sample keys:", sample.keys())
        img = sample["image_options"][0]
        print("Image shape:", img.shape if hasattr(img, 'shape') else type(img))
        print("Correct caption:", sample["caption_options"][0])
        print("Foil caption:", sample["caption_options"][1])
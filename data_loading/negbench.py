import logging
import os
import random
from collections import defaultdict
from typing import List, Dict
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import clip
import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from utils.align import (
    compute_caption_embeddings_intermediate_batch,
    compute_image_embeddings_intermediate_batch,
)

def extract_video_frame(video_path: str, frame_idx: int = 0) -> Image.Image:
    """
    Extract a frame from a video file using OpenCV.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to extract (default: first frame)
        
    Returns:
        PIL Image of the extracted frame
    """
    try:
        import cv2
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from video: {video_path}")
        
        # Convert BGR to RGB and create PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        return pil_image
        
    except ImportError:
        logging.warning("OpenCV not available for video processing. Install with: pip install opencv-python")
        # Return placeholder image
        return Image.new('RGB', (224, 224), color='black')
    except Exception as e:
        logging.warning(f"Could not extract frame from video {video_path}: {e}")
        # Return placeholder image
        return Image.new('RGB', (224, 224), color='black')

class NegBenchDataset(Dataset):
    """
    NegBench Dataset for evaluating negation understanding in vision-language models.
    
    Supports multiple task types and subsets:
    - MCQ (Multiple Choice Questions): 4 captions per image, 1 correct answer
    - Retrieval: Image-caption pairs for retrieval evaluation
    - Binary MCQ: 2 captions per image, 1 correct answer
    
    Available subsets:
    - COCO_val_mcq_llama3.1_rephrased (MCQ)
    - COCO_val_retrieval (Retrieval)
    - COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true (Retrieval)
    - VOC2007_mcq_llama3.1_rephrased (MCQ)
    - VOC2007_retrieval (Retrieval)
    - VOC2007_negated_retrieval_llama3.1_rephrased_affneg_true (Retrieval)
    - msr_vtt_retrieval (Retrieval - Video)
    - msr_vtt_retrieval_rephrased_llama (Retrieval - Video)
    - msr_vtt_mcq_rephrased_llama (MCQ - Video)
    
    CSV structure for MCQ:
        - image_path: Path to image
        - caption_0 to caption_3: Four caption options
        - correct_answer: Index of correct caption (0-3)
        
    CSV structure for Retrieval:
        - filepath/image_path: Path to image
        - caption/captions: Caption text
        
    CSV structure for Binary MCQ:
        - image_path: Path to image  
        - caption_0 to caption_1: Two caption options
        - correct_answer: Index of correct caption (0-1)
    """
    
    # Available subsets and their configurations
    SUBSET_CONFIGS = {
        # COCO Image datasets
        'COCO_val_mcq_llama3.1_rephrased': {
            'task_type': 'mcq',
            'csv_file': 'images/COCO_val_mcq_llama3.1_rephrased.csv',
            'data_type': 'image'
        },
        'COCO_val_retrieval': {
            'task_type': 'retrieval',
            'csv_file': 'images/COCO_val_retrieval.csv',
            'data_type': 'image'
        },
        'COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true': {
            'task_type': 'retrieval',
            'csv_file': 'images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv',
            'data_type': 'image'
        },
        # VOC2007 Image datasets
        'VOC2007_mcq_llama3.1_rephrased': {
            'task_type': 'mcq',
            'csv_file': 'images/VOC2007_mcq_llama3.1_rephrased.csv',
            'data_type': 'image'
        },
        'VOC2007_retrieval': {
            'task_type': 'retrieval',
            'csv_file': 'images/VOC2007_retrieval.csv',
            'data_type': 'image'
        },
        'VOC2007_negated_retrieval_llama3.1_rephrased_affneg_true': {
            'task_type': 'retrieval',
            'csv_file': 'images/VOC2007_negated_retrieval_llama3.1_rephrased_affneg_true.csv',
            'data_type': 'image'
        },
        # MSR-VTT Video datasets
        'msr_vtt_retrieval': {
            'task_type': 'retrieval',
            'csv_file': 'videos/msr_vtt_retrieval.csv',
            'data_type': 'video'
        },
        'msr_vtt_retrieval_rephrased_llama': {
            'task_type': 'retrieval',
            'csv_file': 'videos/msr_vtt_retrieval_rephrased_llama.csv',
            'data_type': 'video'
        },
        'msr_vtt_mcq_rephrased_llama': {
            'task_type': 'mcq',
            'csv_file': 'videos/msr_vtt_mcq_rephrased_llama.csv',
            'data_type': 'video'
        },
        # Additional subsets can be added here
        'all': {
            'task_type': 'mcq',
            'csv_file': 'images/COCO_val_mcq_llama3.1_rephrased.csv',
            'data_type': 'image'
        }
    }
    
    def __init__(
        self,
        data_path: str,
        subset_name: str = 'COCO_val_mcq_llama3.1_rephrased',
        image_preprocess: callable = None,
        **kwargs
    ):
        """
        Initialize NegBench dataset.
        
        Args:
            data_path: Path to NegBench data directory (should contain evaluation_data folder)
            subset_name: Name of the subset to load
            image_preprocess: Image preprocessing function
        """
        self.data_path = data_path
        self.subset_name = subset_name
        self.image_preprocess = image_preprocess
        
        # Get subset configuration
        if subset_name not in self.SUBSET_CONFIGS:
            available_subsets = list(self.SUBSET_CONFIGS.keys())
            raise ValueError(f"Unknown subset '{subset_name}'. Available subsets: {available_subsets}")
        
        subset_config = self.SUBSET_CONFIGS[subset_name]
        self.task_type = subset_config['task_type']
        self.data_type = subset_config['data_type']
        
        # Construct CSV path
        csv_file = subset_config['csv_file']
        self.csv_path = os.path.join(data_path, csv_file)
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # Load and correct CSV data
        self._load_and_correct_data()
        
        # Validate and process dataset based on task type
        self._validate_and_process_data()
        
        # Extract unique captions and images
        self.captions = self.get_captions()
        self.image_paths = self.get_image_paths()
        
        logging.info(f"NegBench {subset_name} ({self.task_type}): {len(self.image_paths)} images, {len(self.captions)} captions")
        
        # Create caption mapping for efficiency
        self.caption_to_idx = {caption: idx for idx, caption in enumerate(self.captions)}
        
        # Determine number of candidates based on task type
        if self.task_type == 'mcq':
            # Count caption columns dynamically
            caption_cols = [col for col in self.df.columns if col.startswith('caption_')]
            self.number_of_candidates = len(caption_cols)
        elif self.task_type == 'binary_mcq':
            self.number_of_candidates = 2
        else:  # retrieval
            self.number_of_candidates = 1
    
    def _load_and_correct_data(self):
        """Load CSV data and correct file paths."""
        # Load CSV data
        self.df = pd.read_csv(self.csv_path)
        logging.info(f"Loaded NegBench dataset from {self.csv_path} with {len(self.df)} samples")
        
        # Apply path corrections based on data type
        if self.data_type == 'image':
            self._correct_coco_paths()
        elif self.data_type == 'video':
            self._correct_msrvtt_paths()
    
    def _correct_coco_paths(self):
        """Correct COCO image paths to point to the actual image location."""
        def fix_coco_path(filepath):
            if pd.isna(filepath):
                return filepath
            return os.path.join(self.data_path, filepath)
        
        # Update both possible path columns
        for col in ['filepath', 'image_path']:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(fix_coco_path)
                logging.info(f"Updated {col} paths for COCO dataset")
                break
    
    def _correct_msrvtt_paths(self):
        """Correct MSR-VTT video paths to point to the actual video location."""
        def fix_msrvtt_path(filepath):
            if pd.isna(filepath):
                return filepath
            return os.path.join(self.data_path, filepath)
        
        # Update both possible path columns
        for col in ['filepath', 'image_path']:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(fix_msrvtt_path)
                logging.info(f"Updated {col} paths for MSR-VTT dataset")
                break
    
    def _validate_and_process_data(self):
        """Validate CSV structure and process data based on task type."""
        
        if self.task_type == 'mcq':
            # Find all caption columns dynamically
            caption_cols = [col for col in self.df.columns if col.startswith('caption_')]
            caption_cols.sort()  # Ensure consistent order
            
            required_cols = ['image_path'] + caption_cols + ['correct_answer']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"MCQ task requires columns {required_cols}, missing: {missing_cols}")
            
            # Validate that we have at least 2 caption options
            if len(caption_cols) < 2:
                raise ValueError(f"MCQ task requires at least 2 caption options, found: {len(caption_cols)}")
                
        elif self.task_type == 'binary_mcq':
            required_cols = ['image_path', 'caption_0', 'caption_1', 'correct_answer']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Binary MCQ task requires columns {required_cols}, missing: {missing_cols}")
                
        elif self.task_type == 'retrieval':
            # Support multiple common column names for retrieval
            image_col = None
            caption_col = None
            
            for col in ['filepath', 'image_path']:
                if col in self.df.columns:
                    image_col = col
                    break
            
            for col in ['caption', 'captions']:
                if col in self.df.columns:
                    caption_col = col
                    break
                    
            if image_col is None or caption_col is None:
                raise ValueError("Retrieval task requires image column (filepath/image_path) and caption column (caption/captions)")
                
            # Standardize column names
            if image_col != 'image_path':
                self.df['image_path'] = self.df[image_col]
            if caption_col != 'caption':
                self.df['caption'] = self.df[caption_col]
        
        # Check if image paths exist (warn if not found)
        missing_count = 0
        for _, row in self.df.iterrows():
            if not os.path.exists(row['image_path']):
                missing_count += 1
                
        if missing_count > 0:
            logging.warning(f"Warning: {missing_count}/{len(self.df)} image paths do not exist")
    
    def get_captions(self) -> List[str]:
        """Extract all unique captions from the dataset."""
        captions = set()
        
        if self.task_type == 'mcq':
            # Find all caption columns dynamically
            caption_cols = [col for col in self.df.columns if col.startswith('caption_')]
            caption_cols.sort()
            for col in caption_cols:
                captions.update(self.df[col].tolist())
        elif self.task_type == 'binary_mcq':
            for i in range(2):
                captions.update(self.df[f'caption_{i}'].tolist())
        else:  # retrieval
            captions.update(self.df['caption'].tolist())
            
        return sorted(list(captions))
    
    def get_image_paths(self) -> List[str]:
        """Extract all unique image paths from the dataset."""
        return sorted(list(set(self.df['image_path'].tolist())))
    
    def get_idx_to_ptr(self, idx: int) -> int:
        """Get mapping from sample index to positive caption index in caption_to_idx."""
        row = self.df.iloc[idx]
        
        if self.task_type in ['mcq', 'binary_mcq']:
            correct_idx = int(row['correct_answer'])
            caption = row[f'caption_{correct_idx}']
        else:  # retrieval
            caption = row['caption']
            
        return self.caption_to_idx[caption]
    
    def get_idx_to_candidates_ptr(self, idx: int) -> List[int]:
        """Get mapping from sample index to negative caption indices."""
        row = self.df.iloc[idx]
        
        if self.task_type in ['mcq', 'binary_mcq']:
            correct_idx = int(row['correct_answer'])
            
            # Get all caption columns dynamically
            caption_cols = [col for col in self.df.columns if col.startswith('caption_')]
            caption_cols.sort()
            
            # Get all captions except the correct one
            negative_captions = []
            for i, col in enumerate(caption_cols):
                if i != correct_idx:
                    negative_captions.append(row[col])
                    
            return [self.caption_to_idx[caption] for caption in negative_captions]
        else:  # retrieval - no explicit negatives, would need to be constructed
            return []
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and preprocess image/video
        image_path = row['image_path']
        
        if self.data_type == 'video':
            # Extract frame from video file
            image = extract_video_frame(image_path, frame_idx=0)  # Extract first frame
        else:
            # Regular image loading
            image = Image.open(image_path).convert('RGB')
            
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        
        if self.task_type == 'mcq':
            # Get caption columns dynamically
            caption_cols = [col for col in self.df.columns if col.startswith('caption_')]
            caption_cols.sort()
            caption_options = [row[col] for col in caption_cols]
            label = int(row['correct_answer'])
            
            return {
                "image_options": image,
                "caption_options": caption_options,
                "label": label,
                "image_path": image_path
            }
            
        elif self.task_type == 'binary_mcq':
            caption_options = [row[f'caption_{i}'] for i in range(2)]
            label = int(row['correct_answer'])
            
            return {
                "image_options": image,
                "caption_options": caption_options,
                "label": label,
                "image_path": image_path
            }
            
        else:  # retrieval
            caption = row['caption']
            
            return {
                "image_options": image,
                "caption": caption,
                "image_path": image_path
            }
    
    def _collate_fn(self, batch):
        """Custom collate function for DataLoader."""
        if self.task_type in ['mcq', 'binary_mcq']:
            batch_images = []
            batch_caption_options = []
            batch_labels = []
            batch_image_paths = []
            
            for sample in batch:
                batch_images.append(sample['image_options'])
                batch_caption_options.append(sample['caption_options'])
                batch_labels.append(sample['label'])
                batch_image_paths.append(sample['image_path'])
            
            batch_images = torch.stack(batch_images)
            
            return {
                'images': batch_images,
                'caption_options': batch_caption_options,
                'labels': batch_labels,
                'image_paths': batch_image_paths
            }
        else:  # retrieval
            batch_images = []
            batch_captions = []
            batch_image_paths = []
            
            for sample in batch:
                batch_images.append(sample['image_options'])
                batch_captions.append(sample['caption'])
                batch_image_paths.append(sample['image_path'])
            
            batch_images = torch.stack(batch_images)
            
            return {
                'images': batch_images,
                'captions': batch_captions,
                'image_paths': batch_image_paths
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
    ) -> tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Evaluate model on NegBench dataset.
        
        For MCQ tasks: Compute accuracy on multiple choice questions
        For Retrieval tasks: Compute image-to-text and text-to-image retrieval metrics
        
        Returns:
            results: Dictionary with evaluation metrics
            embeddings: Dictionary with computed embeddings
        """
        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")
        
        from torch.utils.data import Subset
        
        # Create subset if indices provided
        if indices is not None:
            eval_dataset = Subset(self, indices)
        else:
            eval_dataset = self
        
        # Create DataLoader
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        # Storage for results and embeddings
        all_correct = []
        all_img_embs = []
        all_pos_embs = []
        all_neg_embs = []
        
        # Determine cache name based on CSV filename and task type
        csv_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        
        with EmbeddingCache(
            dataset_name="NegBench",
            subset_name=f"{csv_name}_{self.task_type}",
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Evaluating NegBench {self.task_type}")):
                batch_images = batch['images'].to(device)
                B = batch_images.shape[0]
                
                if B == 0:
                    continue
                
                with torch.no_grad():
                    # Standard CLIP path
                    img_embs = cache.get_or_compute_embeddings(
                        batch_images,
                        "image",
                        compute_image_embeddings_intermediate_batch,
                        intermediate_image_layer_names,
                        start_idx=batch_idx * batch_size
                    )
                    img_embs = img_embs.float() / img_embs.norm(dim=-1, keepdim=True)
                    
                    if self.task_type in ['mcq', 'binary_mcq']:
                        # MCQ evaluation
                        batch_caption_options = batch['caption_options']
                        batch_labels = torch.tensor(batch['labels'], device=device)
                        
                        # Flatten all captions for batch processing
                        all_captions = []
                        for options in batch_caption_options:
                            all_captions.extend(options)
                        
                        # Get caption embeddings
                        cap_embs = cache.get_or_compute_embeddings(
                            all_captions,
                            "text",
                            compute_caption_embeddings_intermediate_batch,
                            intermediate_text_layer_names,
                            start_idx=batch_idx * batch_size * self.number_of_candidates
                        )
                        cap_embs = cap_embs.float() / cap_embs.norm(dim=-1, keepdim=True)
                        
                        # Reshape to [B, num_candidates, D]
                        cap_embs = cap_embs.view(B, self.number_of_candidates, -1)
                        
                        # Compute similarities: [B, num_candidates]
                        similarities = torch.bmm(
                            img_embs.unsqueeze(1),  # [B, 1, D]
                            cap_embs.transpose(1, 2)  # [B, D, num_candidates]
                        ).squeeze(1)  # [B, num_candidates]
                        
                        # Check if predictions are correct
                        predicted = similarities.argmax(dim=1)
                        correct = (predicted == batch_labels).cpu().numpy()
                        all_correct.extend(correct)
                        
                        # Store embeddings for return
                        all_img_embs.append(img_embs.cpu())
                        
                        # Extract positive and negative caption embeddings
                        pos_embs = cap_embs[torch.arange(B), batch_labels]  # [B, D]
                        all_pos_embs.append(pos_embs.cpu())
                        
                        # Create negative embeddings tensor
                        neg_mask = torch.ones(B, self.number_of_candidates, dtype=torch.bool)
                        neg_mask[torch.arange(B), batch_labels] = False
                        neg_embs = cap_embs[neg_mask].view(B, self.number_of_candidates - 1, -1)
                        all_neg_embs.append(neg_embs.cpu())
                        
                    else:  # retrieval
                        batch_captions = batch['captions']
                        
                        # Get caption embeddings
                        cap_embs = cache.get_or_compute_embeddings(
                            batch_captions,
                            "text",
                            compute_caption_embeddings_intermediate_batch,
                            intermediate_text_layer_names,
                            start_idx=batch_idx * batch_size
                        )
                        cap_embs = cap_embs.float() / cap_embs.norm(dim=-1, keepdim=True)
                        
                        # Store embeddings
                        all_img_embs.append(img_embs.cpu())
                        all_pos_embs.append(cap_embs.cpu())
        
        # Compute final results
        if self.task_type in ['mcq', 'binary_mcq']:
            accuracy = np.mean(all_correct)
            results = {
                "accuracy": float(accuracy),
                "text_contrastive_accuracy": float(accuracy)  # Compatibility with other datasets
            }
            
            embeddings = {
                "image_embeddings": torch.cat(all_img_embs, dim=0),
                "caption_embeddings": torch.cat(all_pos_embs, dim=0),
                "negative_caption_embeddings": torch.cat(all_neg_embs, dim=0)
            }
            
        else:  # retrieval
            # For retrieval, we could implement I2T and T2I metrics if needed
            # For now, just return embeddings
            results = {}
            embeddings = {
                "image_embeddings": torch.cat(all_img_embs, dim=0),
                "caption_embeddings": torch.cat(all_pos_embs, dim=0)
            }
        
        return results, embeddings
    
    def split_dataset(
        self, 
        val_ratio: float = 0.1, 
        test_ratio: float = 0.1, 
        seed: int = 42, 
        split_type: str = 'random'
    ) -> Dict[str, Dict[str, List[int]]]:
        """Split dataset into train/val/test splits."""
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
    split_type: Literal['random', 'image'] = 'random'
):
    """Split NegBench dataset into train/val/test."""
    np.random.seed(seed)
    random.seed(seed)
    n = len(dataset)
    
    if split_type == 'image':
        # Group by image path to ensure same image doesn't appear in multiple splits
        image_to_indices = defaultdict(list)
        for idx in range(n):
            row = dataset.df.iloc[idx]
            image_path = row['image_path']
            image_to_indices[image_path].append(idx)
        
        image_paths = list(image_to_indices.keys())
        random.shuffle(image_paths)
        
        n_images = len(image_paths)
        n_test_images = int(n_images * test_ratio)
        n_val_images = int(n_images * val_ratio)
        
        test_images = image_paths[:n_test_images]
        val_images = image_paths[n_test_images:n_test_images + n_val_images]
        train_images = image_paths[n_test_images + n_val_images:]
        
        test_idx = [idx for img in test_images for idx in image_to_indices[img]]
        val_idx = [idx for img in val_images for idx in image_to_indices[img]]
        train_idx = [idx for img in train_images for idx in image_to_indices[img]]
        
    else:  # random
        all_idx = list(range(n))
        random.shuffle(all_idx)
        
        n_test = int(n * test_ratio)
        test_idx = all_idx[:n_test]
        rem_idx = all_idx[n_test:]
        
        adj_val_ratio = val_ratio / (1 - test_ratio)
        train_idx, val_idx = train_test_split(rem_idx, test_size=adj_val_ratio, random_state=seed)
    
    return {
        'train': {'indices': train_idx},
        'val': {'indices': val_idx},
        'test': {'indices': test_idx},
    }


class NegBenchNeg(Dataset):
    """
    NegBench dataset wrapper for negative training similar to other *Neg classes.
    Converts MCQ format to positive/negative pairs for contrastive training.
    """
    
    def __init__(self, dataset: NegBenchDataset, indices: List[int]):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        
        # Ensure we're working with MCQ-type datasets
        if dataset.task_type not in ['mcq', 'binary_mcq']:
            raise ValueError("NegBenchNeg only supports MCQ and binary_mcq task types")
        
        # Index mapping
        self.idx_to_indices = {idx: i for idx, i in enumerate(indices)}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx = self.idx_to_indices[idx]
        sample = self.dataset[sample_idx]
        
        # Extract positive and negative captions
        caption_options = sample['caption_options']
        label = sample['label']
        
        pos_image = sample['image_options']
        pos_text = caption_options[label]
        
        # Get negative captions (all except positive)
        neg_texts = [caption_options[i] for i in range(len(caption_options)) if i != label]
        
        # Randomly select one negative for this sample
        neg_text = random.choice(neg_texts)
        
        # Tokenize captions
        tokenized_caption = clip.tokenize(pos_text).squeeze(0)
        neg_tokenized_caption = clip.tokenize(neg_text).squeeze(0)
        neg_tokenized_captions = clip.tokenize(neg_texts).squeeze(0)
        
        return pos_image, tokenized_caption, neg_tokenized_caption, neg_tokenized_captions
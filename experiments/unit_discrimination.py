"""
Unit Discrimination Experiment

Tests: "How often does the model correctly prefer a positive unit (component/relation) 
over its negative counterpart?"

This is a simple, direct test of compositional understanding:
- For each positive component, compare it against each of its negative components
- For each positive relation, compare it against each of its negative relations
- Report accuracy: % of times model assigns higher similarity to positive than negative

Author: Auto-generated
"""

import os
import sys
import json
import random
import logging
import argparse
import tarfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports from this project are done inside functions to avoid circular imports

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ComponentPair:
    """A positive component paired with a negative component."""
    image_path: str
    image_key: str
    positive: str
    negative: str
    negative_type: str  # e.g., "swap", "attribute", "random"
    original_caption: str = ""


@dataclass
class RelationPair:
    """A positive relation paired with a negative relation."""
    image_path: str
    image_key: str
    positive: str  # Full relation string: "subject relation object"
    negative: str  # Negative relation string
    negative_type: str  # e.g., "swap", "antonym", "negation", "wrong_object", "wrong_subject"
    subject: str = ""
    relation: str = ""
    obj: str = ""
    original_caption: str = ""


@dataclass 
class DiscriminationResult:
    """Result of a single discrimination test."""
    sample_id: str
    unit_type: str  # "component" or "relation"
    negative_type: str
    positive_text: str
    negative_text: str
    score_positive: float
    score_negative: float
    correct: bool  # True if score_positive > score_negative
    margin: float  # score_positive - score_negative


# =============================================================================
# Helper Functions
# =============================================================================

def is_valid_text(text: str) -> bool:
    """Check if text is valid for evaluation."""
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 2:
        return False
    if text.lower() in ['nan', 'none', 'null', '']:
        return False
    return True


# =============================================================================
# Sample Generators
# =============================================================================

class COCOUnitPairGenerator:
    """Generates positive/negative unit pairs from COCO-style JSON data."""
    
    def __init__(
        self,
        json_path: str,
        image_root: str,
        max_samples: int = 5000,
        seed: int = 42,
    ):
        self.json_path = json_path
        self.image_root = image_root
        self.max_samples = max_samples
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Load data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        if isinstance(self.data, dict):
            self.data = [self.data]
        
        logger.info(f"Loaded {len(self.data)} samples from {json_path}")
    
    def _get_image_path(self, sample: Dict) -> Optional[str]:
        """Get image path from sample."""
        # Try different possible keys
        for key in ['image_path', 'img_path', 'image', 'file_name', 'filename']:
            if key in sample:
                img_path = sample[key]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.image_root, img_path)
                if os.path.exists(img_path):
                    return img_path
        
        # Try constructing from image_id
        if 'image_id' in sample:
            img_id = sample['image_id']
            for pattern in [f"{img_id}.jpg", f"COCO_val2014_{img_id:012d}.jpg", 
                           f"COCO_train2014_{img_id:012d}.jpg"]:
                img_path = os.path.join(self.image_root, pattern)
                if os.path.exists(img_path):
                    return img_path
        
        return None
    
    def generate_component_pairs(self) -> List[ComponentPair]:
        """Generate positive/negative component pairs."""
        pairs = []
        
        for sample in tqdm(self.data, desc="Generating component pairs"):
            positive_components = sample.get('positive_components', [])
            negative_components = sample.get('negative_components', {})
            
            if not positive_components or not negative_components:
                continue
            
            image_path = self._get_image_path(sample)
            if not image_path:
                continue
            
            image_key = sample.get('sample_id', sample.get('image_id', str(hash(image_path))))
            original_caption = sample.get('original_caption', sample.get('caption', ''))
            
            # For each positive component, pair with its negatives
            for pos_comp in positive_components:
                if not is_valid_text(pos_comp):
                    continue
                
                # Get negatives for this component
                neg_list = negative_components.get(pos_comp, [])
                
                for neg_info in neg_list:
                    if isinstance(neg_info, dict):
                        neg_text = neg_info.get('negative', neg_info.get('text', ''))
                        neg_type = neg_info.get('change_type', neg_info.get('type', 'unknown'))
                    elif isinstance(neg_info, str):
                        neg_text = neg_info
                        neg_type = 'unknown'
                    else:
                        continue
                    
                    if not is_valid_text(neg_text):
                        continue
                    
                    pairs.append(ComponentPair(
                        image_path=image_path,
                        image_key=f"{image_key}_comp_{len(pairs)}",
                        positive=pos_comp.lower().strip(),
                        negative=neg_text.lower().strip(),
                        negative_type=neg_type,
                        original_caption=original_caption,
                    ))
                    
                    if len(pairs) >= self.max_samples:
                        break
                
                if len(pairs) >= self.max_samples:
                    break
            
            if len(pairs) >= self.max_samples:
                break
        
        logger.info(f"Generated {len(pairs)} component pairs")
        return pairs
    
    def generate_relation_pairs(self) -> List[RelationPair]:
        """Generate positive/negative relation pairs."""
        pairs = []
        
        for sample in tqdm(self.data, desc="Generating relation pairs"):
            relations = sample.get('relations', [])
            
            if not relations:
                continue
            
            image_path = self._get_image_path(sample)
            if not image_path:
                continue
            
            image_key = sample.get('sample_id', sample.get('image_id', str(hash(image_path))))
            original_caption = sample.get('original_caption', sample.get('caption', ''))
            
            for rel in relations:
                if not isinstance(rel, dict):
                    continue
                
                subject = rel.get('subject', '').lower().strip()
                relation_type = rel.get('relation_type', '').lower().strip()
                obj = rel.get('object', '').lower().strip()
                
                if not all([is_valid_text(subject), is_valid_text(relation_type), is_valid_text(obj)]):
                    continue
                
                positive_rel = f"{subject} {relation_type} {obj}"
                
                # Get negatives from within the relation
                negatives = rel.get('negatives', [])
                
                for neg_info in negatives:
                    if not isinstance(neg_info, dict):
                        continue
                    
                    neg_subject = neg_info.get('subject', '').lower().strip()
                    neg_rel_type = neg_info.get('relation_type', '').lower().strip()
                    neg_obj = neg_info.get('object', '').lower().strip()
                    neg_type = neg_info.get('change_type', 'unknown')
                    
                    if not all([is_valid_text(neg_subject), is_valid_text(neg_rel_type), is_valid_text(neg_obj)]):
                        continue
                    
                    negative_rel = f"{neg_subject} {neg_rel_type} {neg_obj}"
                    
                    # Skip if positive and negative are the same
                    if positive_rel.strip() == negative_rel.strip():
                        continue
                    
                    pairs.append(RelationPair(
                        image_path=image_path,
                        image_key=f"{image_key}_rel_{len(pairs)}",
                        positive=positive_rel,
                        negative=negative_rel,
                        negative_type=neg_type,
                        subject=subject,
                        relation=relation_type,
                        obj=obj,
                        original_caption=original_caption,
                    ))
                    
                    if len(pairs) >= self.max_samples:
                        break
                
                if len(pairs) >= self.max_samples:
                    break
            
            if len(pairs) >= self.max_samples:
                break
        
        logger.info(f"Generated {len(pairs)} relation pairs")
        return pairs


class LAIONUnitPairGenerator:
    """Generates positive/negative unit pairs from LAION tar/JSON shards."""
    
    def __init__(
        self,
        data_root: str,
        json_root: Optional[str] = None,
        tar_range: Optional[Tuple[int, int]] = None,
        max_samples: int = 5000,
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ):
        self.data_root = data_root
        self.json_root = json_root or os.path.join(data_root, "laion400m_neg")
        self.tar_range = tar_range
        self.max_samples = max_samples
        self.seed = seed
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cache_temp", "unit_disc_images")
        
        random.seed(seed)
        np.random.seed(seed)
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._discover_shards()
        self._load_data()
        
        self._tar_handles: Dict[int, tarfile.TarFile] = {}
        self._member_indices: Dict[int, Dict[str, str]] = {}
    
    def _discover_shards(self):
        """Discover available JSON shards."""
        all_json_files = sorted(
            f for f in os.listdir(self.json_root)
            if f.endswith(".json") and f[:-5].isdigit()
        )
        all_tar_nums = [int(os.path.splitext(f)[0]) for f in all_json_files]
        
        if self.tar_range is not None:
            start, end = self.tar_range
            self.tar_nums = [t for t in all_tar_nums if start <= t < end]
        else:
            self.tar_nums = all_tar_nums
        
        logger.info(f"[LAION] Found {len(self.tar_nums)} shards")
    
    def _load_data(self):
        """Load all JSON shards."""
        self.data = []
        
        for tar_num in tqdm(self.tar_nums, desc="Loading LAION shards"):
            json_path = os.path.join(self.json_root, f"{tar_num:05d}.json")
            try:
                with open(json_path, 'r') as f:
                    shard_data = json.load(f)
                    if isinstance(shard_data, list):
                        for item in shard_data:
                            item['_tar_num'] = tar_num
                        self.data.extend(shard_data)
                    elif isinstance(shard_data, dict):
                        shard_data['_tar_num'] = tar_num
                        self.data.append(shard_data)
            except Exception as e:
                logger.warning(f"Failed to load {json_path}: {e}")
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def _get_tar_handle(self, tar_num: int) -> tarfile.TarFile:
        """Get or open a tar file handle."""
        if tar_num not in self._tar_handles:
            tar_path = os.path.join(self.data_root, f"{tar_num:05d}.tar")
            self._tar_handles[tar_num] = tarfile.open(tar_path, "r")
        return self._tar_handles[tar_num]
    
    def _get_member_index(self, tar_num: int) -> Dict[str, str]:
        """Build member index for a tar file."""
        if tar_num not in self._member_indices:
            tf = self._get_tar_handle(tar_num)
            index = {}
            for m in tf:
                if m.isfile():
                    base = os.path.splitext(m.name)[0]
                    if base.isdigit() or m.name.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        index[base] = m.name
            self._member_indices[tar_num] = index
        return self._member_indices[tar_num]
    
    def _extract_image(self, tar_num: int, wds_key: str) -> Optional[str]:
        """Extract image from tar to cache."""
        cache_subdir = os.path.join(self.cache_dir, f"{tar_num:05d}")
        os.makedirs(cache_subdir, exist_ok=True)
        
        # Check cache
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            cached_path = os.path.join(cache_subdir, f"{wds_key}{ext}")
            if os.path.exists(cached_path):
                return cached_path
        
        member_index = self._get_member_index(tar_num)
        member_name = member_index.get(wds_key)
        
        if member_name is None and wds_key.isdigit():
            n = int(wds_key)
            for cand in (str(n), f"{n:09d}", f"{n:05d}"):
                if cand in member_index:
                    member_name = member_index[cand]
                    break
        
        if member_name is None:
            return None
        
        try:
            tf = self._get_tar_handle(tar_num)
            member = tf.getmember(member_name)
            ext = os.path.splitext(member_name)[1]
            output_path = os.path.join(cache_subdir, f"{wds_key}{ext}")
            
            with tf.extractfile(member) as src:
                content = src.read()
                with open(output_path, 'wb') as dst:
                    dst.write(content)
            
            return output_path
        except Exception:
            return None
    
    def _get_wds_key(self, sample: Dict) -> Optional[str]:
        """Get wds_key from sample."""
        wds_key = sample.get("wds_key")
        if wds_key is None:
            sample_id = sample.get("sample_id", "")
            if "::" in sample_id:
                wds_key = sample_id.split("::")[1]
        return wds_key
    
    def close(self):
        """Close tar handles."""
        for tf in self._tar_handles.values():
            try:
                tf.close()
            except Exception:
                pass
        self._tar_handles.clear()
    
    def generate_component_pairs(self) -> List[ComponentPair]:
        """Generate component pairs from LAION data."""
        pairs = []
        
        for sample in tqdm(self.data, desc="Generating LAION component pairs"):
            positive_components = sample.get('positive_components', [])
            negative_components = sample.get('negative_components', {})
            
            if not positive_components or not negative_components:
                continue
            
            tar_num = sample.get('_tar_num')
            wds_key = self._get_wds_key(sample)
            
            if tar_num is None or wds_key is None:
                continue
            
            image_path = self._extract_image(tar_num, wds_key)
            if not image_path:
                continue
            
            image_key = sample.get('sample_id', f"{tar_num:05d}::{wds_key}")
            original_caption = sample.get('original_caption', sample.get('caption', ''))
            
            for pos_comp in positive_components:
                if not is_valid_text(pos_comp):
                    continue
                
                neg_list = negative_components.get(pos_comp, [])
                
                for neg_info in neg_list:
                    if isinstance(neg_info, dict):
                        neg_text = neg_info.get('negative', neg_info.get('text', ''))
                        neg_type = neg_info.get('change_type', neg_info.get('type', 'unknown'))
                    elif isinstance(neg_info, str):
                        neg_text = neg_info
                        neg_type = 'unknown'
                    else:
                        continue
                    
                    if not is_valid_text(neg_text):
                        continue
                    
                    pairs.append(ComponentPair(
                        image_path=image_path,
                        image_key=f"{image_key}_comp_{len(pairs)}",
                        positive=pos_comp.lower().strip(),
                        negative=neg_text.lower().strip(),
                        negative_type=neg_type,
                        original_caption=original_caption,
                    ))
                    
                    if len(pairs) >= self.max_samples:
                        break
                
                if len(pairs) >= self.max_samples:
                    break
            
            if len(pairs) >= self.max_samples:
                break
        
        logger.info(f"Generated {len(pairs)} component pairs from LAION")
        return pairs
    
    def generate_relation_pairs(self) -> List[RelationPair]:
        """Generate relation pairs from LAION data."""
        pairs = []
        
        for sample in tqdm(self.data, desc="Generating LAION relation pairs"):
            relations = sample.get('relations', [])
            
            if not relations:
                continue
            
            tar_num = sample.get('_tar_num')
            wds_key = self._get_wds_key(sample)
            
            if tar_num is None or wds_key is None:
                continue
            
            image_path = self._extract_image(tar_num, wds_key)
            if not image_path:
                continue
            
            image_key = sample.get('sample_id', f"{tar_num:05d}::{wds_key}")
            original_caption = sample.get('original_caption', sample.get('caption', ''))
            
            for rel in relations:
                if not isinstance(rel, dict):
                    continue
                
                subject = rel.get('subject', '').lower().strip()
                relation_type = rel.get('relation_type', '').lower().strip()
                obj = rel.get('object', '').lower().strip()
                
                if not all([is_valid_text(subject), is_valid_text(relation_type), is_valid_text(obj)]):
                    continue
                
                positive_rel = f"{subject} {relation_type} {obj}"
                
                negatives = rel.get('negatives', [])
                
                for neg_info in negatives:
                    if not isinstance(neg_info, dict):
                        continue
                    
                    neg_subject = neg_info.get('subject', '').lower().strip()
                    neg_rel_type = neg_info.get('relation_type', '').lower().strip()
                    neg_obj = neg_info.get('object', '').lower().strip()
                    neg_type = neg_info.get('change_type', 'unknown')
                    
                    if not all([is_valid_text(neg_subject), is_valid_text(neg_rel_type), is_valid_text(neg_obj)]):
                        continue
                    
                    negative_rel = f"{neg_subject} {neg_rel_type} {neg_obj}"
                    
                    if positive_rel.strip() == negative_rel.strip():
                        continue
                    
                    pairs.append(RelationPair(
                        image_path=image_path,
                        image_key=f"{image_key}_rel_{len(pairs)}",
                        positive=positive_rel,
                        negative=negative_rel,
                        negative_type=neg_type,
                        subject=subject,
                        relation=relation_type,
                        obj=obj,
                        original_caption=original_caption,
                    ))
                    
                    if len(pairs) >= self.max_samples:
                        break
                
                if len(pairs) >= self.max_samples:
                    break
            
            if len(pairs) >= self.max_samples:
                break
        
        logger.info(f"Generated {len(pairs)} relation pairs from LAION")
        return pairs


# =============================================================================
# Evaluator
# =============================================================================

class UnitDiscriminationEvaluator:
    """Evaluates unit discrimination accuracy for CLIP models."""
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        checkpoint_type: str = "openclip",
        force_openclip: bool = False,
        pretrained: str = "openai",
        clove_weight: float = 0.6,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model {model_name} on {self.device}")
        
        from utils.checkpoint_loader import load_checkpoint_model
        
        if checkpoint_type == "openclip":
            effective_path = model_name
        elif checkpoint_type in ["huggingface", "tripletclip"]:
            if not checkpoint_path:
                raise ValueError(f"checkpoint_path required for checkpoint_type='{checkpoint_type}'")
            effective_path = checkpoint_path
        else:
            if not checkpoint_path:
                raise ValueError(f"checkpoint_path required for checkpoint_type='{checkpoint_type}'")
            effective_path = checkpoint_path
        
        self.model, self.preprocess, self.tokenize = load_checkpoint_model(
            checkpoint_type=checkpoint_type,
            checkpoint_path=effective_path,
            device=self.device,
            base_model=model_name,
            force_openclip=force_openclip,
            pretrained=pretrained,
            clove_weight=clove_weight,
        )
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    @torch.no_grad()
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity between image and text."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        try:
            text_input = self.tokenize([text]).to(self.device)
        except (RuntimeError, ValueError, TypeError):
            text_input = self.tokenize([text], truncate=True).to(self.device)
        
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_input)
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        similarity = (image_features @ text_features.T).item()
        return similarity
    
    def evaluate_component_pair(self, pair: ComponentPair) -> Optional[DiscriminationResult]:
        """Evaluate a single component pair."""
        try:
            image = Image.open(pair.image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {pair.image_path}: {e}")
            return None
        
        score_pos = self.compute_similarity(image, pair.positive)
        score_neg = self.compute_similarity(image, pair.negative)
        
        return DiscriminationResult(
            sample_id=pair.image_key,
            unit_type="component",
            negative_type=pair.negative_type,
            positive_text=pair.positive,
            negative_text=pair.negative,
            score_positive=score_pos,
            score_negative=score_neg,
            correct=(score_pos > score_neg),
            margin=score_pos - score_neg,
        )
    
    def evaluate_relation_pair(self, pair: RelationPair) -> Optional[DiscriminationResult]:
        """Evaluate a single relation pair."""
        try:
            image = Image.open(pair.image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {pair.image_path}: {e}")
            return None
        
        score_pos = self.compute_similarity(image, pair.positive)
        score_neg = self.compute_similarity(image, pair.negative)
        
        return DiscriminationResult(
            sample_id=pair.image_key,
            unit_type="relation",
            negative_type=pair.negative_type,
            positive_text=pair.positive,
            negative_text=pair.negative,
            score_positive=score_pos,
            score_negative=score_neg,
            correct=(score_pos > score_neg),
            margin=score_pos - score_neg,
        )


# =============================================================================
# Analyzer
# =============================================================================

class UnitDiscriminationAnalyzer:
    """Analyzes unit discrimination results."""
    
    def __init__(self, results: List[DiscriminationResult]):
        self.results = results
        
        # Group by unit type and negative type
        self.by_unit_type = defaultdict(list)
        self.by_negative_type = defaultdict(list)
        self.by_unit_and_neg_type = defaultdict(list)
        
        for r in results:
            self.by_unit_type[r.unit_type].append(r)
            self.by_negative_type[r.negative_type].append(r)
            self.by_unit_and_neg_type[(r.unit_type, r.negative_type)].append(r)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all metrics."""
        metrics = {
            'overall': self._compute_group_metrics(self.results),
            'by_unit_type': {},
            'by_negative_type': {},
            'by_unit_and_neg_type': {},
        }
        
        for unit_type, group in self.by_unit_type.items():
            metrics['by_unit_type'][unit_type] = self._compute_group_metrics(group)
        
        for neg_type, group in self.by_negative_type.items():
            metrics['by_negative_type'][neg_type] = self._compute_group_metrics(group)
        
        for (unit_type, neg_type), group in self.by_unit_and_neg_type.items():
            key = f"{unit_type}_{neg_type}"
            metrics['by_unit_and_neg_type'][key] = self._compute_group_metrics(group)
        
        return metrics
    
    def _compute_group_metrics(self, results: List[DiscriminationResult]) -> Dict[str, float]:
        """Compute metrics for a group of results."""
        if not results:
            return {'n': 0, 'accuracy': 0.0}
        
        n = len(results)
        correct = sum(1 for r in results if r.correct)
        margins = [r.margin for r in results]
        
        # Bootstrap CI for accuracy
        acc_ci = self._bootstrap_ci([r.correct for r in results])
        
        return {
            'n': n,
            'accuracy': correct / n,
            'accuracy_ci_low': acc_ci[0],
            'accuracy_ci_high': acc_ci[1],
            'margin_mean': np.mean(margins),
            'margin_median': np.median(margins),
            'margin_std': np.std(margins),
        }
    
    def _bootstrap_ci(self, values: List[bool], n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        values = np.array(values, dtype=float)
        n = len(values)
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        low = np.percentile(bootstrap_means, 100 * alpha / 2)
        high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return (low, high)
    
    def print_summary(self):
        """Print summary of results."""
        metrics = self.compute_metrics()
        
        print("\n" + "=" * 80)
        print("UNIT DISCRIMINATION RESULTS")
        print("=" * 80)
        
        # Overall
        m = metrics['overall']
        print("\nOVERALL:")
        print(f"  N = {m['n']}")
        print(f"  Accuracy: {m['accuracy']:.1%} [{m['accuracy_ci_low']:.1%}, {m['accuracy_ci_high']:.1%}]")
        print(f"  Margin: {m['margin_mean']:.4f} (median: {m['margin_median']:.4f}, std: {m['margin_std']:.4f})")
        
        # By unit type
        print("\n--- By Unit Type ---")
        for unit_type, m in sorted(metrics['by_unit_type'].items()):
            print(f"\n{unit_type.upper()}:")
            print(f"  N = {m['n']}")
            print(f"  Accuracy: {m['accuracy']:.1%} [{m['accuracy_ci_low']:.1%}, {m['accuracy_ci_high']:.1%}]")
            print(f"  Margin: {m['margin_mean']:.4f}")
        
        # By negative type
        print("\n--- By Negative Type ---")
        for neg_type, m in sorted(metrics['by_negative_type'].items()):
            if m['n'] < 10:  # Skip very small groups
                continue
            print(f"\n{neg_type}:")
            print(f"  N = {m['n']}")
            print(f"  Accuracy: {m['accuracy']:.1%} [{m['accuracy_ci_low']:.1%}, {m['accuracy_ci_high']:.1%}]")
        
        print("\n" + "=" * 80)
    
    def save_results(self, output_path: str):
        """Save results to JSON."""
        output = {
            'metrics': self.compute_metrics(),
            'results': [asdict(r) for r in self.results],
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unit Discrimination Experiment")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="ViT-B/32",
                       help="CLIP model name")
    parser.add_argument("--checkpoint_type", type=str, default="openclip",
                       choices=["openclip", "huggingface", "tripletclip", "external", "dac", "clove"],
                       help="Type of checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to checkpoint")
    parser.add_argument("--force_openclip", action="store_true",
                       help="Force OpenCLIP")
    parser.add_argument("--pretrained", type=str, default="openai",
                       help="Pretrained weights for CLOVE")
    parser.add_argument("--clove_weight", type=float, default=0.6,
                       help="CLOVE interpolation weight")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="coco",
                       choices=["laion", "coco"],
                       help="Dataset to use")
    parser.add_argument("--json_folder", type=str, 
                       default="swap_pos_json/coco/",
                       help="JSON folder for COCO")
    parser.add_argument("--image_root", type=str, default=None,
                       help="Image root for COCO")
    
    # LAION arguments
    parser.add_argument("--laion_data_root", type=str, default=None,
                       help="LAION tar files root")
    parser.add_argument("--laion_json_root", type=str, default=None,
                       help="LAION JSON shards root")
    parser.add_argument("--tar_range", type=str, default=None,
                       help="Tar range (e.g., '0,100')")
    parser.add_argument("--laion_cache_dir", type=str, default=None,
                       help="LAION image cache directory")
    
    # Experiment settings
    parser.add_argument("--num_samples", type=int, default=5000,
                       help="Max samples per type")
    parser.add_argument("--output_dir", type=str, default="unit_discrimination_results",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 70)
    print("UNIT DISCRIMINATION EXPERIMENT")
    print("=" * 70)
    print(f"\nModel: {args.model_name}")
    print(f"Checkpoint type: {args.checkpoint_type}")
    if args.checkpoint_path:
        print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.num_samples}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Validate arguments
    if args.dataset == "coco" and not args.image_root:
        logger.error("--image_root required for COCO")
        sys.exit(1)
    if args.dataset == "laion" and not args.laion_data_root:
        logger.error("--laion_data_root required for LAION")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate pairs
    if args.dataset == "laion":
        tar_range = None
        if args.tar_range:
            parts = args.tar_range.split(",")
            tar_range = (int(parts[0]), int(parts[1]))
        
        generator = LAIONUnitPairGenerator(
            data_root=args.laion_data_root,
            json_root=args.laion_json_root,
            tar_range=tar_range,
            max_samples=args.num_samples,
            seed=args.seed,
            cache_dir=args.laion_cache_dir,
        )
        
        component_pairs = generator.generate_component_pairs()
        relation_pairs = generator.generate_relation_pairs()
        generator.close()
    else:
        # COCO
        json_folder = Path(args.json_folder)
        if json_folder.is_file():
            json_path = str(json_folder)
        else:
            json_files = list(json_folder.glob("*.json"))
            if not json_files:
                logger.error(f"No JSON files in {json_folder}")
                sys.exit(1)
            
            if len(json_files) == 1:
                json_path = str(json_files[0])
            else:
                # Combine
                all_data = []
                for jf in json_files:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
                
                combined_path = output_dir / "combined_input.json"
                with open(combined_path, 'w') as f:
                    json.dump(all_data, f)
                json_path = str(combined_path)
        
        generator = COCOUnitPairGenerator(
            json_path=json_path,
            image_root=args.image_root,
            max_samples=args.num_samples,
            seed=args.seed,
        )
        
        component_pairs = generator.generate_component_pairs()
        relation_pairs = generator.generate_relation_pairs()
    
    logger.info(f"Generated {len(component_pairs)} component pairs")
    logger.info(f"Generated {len(relation_pairs)} relation pairs")
    
    # Save pairs
    with open(output_dir / "component_pairs.json", 'w') as f:
        json.dump([asdict(p) for p in component_pairs], f, indent=2)
    with open(output_dir / "relation_pairs.json", 'w') as f:
        json.dump([asdict(p) for p in relation_pairs], f, indent=2)
    
    # Evaluate
    evaluator = UnitDiscriminationEvaluator(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        checkpoint_type=args.checkpoint_type,
        force_openclip=args.force_openclip,
        pretrained=args.pretrained,
        clove_weight=args.clove_weight,
    )
    
    results = []
    
    # Evaluate component pairs
    for pair in tqdm(component_pairs, desc="Evaluating component pairs"):
        result = evaluator.evaluate_component_pair(pair)
        if result:
            results.append(result)
    
    # Evaluate relation pairs
    for pair in tqdm(relation_pairs, desc="Evaluating relation pairs"):
        result = evaluator.evaluate_relation_pair(pair)
        if result:
            results.append(result)
    
    # Analyze
    analyzer = UnitDiscriminationAnalyzer(results)
    analyzer.print_summary()
    analyzer.save_results(str(output_dir / "results.json"))
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

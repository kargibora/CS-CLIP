"""
Half-Truth Ranking Vulnerability Experiment

Extends the half-truth analysis from binary classification (foil wins vs correct wins)
to ranking-based evaluation. Measures how half-truth foils affect retrieval rankings.

Core Hypothesis:
    In image-to-text retrieval, half-truth captions (correct + incorrect foil)
    may rank higher than correct captions, corrupting retrieval quality.

Metrics:
    1. Half-Truth Intrusion Rate: % where half-truth ranks above short-truth
    2. Rank Displacement: Average positions the correct answer drops
    3. MRR Degradation: Impact on Mean Reciprocal Rank
    4. Recall@K: Does correct answer remain in top-K?

Author: Auto-generated for ranking analysis
"""

import sys
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.half_truth_vulnerability import (
    ComponentSample, 
    RelationSample, 
    HalfTruthSampleGenerator,
    is_valid_caption,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RankingResult:
    """Result for a single ranking query.
    
    Tracks how half-truths affect ranking compared to:
    1. Short truth (partial correct caption)
    2. Full truth (complete correct caption / original)
    
    Rank displacement measures how many positions the correct answer drops
    when half-truths are introduced into the gallery.
    """
    sample_id: str
    condition: str
    
    # Ranks (1-indexed, lower is better)
    rank_short_truth: int
    rank_full_truth: int
    rank_half_truth: int
    
    # Scores (cosine similarity)
    score_short_truth: float
    score_full_truth: float
    score_half_truth: float
    
    # Gallery info
    n_candidates: int
    n_distractors: int
    
    # Baseline ranks (without half-truth in gallery) - for displacement calculation
    # These are computed in a separate pass or estimated
    rank_short_truth_baseline: int = 1  # Rank of short truth without half-truth
    rank_full_truth_baseline: int = 1   # Rank of full truth without half-truth
    
    # Captions for reference
    caption_short_truth: str = ""
    caption_full_truth: str = ""
    caption_half_truth: str = ""
    
    @property
    def half_truth_intrusion(self) -> bool:
        """Does half-truth rank above short-truth?"""
        return self.rank_half_truth < self.rank_short_truth
    
    @property
    def half_truth_beats_full(self) -> bool:
        """Does half-truth rank above full-truth?"""
        return self.rank_half_truth < self.rank_full_truth
    
    @property
    def displacement_vs_short(self) -> int:
        """How many positions did short-truth drop due to half-truth?
        
        If half-truth ranks above short-truth, short-truth is displaced by 1.
        """
        return 1 if self.half_truth_intrusion else 0
    
    @property
    def displacement_vs_full(self) -> int:
        """How many positions did full-truth drop due to half-truth?
        
        If half-truth ranks above full-truth, full-truth is displaced by 1.
        """
        return 1 if self.half_truth_beats_full else 0
    
    @property
    def rank_displacement(self) -> int:
        """Alias for displacement_vs_short for backward compatibility."""
        return self.displacement_vs_short
    
    @property
    def mrr_short(self) -> float:
        """MRR contribution from short-truth."""
        return 1.0 / self.rank_short_truth
    
    @property
    def mrr_full(self) -> float:
        """MRR contribution from full-truth."""
        return 1.0 / self.rank_full_truth if self.rank_full_truth > 0 else 0.0
    
    @property
    def score_margin_short_vs_half(self) -> float:
        """Score margin: short_truth - half_truth (positive = short wins)."""
        return self.score_short_truth - self.score_half_truth
    
    @property
    def score_margin_full_vs_half(self) -> float:
        """Score margin: full_truth - half_truth (positive = full wins)."""
        return self.score_full_truth - self.score_half_truth
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'sample_id': self.sample_id,
            'condition': self.condition,
            'rank_short_truth': self.rank_short_truth,
            'rank_full_truth': self.rank_full_truth,
            'rank_half_truth': self.rank_half_truth,
            'rank_short_truth_baseline': self.rank_short_truth_baseline,
            'rank_full_truth_baseline': self.rank_full_truth_baseline,
            'score_short_truth': self.score_short_truth,
            'score_full_truth': self.score_full_truth,
            'score_half_truth': self.score_half_truth,
            'n_candidates': self.n_candidates,
            'n_distractors': self.n_distractors,
            'half_truth_intrusion': self.half_truth_intrusion,
            'half_truth_beats_full': self.half_truth_beats_full,
            'displacement_vs_short': self.displacement_vs_short,
            'displacement_vs_full': self.displacement_vs_full,
            'score_margin_short_vs_half': self.score_margin_short_vs_half,
            'score_margin_full_vs_half': self.score_margin_full_vs_half,
            'mrr_short': self.mrr_short,
            'mrr_full': self.mrr_full,
            'caption_short_truth': self.caption_short_truth,
            'caption_full_truth': self.caption_full_truth,
            'caption_half_truth': self.caption_half_truth,
        }


@dataclass
class RankingMetrics:
    """Aggregate ranking metrics across all samples."""
    
    # Intrusion metrics
    half_truth_intrusion_rate: float  # % where half-truth ranks above short-truth
    half_truth_beats_full_rate: float  # % where half-truth ranks above full-truth
    
    # Rank metrics (mean ranks, lower is better)
    mean_rank_short_truth: float
    mean_rank_full_truth: float
    mean_rank_half_truth: float
    
    # MRR metrics (higher is better)
    mrr_short_truth: float
    mrr_full_truth: float
    mrr_half_truth: float
    
    # Displacement metrics
    mean_displacement_vs_short: float  # Avg displacement of short-truth due to half-truth
    mean_displacement_vs_full: float   # Avg displacement of full-truth due to half-truth
    
    # Score margins (positive = correct wins)
    mean_margin_short_vs_half: float  # Avg(score_short - score_half)
    mean_margin_full_vs_half: float   # Avg(score_full - score_half)
    
    # Recall@K metrics (for short-truth)
    recall_at_1_short: float  # % where short-truth is rank 1
    recall_at_3_short: float  # % where short-truth is in top 3
    recall_at_5_short: float  # % where short-truth is in top 5
    
    # Recall@K metrics (for full-truth)
    recall_at_1_full: float   # % where full-truth is rank 1
    recall_at_3_full: float   # % where full-truth is in top 3
    
    # Sample counts
    n_samples: int
    n_distractors: int  # Average number of distractors
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"RankingMetrics(n={self.n_samples}):\n"
            f"  Half-Truth Intrusion Rate (vs Short): {self.half_truth_intrusion_rate:.1%}\n"
            f"  Half-Truth Beats Full Rate: {self.half_truth_beats_full_rate:.1%}\n"
            f"  Mean Ranks: short={self.mean_rank_short_truth:.2f}, "
            f"full={self.mean_rank_full_truth:.2f}, half={self.mean_rank_half_truth:.2f}\n"
            f"  Mean Displacement: vs_short={self.mean_displacement_vs_short:.3f}, "
            f"vs_full={self.mean_displacement_vs_full:.3f}\n"
            f"  Score Margins: short_vs_half={self.mean_margin_short_vs_half:.4f}, "
            f"full_vs_half={self.mean_margin_full_vs_half:.4f}\n"
            f"  MRR: short={self.mrr_short_truth:.3f}, full={self.mrr_full_truth:.3f}\n"
            f"  Recall@1: short={self.recall_at_1_short:.1%}, full={self.recall_at_1_full:.1%}"
        )


def compute_ranking_metrics(results: List[RankingResult]) -> RankingMetrics:
    """Compute aggregate metrics from individual ranking results."""
    if not results:
        return RankingMetrics(
            half_truth_intrusion_rate=0.0,
            half_truth_beats_full_rate=0.0,
            mean_rank_short_truth=0.0,
            mean_rank_full_truth=0.0,
            mean_rank_half_truth=0.0,
            mrr_short_truth=0.0,
            mrr_full_truth=0.0,
            mrr_half_truth=0.0,
            mean_displacement_vs_short=0.0,
            mean_displacement_vs_full=0.0,
            mean_margin_short_vs_half=0.0,
            mean_margin_full_vs_half=0.0,
            recall_at_1_short=0.0,
            recall_at_3_short=0.0,
            recall_at_5_short=0.0,
            recall_at_1_full=0.0,
            recall_at_3_full=0.0,
            n_samples=0,
            n_distractors=0,
        )
    
    n = len(results)
    
    intrusion_count = sum(1 for r in results if r.half_truth_intrusion)
    beats_full_count = sum(1 for r in results if r.half_truth_beats_full)
    
    ranks_short = [r.rank_short_truth for r in results]
    ranks_full = [r.rank_full_truth for r in results]
    ranks_half = [r.rank_half_truth for r in results]
    
    # Displacement and margin metrics
    displacements_short = [r.displacement_vs_short for r in results]
    displacements_full = [r.displacement_vs_full for r in results]
    margins_short = [r.score_margin_short_vs_half for r in results]
    margins_full = [r.score_margin_full_vs_half for r in results]
    
    return RankingMetrics(
        half_truth_intrusion_rate=intrusion_count / n,
        half_truth_beats_full_rate=beats_full_count / n,
        mean_rank_short_truth=np.mean(ranks_short),
        mean_rank_full_truth=np.mean(ranks_full),
        mean_rank_half_truth=np.mean(ranks_half),
        mrr_short_truth=np.mean([1.0 / r for r in ranks_short]),
        mrr_full_truth=np.mean([1.0 / r for r in ranks_full if r > 0]),
        mrr_half_truth=np.mean([1.0 / r for r in ranks_half]),
        mean_displacement_vs_short=np.mean(displacements_short),
        mean_displacement_vs_full=np.mean(displacements_full),
        mean_margin_short_vs_half=np.mean(margins_short),
        mean_margin_full_vs_half=np.mean(margins_full),
        recall_at_1_short=sum(1 for r in ranks_short if r == 1) / n,
        recall_at_3_short=sum(1 for r in ranks_short if r <= 3) / n,
        recall_at_5_short=sum(1 for r in ranks_short if r <= 5) / n,
        recall_at_1_full=sum(1 for r in ranks_full if r == 1) / n,
        recall_at_3_full=sum(1 for r in ranks_full if r <= 3) / n,
        n_samples=n,
        n_distractors=int(np.mean([r.n_distractors for r in results])),
    )


# =============================================================================
# Distractor Pool
# =============================================================================

class DistractorPool:
    """
    Manages a pool of distractor captions for ranking experiments.
    
    Distractors are sampled from other images to create realistic
    retrieval scenarios where the model must distinguish the correct
    caption from plausible alternatives.
    """
    
    def __init__(
        self,
        samples: List[Any],  # ComponentSample or RelationSample
        seed: int = 42,
    ):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Build pool of distractor captions
        self.distractor_pool = self._build_pool(samples)
        logger.info(f"Built distractor pool with {len(self.distractor_pool)} captions")
    
    def _build_pool(self, samples: List[Any]) -> List[Tuple[str, str]]:
        """
        Build pool of (caption, source_id) tuples.
        
        Extracts various captions from samples to use as distractors.
        """
        pool = []
        
        for sample in samples:
            sample_id = getattr(sample, 'image_key', None)
            
            if isinstance(sample, ComponentSample):
                # Add component captions
                if is_valid_caption(sample.short_correct):
                    pool.append((sample.short_correct, sample_id))
                if is_valid_caption(sample.long_correct):
                    pool.append((sample.long_correct, sample_id))
                if is_valid_caption(sample.second_component):
                    pool.append((sample.second_component, sample_id))
                    
            elif isinstance(sample, RelationSample):
                # Add relation captions
                if is_valid_caption(sample.partial_truth):
                    pool.append((sample.partial_truth, sample_id))
                if is_valid_caption(sample.full_truth):
                    pool.append((sample.full_truth, sample_id))
        
        return pool
    
    def sample_distractors(
        self,
        n: int,
        exclude_id: str,
        exclude_captions: List[str] = None,
    ) -> List[str]:
        """
        Sample n distractor captions, excluding those from the same image.
        
        Args:
            n: Number of distractors to sample
            exclude_id: Sample ID to exclude (same image)
            exclude_captions: List of captions to exclude (already used)
        
        Returns:
            List of distractor captions
        """
        exclude_captions = set(exclude_captions or [])
        
        # Filter eligible distractors
        eligible = [
            (cap, sid) for cap, sid in self.distractor_pool
            if sid != exclude_id and cap not in exclude_captions
        ]
        
        if len(eligible) < n:
            logger.warning(f"Not enough distractors: {len(eligible)} < {n}")
            n = len(eligible)
        
        # Random sample
        sampled = random.sample(eligible, n)
        return [cap for cap, _ in sampled]


# =============================================================================
# Ranking Evaluator
# =============================================================================

class HalfTruthRankingEvaluator:
    """
    Evaluate ranking vulnerability to half-truths.
    
    For each sample (image):
    1. Create a gallery of captions: short-truth, full-truth, half-truth, + distractors
    2. Rank all captions by similarity to the image
    3. Measure where correct answers rank vs. half-truth
    
    This simulates an image-to-text retrieval scenario where half-truths
    may corrupt the ranking.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        checkpoint_type: str = "openclip",
        force_openclip: bool = False,
        pretrained: str = "openai",
        clove_weight: float = 0.6,
        n_distractors: int = 10,
    ):
        """
        Initialize the ranking evaluator.
        
        Args:
            model_name: Base model architecture
            device: Device to use
            checkpoint_path: Path to checkpoint
            checkpoint_type: Type of checkpoint
            force_openclip: Force OpenCLIP
            pretrained: Pretrained weights
            clove_weight: CLOVE interpolation weight
            n_distractors: Number of distractor captions per query
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_distractors = n_distractors
        
        logger.info(f"Loading model {model_name} on {self.device}")
        
        # Use unified checkpoint loader
        from utils.checkpoint_loader import load_checkpoint_model
        
        # Determine effective path
        if checkpoint_type == "openclip":
            effective_path = model_name
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
        
        self.distractor_pool = None
    
    def set_distractor_pool(self, pool: DistractorPool):
        """Set the distractor pool for sampling."""
        self.distractor_pool = pool
    
    @torch.no_grad()
    def compute_similarities(self, image: Image.Image, texts: List[str]) -> np.ndarray:
        """
        Compute CLIP similarities between an image and multiple texts.
        
        Args:
            image: Query image
            texts: List of candidate captions
        
        Returns:
            Array of similarity scores
        """
        # Preprocess image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Tokenize all texts
        try:
            text_inputs = self.tokenize(texts).to(self.device)
        except (RuntimeError, ValueError, TypeError):
            text_inputs = self.tokenize(texts, truncate=True).to(self.device)
        
        # Encode
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_inputs)
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Similarities
        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()
        return similarities
    
    def rank_captions(
        self,
        image: Image.Image,
        captions: List[str],
        caption_types: List[str],
    ) -> Dict[str, Dict]:
        """
        Rank captions by similarity to image.
        
        Args:
            image: Query image
            captions: List of candidate captions
            caption_types: Type label for each caption (e.g., 'short_truth', 'half_truth')
        
        Returns:
            Dict mapping caption_type to {rank, score}
        """
        similarities = self.compute_similarities(image, captions)
        
        # Compute ranks (1-indexed, higher similarity = lower rank)
        sorted_indices = np.argsort(-similarities)
        ranks = np.zeros_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
        
        results = {}
        for i, ctype in enumerate(caption_types):
            results[ctype] = {
                'rank': int(ranks[i]),
                'score': float(similarities[i]),
                'caption': captions[i],
            }
        
        return results
    
    def evaluate_component_sample(
        self,
        sample: ComponentSample,
        condition: str = "component_easy",
    ) -> Optional[RankingResult]:
        """
        Evaluate ranking for a component sample.
        
        Args:
            sample: ComponentSample to evaluate
            condition: Which half-truth condition to test
        
        Returns:
            RankingResult or None if invalid
        """
        # Validate
        if not is_valid_caption(sample.short_correct):
            return None
        if not is_valid_caption(sample.long_correct):
            return None
        
        # Select half-truth based on condition
        if condition == "component_easy":
            half_truth = sample.long_incorrect_easy
        elif condition == "component_hard":
            half_truth = sample.long_incorrect_hard
        elif condition == "component_random":
            half_truth = sample.long_incorrect_random
        else:
            logger.warning(f"Unknown condition: {condition}")
            return None
        
        if not is_valid_caption(half_truth):
            return None
        
        # Load image
        try:
            image = Image.open(sample.image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {sample.image_path}: {e}")
            return None
        
        # Build caption gallery
        captions = [sample.short_correct, sample.long_correct, half_truth]
        caption_types = ['short_truth', 'full_truth', 'half_truth']
        
        # Add distractors
        if self.distractor_pool:
            distractors = self.distractor_pool.sample_distractors(
                n=self.n_distractors,
                exclude_id=sample.image_key,
                exclude_captions=captions,
            )
            captions.extend(distractors)
            caption_types.extend([f'distractor_{i}' for i in range(len(distractors))])
        
        # Rank
        ranking = self.rank_captions(image, captions, caption_types)
        
        return RankingResult(
            sample_id=sample.image_key,
            condition=condition,
            rank_short_truth=ranking['short_truth']['rank'],
            rank_full_truth=ranking['full_truth']['rank'],
            rank_half_truth=ranking['half_truth']['rank'],
            score_short_truth=ranking['short_truth']['score'],
            score_full_truth=ranking['full_truth']['score'],
            score_half_truth=ranking['half_truth']['score'],
            n_candidates=len(captions),
            n_distractors=len(captions) - 3,
            caption_short_truth=sample.short_correct,
            caption_full_truth=sample.long_correct,
            caption_half_truth=half_truth,
        )
    
    def evaluate_relation_sample(
        self,
        sample: RelationSample,
        condition: str = "relation_swap",
    ) -> Optional[RankingResult]:
        """
        Evaluate ranking for a relation sample.
        
        Args:
            sample: RelationSample to evaluate
            condition: Which half-truth condition to test
        
        Returns:
            RankingResult or None if invalid
        """
        # Validate
        if not is_valid_caption(sample.partial_truth):
            return None
        if not is_valid_caption(sample.full_truth):
            return None
        
        # Select half-truth based on condition
        condition_to_field = {
            'relation_swap': 'foil_relation_swap',
            'relation_subject_object_swap': 'foil_relation_swap',
            'relation_antonym': 'foil_relation_antonym',
            'relation_negation': 'foil_relation_negation',
            'relation_wrong_object': 'foil_object_wrong',
            'object_wrong': 'foil_object_wrong',
            'relation_wrong_subject': 'foil_subject_wrong',
            'subject_wrong': 'foil_subject_wrong',
            'relation_wrong_relation': 'foil_relation_wrong',
            'relation_wrong': 'foil_relation_wrong',
            'relation_wrong_attribute': 'foil_attribute_wrong',
            'attribute_wrong': 'foil_attribute_wrong',
        }
        
        field = condition_to_field.get(condition)
        if not field:
            logger.warning(f"Unknown condition: {condition}")
            return None
        
        half_truth = getattr(sample, field, "")
        if not is_valid_caption(half_truth):
            return None
        
        # Load image
        try:
            image = Image.open(sample.image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {sample.image_path}: {e}")
            return None
        
        # Build caption gallery
        captions = [sample.partial_truth, sample.full_truth, half_truth]
        caption_types = ['short_truth', 'full_truth', 'half_truth']
        
        # Add distractors
        if self.distractor_pool:
            distractors = self.distractor_pool.sample_distractors(
                n=self.n_distractors,
                exclude_id=sample.image_key,
                exclude_captions=captions,
            )
            captions.extend(distractors)
            caption_types.extend([f'distractor_{i}' for i in range(len(distractors))])
        
        # Rank
        ranking = self.rank_captions(image, captions, caption_types)
        
        return RankingResult(
            sample_id=sample.image_key,
            condition=condition,
            rank_short_truth=ranking['short_truth']['rank'],
            rank_full_truth=ranking['full_truth']['rank'],
            rank_half_truth=ranking['half_truth']['rank'],
            score_short_truth=ranking['short_truth']['score'],
            score_full_truth=ranking['full_truth']['score'],
            score_half_truth=ranking['half_truth']['score'],
            n_candidates=len(captions),
            n_distractors=len(captions) - 3,
            caption_short_truth=sample.partial_truth,
            caption_full_truth=sample.full_truth,
            caption_half_truth=half_truth,
        )
    
    def evaluate_all_components(
        self,
        samples: List[ComponentSample],
        conditions: List[str] = None,
        show_progress: bool = True,
    ) -> Dict[str, List[RankingResult]]:
        """
        Evaluate all component samples.
        
        Args:
            samples: List of ComponentSamples
            conditions: Conditions to evaluate
            show_progress: Show progress bar
        
        Returns:
            Dict mapping condition -> list of results
        """
        if conditions is None:
            conditions = ["component_easy", "component_hard", "component_random"]
        
        results = {cond: [] for cond in conditions}
        
        iterator = tqdm(samples, desc="Evaluating components") if show_progress else samples
        for sample in iterator:
            for condition in conditions:
                result = self.evaluate_component_sample(sample, condition)
                if result:
                    results[condition].append(result)
        
        return results
    
    def evaluate_all_relations(
        self,
        samples: List[RelationSample],
        conditions: List[str] = None,
        show_progress: bool = True,
    ) -> Dict[str, List[RankingResult]]:
        """
        Evaluate all relation samples.
        
        Args:
            samples: List of RelationSamples
            conditions: Conditions to evaluate
            show_progress: Show progress bar
        
        Returns:
            Dict mapping condition -> list of results
        """
        if conditions is None:
            conditions = [
                "relation_swap", "relation_antonym", "relation_negation",
                "object_wrong", "subject_wrong", "attribute_wrong"
            ]
        
        results = {cond: [] for cond in conditions}
        
        iterator = tqdm(samples, desc="Evaluating relations") if show_progress else samples
        for sample in iterator:
            for condition in conditions:
                result = self.evaluate_relation_sample(sample, condition)
                if result:
                    results[condition].append(result)
        
        return results


# =============================================================================
# Analysis & Visualization
# =============================================================================

class RankingAnalyzer:
    """Analyze and visualize ranking results."""
    
    def __init__(self, results: Dict[str, List[RankingResult]]):
        self.results = results
        self.metrics = {
            cond: compute_ranking_metrics(res) 
            for cond, res in results.items()
        }
    
    def summary_table(self) -> pd.DataFrame:
        """Create summary table of metrics by condition."""
        rows = []
        for cond, metrics in self.metrics.items():
            rows.append({
                'Condition': cond,
                'N': metrics.n_samples,
                'Intrusion (vs Short)': f"{metrics.half_truth_intrusion_rate:.1%}",
                'Beats Full': f"{metrics.half_truth_beats_full_rate:.1%}",
                'Disp. Short': f"{metrics.mean_displacement_vs_short:.3f}",
                'Disp. Full': f"{metrics.mean_displacement_vs_full:.3f}",
                'Margin (S-H)': f"{metrics.mean_margin_short_vs_half:.4f}",
                'Margin (F-H)': f"{metrics.mean_margin_full_vs_half:.4f}",
                'MRR (Short)': f"{metrics.mrr_short_truth:.3f}",
                'MRR (Full)': f"{metrics.mrr_full_truth:.3f}",
                'R@1 Short': f"{metrics.recall_at_1_short:.1%}",
                'R@1 Full': f"{metrics.recall_at_1_full:.1%}",
            })
        return pd.DataFrame(rows)
    
    def plot_intrusion_rates(self, save_path: Optional[str] = None):
        """Plot half-truth intrusion rates by condition."""
        import matplotlib.pyplot as plt
        
        conditions = list(self.metrics.keys())
        rates = [self.metrics[c].half_truth_intrusion_rate for c in conditions]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(conditions)), rates, color='#e74c3c', alpha=0.8)
        
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylabel('Half-Truth Intrusion Rate')
        ax.set_title('How Often Does Half-Truth Rank Above Correct Answer?')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{rate:.1%}', ha='center', va='bottom', fontsize=10)
        
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_rank_distributions(self, save_path: Optional[str] = None):
        """Plot distribution of ranks for short-truth vs half-truth."""
        import matplotlib.pyplot as plt
        
        # Aggregate all results
        all_results = []
        for res_list in self.results.values():
            all_results.extend(res_list)
        
        ranks_short = [r.rank_short_truth for r in all_results]
        ranks_half = [r.rank_half_truth for r in all_results]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Histogram
        ax = axes[0]
        max_rank = max(max(ranks_short), max(ranks_half))
        bins = range(1, min(max_rank + 2, 20))
        ax.hist(ranks_short, bins=bins, alpha=0.7, label='Short Truth', color='#3498db')
        ax.hist(ranks_half, bins=bins, alpha=0.7, label='Half Truth', color='#e74c3c')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Count')
        ax.legend()
        ax.set_title('Rank Distribution')
        
        # Right: Box plot by condition
        ax = axes[1]
        conditions = list(self.results.keys())
        data_short = [[r.rank_short_truth for r in self.results[c]] for c in conditions]
        data_half = [[r.rank_half_truth for r in self.results[c]] for c in conditions]
        
        positions_short = np.arange(len(conditions)) - 0.2
        positions_half = np.arange(len(conditions)) + 0.2
        
        bp1 = ax.boxplot(data_short, positions=positions_short, widths=0.35,
                        patch_artist=True, boxprops=dict(facecolor='#3498db', alpha=0.7))
        bp2 = ax.boxplot(data_half, positions=positions_half, widths=0.35,
                        patch_artist=True, boxprops=dict(facecolor='#e74c3c', alpha=0.7))
        
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylabel('Rank (lower is better)')
        ax.set_title('Rank by Condition')
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Short Truth', 'Half Truth'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_mrr_comparison(self, save_path: Optional[str] = None):
        """Compare MRR across conditions."""
        import matplotlib.pyplot as plt
        
        conditions = list(self.metrics.keys())
        mrr_short = [self.metrics[c].mrr_short_truth for c in conditions]
        mrr_full = [self.metrics[c].mrr_full_truth for c in conditions]
        mrr_half = [self.metrics[c].mrr_half_truth for c in conditions]
        
        x = np.arange(len(conditions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, mrr_short, width, label='Short Truth', color='#3498db', alpha=0.8)
        ax.bar(x, mrr_full, width, label='Full Truth', color='#27ae60', alpha=0.8)
        ax.bar(x + width, mrr_half, width, label='Half Truth', color='#e74c3c', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylabel('Mean Reciprocal Rank')
        ax.set_title('MRR Comparison: Correct vs Half-Truth Captions')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def save_results(self, output_dir: str, prefix: str = "ranking"):
        """Save results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        all_results = {}
        for cond, res_list in self.results.items():
            all_results[cond] = [r.to_dict() for r in res_list]
        
        with open(output_dir / f"{prefix}_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save metrics summary
        metrics_dict = {cond: m.to_dict() for cond, m in self.metrics.items()}
        with open(output_dir / f"{prefix}_metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Save summary table as CSV
        self.summary_table().to_csv(output_dir / f"{prefix}_summary.csv", index=False)
        
        logger.info(f"Saved results to {output_dir}")


# =============================================================================
# Multi-Model Comparison
# =============================================================================

def compare_models_ranking(
    models_config: Dict[str, Dict],
    component_samples: List[ComponentSample],
    relation_samples: List[RelationSample],
    n_distractors: int = 10,
    output_dir: str = "ranking_results",
) -> Dict[str, RankingAnalyzer]:
    """
    Compare multiple models on ranking task.
    
    Args:
        models_config: Dict of model_name -> config dict with keys:
            - checkpoint_path, checkpoint_type, base_model, etc.
        component_samples: List of ComponentSamples
        relation_samples: List of RelationSamples
        n_distractors: Number of distractors
        output_dir: Output directory
    
    Returns:
        Dict of model_name -> RankingAnalyzer
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build distractor pool
    all_samples = list(component_samples) + list(relation_samples)
    distractor_pool = DistractorPool(all_samples)
    
    analyzers = {}
    
    for model_name, config in models_config.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*60}")
        
        # Create evaluator
        evaluator = HalfTruthRankingEvaluator(
            model_name=config.get('base_model', 'ViT-B/32'),
            checkpoint_path=config.get('checkpoint_path'),
            checkpoint_type=config.get('checkpoint_type', 'openclip'),
            n_distractors=n_distractors,
        )
        evaluator.set_distractor_pool(distractor_pool)
        
        # Evaluate
        component_results = evaluator.evaluate_all_components(component_samples)
        relation_results = evaluator.evaluate_all_relations(relation_samples)
        
        # Merge results
        all_results = {**component_results, **relation_results}
        
        # Analyze
        analyzer = RankingAnalyzer(all_results)
        analyzers[model_name] = analyzer
        
        # Print summary
        print(f"\n{model_name} Summary:")
        print(analyzer.summary_table().to_string())
        
        # Save
        model_output_dir = output_dir / model_name.replace('/', '_')
        analyzer.save_results(model_output_dir)
        analyzer.plot_intrusion_rates(model_output_dir / "intrusion_rates.png")
        analyzer.plot_rank_distributions(model_output_dir / "rank_distributions.png")
        analyzer.plot_mrr_comparison(model_output_dir / "mrr_comparison.png")
    
    return analyzers


def plot_model_comparison(
    analyzers: Dict[str, RankingAnalyzer],
    save_path: Optional[str] = None,
):
    """Create comparison plot across models."""
    import matplotlib.pyplot as plt
    
    model_names = list(analyzers.keys())
    
    # Aggregate intrusion rate across all conditions
    intrusion_rates = []
    for name in model_names:
        all_results = []
        for res_list in analyzers[name].results.values():
            all_results.extend(res_list)
        rate = sum(1 for r in all_results if r.half_truth_intrusion) / len(all_results)
        intrusion_rates.append(rate)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c' if 'CLIP' in n and 'CS' not in n else '#27ae60' for n in model_names]
    bars = ax.bar(range(len(model_names)), intrusion_rates, color=colors, alpha=0.8)
    
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Half-Truth Intrusion Rate')
    ax.set_title('Ranking Vulnerability: How Often Does Half-Truth Beat Correct?')
    ax.set_ylim(0, 1)
    
    for bar, rate in zip(bars, intrusion_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{rate:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Half-Truth Ranking Evaluation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="ViT-B/32",
                       help="CLIP model name (e.g., 'ViT-B-32', 'ViT-L-14')")
    parser.add_argument("--checkpoint_type", type=str, default="openclip",
                       choices=["openclip", "huggingface", "tripletclip", "external", "dac", "clove"],
                       help="Type of checkpoint to load")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to checkpoint file or HuggingFace model ID")
    parser.add_argument("--force_openclip", action="store_true",
                       help="Force using OpenCLIP instead of OpenAI CLIP")
    parser.add_argument("--pretrained", type=str, default="openai",
                       help="Pretrained weights for CLOVE (e.g., 'openai', 'laion2b_s34b_b79k')")
    parser.add_argument("--clove_weight", type=float, default=0.6,
                       help="Interpolation weight for CLOVE checkpoint (0-1, default 0.6)")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="coco",
                       choices=["laion", "coco"],
                       help="Dataset to use: 'laion' or 'coco'")
    parser.add_argument("--json_folder", type=str, 
                       default="swap_pos_json/coco/",
                       help="Folder containing JSON files with structured captions (for COCO)")
    parser.add_argument("--image_root", type=str, default=None,
                       help="Root directory for images (required for COCO, ignored for LAION)")
    
    # LAION-specific arguments
    parser.add_argument("--laion_data_root", type=str, default=None,
                       help="Root directory containing LAION tar files")
    parser.add_argument("--laion_json_root", type=str, default=None,
                       help="Root directory containing LAION JSON shards")
    parser.add_argument("--tar_range", type=str, default=None,
                       help="Tar range to load, format: 'start,end' (e.g., '0,100')")
    parser.add_argument("--laion_cache_dir", type=str, default=None,
                       help="Directory to cache extracted LAION images")
    
    # Ranking-specific arguments
    parser.add_argument("--n_distractors", type=int, default=10,
                       help="Number of distractor captions per query")
    
    # Experiment settings
    parser.add_argument("--num_samples", type=int, default=5000,
                       help="Maximum samples per condition")
    parser.add_argument("--output_dir", type=str, default="ranking_results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Comparison mode
    parser.add_argument("--compare_models", action="store_true",
                       help="Run comparison across multiple models")
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 70)
    print("HALF-TRUTH RANKING VULNERABILITY EXPERIMENT")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Checkpoint type: {args.checkpoint_type}")
    if args.checkpoint_path:
        print(f"  Checkpoint path: {args.checkpoint_path}")
    if args.checkpoint_type == "clove":
        print(f"  Pretrained: {args.pretrained}")
        print(f"  CLOVE weight: {args.clove_weight}")
    print(f"  Force OpenCLIP: {args.force_openclip}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Num distractors: {args.n_distractors}")
    print(f"  Output directory: {args.output_dir}")
    if args.dataset == "coco":
        print(f"  JSON folder: {args.json_folder}")
        print(f"  Image root: {args.image_root}")
    else:
        print(f"  LAION data root: {args.laion_data_root}")
        print(f"  LAION JSON root: {args.laion_json_root}")
        print(f"  Tar range: {args.tar_range}")
    print("=" * 70)
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate arguments based on dataset
    if args.dataset == "coco":
        if not args.image_root:
            logger.error("--image_root is required for COCO dataset")
            sys.exit(1)
    elif args.dataset == "laion":
        if not args.laion_data_root:
            logger.error("--laion_data_root is required for LAION dataset")
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples based on dataset type
    if args.dataset == "laion":
        # Parse tar range
        tar_range = None
        if args.tar_range:
            try:
                parts = args.tar_range.split(",")
                tar_range = (int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                logger.error(f"Invalid tar_range format: {args.tar_range}. Expected 'start,end'")
                sys.exit(1)
        
        # Import LAION generator
        from experiments.half_truth_vulnerability import LAIONHalfTruthSampleGenerator
        
        generator = LAIONHalfTruthSampleGenerator(
            data_root=args.laion_data_root,
            json_root=args.laion_json_root,
            tar_range=tar_range,
            max_samples_per_condition=args.num_samples,
            seed=args.seed,
            cache_dir=args.laion_cache_dir,
        )
        
        component_samples = generator.generate_component_samples()
        relation_samples = generator.generate_relation_samples()
        component_samples = generator.add_random_component_foils(component_samples, seed=args.seed)
        generator.close()
        
    else:
        # COCO dataset - handle directory of JSON files
        json_folder = Path(args.json_folder)
        
        if json_folder.is_file():
            # Single JSON file provided
            json_path = str(json_folder)
        else:
            # Folder provided - look for combined JSON or merge files
            combined_json = json_folder / "combined_data.json"
            if combined_json.exists():
                json_path = str(combined_json)
            else:
                # Try to find any JSON file
                json_files = list(json_folder.glob("*.json"))
                if not json_files:
                    logger.error(f"No JSON files found in {json_folder}")
                    sys.exit(1)
                
                if len(json_files) == 1:
                    json_path = str(json_files[0])
                else:
                    # Combine multiple JSON files
                    logger.info(f"Combining {len(json_files)} JSON files...")
                    all_data = []
                    for jf in json_files:
                        with open(jf, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_data.extend(data)
                            elif isinstance(data, dict):
                                all_data.append(data)
                    
                    # Save combined data
                    combined_path = output_dir / "combined_input_data.json"
                    with open(combined_path, 'w') as f:
                        json.dump(all_data, f)
                    json_path = str(combined_path)
                    logger.info(f"Combined {len(all_data)} samples into {combined_path}")
        
        # Generate samples
        generator = HalfTruthSampleGenerator(
            json_path=json_path,
            image_root=args.image_root,
            max_samples_per_condition=args.num_samples,
            seed=args.seed,
        )
        
        component_samples = generator.generate_component_samples()
        relation_samples = generator.generate_relation_samples()
        component_samples = generator.add_random_component_foils(component_samples, seed=args.seed)
    
    logger.info(f"Generated {len(component_samples)} component samples")
    logger.info(f"Generated {len(relation_samples)} relation samples")
    
    # Save samples for reproducibility
    from dataclasses import asdict as dc_asdict
    with open(output_dir / "component_samples.json", 'w') as f:
        json.dump([dc_asdict(s) for s in component_samples], f, indent=2)
    with open(output_dir / "relation_samples.json", 'w') as f:
        json.dump([dc_asdict(s) for s in relation_samples], f, indent=2)
    
    # Build distractor pool
    all_samples = component_samples + relation_samples
    distractor_pool = DistractorPool(all_samples, seed=args.seed)
    
    if args.compare_models:
        # Define models to compare
        models_config = {
            "CLIP": {
                'base_model': args.model_name,
                'checkpoint_type': 'openclip',
                'checkpoint_path': None,
            },
            # Add more models here as needed
        }
        
        analyzers = compare_models_ranking(
            models_config=models_config,
            component_samples=component_samples,
            relation_samples=relation_samples,
            n_distractors=args.n_distractors,
            output_dir=str(output_dir),
        )
        
        # Create cross-model comparison
        plot_model_comparison(analyzers, output_dir / "model_comparison.png")
        
    else:
        # Single model evaluation
        evaluator = HalfTruthRankingEvaluator(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            checkpoint_type=args.checkpoint_type,
            force_openclip=args.force_openclip,
            pretrained=args.pretrained,
            clove_weight=args.clove_weight,
            n_distractors=args.n_distractors,
        )
        evaluator.set_distractor_pool(distractor_pool)
        
        # Evaluate
        logger.info("Evaluating component samples...")
        component_results = evaluator.evaluate_all_components(component_samples)
        
        logger.info("Evaluating relation samples...")
        relation_results = evaluator.evaluate_all_relations(relation_samples)
        
        # Merge and analyze
        all_results = {**component_results, **relation_results}
        analyzer = RankingAnalyzer(all_results)
        
        # Print summary
        print("\n" + "="*70)
        print("RANKING ANALYSIS SUMMARY")
        print("="*70)
        print(analyzer.summary_table().to_string())
        
        # Print detailed metrics
        print("\nDetailed Metrics by Condition:")
        for cond, metrics in analyzer.metrics.items():
            print(f"\n{cond}:")
            print(metrics)
        
        # Save results
        analyzer.save_results(output_dir)
        analyzer.plot_intrusion_rates(output_dir / "intrusion_rates.png")
        analyzer.plot_rank_distributions(output_dir / "rank_distributions.png")
        analyzer.plot_mrr_comparison(output_dir / "mrr_comparison.png")
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

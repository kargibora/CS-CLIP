"""
Caption Consistency Test

Tests the consistency of CLIP scores as we progressively build up captions
from components to relations to the full caption.

Hypothesis: If CLIP has proper compositional understanding, scores should follow
a logical ordering based on semantic coverage:
    s(comp_1) <= s(comp_1 + comp_2) <= s(comp_1 + comp_2 + rel) <= s(full_caption)

The test measures:
1. Individual component scores: s(comp_i)
2. Aggregated component scores: s(comp_1 and comp_2 and ...)
3. Component + relation scores: s(comps where rel_1)
4. Full caption score: s(original_caption)

This reveals whether CLIP's scoring is consistent with semantic inclusion.

Author: Auto-generated
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.clip_wrapper import load_clip_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CaptionBuildupStep:
    """A single step in building up a caption."""
    step_type: str          # "component", "component_aggregate", "relation", "full"
    caption: str            # The caption at this step
    description: str        # Human-readable description of what was added
    score: float = 0.0      # CLIP similarity score


@dataclass 
class ConsistencyResult:
    """Results for a single image's caption buildup."""
    image_path: str
    image_key: str
    original_caption: str
    
    # Components (individual)
    components: List[str]
    component_scores: List[float]
    
    # Aggregated components (cumulative)
    component_aggregates: List[str]       # ["cat", "cat and floor", "cat and floor and ball"]
    component_aggregate_scores: List[float]
    
    # Relations
    relations: List[str]                  # ["cat scratching floor"]
    relation_scores: List[float]
    
    # Full buildup with relations
    buildup_with_relations: List[str]     # ["cat and floor", "cat and floor where cat scratching floor"]
    buildup_with_relations_scores: List[float]
    
    # Full caption
    full_caption_score: float
    
    # Analysis metrics
    components_monotonic: bool = False    # Are component scores monotonically increasing?
    relations_add_value: bool = False     # Do relations increase score over components alone?
    full_highest: bool = False            # Is full caption score highest?
    
    # Detailed step-by-step buildup
    buildup_steps: List[CaptionBuildupStep] = None


@dataclass
class ConsistencyAnalysis:
    """Aggregate analysis across all samples."""
    n_samples: int
    
    # Monotonicity rates
    component_monotonic_rate: float       # % of samples where comp scores increase with additions
    relation_adds_value_rate: float       # % of samples where adding relation increases score
    full_highest_rate: float              # % of samples where full caption has highest score
    
    # Average patterns
    avg_first_component_score: float
    avg_all_components_score: float
    avg_with_relations_score: float
    avg_full_caption_score: float
    
    # Score deltas
    avg_delta_add_component: float        # Avg change when adding a component
    avg_delta_add_relation: float         # Avg change when adding a relation
    avg_delta_to_full: float              # Avg change from buildup to full caption


# =============================================================================
# Sample Generation
# =============================================================================

class ConsistencySampleGenerator:
    """Generates samples for consistency testing."""
    
    def __init__(
        self,
        json_path: str,
        image_root: str,
        max_samples: int = 1000,
        min_components: int = 2,
        max_components: int = 5,
        seed: int = 42,
    ):
        self.json_path = json_path
        self.image_root = image_root
        self.max_samples = max_samples
        self.min_components = min_components
        self.max_components = max_components
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Load data
        logger.info(f"Loading data from {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} samples")
    
    def generate_samples(self) -> List[Dict]:
        """Generate samples suitable for consistency testing.
        
        Returns samples with:
        - At least min_components components
        - At least one relation
        - Valid image path
        """
        samples = []
        
        for sample in tqdm(self.data, desc="Filtering samples"):
            positive_components = sample.get('positive_components', [])
            relations = sample.get('relations', [])
            
            # Filter by component count
            if len(positive_components) < self.min_components:
                continue
            if len(positive_components) > self.max_components:
                # Take first max_components
                positive_components = positive_components[:self.max_components]
            
            # Need at least one relation
            if not relations:
                continue
            
            # Validate image
            image_path = sample.get('image_path', '')
            if image_path and self.image_root and not image_path.startswith('/'):
                image_path = os.path.join(self.image_root, image_path)
            
            if not image_path or not os.path.exists(image_path):
                continue
            
            samples.append({
                'image_path': image_path,
                'image_key': sample.get('sample_id', ''),
                'original_caption': sample.get('original_caption', ''),
                'positive_components': positive_components,
                'relations': relations,
            })
            
            if len(samples) >= self.max_samples:
                break
        
        logger.info(f"Generated {len(samples)} valid samples for consistency testing")
        return samples


# =============================================================================
# Evaluation
# =============================================================================

class ConsistencyEvaluator:
    """Evaluates caption consistency for CLIP models."""
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        force_openclip: bool = False,
        component_connector: str = " and ",
        relation_connector: str = " where ",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.component_connector = component_connector
        self.relation_connector = relation_connector
        
        logger.info(f"Loading model {model_name} on {self.device}")
        
        if checkpoint_path:
            self.model, self.preprocess = load_clip_model(model_name, self.device, force_openclip=force_openclip)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            self.model, self.preprocess = load_clip_model(model_name, self.device, force_openclip=force_openclip)
        
        self.model.eval()
        
        # Get tokenizer
        try:
            import clip
            self.tokenize = clip.tokenize
        except ImportError:
            import open_clip
            self.tokenize = open_clip.get_tokenizer(model_name)
    
    @torch.no_grad()
    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity between image and text."""
        if not text or not text.strip():
            return 0.0
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = self.tokenize([text]).to(self.device)
        
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_input)
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        similarity = (image_features @ text_features.T).item()
        return similarity
    
    @torch.no_grad()
    def compute_similarities_batch(self, image: Image.Image, texts: List[str]) -> List[float]:
        """Compute CLIP similarities for multiple texts efficiently."""
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [0.0] * len(texts)
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_inputs = self.tokenize(valid_texts).to(self.device)
        
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_inputs)
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()
        
        # Map back to original indices (handle empty texts)
        result = []
        valid_idx = 0
        for t in texts:
            if t and t.strip():
                result.append(float(similarities[valid_idx]))
                valid_idx += 1
            else:
                result.append(0.0)
        
        return result
    
    def format_relation(self, rel: Dict) -> str:
        """Format a relation as a string."""
        subject = rel.get('subject', '').lower().strip()
        relation_type = rel.get('relation_type', '').lower().strip()
        obj = rel.get('object', '').lower().strip()
        
        if all([subject, relation_type, obj]):
            return f"{subject} {relation_type} {obj}"
        return ""
    
    def evaluate_sample(self, sample: Dict) -> ConsistencyResult:
        """Evaluate consistency for a single sample.
        
        Builds up captions step by step:
        1. Individual components: "cat", "floor", "ball"
        2. Cumulative components: "cat", "cat and floor", "cat and floor and ball"
        3. Relations: "cat scratching floor"
        4. Components + relations: "cat and floor where cat scratching floor"
        5. Full caption: original caption
        """
        image_path = sample['image_path']
        image_key = sample['image_key']
        original_caption = sample['original_caption']
        components = [c.lower().strip() for c in sample['positive_components']]
        relations = sample['relations']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
        
        buildup_steps = []
        
        # ================================================================
        # Step 1: Individual component scores
        # ================================================================
        component_scores = self.compute_similarities_batch(image, components)
        
        for i, (comp, score) in enumerate(zip(components, component_scores)):
            buildup_steps.append(CaptionBuildupStep(
                step_type="component",
                caption=comp,
                description=f"Component {i+1}: {comp}",
                score=score
            ))
        
        # ================================================================
        # Step 2: Cumulative component aggregates
        # ================================================================
        component_aggregates = []
        current_aggregate = ""
        
        for i, comp in enumerate(components):
            if i == 0:
                current_aggregate = comp
            else:
                current_aggregate = f"{current_aggregate}{self.component_connector}{comp}"
            component_aggregates.append(current_aggregate)
        
        component_aggregate_scores = self.compute_similarities_batch(image, component_aggregates)
        
        for i, (agg, score) in enumerate(zip(component_aggregates, component_aggregate_scores)):
            buildup_steps.append(CaptionBuildupStep(
                step_type="component_aggregate",
                caption=agg,
                description=f"Components 1-{i+1} combined",
                score=score
            ))
        
        # ================================================================
        # Step 3: Individual relation scores
        # ================================================================
        relation_strings = []
        for rel in relations:
            if isinstance(rel, dict):
                rel_str = self.format_relation(rel)
                if rel_str:
                    relation_strings.append(rel_str)
        
        relation_scores = self.compute_similarities_batch(image, relation_strings) if relation_strings else []
        
        for i, (rel_str, score) in enumerate(zip(relation_strings, relation_scores)):
            buildup_steps.append(CaptionBuildupStep(
                step_type="relation",
                caption=rel_str,
                description=f"Relation {i+1}: {rel_str}",
                score=score
            ))
        
        # ================================================================
        # Step 4: Components + relations buildup
        # ================================================================
        buildup_with_relations = []
        buildup_with_relations_scores = []
        
        # Start with all components
        all_components_str = component_aggregates[-1] if component_aggregates else ""
        
        if all_components_str and relation_strings:
            # Add relations one by one
            current_buildup = all_components_str
            buildup_with_relations.append(current_buildup)
            
            for rel_str in relation_strings:
                current_buildup = f"{current_buildup}{self.relation_connector}{rel_str}"
                buildup_with_relations.append(current_buildup)
            
            buildup_with_relations_scores = self.compute_similarities_batch(image, buildup_with_relations)
            
            for i, (buildup, score) in enumerate(zip(buildup_with_relations, buildup_with_relations_scores)):
                if i == 0:
                    desc = "All components"
                else:
                    desc = f"All components + {i} relation(s)"
                
                buildup_steps.append(CaptionBuildupStep(
                    step_type="buildup",
                    caption=buildup,
                    description=desc,
                    score=score
                ))
        
        # ================================================================
        # Step 5: Full caption score
        # ================================================================
        full_caption_score = self.compute_similarity(image, original_caption) if original_caption else 0.0
        
        buildup_steps.append(CaptionBuildupStep(
            step_type="full",
            caption=original_caption,
            description="Original full caption",
            score=full_caption_score
        ))
        
        # ================================================================
        # Compute analysis metrics
        # ================================================================
        
        # Check if component aggregate scores are monotonically increasing
        components_monotonic = all(
            component_aggregate_scores[i] <= component_aggregate_scores[i+1] + 0.001  # small tolerance
            for i in range(len(component_aggregate_scores) - 1)
        ) if len(component_aggregate_scores) > 1 else True
        
        # Check if adding relations increases score
        if buildup_with_relations_scores and len(buildup_with_relations_scores) > 1:
            relations_add_value = buildup_with_relations_scores[-1] > buildup_with_relations_scores[0]
        else:
            relations_add_value = False
        
        # Check if full caption has highest score
        all_scores = component_scores + component_aggregate_scores + relation_scores + buildup_with_relations_scores
        full_highest = full_caption_score >= max(all_scores) if all_scores else True
        
        return ConsistencyResult(
            image_path=image_path,
            image_key=image_key,
            original_caption=original_caption,
            components=components,
            component_scores=component_scores,
            component_aggregates=component_aggregates,
            component_aggregate_scores=component_aggregate_scores,
            relations=relation_strings,
            relation_scores=relation_scores,
            buildup_with_relations=buildup_with_relations,
            buildup_with_relations_scores=buildup_with_relations_scores,
            full_caption_score=full_caption_score,
            components_monotonic=components_monotonic,
            relations_add_value=relations_add_value,
            full_highest=full_highest,
            buildup_steps=buildup_steps,
        )


# =============================================================================
# Analysis
# =============================================================================

class ConsistencyAnalyzer:
    """Analyzes consistency test results."""
    
    def __init__(self, results: List[ConsistencyResult]):
        self.results = [r for r in results if r is not None]
    
    def compute_analysis(self) -> ConsistencyAnalysis:
        """Compute aggregate analysis metrics."""
        n = len(self.results)
        if n == 0:
            return None
        
        # Monotonicity rates
        component_monotonic_rate = sum(1 for r in self.results if r.components_monotonic) / n
        relation_adds_value_rate = sum(1 for r in self.results if r.relations_add_value) / n
        full_highest_rate = sum(1 for r in self.results if r.full_highest) / n
        
        # Average scores at different stages
        avg_first_component_score = np.mean([r.component_scores[0] for r in self.results if r.component_scores])
        avg_all_components_score = np.mean([r.component_aggregate_scores[-1] for r in self.results if r.component_aggregate_scores])
        avg_with_relations_score = np.mean([r.buildup_with_relations_scores[-1] for r in self.results if r.buildup_with_relations_scores])
        avg_full_caption_score = np.mean([r.full_caption_score for r in self.results])
        
        # Score deltas
        deltas_add_component = []
        for r in self.results:
            if len(r.component_aggregate_scores) > 1:
                for i in range(len(r.component_aggregate_scores) - 1):
                    deltas_add_component.append(r.component_aggregate_scores[i+1] - r.component_aggregate_scores[i])
        
        deltas_add_relation = []
        for r in self.results:
            if len(r.buildup_with_relations_scores) > 1:
                # First entry is just components, rest include relations
                for i in range(len(r.buildup_with_relations_scores) - 1):
                    deltas_add_relation.append(r.buildup_with_relations_scores[i+1] - r.buildup_with_relations_scores[i])
        
        deltas_to_full = []
        for r in self.results:
            if r.buildup_with_relations_scores:
                deltas_to_full.append(r.full_caption_score - r.buildup_with_relations_scores[-1])
            elif r.component_aggregate_scores:
                deltas_to_full.append(r.full_caption_score - r.component_aggregate_scores[-1])
        
        return ConsistencyAnalysis(
            n_samples=n,
            component_monotonic_rate=component_monotonic_rate,
            relation_adds_value_rate=relation_adds_value_rate,
            full_highest_rate=full_highest_rate,
            avg_first_component_score=float(avg_first_component_score),
            avg_all_components_score=float(avg_all_components_score),
            avg_with_relations_score=float(avg_with_relations_score) if not np.isnan(avg_with_relations_score) else 0.0,
            avg_full_caption_score=float(avg_full_caption_score),
            avg_delta_add_component=float(np.mean(deltas_add_component)) if deltas_add_component else 0.0,
            avg_delta_add_relation=float(np.mean(deltas_add_relation)) if deltas_add_relation else 0.0,
            avg_delta_to_full=float(np.mean(deltas_to_full)) if deltas_to_full else 0.0,
        )
    
    def print_summary(self):
        """Print summary of results."""
        analysis = self.compute_analysis()
        
        if analysis is None:
            print("No valid results to analyze.")
            return
        
        print("\n" + "=" * 80)
        print("CAPTION CONSISTENCY TEST RESULTS")
        print("=" * 80)
        
        print(f"\nSamples analyzed: {analysis.n_samples}")
        
        print("\n--- Consistency Rates ---")
        print(f"  Component scores monotonically increasing: {analysis.component_monotonic_rate:.1%}")
        print(f"  Adding relations increases score: {analysis.relation_adds_value_rate:.1%}")
        print(f"  Full caption has highest score: {analysis.full_highest_rate:.1%}")
        
        print("\n--- Average Scores at Each Stage ---")
        print(f"  First component only:    {analysis.avg_first_component_score:.4f}")
        print(f"  All components combined: {analysis.avg_all_components_score:.4f}")
        print(f"  With relations added:    {analysis.avg_with_relations_score:.4f}")
        print(f"  Full original caption:   {analysis.avg_full_caption_score:.4f}")
        
        print("\n--- Average Score Changes ---")
        print(f"  Δ when adding a component:  {analysis.avg_delta_add_component:+.4f}")
        print(f"  Δ when adding a relation:   {analysis.avg_delta_add_relation:+.4f}")
        print(f"  Δ from buildup to full:     {analysis.avg_delta_to_full:+.4f}")
        
        print("\n--- Interpretation ---")
        if analysis.component_monotonic_rate > 0.7:
            print("  ✓ Good: Adding components generally increases similarity")
        else:
            print("  ✗ Poor: Adding components doesn't reliably increase similarity")
        
        if analysis.relation_adds_value_rate > 0.5:
            print("  ✓ Good: Relations add semantic value")
        else:
            print("  ✗ Poor: Relations don't reliably increase similarity")
        
        if analysis.full_highest_rate > 0.5:
            print("  ✓ Good: Full captions usually score highest")
        else:
            print("  ✗ Poor: Partial captions often outscore full captions")
        
        print("\n" + "=" * 80)
    
    def get_extreme_examples(self, n: int = 5, extreme_type: str = "worst") -> List[ConsistencyResult]:
        """Get examples with extreme consistency patterns."""
        
        # Score each result by how "inconsistent" it is
        scored_results = []
        for r in self.results:
            inconsistency_score = 0
            
            # Penalize non-monotonic component scores
            if not r.components_monotonic:
                inconsistency_score += 1
            
            # Penalize if relations don't add value
            if not r.relations_add_value:
                inconsistency_score += 1
            
            # Penalize if full caption not highest
            if not r.full_highest:
                inconsistency_score += 1
            
            # Also consider the magnitude of violations
            if r.component_aggregate_scores and len(r.component_aggregate_scores) > 1:
                for i in range(len(r.component_aggregate_scores) - 1):
                    if r.component_aggregate_scores[i] > r.component_aggregate_scores[i+1]:
                        inconsistency_score += abs(r.component_aggregate_scores[i] - r.component_aggregate_scores[i+1])
            
            scored_results.append((inconsistency_score, r))
        
        # Sort by inconsistency
        scored_results.sort(key=lambda x: x[0], reverse=(extreme_type == "worst"))
        
        return [r for _, r in scored_results[:n]]
    
    def save_results(self, output_path: str):
        """Save results to JSON."""
        analysis = self.compute_analysis()
        
        output = {
            'analysis': asdict(analysis) if analysis else None,
            'results': [asdict(r) for r in self.results],
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Saved results to {output_path}")


# =============================================================================
# Visualization
# =============================================================================

def plot_consistency_results(results: List[ConsistencyResult], output_dir: str):
    """Create visualization plots for consistency results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping plots")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid results
    results = [r for r in results if r is not None]
    
    if not results:
        return
    
    # ================================================================
    # Plot 1: Score progression across stages
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect scores at each stage
    first_component_scores = [r.component_scores[0] for r in results if r.component_scores]
    all_component_scores = [r.component_aggregate_scores[-1] for r in results if r.component_aggregate_scores]
    with_relations_scores = [r.buildup_with_relations_scores[-1] for r in results if r.buildup_with_relations_scores]
    full_scores = [r.full_caption_score for r in results]
    
    stages = ['First\nComponent', 'All\nComponents', 'With\nRelations', 'Full\nCaption']
    all_stage_scores = [first_component_scores, all_component_scores, with_relations_scores, full_scores]
    
    # Box plot
    bp = ax.boxplot(all_stage_scores, labels=stages, patch_artist=True)
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add mean line
    means = [np.mean(scores) for scores in all_stage_scores]
    ax.plot(range(1, len(stages)+1), means, 'ko-', markersize=10, linewidth=2, label='Mean')
    
    ax.set_ylabel('CLIP Similarity Score', fontsize=12)
    ax.set_title('Score Progression: Component → Relation → Full Caption', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_progression.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ================================================================
    # Plot 2: Score changes (deltas) distribution
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Delta: adding components
    ax = axes[0]
    deltas_add_component = []
    for r in results:
        if len(r.component_aggregate_scores) > 1:
            for i in range(len(r.component_aggregate_scores) - 1):
                deltas_add_component.append(r.component_aggregate_scores[i+1] - r.component_aggregate_scores[i])
    
    if deltas_add_component:
        ax.hist(deltas_add_component, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
        ax.axvline(x=np.mean(deltas_add_component), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(deltas_add_component):.4f}')
        ax.set_xlabel('Δ Score')
        ax.set_ylabel('Count')
        ax.set_title('Score Change: Adding Component')
        ax.legend()
    
    # Delta: adding relations
    ax = axes[1]
    deltas_add_relation = []
    for r in results:
        if len(r.buildup_with_relations_scores) > 1:
            for i in range(len(r.buildup_with_relations_scores) - 1):
                deltas_add_relation.append(r.buildup_with_relations_scores[i+1] - r.buildup_with_relations_scores[i])
    
    if deltas_add_relation:
        ax.hist(deltas_add_relation, bins=50, edgecolor='black', alpha=0.7, color='#f39c12')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
        ax.axvline(x=np.mean(deltas_add_relation), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(deltas_add_relation):.4f}')
        ax.set_xlabel('Δ Score')
        ax.set_ylabel('Count')
        ax.set_title('Score Change: Adding Relation')
        ax.legend()
    
    # Delta: buildup to full caption
    ax = axes[2]
    deltas_to_full = []
    for r in results:
        if r.buildup_with_relations_scores:
            deltas_to_full.append(r.full_caption_score - r.buildup_with_relations_scores[-1])
        elif r.component_aggregate_scores:
            deltas_to_full.append(r.full_caption_score - r.component_aggregate_scores[-1])
    
    if deltas_to_full:
        ax.hist(deltas_to_full, bins=50, edgecolor='black', alpha=0.7, color='#e74c3c')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
        ax.axvline(x=np.mean(deltas_to_full), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(deltas_to_full):.4f}')
        ax.set_xlabel('Δ Score')
        ax.set_ylabel('Count')
        ax.set_title('Score Change: Buildup → Full Caption')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_deltas.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ================================================================
    # Plot 3: Consistency rates pie chart
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Component monotonicity
    ax = axes[0]
    monotonic_count = sum(1 for r in results if r.components_monotonic)
    non_monotonic_count = len(results) - monotonic_count
    ax.pie([monotonic_count, non_monotonic_count], 
           labels=['Monotonic', 'Non-monotonic'],
           colors=['#2ecc71', '#e74c3c'],
           autopct='%1.1f%%',
           startangle=90)
    ax.set_title('Component Score Monotonicity')
    
    # Relations add value
    ax = axes[1]
    adds_value_count = sum(1 for r in results if r.relations_add_value)
    no_value_count = len(results) - adds_value_count
    ax.pie([adds_value_count, no_value_count],
           labels=['Adds Value', 'No Added Value'],
           colors=['#2ecc71', '#e74c3c'],
           autopct='%1.1f%%',
           startangle=90)
    ax.set_title('Relations Add Semantic Value')
    
    # Full highest
    ax = axes[2]
    full_highest_count = sum(1 for r in results if r.full_highest)
    not_highest_count = len(results) - full_highest_count
    ax.pie([full_highest_count, not_highest_count],
           labels=['Full Highest', 'Partial Higher'],
           colors=['#2ecc71', '#e74c3c'],
           autopct='%1.1f%%',
           startangle=90)
    ax.set_title('Full Caption Has Highest Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'consistency_rates.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plots to {output_dir}")


def print_example_buildup(result: ConsistencyResult, example_num: int = 1):
    """Print detailed buildup for a single example."""
    print(f"\n{'='*80}")
    print(f"EXAMPLE {example_num}: {result.image_key}")
    print(f"{'='*80}")
    
    print(f"\nImage: {result.image_path}")
    print(f"Original caption: \"{result.original_caption}\"")
    
    print(f"\n--- Components ({len(result.components)}) ---")
    for i, (comp, score) in enumerate(zip(result.components, result.component_scores)):
        print(f"  {i+1}. \"{comp}\" → score: {score:.4f}")
    
    print(f"\n--- Component Buildup ---")
    for i, (agg, score) in enumerate(zip(result.component_aggregates, result.component_aggregate_scores)):
        delta = ""
        if i > 0:
            d = score - result.component_aggregate_scores[i-1]
            delta = f" (Δ: {d:+.4f})"
        print(f"  {i+1}. \"{agg}\" → score: {score:.4f}{delta}")
    
    if result.relations:
        print(f"\n--- Relations ({len(result.relations)}) ---")
        for i, (rel, score) in enumerate(zip(result.relations, result.relation_scores)):
            print(f"  {i+1}. \"{rel}\" → score: {score:.4f}")
    
    if result.buildup_with_relations:
        print(f"\n--- Buildup with Relations ---")
        for i, (buildup, score) in enumerate(zip(result.buildup_with_relations, result.buildup_with_relations_scores)):
            delta = ""
            if i > 0:
                d = score - result.buildup_with_relations_scores[i-1]
                delta = f" (Δ: {d:+.4f})"
            print(f"  {i+1}. \"{buildup[:80]}...\" → score: {score:.4f}{delta}")
    
    print(f"\n--- Full Caption ---")
    print(f"  \"{result.original_caption}\" → score: {result.full_caption_score:.4f}")
    
    print(f"\n--- Consistency Analysis ---")
    print(f"  Components monotonic: {'✓' if result.components_monotonic else '✗'}")
    print(f"  Relations add value:  {'✓' if result.relations_add_value else '✗'}")
    print(f"  Full caption highest: {'✓' if result.full_highest else '✗'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Caption Consistency Test")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="ViT-B-32",
                       help="CLIP model name")
    parser.add_argument("--checkpoint_type", type=str, default="openclip",
                       choices=["openclip", "external", "local", "huggingface"],
                       help="Type of checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to checkpoint file")
    
    # Dataset arguments
    parser.add_argument("--json_folder", type=str, required=True,
                       help="Folder containing JSON files with structured captions")
    parser.add_argument("--image_root", type=str, required=True,
                       help="Root directory for images")
    
    # Experiment settings
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to evaluate")
    parser.add_argument("--min_components", type=int, default=2,
                       help="Minimum number of components per sample")
    parser.add_argument("--max_components", type=int, default=5,
                       help="Maximum number of components per sample")
    parser.add_argument("--output_dir", type=str, default="consistency_results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Output options
    parser.add_argument("--show_examples", type=int, default=5,
                       help="Number of example buildups to print")
    parser.add_argument("--component_connector", type=str, default=" and ",
                       help="Connector between components")
    parser.add_argument("--relation_connector", type=str, default=" where ",
                       help="Connector before relations")
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 70)
    print("CAPTION CONSISTENCY TEST")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Checkpoint type: {args.checkpoint_type}")
    if args.checkpoint_path:
        print(f"  Checkpoint path: {args.checkpoint_path}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Components: {args.min_components}-{args.max_components}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine force_openclip
    force_openclip = args.checkpoint_type in ["openclip", "external", "local"]
    
    # Find and combine JSON files
    json_folder = Path(args.json_folder)
    if json_folder.is_file():
        json_path = str(json_folder)
    else:
        json_files = list(json_folder.glob("*.json"))
        if not json_files:
            logger.error(f"No JSON files found in {json_folder}")
            sys.exit(1)
        
        # Combine JSON files
        all_data = []
        for jf in json_files:
            with open(jf, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        
        json_path = str(output_dir / "combined_input_data.json")
        with open(json_path, 'w') as f:
            json.dump(all_data, f)
        logger.info(f"Combined {len(json_files)} JSON files into {json_path}")
    
    # Generate samples
    generator = ConsistencySampleGenerator(
        json_path=json_path,
        image_root=args.image_root,
        max_samples=args.num_samples,
        min_components=args.min_components,
        max_components=args.max_components,
        seed=args.seed,
    )
    
    samples = generator.generate_samples()
    
    if not samples:
        logger.error("No valid samples found!")
        sys.exit(1)
    
    # Save samples
    with open(output_dir / "samples.json", 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Evaluate
    checkpoint_path = args.checkpoint_path if args.checkpoint_type != "openclip" else None
    
    evaluator = ConsistencyEvaluator(
        model_name=args.model_name,
        checkpoint_path=checkpoint_path,
        force_openclip=force_openclip,
        component_connector=args.component_connector,
        relation_connector=args.relation_connector,
    )
    
    results = []
    for sample in tqdm(samples, desc="Evaluating consistency"):
        result = evaluator.evaluate_sample(sample)
        if result:
            results.append(result)
    
    logger.info(f"Evaluated {len(results)} samples successfully")
    
    # Analyze
    analyzer = ConsistencyAnalyzer(results)
    analyzer.print_summary()
    analyzer.save_results(str(output_dir / "results.json"))
    
    # Create plots
    plot_consistency_results(results, str(output_dir))
    
    # Show examples
    if args.show_examples > 0:
        print("\n" + "=" * 80)
        print(f"EXAMPLE BUILDUPS (showing {args.show_examples} samples)")
        print("=" * 80)
        
        # Show a mix of good and bad examples
        worst_examples = analyzer.get_extreme_examples(n=args.show_examples // 2, extreme_type="worst")
        best_examples = analyzer.get_extreme_examples(n=args.show_examples // 2, extreme_type="best")
        
        print("\n--- Most Inconsistent Examples ---")
        for i, r in enumerate(worst_examples):
            print_example_buildup(r, example_num=i+1)
        
        print("\n--- Most Consistent Examples ---")
        for i, r in enumerate(best_examples):
            print_example_buildup(r, example_num=i+1)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

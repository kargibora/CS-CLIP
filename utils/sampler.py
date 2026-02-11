from typing import List, Tuple, Dict, Optional
import random

# =====================================================================
# Structured Positive-Negative Pair Sampling
# =====================================================================

def _clean_relation_type(relation_type: str) -> str:
    """Clean up relation type string for caption formatting."""
    rel = relation_type.strip()
    rel = rel.replace("_", " ")
    return rel


def _format_component_caption(component: str, extra_components: Optional[List[str]] = None) -> str:
    """Format a component caption with optional additional components."""
    parts = [component]
    if extra_components:
        parts.extend(extra_components)
    random.shuffle(parts)
    return " and ".join(parts)


def _format_relation_caption(rel: Dict) -> str:
    """Format a relation dictionary into a caption string."""
    subject = (rel.get("subject") or "").strip()
    relation_type = (rel.get("relation_type") or "").strip()
    obj = (rel.get("object") or "").strip()

    if not subject or not relation_type or not obj:
        return ""

    rel_phrase = _clean_relation_type(relation_type)
    return f"{subject} {rel_phrase} {obj}"


class StructuredSampler:
    """
    Sampler for generating structured positive-negative caption pairs.
    
    This sampler creates targeted negative examples by:
    1. Component pairs: Replacing a component with its negative while optionally
       keeping context (other components)
    2. Relation pairs: Creating relation-based negatives through:
       - Relation opposite (e.g., "on" → "under")
       - Subject-object swap
       - Subject replacement (replace subject with its negative)
       - Object replacement (replace object with its negative)
    3. Binding pairs: Swapping nouns between two components while keeping their
       attributes (tests attribute-object binding)
       - E.g., "blue dog" + "red car" → pos: "blue dog", neg: "blue car"
    
    This provides much stronger supervision than random negatives.
    """
    
    def __init__(
        self,
        structured_relation_prob: float = 0.5,
        use_context_in_component_pairs: bool = True,
        binding_negative_prob: float = 0.0,
    ):
        """
        Initialize the structured sampler.
        
        Args:
            structured_relation_prob: Probability of trying relation pairs first (vs component pairs)
            use_context_in_component_pairs: If True, sometimes include additional
                components for context in component pairs
            binding_negative_prob: Probability of sampling binding pairs (noun swaps between
                components). If > 0, will try binding pairs first with this probability.
        """
        self.structured_relation_prob = structured_relation_prob
        self.use_context_in_component_pairs = use_context_in_component_pairs
        self.binding_negative_prob = binding_negative_prob

    def _sample_binding_pair(self, sample: Dict) -> Tuple[str, str, Dict]:
        """
        Sample a binding negative pair from pre-generated binding_negatives.
        
        Binding negatives test attribute-object binding by keeping attributes
        but swapping nouns between components.
        
        Example:
            "blue dog" + "red car" → pos: "blue dog", neg: "blue car"
            
        Returns:
            Tuple of (positive_caption, negative_caption, metadata)
            Returns ("", "", {}) if no valid pair can be created
        """
        binding_negatives = sample.get("binding_negatives", []) or []
        
        if not binding_negatives:
            return "", "", {}
        
        # Pick a random binding pair
        binding = random.choice(binding_negatives)
        
        component_1 = binding.get("component_1", "").strip()
        component_2 = binding.get("component_2", "").strip()
        binding_neg_1 = binding.get("binding_neg_1", "").strip()
        binding_neg_2 = binding.get("binding_neg_2", "").strip()
        
        if not component_1 or not binding_neg_1:
            return "", "", {}
        
        # Randomly choose which component to use as positive
        # Option 1: component_1 as positive, binding_neg_1 as negative
        # Option 2: component_2 as positive, binding_neg_2 as negative
        if random.random() < 0.5:
            pos_caption = component_1
            neg_caption = binding_neg_1
            meta = {
                "pair_type": "binding",
                "original_component": component_1,
                "paired_component": component_2,
                "swapped_noun": binding.get("swapped_noun_1", ""),
            }
        else:
            if not component_2 or not binding_neg_2:
                # Fall back to first option
                pos_caption = component_1
                neg_caption = binding_neg_1
                meta = {
                    "pair_type": "binding",
                    "original_component": component_1,
                    "paired_component": component_2,
                    "swapped_noun": binding.get("swapped_noun_1", ""),
                }
            else:
                pos_caption = component_2
                neg_caption = binding_neg_2
                meta = {
                    "pair_type": "binding",
                    "original_component": component_2,
                    "paired_component": component_1,
                    "swapped_noun": binding.get("swapped_noun_2", ""),
                }
        
        if pos_caption == neg_caption:
            return "", "", {}
        
        return pos_caption, neg_caption, meta

    def _sample_structured_component_pair(self, sample: Dict) -> Tuple[str, str, Dict]:
        """
        Sample a structured component-based positive-negative pair.
        
        Returns:
            Tuple of (positive_caption, negative_caption, metadata)
            Returns ("", "", {}) if no valid pair can be created
        """
        positive_components = sample.get("positive_components", []) or []
        negative_components = sample.get("negative_components", {}) or {}

        if not positive_components or not negative_components:
            return "", "", {}

        # Find components that have negatives
        candidates = [
            comp for comp in positive_components
            if comp in negative_components and negative_components[comp]
        ]
        if not candidates:
            return "", "", {}

        # Pick a component and its negative
        component = random.choice(candidates)
        neg_options = negative_components[component]
        neg_choice = random.choice(neg_options)

        if isinstance(neg_choice, dict):
            neg_text = (neg_choice.get("negative") or "").strip()
            change_type = (neg_choice.get("change_type") or "").strip()
        else:
            neg_text = str(neg_choice).strip()
            change_type = ""

        if not neg_text:
            return "", "", {}

        # Optionally add context from other components
        other_components = [c for c in positive_components if c != component]
        extra: List[str] = []
        if other_components and self.use_context_in_component_pairs:
            if random.random() < 0.5:  # 50% chance to add one extra component
                extra = [random.choice(other_components)]

        pos_caption = _format_component_caption(component, extra_components=extra)
        neg_caption = _format_component_caption(neg_text, extra_components=extra)

        meta = {
            "pair_type": "component",
            "component": component,
            "change_type": change_type,
        }
        return pos_caption, neg_caption, meta

    def _sample_structured_relation_pair(self, sample: Dict) -> Tuple[str, str, Dict]:
        """
        Sample a structured relation-based positive-negative pair.
        
        Supports TWO formats:
        
        NEW FORMAT (negatives embedded in relations):
        {
            "relations": [
                {
                    "subject": "cat",
                    "relation_type": "is on",
                    "object": "table",
                    "negatives": [
                        {"relation_type": "is under", "change_type": "antonym"},
                        {"subject": "table", "object": "cat", "change_type": "subject_object_swap"}
                    ]
                }
            ]
        }
        
        OLD FORMAT (separate negative_relations list):
        {
            "relations": [...],
            "negative_relations": [
                {"subject": "cat", "relation_type": "is under", "object": "table", "change_type": "relation_opposite"}
            ]
        }
        
        Strategies tried in order:
        1. Embedded negatives in relation (NEW FORMAT - antonym, negation, swap)
        2. Explicit negative_relations list (OLD FORMAT - relation_opposite, subject_object_swap)
        3. Synthetic subject_replace (replace subject with its component negative)
        4. Synthetic object_replace (replace object with its component negative)
        
        Returns:
            Tuple of (positive_caption, negative_caption, metadata)
            Returns ("", "", {}) if no valid pair can be created
        """
        relations = sample.get("relations", []) or []
        negative_relations = sample.get("negative_relations", []) or []
        negative_components = sample.get("negative_components", {}) or {}

        if not relations:
            return "", "", {}

        # Try up to 10 relations to find one with a valid negative
        max_trials = min(10, len(relations))
        shuffled_relations = list(relations)
        random.shuffle(shuffled_relations)
        
        for pos_rel in shuffled_relations[:max_trials]:
            subj = (pos_rel.get("subject") or "").strip()
            obj = (pos_rel.get("object") or "").strip()
            rel_type = (pos_rel.get("relation_type") or "").strip()

            if not subj or not obj or not rel_type:
                continue

            candidate_negs: List[Tuple[str, Dict]] = []

            # ============================================================
            # Strategy 1: NEW FORMAT - Embedded negatives within relation
            # ============================================================
            embedded_negatives = pos_rel.get("negatives", []) or []
            for neg in embedded_negatives:
                if not isinstance(neg, dict):
                    continue
                    
                change_type = (neg.get("change_type") or "").strip().lower()
                
                if change_type in ["swap", "subject_object_swap"]:
                    # Subject-object swap
                    swap_subj = neg.get("subject", neg.get("swap_subject", "")).strip()
                    swap_obj = neg.get("object", neg.get("swap_object", "")).strip()
                    
                    # If swap_subject/swap_object not provided, swap original
                    if not swap_subj and not swap_obj:
                        swap_subj = obj
                        swap_obj = subj
                    
                    if swap_subj and swap_obj:
                        neg_rel = {
                            "subject": swap_subj,
                            "relation_type": rel_type,
                            "object": swap_obj,
                            "change_type": "subject_object_swap",
                        }
                        candidate_negs.append(("subject_object_swap", neg_rel))
                
                elif change_type in ["antonym", "negation", "relation_change", "relation_opposite"]:
                    # Relation type change (antonym or negation)
                    new_rel_type = neg.get("relation_type", "").strip()
                    if new_rel_type and new_rel_type != rel_type:
                        neg_rel = {
                            "subject": subj,
                            "relation_type": new_rel_type,
                            "object": obj,
                            "change_type": change_type,
                        }
                        candidate_negs.append((change_type, neg_rel))

            # ============================================================
            # Strategy 2: OLD FORMAT - Explicit negative_relations list
            # ============================================================
            if not candidate_negs and negative_relations:
                for neg_rel in negative_relations:
                    change_type = (neg_rel.get("change_type") or "").strip()
                    n_subj = (neg_rel.get("subject") or "").strip()
                    n_obj = (neg_rel.get("object") or "").strip()
                    if not n_subj or not n_obj:
                        continue

                    if change_type == "relation_opposite":
                        if n_subj == subj and n_obj == obj:
                            candidate_negs.append(("relation_opposite", neg_rel))

                    elif change_type == "subject_object_swap":
                        if n_subj == obj and n_obj == subj:
                            candidate_negs.append(("subject_object_swap", neg_rel))

            # ============================================================
            # Strategy 3: Synthetic subject_replace
            # ============================================================
            if subj in negative_components and negative_components[subj]:
                neg_choice = random.choice(negative_components[subj])
                if isinstance(neg_choice, dict):
                    neg_subject = (neg_choice.get("negative") or "").strip()
                else:
                    neg_subject = str(neg_choice).strip()
                if neg_subject:
                    neg_rel = {
                        "subject": neg_subject,
                        "relation_type": rel_type,
                        "object": obj,
                        "change_type": "subject_replace",
                    }
                    candidate_negs.append(("subject_replace", neg_rel))

            # ============================================================
            # Strategy 4: Synthetic object_replace
            # ============================================================
            if obj in negative_components and negative_components[obj]:
                neg_choice = random.choice(negative_components[obj])
                if isinstance(neg_choice, dict):
                    neg_object = (neg_choice.get("negative") or "").strip()
                else:
                    neg_object = str(neg_choice).strip()
                if neg_object:
                    neg_rel = {
                        "subject": subj,
                        "relation_type": rel_type,
                        "object": neg_object,
                        "change_type": "object_replace",
                    }
                    candidate_negs.append(("object_replace", neg_rel))

            # ============================================================
            # Select from candidates
            # ============================================================
            if candidate_negs:
                change_type, neg_rel = random.choice(candidate_negs)
                pos_caption = _format_relation_caption(pos_rel)
                neg_caption = _format_relation_caption(neg_rel)

                if not pos_caption or not neg_caption or pos_caption == neg_caption:
                    continue

                meta = {
                    "pair_type": "relation",
                    "change_type": change_type,
                    "pos_relation": pos_rel,
                    "neg_relation": neg_rel,
                }
                return pos_caption, neg_caption, meta

        return "", "", {}

    def sample_structured_positive_and_negative(self, sample: Dict) -> Tuple[str, str, Dict]:
        """
        Sample a structured positive-negative caption pair.
        
        Sampling order:
        1. Try binding pairs first with probability binding_negative_prob
        2. Try relation pairs or component pairs based on structured_relation_prob,
           with fallback to the other type if the first fails.
        
        Returns:
            Tuple of (positive_caption, negative_caption, metadata)
            Returns ("", "", {}) if no valid pair can be created
        """
        # Try binding pairs first with probability binding_negative_prob
        if self.binding_negative_prob > 0 and random.random() < self.binding_negative_prob:
            pos, neg, meta = self._sample_binding_pair(sample)
            if pos and neg:
                return pos, neg, meta
            # Binding pairs not available, fall through to other methods
        
        # Try relation first with probability structured_relation_prob
        if random.random() < self.structured_relation_prob:
            pos, neg, meta = self._sample_structured_relation_pair(sample)
            if pos and neg:
                return pos, neg, meta

            # Fallback to component pairs
            pos, neg, meta = self._sample_structured_component_pair(sample)
            if pos and neg:
                return pos, neg, meta
        else:
            # Try component first
            pos, neg, meta = self._sample_structured_component_pair(sample)
            if pos and neg:
                return pos, neg, meta

            # Fallback to relation pairs
            pos, neg, meta = self._sample_structured_relation_pair(sample)
            if pos and neg:
                return pos, neg, meta

        return "", "", {}


# =====================================================================
# Legacy Component-Based Sampling (Non-Structured)
# =====================================================================

class LegacySampler:
    """
    Legacy sampler for component-based positive/negative caption sampling.
    
    This uses the original random sampling approach where:
    - Positives: Sample random subset of components, optionally add relation
    - Negatives: Single negative using in-place replacement, component swap, or relation swap
    
    Used as fallback when structured sampling is disabled or for single-negative mode.
    """
    
    def __init__(
        self,
        max_components_per_sample: Optional[int] = None,
        max_positive_components_with_negative: Optional[int] = None,
        sample_relations: bool = False,
        sample_relation_or_components: str = "both",
        relation_sample_prob: float = 0.8,
        negative_relation_sample_prob: float = 0.8,
        inplace_replacement_prob: float = 0.7,
    ):
        """
        Initialize legacy sampler.
        
        Args:
            max_components_per_sample: Max components per positive caption (None = random)
            max_positive_components_with_negative: Max positive components in negative caption
            sample_relations: Whether to sample relations for positive captions
            sample_relation_or_components: "both", "relation_only", "components_only", or "either"
            relation_sample_prob: Probability of sampling relation for positive caption
            negative_relation_sample_prob: Probability of using negative relation vs component
            inplace_replacement_prob: Probability of in-place component replacement for negatives
        """
        self.max_components_per_sample = max_components_per_sample
        self.max_positive_components_with_negative = max_positive_components_with_negative
        self.sample_relations = sample_relations
        self.sample_relation_or_components = sample_relation_or_components
        self.relation_sample_prob = relation_sample_prob
        self.negative_relation_sample_prob = negative_relation_sample_prob
        self.inplace_replacement_prob = inplace_replacement_prob
    
    def sample_component_positive(self, sample: Dict) -> Tuple[str, int]:
        """
        Sample a positive caption from components, optionally with relation.
        
        Returns:
            Tuple of (caption, num_components_used)
        """
        positive_components = sample.get("positive_components", [])
        relations = sample.get("relations", [])
        
        # Handle "either" mode - randomly choose relation OR components
        if self.sample_relation_or_components == "either":
            if self.sample_relations and relations and random.random() < self.relation_sample_prob:
                relation = random.choice(relations)
                subject = relation.get("subject", "")
                relation_type = relation.get("relation_type", "")
                obj = relation.get("object", "")
                
                if subject and relation_type and obj:
                    relation_caption = f"{subject} {relation_type} {obj}"
                    return relation_caption, 0
            
            if positive_components:
                num_components = len(positive_components)
                if num_components == 1:
                    return positive_components[0], 1
                else:
                    if self.max_components_per_sample is None:
                        num_to_sample = random.randint(1, num_components)
                    else:
                        num_to_sample = min(self.max_components_per_sample, num_components)
                    
                    selected_components = random.sample(positive_components, num_to_sample)
                    return " and ".join(selected_components), num_to_sample
            
            return sample.get("original_caption", sample.get("caption", "")), 0
        
        # Handle relation_only mode
        if self.sample_relation_or_components == "relation_only":
            if self.sample_relations and relations:
                relation = random.choice(relations)
                subject = relation.get("subject", "")
                relation_type = relation.get("relation_type", "")
                obj = relation.get("object", "")
                
                if subject and relation_type and obj:
                    return f"{subject} {relation_type} {obj}", 0
            
            return sample.get("original_caption", sample.get("caption", "")), 0
        
        if not positive_components:
            return sample.get("original_caption", sample.get("caption", "")), 0
        
        # Sample components
        num_components = len(positive_components)
        if num_components == 1:
            caption = positive_components[0]
            num_sampled = 1
        else:
            if self.max_components_per_sample is None:
                num_to_sample = random.randint(1, num_components)
            else:
                num_to_sample = min(self.max_components_per_sample, num_components)
            
            selected_components = random.sample(positive_components, num_to_sample)
            caption = " and ".join(selected_components)
            num_sampled = num_to_sample
        
        # Handle relation sampling based on mode
        if self.sample_relation_or_components == "components_only":
            return caption, num_sampled
        
        # Default "both" mode: Optionally add relation
        if self.sample_relations and random.random() < self.relation_sample_prob:
            if relations:
                relation = random.choice(relations)
                subject = relation.get("subject", "")
                relation_type = relation.get("relation_type", "")
                obj = relation.get("object", "")
                
                if subject and relation_type and obj:
                    caption = f"{caption} where {subject} {relation_type} {obj}"
        
        return caption, num_sampled
    
    def sample_component_negative(
        self, 
        sample: Dict,
        use_swap_negatives: bool = True
    ) -> str:
        """
        Sample a negative caption using multiple strategies.
        
        Supports TWO formats for relation negatives:
        
        NEW FORMAT (negatives embedded in relations):
        {"relations": [{"subject": "cat", "relation_type": "is on", "object": "table", 
                        "negatives": [{"relation_type": "is under", "change_type": "antonym"}]}]}
        
        OLD FORMAT (separate negative_relations list):
        {"negative_relations": [{"subject": "cat", "relation_type": "is under", "object": "table"}]}
        
        Priority:
        1. Pre-generated swap_negatives (if available and use_swap_negatives=True)
        2. In-place component replacement
        3. Negative relation triplet (NEW or OLD format)
        4. Component replacement with random other components
        
        Args:
            sample: Sample dict with negative data
            use_swap_negatives: Whether to use pre-generated swap_negatives first
        
        Returns:
            Negative caption string (empty string if no negatives available)
        """
        # Priority 1: Use pre-generated swap_negatives
        if use_swap_negatives:
            swap_negatives = sample.get("swap_negatives", [])
            if swap_negatives and isinstance(swap_negatives, list):
                # Randomly select one swap negative
                swap_choice = random.choice(swap_negatives)
                if isinstance(swap_choice, dict):
                    neg_text = swap_choice.get("negative", "")
                    if neg_text and neg_text.strip():
                        return neg_text.strip()
        
        negative_components = sample.get("negative_components", {})
        
        # Priority 2: In-place replacement
        if self.inplace_replacement_prob > 0 and random.random() < self.inplace_replacement_prob:
            inplace_result = self._sample_inplace_replacement_negative(sample)
            if inplace_result:
                return inplace_result
        
        # Priority 3: Negative relation triplet - try to get one from either format
        neg_relation_result = self._sample_negative_relation_caption(sample)
        if neg_relation_result:
            return neg_relation_result
        
        # Priority 4: Component replacement
        positive_components = sample.get("positive_components", [])
        
        if not positive_components or not negative_components:
            return ""
        
        components_with_negatives = [
            comp for comp in positive_components 
            if comp in negative_components and negative_components[comp]
        ]
        
        if not components_with_negatives:
            return ""
        
        component_to_replace = random.choice(components_with_negatives)
        negative_options = negative_components[component_to_replace]
        negative_choice = random.choice(negative_options)
        
        if isinstance(negative_choice, dict):
            negative_text = negative_choice.get("negative", "")
        else:
            negative_text = str(negative_choice)
        
        if not negative_text:
            return ""
        
        other_components = [comp for comp in positive_components if comp != component_to_replace]
        
        if other_components:
            if self.max_positive_components_with_negative is None:
                num_to_sample = random.randint(0, len(other_components))
            else:
                max_allowed = min(self.max_positive_components_with_negative, len(other_components))
                num_to_sample = random.randint(0, max_allowed)
            
            sampled_others = random.sample(other_components, num_to_sample) if num_to_sample > 0 else []
        else:
            sampled_others = []
        
        all_components = sampled_others + [negative_text]
        random.shuffle(all_components)
        
        return " and ".join(all_components)
    
    def _sample_negative_relation_caption(self, sample: Dict) -> str:
        """
        Sample a negative relation caption, supporting both NEW and OLD formats.
        
        NEW FORMAT: negatives embedded in relations
        OLD FORMAT: separate negative_relations list
        
        Returns:
            Negative relation caption string, or empty string if none available
        """
        relations = sample.get("relations", []) or []
        negative_relations = sample.get("negative_relations", []) or []
        positive_components = sample.get("positive_components", [])
        
        # Collect all available negative relations from both formats
        all_neg_relations = []
        
        # NEW FORMAT: Extract negatives embedded in relations
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            
            subj = (rel.get("subject") or "").strip()
            obj = (rel.get("object") or "").strip()
            rel_type = (rel.get("relation_type") or "").strip()
            
            if not subj or not obj or not rel_type:
                continue
            
            embedded_negatives = rel.get("negatives", []) or []
            for neg in embedded_negatives:
                if not isinstance(neg, dict):
                    continue
                
                change_type = (neg.get("change_type") or "").strip().lower()
                
                if change_type in ["swap", "subject_object_swap"]:
                    # Subject-object swap
                    swap_subj = neg.get("subject", neg.get("swap_subject", obj)).strip()
                    swap_obj = neg.get("object", neg.get("swap_object", subj)).strip()
                    if swap_subj and swap_obj:
                        all_neg_relations.append({
                            "subject": swap_subj,
                            "relation_type": rel_type,
                            "object": swap_obj,
                        })
                elif change_type in ["antonym", "negation", "relation_change", "relation_opposite"]:
                    # Relation type change
                    new_rel_type = neg.get("relation_type", "").strip()
                    if new_rel_type:
                        all_neg_relations.append({
                            "subject": subj,
                            "relation_type": new_rel_type,
                            "object": obj,
                        })
        
        # OLD FORMAT: Use negative_relations list directly
        for neg_rel in negative_relations:
            if isinstance(neg_rel, dict):
                n_subj = (neg_rel.get("subject") or "").strip()
                n_rel = (neg_rel.get("relation_type") or "").strip()
                n_obj = (neg_rel.get("object") or "").strip()
                if n_subj and n_rel and n_obj:
                    all_neg_relations.append(neg_rel)
        
        # Decide whether to use relation negatives based on probability
        if not all_neg_relations:
            return ""
        
        if random.random() >= self.negative_relation_sample_prob:
            return ""
        
        # Select a random negative relation
        negative_relation = random.choice(all_neg_relations)
        subject = negative_relation.get("subject", "")
        relation_type = negative_relation.get("relation_type", "")
        obj = negative_relation.get("object", "")
        
        if not subject or not relation_type or not obj:
            return ""
        
        # Build caption with optional positive components
        if positive_components:
            num_components = len(positive_components)
            if num_components == 1:
                caption = positive_components[0]
            else:
                num_to_sample = random.randint(1, num_components)
                selected_components = random.sample(positive_components, num_to_sample)
                caption = " and ".join(selected_components)
        else:
            caption = ""
        
        # Clean up relation type
        clean_relation = relation_type.replace("spatial_", "").replace("action_", "").replace("attribute_", "")
        clean_relation = clean_relation.replace("_", " ")
        
        if caption:
            return f"{caption} where {subject} {clean_relation} {obj}"
        else:
            return f"{subject} {clean_relation} {obj}"
    
    def _sample_inplace_replacement_negative(self, sample: Dict) -> str:
        """Sample negative by replacing component in original caption."""
        original_caption = sample.get("original_caption", sample.get("caption", ""))
        positive_components = sample.get("positive_components", [])
        negative_components = sample.get("negative_components", {})
        
        if not original_caption or not positive_components or not negative_components:
            return ""
        
        valid_components = [
            comp for comp in positive_components
            if comp in negative_components and negative_components[comp]
        ]
        
        if not valid_components:
            return ""
        
        random.shuffle(valid_components)
        
        # Try exact match first
        for component in valid_components:
            if component not in original_caption:
                continue
            
            negative_options = negative_components[component]
            if not isinstance(negative_options, list) or len(negative_options) == 0:
                continue
            
            negative_choice = random.choice(negative_options)
            negative_text = negative_choice.get("negative", "") if isinstance(negative_choice, dict) else str(negative_choice)
            
            if not negative_text:
                continue
            
            negative_caption = original_caption.replace(component, negative_text, 1)
            
            if negative_caption != original_caption:
                return negative_caption
        
        return ""


# =====================================================================
# Original Caption Negative Sampling
# =====================================================================

class OriginalCaptionNegativeSampler:
    """
    Sampler for generating negatives for the original (full) caption.
    
    This is a thin wrapper around LegacySampler.sample_component_negative()
    that provides control over swap_negative usage probability.
    
    Strategies (handled by LegacySampler):
    1. Swap negatives (noisy) - pre-generated word-shuffled negatives
    2. Inplace replacement - replace component with negative in original caption  
    3. Negative relation - use a negative relation triplet
    4. Component replacement - negative component + other positive components
    """
    
    def __init__(
        self,
        swap_negative_prob: float = 0.3,
        inplace_replacement_prob: float = 0.7,
        negative_relation_sample_prob: float = 0.5,
    ):
        """
        Initialize the original caption negative sampler.
        
        Args:
            swap_negative_prob: Probability of using swap_negatives (noisy word shuffle)
            inplace_replacement_prob: Probability of using inplace replacement vs relation
            negative_relation_sample_prob: Probability of using negative relation vs component
        """
        self.swap_negative_prob = swap_negative_prob
        
        # Create internal LegacySampler with the configured probabilities
        self._legacy_sampler = LegacySampler(
            inplace_replacement_prob=inplace_replacement_prob,
            negative_relation_sample_prob=negative_relation_sample_prob,
        )
    
    def sample_negative(self, sample: Dict) -> str:
        """
        Sample a negative for the original caption.
        
        Args:
            sample: Sample dict with all negative data
            
        Returns:
            Negative caption string, or empty string if all methods fail
        """
        # Decide whether to use swap_negatives based on probability
        use_swap = random.random() < self.swap_negative_prob
        
        # Delegate to LegacySampler which handles all the strategies
        return self._legacy_sampler.sample_component_negative(
            sample, 
            use_swap_negatives=use_swap
        )



from typing import List, Tuple, Dict, Optional
import random

def _clean_relation_type(relation_type: str) -> str:
    """Clean up relation type string for caption formatting."""
    rel = relation_type.strip()
    rel = rel.replace("_", " ")
    return rel


def _format_entity_caption(entity: str, extra_entities: Optional[List[str]] = None) -> str:
    """Format an entity caption with optional additional entities."""
    parts = [entity]
    if extra_entities:
        parts.extend(extra_entities)
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
    """
    
    def __init__(
        self,
        structured_relation_prob: float = 0.5,
        use_context_in_entity_pairs: bool = True,
    ):
        """
        Initialize the structured sampler.
        
        Args:
            structured_relation_prob: Probability of trying relation pairs first
            use_context_in_entity_pairs: If True, sometimes include additional entities
        """
        self.structured_relation_prob = structured_relation_prob
        self.use_context_in_entity_pairs = use_context_in_entity_pairs

    def _sample_structured_entity_pair(self, sample: Dict) -> Tuple[str, str, Dict]:
        """
        Sample a structured entity positive-negative pair.
        
        Returns:
            Tuple of (positive_caption, negative_caption, metadata)
            Returns ("", "", {}) if no valid pair can be created
        """
        positive_entities = sample.get("entities", []) or []
        negative_entities = sample.get("negative_entities", {}) or {}

        if not positive_entities or not negative_entities:
            return "", "", {}

        candidates = [
            entity for entity in positive_entities
            if entity in negative_entities and negative_entities[entity]
        ]
        if not candidates:
            return "", "", {}

        entity = random.choice(candidates)
        neg_options = negative_entities[entity]
        neg_choice = random.choice(neg_options)

        if isinstance(neg_choice, dict):
            neg_text = (neg_choice.get("negative") or "").strip()
            change_type = (neg_choice.get("change_type") or "").strip()
        else:
            neg_text = str(neg_choice).strip()
            change_type = ""

        if not neg_text:
            return "", "", {}

        other_entities = [c for c in positive_entities if c != entity]
        extra: List[str] = []
        if other_entities and self.use_context_in_entity_pairs:
            if random.random() < 0.5:
                extra = [random.choice(other_entities)]

        pos_caption = _format_entity_caption(entity, extra_entities=extra)
        neg_caption = _format_entity_caption(neg_text, extra_entities=extra)

        meta = {
            "pair_type": "entity",
            "entity": entity,
            "change_type": change_type,
        }
        return pos_caption, neg_caption, meta

    def _sample_structured_relation_pair(self, sample: Dict) -> Tuple[str, str, Dict]:
        """
        Sample a structured relation-based positive-negative pair.
        
        Strategies tried in order:
        1. Relation negatives embedded in the relation object
        2. Subject replacement using entity negatives
        3. Object replacement using entity negatives
        
        Returns:
            Tuple of (positive_caption, negative_caption, metadata)
            Returns ("", "", {}) if no valid pair can be created
        """
        relations = sample.get("relations", []) or []
        negative_entities = sample.get("negative_entities", {}) or {}

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

            if subj in negative_entities and negative_entities[subj]:
                neg_choice = random.choice(negative_entities[subj])
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

            if obj in negative_entities and negative_entities[obj]:
                neg_choice = random.choice(negative_entities[obj])
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
        
        Try relation pairs or entity pairs based on structured_relation_prob,
        with fallback to the other type if the first fails.
        
        Returns:
            Tuple of (positive_caption, negative_caption, metadata)
            Returns ("", "", {}) if no valid pair can be created
        """
        # Try relation first with probability structured_relation_prob
        if random.random() < self.structured_relation_prob:
            pos, neg, meta = self._sample_structured_relation_pair(sample)
            if pos and neg:
                return pos, neg, meta

            pos, neg, meta = self._sample_structured_entity_pair(sample)
            if pos and neg:
                return pos, neg, meta
        else:
            pos, neg, meta = self._sample_structured_entity_pair(sample)
            if pos and neg:
                return pos, neg, meta

            pos, neg, meta = self._sample_structured_relation_pair(sample)
            if pos and neg:
                return pos, neg, meta

        return "", "", {}

# =====================================================================
# Original Caption Negative Sampling
# =====================================================================

class OriginalCaptionNegativeSampler:
    """
    Sampler for generating negatives for the original (full) caption.
    
    Strategies:
    1. Swap negatives from post-processing
    2. In-place entity replacement in the original caption
    3. Relation negatives
    4. Entity-list replacement
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
            negative_relation_sample_prob: Probability of using relation negatives before entity-list replacement
        """
        self.swap_negative_prob = swap_negative_prob
        self.inplace_replacement_prob = inplace_replacement_prob
        self.negative_relation_sample_prob = negative_relation_sample_prob
    
    def sample_negative(self, sample: Dict) -> str:
        """
        Sample a negative for the original caption.
        
        Args:
            sample: Sample dict with all negative data
            
        Returns:
            Negative caption string, or empty string if all methods fail
        """
        use_swap = random.random() < self.swap_negative_prob
        if use_swap:
            swap_negatives = sample.get("swap_negatives", [])
            if swap_negatives and isinstance(swap_negatives, list):
                swap_choice = random.choice(swap_negatives)
                if isinstance(swap_choice, dict):
                    neg_text = (swap_choice.get("negative") or "").strip()
                    if neg_text:
                        return neg_text

        if self.inplace_replacement_prob > 0 and random.random() < self.inplace_replacement_prob:
            inplace_result = self._sample_inplace_replacement_negative(sample)
            if inplace_result:
                return inplace_result

        neg_relation_result = self._sample_negative_relation_caption(sample)
        if neg_relation_result:
            return neg_relation_result

        return self._sample_entity_replacement_negative(sample)

    def _sample_entity_replacement_negative(self, sample: Dict) -> str:
        positive_entities = sample.get("entities", [])
        negative_entities = sample.get("negative_entities", {})

        if not positive_entities or not negative_entities:
            return ""

        entities_with_negatives = [
            entity for entity in positive_entities
            if entity in negative_entities and negative_entities[entity]
        ]
        if not entities_with_negatives:
            return ""

        entity_to_replace = random.choice(entities_with_negatives)
        negative_options = negative_entities[entity_to_replace]
        negative_choice = random.choice(negative_options)

        if isinstance(negative_choice, dict):
            negative_text = (negative_choice.get("negative") or "").strip()
        else:
            negative_text = str(negative_choice).strip()

        if not negative_text:
            return ""

        other_entities = [entity for entity in positive_entities if entity != entity_to_replace]
        sampled_others = []
        if other_entities:
            num_to_sample = random.randint(0, len(other_entities))
            if num_to_sample > 0:
                sampled_others = random.sample(other_entities, num_to_sample)

        all_entities = sampled_others + [negative_text]
        random.shuffle(all_entities)
        return " and ".join(all_entities)

    def _sample_negative_relation_caption(self, sample: Dict) -> str:
        relations = sample.get("relations", []) or []
        positive_entities = sample.get("entities", [])

        all_neg_relations = []

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
                    swap_subj = neg.get("subject", neg.get("swap_subject", obj)).strip()
                    swap_obj = neg.get("object", neg.get("swap_object", subj)).strip()
                    if swap_subj and swap_obj:
                        all_neg_relations.append(
                            {"subject": swap_subj, "relation_type": rel_type, "object": swap_obj}
                        )
                elif change_type in ["antonym", "negation", "relation_change", "relation_opposite"]:
                    new_rel_type = (neg.get("relation_type") or "").strip()
                    if new_rel_type:
                        all_neg_relations.append(
                            {"subject": subj, "relation_type": new_rel_type, "object": obj}
                        )

        if not all_neg_relations or random.random() >= self.negative_relation_sample_prob:
            return ""

        negative_relation = random.choice(all_neg_relations)
        subject = (negative_relation.get("subject") or "").strip()
        relation_type = (negative_relation.get("relation_type") or "").strip()
        obj = (negative_relation.get("object") or "").strip()
        if not subject or not relation_type or not obj:
            return ""

        caption = ""
        if positive_entities:
            if len(positive_entities) == 1:
                caption = positive_entities[0]
            else:
                num_to_sample = random.randint(1, len(positive_entities))
                caption = " and ".join(random.sample(positive_entities, num_to_sample))

        clean_relation = relation_type.replace("spatial_", "").replace("action_", "").replace("attribute_", "")
        clean_relation = clean_relation.replace("_", " ")
        if caption:
            return f"{caption} where {subject} {clean_relation} {obj}"
        return f"{subject} {clean_relation} {obj}"

    def _sample_inplace_replacement_negative(self, sample: Dict) -> str:
        original_caption = sample.get("original_caption", sample.get("caption", ""))
        positive_entities = sample.get("entities", [])
        negative_entities = sample.get("negative_entities", {})

        if not original_caption or not positive_entities or not negative_entities:
            return ""

        valid_entities = [
            entity for entity in positive_entities
            if entity in negative_entities and negative_entities[entity]
        ]
        if not valid_entities:
            return ""

        random.shuffle(valid_entities)
        for entity in valid_entities:
            if entity not in original_caption:
                continue

            negative_options = negative_entities[entity]
            if not isinstance(negative_options, list) or not negative_options:
                continue

            negative_choice = random.choice(negative_options)
            negative_text = negative_choice.get("negative", "") if isinstance(negative_choice, dict) else str(negative_choice)
            negative_text = negative_text.strip()
            if not negative_text:
                continue

            negative_caption = original_caption.replace(entity, negative_text, 1)
            if negative_caption != original_caption:
                return negative_caption

        return ""

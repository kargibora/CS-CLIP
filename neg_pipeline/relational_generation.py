"""
Unified Component & Relational Generation
Extracts components, their relationships, and generates negative variants.

Key features:
1. Extract positive components from captions
2. Generate negative component variants (semantically different but similar category)
3. Extract relations between components (if they exist)
4. Generate negative relations (reversed, swapped, or modified)

Output: Single unified JSON with positive/negative components and relations
"""

from typing import List, Dict, Any
from tqdm import tqdm
import math
import json


    # SHARED system message - cached by vLLM across ALL batches for prefix caching
def extract_relational_components_batched(
    llm,
    captions: List[str],
    batch_size: int = 16
) -> List[Dict[str, Any]]:
    """Extract visual components and relations from captions."""
    
    # SHARED system message - cached by vLLM across ALL batches for prefix caching
    system_message = """Extract visual components and relations. Return ONLY valid JSON:
{"components": ["item1", "item2"], "relations": [{"subject": "item1", "relation_type": "is action", "object": "item2"}]}

═══════════════════════════════════════════════════════════
COMPONENT EXTRACTION RULES:
═══════════════════════════════════════════════════════════

1. Extract ONLY what's described - no hallucination
2. Keep attributes WITH objects: "red car" (not separate "red" and "car")
3. NO overlapping components - each distinct object once with full description
4. Include counting numbers: "three dogs", "five people"
5. Match input language exactly
6. Max 12 components, 12 relations

═══════════════════════════════════════════════════════════
RELATION EXTRACTION RULES - CRITICAL:
═══════════════════════════════════════════════════════════

1. GRAMMATICALLY COMPLETE: Relations must form valid sentences
   Format: "subject + relation_type + object" = complete sentence
   
   ✓ CORRECT: "is holding", "is sitting on", "is next to", "is wearing"
   ✗ WRONG: "holding", "sitting on", "next to", "wearing"
   
2. RELATION TYPES TO CAPTURE:
   - SPATIAL: "is on", "is under", "is next to", "is in front of", "is behind", "is inside", "is above", "is below"
   - ACTIONS: "is holding", "is eating", "is riding", "is playing with", "is looking at", "is touching"
   - POSSESSION: "has", "is wearing", "is carrying", "owns"
   - COMPARISON: "is larger than", "is smaller than", "is taller than", "is similar to"
   - EXISTENCE: "is with", "is accompanied by", "contains", "includes"
   - STATE: "is covered by", "is surrounded by", "is filled with", "is made of"

3. SUBJECT AND OBJECT must be from component list (exact match)

═══════════════════════════════════════════════════════════
EXAMPLES:
═══════════════════════════════════════════════════════════

"A man holding a red umbrella"
{"components": ["man", "red umbrella"], "relations": [{"subject": "man", "relation_type": "is holding", "object": "red umbrella"}]}

"Cat sitting on the wooden table"
{"components": ["cat", "wooden table"], "relations": [{"subject": "cat", "relation_type": "is sitting on", "object": "wooden table"}]}

"The dog is larger than the cat"
{"components": ["dog", "cat"], "relations": [{"subject": "dog", "relation_type": "is larger than", "object": "cat"}]}

"Woman with a child near the fountain"
{"components": ["woman", "child", "fountain"], "relations": [{"subject": "woman", "relation_type": "is with", "object": "child"}, {"subject": "woman", "relation_type": "is near", "object": "fountain"}]}

"Three red apples in a blue bowl"
{"components": ["three red apples", "blue bowl"], "relations": [{"subject": "three red apples", "relation_type": "are in", "object": "blue bowl"}]}

"Gato sobre la mesa"
{"components": ["gato", "mesa"], "relations": [{"subject": "gato", "relation_type": "está sobre", "object": "mesa"}]}

"Bird flying above the trees"
{"components": ["bird", "trees"], "relations": [{"subject": "bird", "relation_type": "is flying above", "object": "trees"}]}

"The coffee cup next to the laptop"
{"components": ["coffee cup", "laptop"], "relations": [{"subject": "coffee cup", "relation_type": "is next to", "object": "laptop"}]}"""""

    # Process in batches
    all_results = []
    num_batches = math.ceil(len(captions) / batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting components"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(captions))
        batch_captions = captions[start_idx:end_idx]
        
        # Escape braces in captions to prevent format() errors
        safe_captions = [cap.replace("{", "{{").replace("}", "}}") for cap in batch_captions]
        
        # MINIMAL user prompts - just the caption (cache-friendly)
        batch_prompts = [cap for cap in safe_captions]
        
        # Generate with deterministic settings for structured output
        batch_outputs = llm.generate(
            batch_prompts,
            max_new_tokens=800,  # Reduced - enforce conciseness
            temperature=0.3,  # Deterministic for structured output
            repetition_penalty=1.0,  # Neutral - don't distort JSON
            use_relational_scheme=True,
            system_message=system_message,  # Shared prefix - cached
            stop=["\n\n", "Input:", "Extract from:"]  # Stop sequences
        )
        
        # Process batch outputs
        for i, output in enumerate(batch_outputs):
            if isinstance(output, dict):
                components = output.get('components', [])
                relations = output.get('relations', [])
                
                # Minimal filtering
                valid_components = []
                for comp in components:
                    comp_str = str(comp).strip()
                    
                    # Only reject obvious placeholders
                    if (comp_str.lower() not in ['x', 'y', 'string', '...', '___'] and
                        len(comp_str) >= 2):
                        valid_components.append(comp_str)
                
                result = {
                    'original_caption': batch_captions[i],
                    'components': valid_components,
                    'relations': relations,
                }
            else:
                result = {
                    'original_caption': batch_captions[i],
                    'components': [],
                    'relations': []
                }
            all_results.append(result)
    
    total_components = sum(len(r['components']) for r in all_results)
    total_relations = sum(len(r['relations']) for r in all_results)
    captions_with_relations = sum(1 for r in all_results if r['relations'])
    
    print(f"Extracted components from {len(all_results)} captions:")
    print(f"  - {total_components} total components")
    print(f"  - {total_relations} total relations")
    print(f"  - {captions_with_relations} captions have relations")
    
    return all_results


def generate_component_negatives_batched(
    llm,
    components_list: List[List[str]],
    original_captions: List[str],
    batch_size: int = 16,
    n_neg_per_component: int = 2,
    n_binding_pairs: int = 2,  # NEW: Number of binding pairs per CAPTION (not per component)
) -> List[Dict[str, Any]]:
    """Generate hard negative component variants AND attribute binding pairs.
    
    Two types of negatives:
    1. Component negatives: Direct variations per component (e.g., "red car" → "blue car", "truck")
    2. Binding pairs: Attribute swaps between TWO components (e.g., "red car" + "blue sky" → "blue car" + "red sky")
    
    Args:
        llm: Language model wrapper
        components_list: List of component lists per caption
        original_captions: Original captions for context
        batch_size: Batch size for generation
        n_neg_per_component: Number of regular negatives per component
        n_binding_pairs: Number of binding pairs per CAPTION (tests attribute-object binding)
        
    Returns:
        List of dicts with 'negative_components' and 'binding_pairs'
    """
    
    # SHARED system message - SIMPLIFIED and focused on balanced negatives
    system_message = """Generate component negatives for vision-language training. Return ONLY valid JSON:
{"negative_variants": [{"negative": "variant", "change_type": "type"}]}

═══════════════════════════════════════════════════════════
GOAL: Create BALANCED negatives - clearly different but same category
═══════════════════════════════════════════════════════════

DIFFICULTY BALANCE:
- TOO EASY: "cat" → "airplane" (completely unrelated - trivial to distinguish)
- TOO HARD: "boat" → "canoe" (nearly identical - impossible to distinguish)  
- JUST RIGHT: "cat" → "dog" (same category, clearly different appearance)

SYNONYM CHECK - REJECT these pairs:
- bike/bicycle, car/automobile, couch/sofa, pants/trousers
- big/large, small/little, happy/joyful, fast/quick
- boat/ship (too similar!), cup/mug (too similar!)

═══════════════════════════════════════════════════════════
CHANGE TYPES (pick 2-3 diverse negatives):
═══════════════════════════════════════════════════════════

1. ATTRIBUTE_CHANGE - Change a visual property, keep object:
   Colors: "red car" → "blue car", "white dog" → "brown dog"
   Count: "three apples" → "five apples", "two birds" → "one bird"  
   Size: "large pizza" → "small pizza", "tall building" → "short building"
   State: "open door" → "closed door", "full glass" → "empty glass"
   Age/Gender: "young man" → "old man", "boy" → "girl"

2. OBJECT_CHANGE - Replace with DIFFERENT object from SAME category:
   Animals: "cat" → "dog", "horse" → "cow", "eagle" → "owl"
   Vehicles: "car" → "bus", "bicycle" → "motorcycle", "train" → "truck"
   Furniture: "chair" → "bench", "table" → "desk", "bed" → "couch"
   Places: "beach" → "mountain", "kitchen" → "bathroom", "park" → "street"
   Food: "apple" → "orange", "pizza" → "burger", "cake" → "pie"
   Clothing: "shirt" → "jacket", "hat" → "helmet", "shoes" → "boots"

═══════════════════════════════════════════════════════════
EXAMPLES:
═══════════════════════════════════════════════════════════

"red car"
{"negative_variants": [
  {"negative": "blue car", "change_type": "attribute_change"},
  {"negative": "red truck", "change_type": "object_change"}
]}

"three cats"
{"negative_variants": [
  {"negative": "one cat", "change_type": "attribute_change"},
  {"negative": "three dogs", "change_type": "object_change"}
]}

"woman running"
{"negative_variants": [
  {"negative": "man running", "change_type": "attribute_change"},
  {"negative": "woman walking", "change_type": "attribute_change"}
]}

"wooden table"
{"negative_variants": [
  {"negative": "metal table", "change_type": "attribute_change"},
  {"negative": "wooden chair", "change_type": "object_change"}
]}

"beach"
{"negative_variants": [
  {"negative": "forest", "change_type": "object_change"},
  {"negative": "mountain", "change_type": "object_change"}
]}

"sleeping dog"
{"negative_variants": [
  {"negative": "running dog", "change_type": "attribute_change"},
  {"negative": "sleeping cat", "change_type": "object_change"}
]}

═══════════════════════════════════════════════════════════
STRICT RULES:
═══════════════════════════════════════════════════════════

1. SAME CATEGORY: dog→cat ✓, dog→table ✗
2. NO SYNONYMS: bike→bicycle ✗, car→automobile ✗
3. NOT TOO SIMILAR: boat→canoe ✗, cup→mug ✗, sofa→couch ✗
4. VISUALLY DISTINCT: Must look clearly different in an image
5. KEEP LANGUAGE: Spanish input → Spanish output
6. 2-3 NEGATIVES: Mix of attribute_change and object_change
7. NO PLACEHOLDERS: No "variant1", "X", "example"

SKIP if component is abstract text ("Sign up", "Call now") → return empty []"""



    # Initialize results
    caption_results = {}
    for caption_idx in range(len(components_list)):
        caption_results[caption_idx] = {
            'original_caption': original_captions[caption_idx],
            'positive_components': components_list[caption_idx],
            'negative_components': {},
            'binding_pairs': []  # NEW: Store binding pairs at caption level (not per component)
        }
    
    prompts = []
    component_indices = []
    
    for caption_idx, components in enumerate(components_list):
        if not components:
            continue
        for comp in components:
            # Escape braces in component/caption to prevent format() errors
            safe_comp = str(comp).replace("{", "{{").replace("}", "}}")
            safe_caption = original_captions[caption_idx].replace("{", "{{").replace("}", "}}")
            
            # MINIMAL prompt - just component and context (cache-friendly)
            prompt = f"{safe_comp}\nContext: {safe_caption}"

            prompts.append(prompt)
            component_indices.append((caption_idx, comp))
    
    if not prompts:
        return [caption_results[i] for i in sorted(caption_results.keys())]
    
    print(f"Generating hard negatives for {len(prompts)} components...")
    
    # Generate with HIGHER temperature for creativity
    all_outputs = []
    num_batches = math.ceil(len(prompts) / batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc="Component negatives"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        
        batch_outputs = llm.generate(
            batch_prompts,
            max_new_tokens=400,  # Reduced - enforce conciseness (max 3 variants)
            temperature=0.3,  # Low but non-zero for diversity while maintaining quality
            use_component_negative_scheme=True,
            repetition_penalty=1.0,  # Neutral - don't distort JSON
            system_message=system_message,  # Shared prefix - cached
            stop=["\n\n", "Component:", "Context:"]  # Stop sequences
        )
        all_outputs.extend(batch_outputs)
    
    # Process outputs
    successful_negs = 0
    empty_negs = 0
    rejected_low_quality = 0
    rejected_synonyms = 0  # NEW: Track synonym rejections separately
    rejected_component_copies = 0
    rejected_duplicates = 0
    binding_pairs_by_caption = {}  # Collect binding pairs by caption
    
    # Track used negatives per caption to prevent duplicates
    used_negatives_per_caption = {}
    
    for i, output in enumerate(all_outputs):
        caption_idx, original_comp = component_indices[i]
        
        # Initialize set of used negatives for this caption
        if caption_idx not in used_negatives_per_caption:
            used_negatives_per_caption[caption_idx] = set()
        
        # Get all components from this caption to check for copying
        all_components_in_caption = components_list[caption_idx]
        
        valid_variants = []
        
        if isinstance(output, dict):
            # Process regular negative variants
            negative_variants = output.get('negative_variants', [])
            
            for var in negative_variants:
                neg = var.get('negative', '').strip()
                change_type = var.get('change_type', '').strip()
                
                if neg and isinstance(neg, str):
                    neg_lower = neg.lower()
                    orig_lower = original_comp.lower()
                    
                    # Reject obvious placeholders
                    bad_patterns = ['x', 'y', 'z', 'variant', 'alternative', 'example', 
                                   'string', 'object', '...', '___', 'false', 'placeholder',
                                   'true', 'none', 'null', 'variant1', 'variant2', 'variente1', 'variente2']
                    
                    # Check for minimal numeric changes (iPhone 5 -> iPhone 4)
                    def has_only_number_change(orig, neg):
                        """Check if only numbers differ between strings."""
                        import re
                        # Remove all digits
                        orig_no_nums = re.sub(r'\d+', '', orig.lower())
                        neg_no_nums = re.sub(r'\d+', '', neg.lower())
                        # If text is identical after removing numbers, it's just a number change
                        return orig_no_nums == neg_no_nums and orig_no_nums != ''
                    
                    # Check for synonyms and too-similar pairs
                    def is_synonym_or_too_similar(orig, neg):
                        """Reject synonyms and nearly-identical pairs."""
                        orig_l = orig.lower().strip()
                        neg_l = neg.lower().strip()
                        
                        # Known synonym pairs (bidirectional)
                        synonym_pairs = [
                            # Vehicles
                            ('car', 'automobile'), ('car', 'auto'), ('bike', 'bicycle'),
                            ('boat', 'ship'), ('boat', 'canoe'), ('boat', 'kayak'),
                            ('plane', 'airplane'), ('plane', 'aircraft'),
                            # Furniture
                            ('couch', 'sofa'), ('sofa', 'couch'), ('cup', 'mug'),
                            ('curtain', 'drape'), ('rug', 'carpet'),
                            # Clothing
                            ('pants', 'trousers'), ('shirt', 'top'), ('sweater', 'jumper'),
                            ('sneakers', 'trainers'), ('cap', 'hat'),
                            # Animals (too similar)
                            ('puppy', 'dog'), ('kitten', 'cat'), ('pup', 'dog'),
                            # People
                            ('guy', 'man'), ('lady', 'woman'), ('kid', 'child'),
                            ('infant', 'baby'), ('teen', 'teenager'),
                            # Adjectives
                            ('big', 'large'), ('small', 'little'), ('tiny', 'small'),
                            ('happy', 'joyful'), ('sad', 'unhappy'), ('angry', 'mad'),
                            ('fast', 'quick'), ('slow', 'sluggish'),
                            # Common synonyms
                            ('rock', 'stone'), ('woods', 'forest'), ('road', 'street'),
                            ('store', 'shop'), ('movie', 'film'),
                        ]
                        
                        # Check if pair is in synonym list
                        for s1, s2 in synonym_pairs:
                            if (s1 in orig_l and s2 in neg_l) or (s2 in orig_l and s1 in neg_l):
                                return True
                        
                        # Check for plural/singular only changes
                        import re
                        orig_singular = re.sub(r's$', '', orig_l)
                        neg_singular = re.sub(r's$', '', neg_l)
                        if orig_singular == neg_singular and orig_l != neg_l:
                            return True
                        
                        # Check for -ing/-ed variations of same verb
                        orig_stem = re.sub(r'(ing|ed)$', '', orig_l)
                        neg_stem = re.sub(r'(ing|ed)$', '', neg_l)
                        if len(orig_stem) > 3 and orig_stem == neg_stem and orig_l != neg_l:
                            return True
                        
                        # Check high word overlap (>80% same words = too similar)
                        orig_words = set(orig_l.split())
                        neg_words = set(neg_l.split())
                        if orig_words and neg_words:
                            overlap = len(orig_words & neg_words) / max(len(orig_words), len(neg_words))
                            if overlap > 0.8 and len(orig_words) > 1:
                                return True
                        
                        return False
                    
                    # Check if negative is just copying another component from the same caption
                    def is_component_copy(neg_str, components):
                        """Check if negative is identical to another component in the caption."""
                        neg_normalized = neg_str.lower().strip()
                        for comp in components:
                            comp_normalized = comp.lower().strip()
                            # Exact match or substring match (catches "Michael Raymond-James" in longer text)
                            if neg_normalized == comp_normalized or neg_normalized in comp_normalized or comp_normalized in neg_normalized:
                                return True
                        return False
                    
                    # Check if negative contains substantial parts of another component
                    def contains_other_component(neg_str, original, components):
                        """Check if negative contains significant parts of other components."""
                        neg_words = set(neg_str.lower().split())
                        
                        for comp in components:
                            if comp == original:
                                continue
                            comp_words = set(comp.lower().split())
                            # If negative shares >50% words with another component, likely a copy
                            if comp_words and len(neg_words & comp_words) / len(comp_words) > 0.5:
                                return True
                        return False
                    
                    # Check for generic/vague terms
                    generic_terms = ['item', 'thing', 'stuff', 'vehicle', 'device', 
                                   'product', 'something', 'anything']
                    
                    # CRITICAL: Check if negative is copying another component
                    if is_component_copy(neg, all_components_in_caption):
                        rejected_component_copies += 1
                        continue
                    
                    # Check if negative contains substantial parts of other components
                    if contains_other_component(neg, original_comp, all_components_in_caption):
                        rejected_component_copies += 1
                        continue
                    
                    # Check if this negative was already used for another component in same caption
                    neg_normalized = neg.lower().strip()
                    if neg_normalized in used_negatives_per_caption[caption_idx]:
                        rejected_duplicates += 1
                        continue
                    
                    # NEW: Check for synonyms and too-similar pairs
                    if is_synonym_or_too_similar(original_comp, neg):
                        rejected_synonyms += 1
                        continue
                    
                    is_valid = (
                        neg_lower not in bad_patterns and
                        neg_lower not in generic_terms and
                        len(neg) > 1 and 
                        neg_lower != orig_lower and
                        not has_only_number_change(original_comp, neg)
                    )
                    
                    if is_valid:
                        # Normalize change type
                        if any(w in change_type.lower() for w in ['entity', 'person', 'role']):
                            norm_type = 'entity_change'
                        elif any(w in change_type.lower() for w in ['color', 'size', 'material', 'shape', 'attribute']):
                            norm_type = 'attribute_change'
                        else:
                            norm_type = 'object_change'
                        
                        valid_variants.append({
                            'negative': neg,
                            'change_type': norm_type
                        })
                        
                        # Mark this negative as used for this caption
                        used_negatives_per_caption[caption_idx].add(neg_normalized)
                    else:
                        rejected_low_quality += 1
            
            # NEW: Collect binding pairs (per caption, not per component)
            # Store them temporarily - we'll deduplicate later
            binding_pairs = output.get('binding_pairs', [])
            
            if binding_pairs and caption_idx not in binding_pairs_by_caption:
                binding_pairs_by_caption[caption_idx] = []
            
            for pair in binding_pairs:
                comp_1 = pair.get('component_1', '').strip()
                comp_2 = pair.get('component_2', '').strip()
                orig_1 = pair.get('original_1', '').strip()
                orig_2 = pair.get('original_2', '').strip()
                attr_1 = pair.get('attribute_1', '').strip()
                attr_2 = pair.get('attribute_2', '').strip()
                
                # Validate binding pair
                if all([comp_1, comp_2, orig_1, orig_2, attr_1, attr_2]):
                    # Check that originals are different
                    if orig_1.lower() != orig_2.lower():
                        binding_pairs_by_caption[caption_idx].append({
                            'component_1': comp_1,
                            'component_2': comp_2,
                            'original_1': orig_1,
                            'original_2': orig_2,
                            'attribute_1': attr_1,
                            'attribute_2': attr_2
                        })
        
        # Add to results
        caption_results[caption_idx]['negative_components'][original_comp] = valid_variants
        
        if valid_variants:
            successful_negs += 1
        else:
            empty_negs += 1
    
    # Add binding pairs to results (deduplicated per caption)
    total_binding_pairs = 0
    for caption_idx, pairs in binding_pairs_by_caption.items():
        # Deduplicate pairs (same components in any order)
        unique_pairs = []
        seen = set()
        
        for pair in pairs:
            # Create normalized key for deduplication
            key = tuple(sorted([
                (pair['original_1'], pair['component_1']),
                (pair['original_2'], pair['component_2'])
            ]))
            
            if key not in seen:
                seen.add(key)
                unique_pairs.append(pair)
        
        caption_results[caption_idx]['binding_pairs'] = unique_pairs
        total_binding_pairs += len(unique_pairs)
    
    captions_with_bindings = len(binding_pairs_by_caption)
    
    print(f"Component negatives: {successful_negs} with alternatives, {empty_negs} empty")
    print(f"Rejected: {rejected_low_quality} low quality, {rejected_synonyms} synonyms/too-similar, "
          f"{rejected_component_copies} component copies, {rejected_duplicates} duplicates")
    print(f"Binding pairs: {total_binding_pairs} pairs across {captions_with_bindings} captions")
    
    results = [caption_results[i] for i in sorted(caption_results.keys())]
    
    total_variants = sum(
        len(neg_list) 
        for r in results 
        for neg_list in r['negative_components'].values()
    )
    
    print(f"✓ Generated {total_variants} component negative variants and {total_binding_pairs} binding pairs")
    return results


def generate_relational_negatives_batched(
    llm,
    relational_data: List[Dict[str, Any]],
    batch_size: int = 16,
    n_negatives: int = 3,
) -> List[Dict[str, Any]]:
    """
    Generate negative relations and store them within each relation object.
    
    Output structure for each relation:
    {
        "subject": "cat",
        "relation_type": "is sitting on",
        "object": "table",
        "negative_relations": [
            {"relation_type": "is under", "change_type": "antonym"},
            {"relation_type": "is not sitting on", "change_type": "negation"},
            {"subject": "table", "relation_type": "is sitting on", "object": "cat", "change_type": "swap"}
        ]
    }
    """
    
    # SHARED system message - focused on per-relation negatives
    system_message = """Generate negative relations for each input relation. Return ONLY valid JSON:
{"relation_negatives": [{"original_relation": "is holding", "negatives": [{"relation_type": "is not holding", "change_type": "negation"}, {"relation_type": "is dropping", "change_type": "antonym"}]}]}

═══════════════════════════════════════════════════════════
GOAL: For EACH relation, generate 2-3 negative variants
═══════════════════════════════════════════════════════════

NEGATIVE TYPES (in order of preference):

1. **ANTONYM** (PREFERRED) - Use semantic opposite:
   SPATIAL:
   - "is on" → "is under"
   - "is above" → "is below"
   - "is in front of" → "is behind"
   - "is inside" → "is outside"
   - "is near" → "is far from"
   - "is next to" → "is away from"
   - "is left of" → "is right of"
   
   ACTIONS:
   - "is holding" → "is dropping"
   - "is sitting on" → "is standing on"
   - "is walking towards" → "is walking away from"
   - "is pushing" → "is pulling"
   - "is opening" → "is closing"
   - "is eating" → "is not eating" (no clear antonym)
   
   STATES:
   - "is with" → "is without"
   - "is attached to" → "is detached from"
   - "is covered by" → "is exposed"
   - "is filled with" → "is empty of"
   
   COMPARISON:
   - "is larger than" → "is smaller than"
   - "is taller than" → "is shorter than"
   - "is older than" → "is younger than"

2. **NEGATION** - Add "not" (use when no clear antonym):
   - "is holding" → "is not holding"
   - "is wearing" → "is not wearing"
   - "is looking at" → "is not looking at"
   - "has" → "does not have"

3. **SWAP** - Reverse subject/object (only if creates different scene):
   - Original: subject="cat", relation="is on", object="mat"
   - Swap: subject="mat", relation="is on", object="cat"
   - Only include if visually different!

═══════════════════════════════════════════════════════════
OUTPUT FORMAT:
═══════════════════════════════════════════════════════════

For each original relation, output its negatives:

{"relation_negatives": [
  {
    "original_relation": "is sitting on",
    "original_subject": "cat",
    "original_object": "mat",
    "negatives": [
      {"relation_type": "is standing on", "change_type": "antonym"},
      {"relation_type": "is under", "change_type": "antonym"},
      {"relation_type": "is not sitting on", "change_type": "negation"},
      {"swap_subject": "mat", "swap_object": "cat", "change_type": "swap"}
    ]
  }
]}

═══════════════════════════════════════════════════════════
RULES:
═══════════════════════════════════════════════════════════

1. GRAMMATICALLY COMPLETE: Keep "is/are" prefix
   ✓ "is sitting on" → "is standing on"
   ✗ "sitting on" → "standing on"

2. PREFER ANTONYMS: Only use negation if no clear antonym exists

3. NO SYNONYMS: 
   ✗ "is near" → "is close to" (same meaning!)
   ✓ "is near" → "is far from" (opposite)

4. MATCH SUBJECT/OBJECT: Use exact component names from input

5. VISUALLY DISTINCT: Each negative must create different visual scene

═══════════════════════════════════════════════════════════
EXAMPLES:
═══════════════════════════════════════════════════════════

Input: "cat is sitting on mat" | Components: [cat, mat]
{"relation_negatives": [{"original_relation": "is sitting on", "original_subject": "cat", "original_object": "mat", "negatives": [{"relation_type": "is under", "change_type": "antonym"}, {"relation_type": "is standing on", "change_type": "antonym"}, {"relation_type": "is not sitting on", "change_type": "negation"}, {"swap_subject": "mat", "swap_object": "cat", "change_type": "swap"}]}]}

Input: "man is holding umbrella" | Components: [man, umbrella]
{"relation_negatives": [{"original_relation": "is holding", "original_subject": "man", "original_object": "umbrella", "negatives": [{"relation_type": "is dropping", "change_type": "antonym"}, {"relation_type": "is not holding", "change_type": "negation"}]}]}

Input: "dog is with owner" | Components: [dog, owner]
{"relation_negatives": [{"original_relation": "is with", "original_subject": "dog", "original_object": "owner", "negatives": [{"relation_type": "is without", "change_type": "antonym"}, {"relation_type": "is away from", "change_type": "antonym"}, {"relation_type": "is not with", "change_type": "negation"}]}]}

Input: "book is on table" | Components: [book, table]
{"relation_negatives": [{"original_relation": "is on", "original_subject": "book", "original_object": "table", "negatives": [{"relation_type": "is under", "change_type": "antonym"}, {"relation_type": "is beside", "change_type": "antonym"}, {"relation_type": "is not on", "change_type": "negation"}, {"swap_subject": "table", "swap_object": "book", "change_type": "swap"}]}]}

Input: "car is larger than bicycle" | Components: [car, bicycle]
{"relation_negatives": [{"original_relation": "is larger than", "original_subject": "car", "original_object": "bicycle", "negatives": [{"relation_type": "is smaller than", "change_type": "antonym"}, {"relation_type": "is not larger than", "change_type": "negation"}, {"swap_subject": "bicycle", "swap_object": "car", "change_type": "swap"}]}]}

Input: "gato está sobre la mesa" | Components: [gato, mesa]
{"relation_negatives": [{"original_relation": "está sobre", "original_subject": "gato", "original_object": "mesa", "negatives": [{"relation_type": "está debajo de", "change_type": "antonym"}, {"relation_type": "no está sobre", "change_type": "negation"}, {"swap_subject": "mesa", "swap_object": "gato", "change_type": "swap"}]}]}"""

    # Build prompts - one per caption with all its relations
    prompts = []
    prompt_metadata = []  # Store (caption_idx, relations) for each prompt
    
    for idx, data in enumerate(relational_data):
        relations = data.get('relations', [])
        if not relations:
            continue
        
        original_caption = data.get('original_caption', '')
        components = data.get('components', [])
        
        # Escape braces to prevent format() errors
        safe_caption = original_caption.replace("{", "{{").replace("}", "}}")
        
        # Format components list
        components_str = ", ".join(components[:12])
        
        # Format each relation as "subject relation_type object"
        relations_formatted = []
        for r in relations[:8]:
            subj = r.get('subject', '')
            rel = r.get('relation_type', '')
            obj = r.get('object', '')
            if subj and rel and obj:
                relations_formatted.append(f"{subj} {rel} {obj}")
        
        if not relations_formatted:
            continue
        
        relations_str = " | ".join(relations_formatted)
        
        # Prompt format
        prompt = f"{safe_caption}\nComponents: [{components_str}]\nRelations: {relations_str}"
        
        prompts.append(prompt)
        prompt_metadata.append((idx, relations[:8], components))
    
    if not prompts:
        print("⚠ No relations found for negative generation")
        return relational_data  # Return original data unchanged
    
    print(f"✓ Generating relational negatives for {len(prompts)} captions...")
    
    all_outputs = []
    num_batches = math.ceil(len(prompts) / batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc="Relational negatives"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        
        try:
            batch_outputs = llm.generate(
                batch_prompts,
                max_new_tokens=800,  # Enough for complete triplet JSON
                temperature=0.3,  # Deterministic for structured output
                use_relational_negative_scheme=True,
                repetition_penalty=1.0,  # Neutral - don't distort JSON
                system_message=system_message,  # Shared prefix - cached
                stop=["\n\n", "Caption:", "Components:"],  # Stop sequences
            )
            all_outputs.extend(batch_outputs)
        except Exception as e:
            print(f"\n⚠️ Error in batch {batch_idx} (prompts {start_idx}-{end_idx}): {e}")
            print(f"   Skipping {len(batch_prompts)} prompts in this batch")
            # Add empty results for failed batch
            for _ in batch_prompts:
                all_outputs.append({"relation_negatives": []})
            continue
    
    # Process outputs - store negatives within each relation object
    results = []
    total_negatives = 0
    
    for i, output in enumerate(all_outputs):
        original_idx, original_relations, original_components = prompt_metadata[i]
        original_data = relational_data[original_idx]
        
        # Create a copy of relations with negatives added to each
        updated_relations = []
        
        if isinstance(output, dict):
            rel_negs_list = output.get('relation_negatives', [])
            
            # Build a mapping from original relation to its negatives
            relation_to_negatives = {}
            for rel_neg in rel_negs_list:
                orig_rel = rel_neg.get('original_relation', '').strip()
                orig_subj = rel_neg.get('original_subject', '').strip()
                orig_obj = rel_neg.get('original_object', '').strip()
                negatives = rel_neg.get('negatives', [])
                
                if orig_rel and orig_subj and orig_obj:
                    key = (orig_subj, orig_rel, orig_obj)
                    relation_to_negatives[key] = negatives
            
            # Update each relation with its negatives
            for rel in original_relations:
                subj = rel.get('subject', '').strip()
                rel_type = rel.get('relation_type', '').strip()
                obj = rel.get('object', '').strip()
                
                key = (subj, rel_type, obj)
                negatives = relation_to_negatives.get(key, [])
                
                # Process and validate negatives
                valid_negatives = []
                for neg in negatives[:5]:  # Allow up to 5 negatives per relation
                    change_type = neg.get('change_type', '').lower()
                    
                    if change_type == 'swap':
                        # Subject-object swap
                        swap_subj = neg.get('swap_subject', '').strip()
                        swap_obj = neg.get('swap_object', '').strip()
                        
                        # Validate swap components exist in original
                        if swap_subj in original_components and swap_obj in original_components:
                            valid_negatives.append({
                                'subject': swap_subj,
                                'relation_type': rel_type,  # Keep same relation
                                'object': swap_obj,
                                'change_type': 'subject_object_swap'
                            })
                    else:
                        # Relation type change (antonym or negation)
                        new_rel_type = neg.get('relation_type', '').strip()
                        if new_rel_type and new_rel_type != rel_type:
                            valid_negatives.append({
                                'subject': subj,
                                'relation_type': new_rel_type,
                                'object': obj,
                                'change_type': change_type if change_type in ['antonym', 'negation'] else 'antonym'
                            })
                
                # Add updated relation with negatives inside
                updated_rel = dict(rel)
                updated_rel['negatives'] = valid_negatives
                updated_relations.append(updated_rel)
                total_negatives += len(valid_negatives)
        else:
            # No valid output - keep relations without negatives
            for rel in original_relations:
                updated_rel = dict(rel)
                updated_rel['negatives'] = []
                updated_relations.append(updated_rel)
        
        results.append({
            'original_caption': original_data['original_caption'],
            'components': original_data['components'],
            'relations': updated_relations,
            'original_index': original_idx
        })
    
    print(f"✓ Generated {total_negatives} relational negatives")
    return results
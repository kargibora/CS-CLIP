"""
Component-based negative generation.

This module generates negatives by:
1. Taking positive components extracted from captions
2. Generating negative variants of each component
3. Randomly mixing positive and negative components to create negative captions
"""

from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import random
import copy


def generate_component_negatives_batched(
    llm,
    components_list: List[List[str]],
    original_captions: List[str],
    batch_size: int = 16,
    n_neg_per_component: int = 2,
    hardness: str = "hard"
) -> List[Dict[str, Any]]:
    """
    Generate negative variants for each component.
    
    Args:
        llm: VLLMWrapper instance
        components_list: List of component lists, one per caption
        original_captions: Original captions for context
        batch_size: Batch size for LLM generation
        n_neg_per_component: Number of negative variants per component
        hardness: Difficulty level (easy/medium/hard)
        
    Returns:
        List of dicts with positive components and their negative variants
    """
    
    few_shot_examples = """
# Example 1: Person with attribute
Input:
Original Caption: "Close-up of man with wristwatch holding gift box"
Component: "man with wristwatch"

Output:
{
    "negative_variants": [
        {"negative": "woman with wristwatch", "change_type": "entity", "changed_element": "man→woman"},
        {"negative": "man with bracelet", "change_type": "attribute", "changed_element": "wristwatch→bracelet"}
    ]
}

# Example 2: Object with material/attribute
Input:
Original Caption: "Red wooden table in a room"
Component: "red wooden table"

Output:
{
    "negative_variants": [
        {"negative": "blue wooden table", "change_type": "color", "changed_element": "red→blue"},
        {"negative": "red metal table", "change_type": "material", "changed_element": "wooden→metal"}
    ]
}

# Example 3: Object only
Input:
Original Caption: "A cat and a dog playing"
Component: "gift box"

Output:
{
    "negative_variants": [
        {"negative": "plastic box", "change_type": "attribute", "changed_element": "gift→plastic"},
        {"negative": "wrapped package", "change_type": "object", "changed_element": "box→package"}
    ]
}

# Example 4: Action/pose
Input:
Original Caption: "Woman sitting on chair"
Component: "woman sitting"

Output:
{
    "negative_variants": [
        {"negative": "woman standing", "change_type": "action", "changed_element": "sitting→standing"},
        {"negative": "man sitting", "change_type": "entity", "changed_element": "woman→man"}
    ]
}

# Example 5: Spatial relation
Input:
Original Caption: "Book on top of table"
Component: "book on top"

Output:
{
    "negative_variants": [
        {"negative": "book underneath", "change_type": "spatial", "changed_element": "on top→underneath"},
        {"negative": "laptop on top", "change_type": "object", "changed_element": "book→laptop"}
    ]
}

# Example 6: Multiple attributes
Input:
Original Caption: "Large brown dog running fast"
Component: "large brown dog"

Output:
{
    "negative_variants": [
        {"negative": "small brown dog", "change_type": "size", "changed_element": "large→small"},
        {"negative": "large white dog", "change_type": "color", "changed_element": "brown→white"}
    ]
}

# Example 7: Scene/location
Input:
Original Caption: "People walking in sunny park"
Component: "sunny park"

Output:
{
    "negative_variants": [
        {"negative": "rainy park", "change_type": "weather", "changed_element": "sunny→rainy"},
        {"negative": "sunny beach", "change_type": "location", "changed_element": "park→beach"}
    ]
}
"""

    if hardness == "easy":
        hardness_prompt = (
            "For EASY negatives, make dramatic, highly visible changes:\n"
            "- Change to completely different categories (man→robot, dog→elephant)\n"
            "- Use extreme contrasts (red→purple, large→microscopic)\n"
            "- Pick very different materials/attributes (wooden→crystal, sunny→stormy)\n"
            "The changes should be immediately obvious and visually striking."
        )
    elif hardness == "hard":
        hardness_prompt = (
            "For HARD negatives, make subtle but meaningful changes:\n"
            "- Change within same category (man→woman, dog→cat, red→orange)\n"
            "- Use plausible alternatives (wooden→metal, sitting→standing)\n"
            "- Keep similar visual appearance but different semantics\n"
            "The changes should be noticeable but require attention to distinguish."
        )
    else:  # medium
        hardness_prompt = (
            "For MEDIUM negatives, make clear but related changes:\n"
            "- Change to related categories (man→boy, dog→wolf, red→yellow)\n"
            "- Use noticeably different but plausible alternatives\n"
            "- Make changes that are clear but not extreme\n"
            "The changes should be obvious without being jarring."
        )

    system_prompt = f"""
You are an expert at generating visually distinct negative variants of image caption components.

**TASK**: Given a component from an image caption, generate negative variants by changing ONE element at a time.

**RULES**:
1. Change ONLY ONE element (entity, attribute, action, material, color, size, etc.)
2. Keep the rest of the component UNCHANGED
3. All changes must be VISUALLY OBSERVABLE in an image
4. Generate exactly {n_neg_per_component} diverse negative variants
5. Each variant should change a DIFFERENT element/aspect
6. Specify what was changed (e.g., "man→woman", "red→blue")

**CHANGE TYPES**:
- entity: Change the main object/person (man→woman, cat→dog)
- attribute: Change describing word (gift→plastic, large→small)
- color: Change color (red→blue, white→black)
- material: Change material (wooden→metal, glass→plastic)
- size: Change size (large→small, tiny→huge)
- action: Change action/pose (sitting→standing, running→walking)
- spatial: Change spatial relation (on top→underneath, left→right)
- location: Change location/scene (park→beach, room→garden)
- number: Change quantity (one→three, many→few)
- weather: Change weather/lighting (sunny→rainy, bright→dark)
- object: Change related object while keeping attributes

**HARDNESS LEVEL**: {hardness}
{hardness_prompt}

**AVOID**:
- Abstract or non-visual changes
- Near-synonyms (photo→picture, car→vehicle)
- Changes that don't make semantic sense
- Removing elements entirely

**OUTPUT FORMAT** (valid JSON):
{{
    "negative_variants": [
        {{
            "negative": "modified component text",
            "change_type": "type of change",
            "changed_element": "original→new"
        }},
        ...
    ]
}}
"""

    # Flatten all components with metadata
    prompts = []
    meta = []
    
    for cap_idx, (components, original_caption) in enumerate(zip(components_list, original_captions)):
        for comp in components:
            user_prompt = (
                few_shot_examples.strip() + "\n\n"
                f'Original Caption: "{original_caption}"\n'
                f'Component: "{comp}"\n\n'
                f'Generate {n_neg_per_component} negative variants of this component:\n'
                f'Output:'
            )
            
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            prompt = llm.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            meta.append({
                "caption_idx": cap_idx,
                "component": comp,
                "original_caption": original_caption
            })
    
    print(f"Component negative generation: {len(prompts)} components, batch_size={batch_size}")
    
    # Generate in batches
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating component negatives"):
        batch_prompts = prompts[i:i + batch_size]
        batch_outputs = llm.generate(
            batch_prompts,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            batch_size=batch_size,
            num_return_sequences=1,
            use_component_negative_scheme=True
        )
        outputs.extend(batch_outputs)
    
    # Organize results by caption
    results = [{"components": [], "negative_variants": {}} for _ in range(len(components_list))]
    
    for i, output in enumerate(outputs):
        m = meta[i]
        cap_idx = m["caption_idx"]
        component = m["component"]
        
        # Add component if not already there
        if component not in results[cap_idx]["components"]:
            results[cap_idx]["components"].append(component)
        
        # Add negative variants
        negative_variants = output.get("negative_variants", [])
        results[cap_idx]["negative_variants"][component] = negative_variants
    
    return results


def generate_mixed_negatives(
    positive_components: List[str],
    negative_variants: Dict[str, List[Dict[str, str]]],
    n_negatives: int = 3,
    strategy: str = "random"
) -> List[Dict[str, Any]]:
    """
    Generate negative captions by mixing positive and negative components.
    
    Args:
        positive_components: List of positive components
        negative_variants: Dict mapping components to their negative variants
        n_negatives: Number of negative captions to generate
        strategy: Mixing strategy (random, single, multiple)
        
    Returns:
        List of negative caption dictionaries
    """
    negatives = []
    
    if strategy == "random" or strategy == "mixed":
        # Randomly replace components with their negatives
        for _ in range(n_negatives):
            # Decide how many components to change (1 to all)
            num_to_change = random.randint(1, len(positive_components))
            components_to_change = random.sample(range(len(positive_components)), num_to_change)
            
            mixed_components = []
            changed_info = []
            
            for idx, comp in enumerate(positive_components):
                if idx in components_to_change and comp in negative_variants and negative_variants[comp]:
                    # Use negative variant
                    neg_variant = random.choice(negative_variants[comp])
                    mixed_components.append(neg_variant["negative"])
                    changed_info.append({
                        "original": comp,
                        "negative": neg_variant["negative"],
                        "change_type": neg_variant["change_type"],
                        "changed_element": neg_variant["changed_element"]
                    })
                else:
                    # Keep positive
                    mixed_components.append(comp)
            
            if mixed_components and changed_info:  # Only add if we made changes
                negatives.append({
                    "negative_caption": " and ".join(mixed_components),
                    "components": mixed_components,
                    "changes": changed_info,
                    "num_changes": len(changed_info),
                    "strategy": "mixed"
                })
    
    elif strategy == "single":
        # Replace only one component at a time
        for comp in positive_components:
            if comp in negative_variants and negative_variants[comp]:
                for neg_variant in negative_variants[comp][:n_negatives]:
                    mixed_components = positive_components.copy()
                    idx = positive_components.index(comp)
                    mixed_components[idx] = neg_variant["negative"]
                    
                    negatives.append({
                        "negative_caption": " and ".join(mixed_components),
                        "components": mixed_components,
                        "changes": [{
                            "original": comp,
                            "negative": neg_variant["negative"],
                            "change_type": neg_variant["change_type"],
                            "changed_element": neg_variant["changed_element"]
                        }],
                        "num_changes": 1,
                        "strategy": "single"
                    })
    
    elif strategy == "all":
        # Replace all components
        all_have_negatives = all(comp in negative_variants and negative_variants[comp] for comp in positive_components)
        
        if all_have_negatives:
            for _ in range(n_negatives):
                mixed_components = []
                changed_info = []
                
                for comp in positive_components:
                    neg_variant = random.choice(negative_variants[comp])
                    mixed_components.append(neg_variant["negative"])
                    changed_info.append({
                        "original": comp,
                        "negative": neg_variant["negative"],
                        "change_type": neg_variant["change_type"],
                        "changed_element": neg_variant["changed_element"]
                    })
                
                negatives.append({
                    "negative_caption": " and ".join(mixed_components),
                    "components": mixed_components,
                    "changes": changed_info,
                    "num_changes": len(changed_info),
                    "strategy": "all"
                })
    
    return negatives

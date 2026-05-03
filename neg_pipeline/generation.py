# generation.py

from typing import List, Dict, Optional, Tuple, Any
import re
import random
from datasets import Dataset
import copy

# Use LLMWrapper for all LLM access
from .llm_utils import VLLMWrapper
from tqdm import tqdm



from typing import List
import re

def get_fewshot_examples(hardness: str) -> str:
    if hardness == "easy":
        return """
        # Example 1: Color change - EASY (dramatic color change)
        Input:
        Caption: "A red apple on a table."
        Attribute: red
        Object: apple

        Output:
        {
            "attribute_negative": "purple",
            "object_negative": "basketball",
            "swap_negative": null,
            "attribute_concept": "color"
        }

        # Example 2: Size with swap - EASY (extreme size difference)
        Input:
        Caption: "A large dog and a small cat."
        Attribute: large
        Object: dog

        Output:
        {
            "attribute_negative": "microscopic",
            "object_negative": "elephant",
            "swap_negative": ["large", "small"],
            "attribute_concept": "size"
        }

        # Example 3: Material - EASY (very different material)
        Input:
        Caption: "A wooden chair in the room."
        Attribute: wooden
        Object: chair

        Output:
        {
            "attribute_negative": "crystal",
            "object_negative": "helicopter",
            "swap_negative": null,
            "attribute_concept": "material"
        }

        # Example 4: Number - EASY (large number change)
        Input:
        Caption: "Three birds on a wire."
        Attribute: three
        Object: birds

        Output:
        {
            "attribute_negative": "fifty",
            "object_negative": "airplanes",
            "swap_negative": null,
            "attribute_concept": "number"
        }

        # Example 5: Pattern with swap - EASY
        Input:
        Caption: "A striped shirt and a plain jacket."
        Attribute: striped
        Object: shirt

        Output:
        {
            "attribute_negative": "polka-dotted",
            "object_negative": "flag",
            "swap_negative": ["striped", "plain"],
            "attribute_concept": "pattern"
        }

        # Example 6: Abstract attribute (null)
        Input:
        Caption: "A photo of two giraffes."
        Attribute: of
        Object: giraffes

        Output:
        {
            "attribute_negative": null,
            "object_negative": "rockets",
            "swap_negative": null,
            "attribute_concept": "other"
        }

        # Example 7: Spatial relation - EASY
        Input:
        Caption: "A book to the left of a vase."
        Attribute: left
        Object: book

        Output:
        {
            "attribute_negative": "underneath",
            "object_negative": "elephant",
            "swap_negative": ["book", "vase"],
            "attribute_concept": "spatial_relation"
        }

        # Example 8: Generic object - return null
        Input:
        Caption: "A beautiful photograph of a sunset."
        Attribute: beautiful
        Object: photograph

        Output:
        {
            "attribute_negative": null,
            "object_negative": null,
            "swap_negative": null,
            "attribute_concept": "other"
        }
        """
    
    elif hardness == "hard":
        return """
        # Example 1: Color change - HARD (different but natural color)
        Input:
        Caption: "A red apple on a table."
        Attribute: red
        Object: apple

        Output:
        {
            "attribute_negative": "green",
            "object_negative": "tomato",
            "swap_negative": null,
            "attribute_concept": "color"
        }

        # Example 2: Size with swap - HARD (clear size difference)
        Input:
        Caption: "A large dog and a small cat."
        Attribute: large
        Object: dog

        Output:
        {
            "attribute_negative": "small",
            "object_negative": "horse",
            "swap_negative": ["large", "small"],
            "attribute_concept": "size"
        }

        # Example 3: Material - HARD (visually different material)
        Input:
        Caption: "A wooden chair in the room."
        Attribute: wooden
        Object: chair

        Output:
        {
            "attribute_negative": "metal",
            "object_negative": "table",
            "swap_negative": null,
            "attribute_concept": "material"
        }

        # Example 4: Number - HARD (noticeably different count)
        Input:
        Caption: "Three birds on a wire."
        Attribute: three
        Object: birds

        Output:
        {
            "attribute_negative": "seven",
            "object_negative": "squirrels",
            "swap_negative": null,
            "attribute_concept": "number"
        }

        # Example 5: Pattern with swap - HARD (visually distinct pattern)
        Input:
        Caption: "A striped shirt and a plain jacket."
        Attribute: striped
        Object: shirt

        Output:
        {
            "attribute_negative": "checkered",
            "object_negative": "tie",
            "swap_negative": ["striped", "plain"],
            "attribute_concept": "pattern"
        }

        # Example 6: Abstract attribute (null)
        Input:
        Caption: "A photo of two giraffes."
        Attribute: of
        Object: giraffes

        Output:
        {
            "attribute_negative": null,
            "object_negative": "zebras",
            "swap_negative": null,
            "attribute_concept": "other"
        }

        # Example 7: Action - HARD (visually different action)
        Input:
        Caption: "A person walking on the beach."
        Attribute: walking
        Object: person

        Output:
        {
            "attribute_negative": "sitting",  # Clearly different pose/action
            "object_negative": "dog",  # Different entity entirely
            "swap_negative": null,
            "attribute_concept": "action"
        }

        # Example 8: Spatial relation - HARD
        Input:
        Caption: "A book to the left of a vase."
        Attribute: left
        Object: book

        Output:
        {
            "attribute_negative": "behind",  # Different spatial arrangement
            "object_negative": "laptop",  # Different but plausible object
            "swap_negative": ["book", "vase"],
            "attribute_concept": "spatial_relation"
        }

        # Example 9: Generic object - return null
        Input:
        Caption: "A beautiful photograph of a sunset."
        Attribute: beautiful
        Object: photograph

        Output:
        {
            "attribute_negative": null,
            "object_negative": null,
            "swap_negative": null,
            "attribute_concept": "other"
        }

        # Example 10: State change - visually obvious
        Input:
        Caption: "An open door to the garden"
        Attribute: open
        Object: door

        Output:
        {
            "attribute_negative": "closed",  # Visually opposite state
            "object_negative": "window",  # Different opening type
            "swap_negative": null,
            "attribute_concept": "state"
        }

        # Example 11: Near-synonyms NOT allowed
        Input:
        Caption: "An interview with the CEO"
        Attribute: interview
        Object: CEO

        Output:
        {
            "attribute_negative": null,  # 'meeting' or 'talk' too similar
            "object_negative": null,  # Too abstract to have visual alternative
            "swap_negative": null,
            "attribute_concept": "other"
        }
    """
    
        
def parse_thinking_content(messages):
    messages = copy.deepcopy(messages)
    for message in messages:
        if message["role"] == "assistant" and (m := re.match(r"<think>\n(.+)</think>\n\n", message["content"], flags=re.DOTALL)):
            message["content"] = message["content"][len(m.group(0)):]
            if thinking_content := m.group(1).strip():
                message["reasoning_content"] = thinking_content
    return messages


def generate_joint_negatives_batched_v2(
    llm,  # LLMWrapper
    captions: List[str],
    adjectives: List[str],
    objects: List[str],
    concepts: List[str],
    n_neg: int = 2,
    batch_size: int = 16,
    hardness: str = "hard"
) -> List[List[Dict[str, Any]]]:
    """
    For each (caption, attribute, object, concept), generate:
      - Attribute negative (change the attribute, which could be an adjective, adverb, relation, or number)
      - Object negative (change object)
      - Attribute concept classification (classify attribute into concept category)
    Returns a list of lists: each inner list is for one caption, each item is a dict with the negatives and concept.
    """
    import re

    # Define valid concepts for classification
    valid_concepts = [
        "color",              # red, blue, green, yellow, purple, orange, pink, brown, black, white
        "size",               # large, small, big, tiny, huge, massive, mini, enormous
        "shape",              # round, square, circular, rectangular, triangular, oval, curved
        "material",           # wooden, metal, plastic, glass, leather, fabric, ceramic, stone
        "texture",            # smooth, rough, soft, hard, fuzzy, silky, bumpy, coarse, shiny
        "pattern",            # striped, dotted, checkered, floral, plain, solid, spotted
        "number",             # one, two, three, many, few, several, couple, dozen
        "spatial_relation",   # left, right, above, below, behind, front, between, next to, inside, outside
        "orientation",        # vertical, horizontal, diagonal, upright, sideways, tilted, rotated
        "state",              # open, closed, empty, full, on, off, wet, dry, broken, intact
        "lighting",           # bright, dark, dim, sunny, shadowy, illuminated, glowing
        "weather",            # sunny, cloudy, rainy, snowy, windy, stormy, foggy
        "action",             # running, sitting, jumping, eating, sleeping, flying, walking
        "manner",             # quickly, slowly, carefully, gracefully, awkwardly, smoothly
        "pose",               # standing, lying, crouching, leaning, stretching, bent
        "existence",          # with, without, having, lacking, containing, missing
        "comparative",        # larger, smaller, taller, shorter, more, less, better, worse
        "possession",         # owned, borrowed, shared, personal, public, private
        "age",                # young, old, new, ancient, modern, vintage, fresh, aged
        "emotion",            # happy, sad, angry, excited, calm, surprised, worried
        "other"               # fallback for unclassified attributes
    ]

    joint_few_shot_easy = """
    # Example 1: Abstract/grammatical attribute (attribute_negative null)
    Input:
    Caption: "A photo of two giraffes."
    Attribute: of
    Object: giraffes

    Output:
    {
    "attribute_negative": null,
    "object_negative": "A photo of two cars.",
    "swap_negative": null,
    "attribute_concept": "other"
    }

    # Example 2: Number (easy - change to non-animal)
    Input:
    Caption: "A photo of two giraffes."
    Attribute: two
    Object: giraffes

    Output:
    {
    "attribute_negative": "A photo of five giraffes.",
    "object_negative": "A photo of two boats.",
    "swap_negative": null,
    "attribute_concept": "number"
    }

    # Example 3: Color (easy - wild change)
    Input:
    Caption: "A red apple on a table."
    Attribute: red
    Object: apple

    Output:
    {
    "attribute_negative": "A blue apple on a table.",
    "object_negative": "A red phone on a table.",
    "swap_negative": null,
    "attribute_concept": "color"
    }

    # Example 4: Place category (easy - dramatic change)
    Input:
    Caption: "A lake under a blue sky."
    Attribute: lake
    Object: lake

    Output:
    {
    "attribute_negative": null,
    "object_negative": "A city center under a blue sky.",
    "swap_negative": null,
    "attribute_concept": "other"
    }

    # Example 5: Swap with dramatic category difference
    Input:
    Caption: "A large dog and a small cat."
    Attribute: large
    Object: dog

    Output:
    {
    "attribute_negative": "A tiny dog and a small cat.",
    "object_negative": "A large chair and a small cat.",
    "swap_negative": "A large cat and a small dog.",
    "attribute_concept": "size"
    }

    # Example 6: Existence/possession, easy category change
    Input:
    Caption: "A woman with glasses reading a book."
    Attribute: with
    Object: woman

    Output:
    {
    "attribute_negative": "A woman without glasses reading a book.",
    "object_negative": "A robot with glasses reading a book.",
    "swap_negative": null,
    "attribute_concept": "existence"
    }

    # Example 7: Comparative, obvious category change
    Input:
    Caption: "A taller man stands next to a shorter woman."
    Attribute: taller
    Object: man

    Output:
    {
    "attribute_negative": "A shorter man stands next to a shorter woman.",
    "object_negative": "A taller horse stands next to a shorter woman.",
    "swap_negative": "A taller woman stands next to a shorter man.",
    "attribute_concept": "comparative"
    }

    # Example 8: Swap colors
    Input:
    Caption: "green fruits and vegetables in the shape of bathroom weighing scales over a white background"
    Attribute: green
    Object: fruits

    Output:
    {
    "attribute_negative": "yellow fruits and vegetables in the shape of bathroom weighing scales over a white background.",
    "object_negative": "green cars and vegetables in the shape of bathroom weighing scales over a white background.",
    "swap_negative": "white apples and vegetables in the shape of bathroom weighing scales over a green background.",
    "attribute_concept": "color"
    }

    # Example 9: Attribute removal is not allowed
    Input:
    Caption: "misty light over the valley , with a griffin vulture soaring"
    Attribute: misty
    Object: light

    Output:
    {
    "attribute_negative": "sunny light over the valley , with a griffin vulture soaring",
    "object_negative": "misty cloud over the valley , with a griffin vulture soaring",
    "swap_negative": null,  // Only one attribute-object pair, cannot swap.
    "attribute_concept": "weather"
    }

    """

    joint_few_shot_hard = """
    # Example 1: Abstract/grammatical attribute (attribute_negative null)
    Input:
    Caption: "A photo of two giraffes."
    Attribute: of
    Object: giraffes

    Output:
    {
    "attribute_negative": null,
    "object_negative": "A photo of two lions.",
    "swap_negative": null,
    "attribute_concept": "other"
    }

    # Example 2: Number (hard - plausible, subtle change)
    Input:
    Caption: "A photo of two giraffes."
    Attribute: two
    Object: giraffes

    Output:
    {
    "attribute_negative": "A photo of three giraffes.",
    "object_negative": "A photo of two zebras.",
    "swap_negative": null,
    "attribute_concept": "number"
    }

    # Example 3: Color (hard - natural, plausible color change)
    Input:
    Caption: "A red apple on a table."
    Attribute: red
    Object: apple

    Output:
    {
    "attribute_negative": "A green apple on a table.",
    "object_negative": "A red pear on a table.",
    "swap_negative": null,
    "attribute_concept": "color"
    }

    # Example 4: Place/object (hard - same broad category, subtle shift)
    Input:
    Caption: "A lake under a blue sky."
    Attribute: lake
    Object: lake

    Output:
    {
    "attribute_negative": null,
    "object_negative": "A sea under a blue sky.",
    "swap_negative": null,
    "attribute_concept": "other"
    }

    # Example 5: Swap within closely related objects/attributes
    Input:
    Caption: "A large dog and a small cat."
    Attribute: large
    Object: dog

    Output:
    {
    "attribute_negative": "A medium dog and a small cat.",
    "object_negative": "A large wolf and a small cat.",
    "swap_negative": "A large cat and a small dog.",
    "attribute_concept": "size"
    }

    # Example 6: Existence/possession, subtle object shift
    Input:
    Caption: "A woman with glasses reading a book."
    Attribute: with
    Object: woman

    Output:
    {
    "attribute_negative": "A woman without glasses reading a book.",
    "object_negative": "A man with glasses reading a book.",
    "swap_negative": null,
    "attribute_concept": "existence"
    }

    # Example 7: Comparative, subtle but plausible
    Input:
    Caption: "A taller man stands next to a shorter woman."
    Attribute: taller
    Object: man

    Output:
    {
    "attribute_negative": "A shorter man stands next to a shorter woman.",
    "object_negative": "A taller boy stands next to a shorter woman.",
    "swap_negative": "A taller woman stands next to a shorter man.",
    "attribute_concept": "comparative"
    }

    # Example 8: Swap colors
    Input:
    Caption: "green fruits and vegetables in the shape of bathroom weighing scales over a white background"
    Attribute: green
    Object: fruits

    Output:
    {
    "attribute_negative": "yellow fruits and vegetables in the shape of bathroom weighing scales over a white background.",
    "object_negative": "green apples and vegetables in the shape of bathroom weighing scales over a white background.",
    "swap_negative": "white apples and vegetables in the shape of bathroom weighing scales over a green background.",
    "attribute_concept": "color"
    }

    # Example 9: Attribute removal is not allowed
    Input:
    Caption: "misty light over the valley , with a griffin vulture soaring"
    Attribute: misty
    Object: light

    Output:
    {
    "attribute_negative": "sunny light over the valley , with a griffin vulture soaring",
    "object_negative": "misty cloud over the valley , with a griffin vulture soaring",
    "swap_negative": null,  // Only one attribute-object pair, cannot swap.
    "attribute_concept": "weather"
    }

    # Example: Abstract or not visually observable
    Input:
    Caption: "setup your professional ★wordpress★ site within ►6hours◄"
    Attribute: professional
    Object: site

    Output:
    {
        "attribute_negative": null,
        "object_negative": null,
        "swap_negative": null,
        "attribute_concept": "other"
    }
    """

    if hardness == "easy":
            hardness_prompt = "For easy negatives, replace with something highly visually and semantically distinct, "
            "but still plausible in the scene. For example, replace a place with a very different type of place, "
            "or an animal with a very different animal. The change should be obvious and striking, "
            "but the caption must remain plausible and grammatically correct."
            joint_few_shot = joint_few_shot_easy
    elif hardness == "hard":
            hardness_prompt = "For hard negatives, replace with something visually similar but meaningfully different, "
            "requiring careful observation to distinguish. Avoid changes that are so subtle as to be indistinguishable. "
            "All changes should still be visually and semantically identifiable in an image and must make sense in context."
            joint_few_shot = joint_few_shot_hard
    else:
            hardness_prompt = "For medium negatives, replace with something from the same broad category, "
            "but with a clear and noticeable difference. The change should be meaningful and plausible, "
            "but not jarring or overly subtle."
            joint_few_shot = joint_few_shot_hard

    prompts = []
    meta = []
    for caption, adj, obj, concept in zip(captions, adjectives, objects, concepts):
        system_prompt = f"""
        You are an expert at generating visually distinct negative captions for compositionality evaluation.

        **IMPORTANT RULE:**
        Only suggest changes that correspond to attributes or objects that are **visually distinguishable in an image**. Do NOT generate negatives for abstract, functional, or context-only changes (e.g., "blog" vs "site", "photo", "image", "view", or branding words).

        **TASK**: Given a caption, an attribute, and an object:
        - Generate up to three negatives:
        1. **attribute_negative**: Change ONLY the specified attribute to a visually distinct, plausible alternative for the same object. Keep all other words, including the object, unchanged.
        2. **object_negative**: Change ONLY the specified object to a visually plausible alternative for the same attribute. Keep the attribute and all other words unchanged.
        3. **swap_negative**: Swap the *positions* or *attributes* between two objects in the caption, but ONLY if two objects/attributes are present and share a comparable attribute. If not possible, return null.

        - Also, classify the attribute into one of these visual concept categories:
        [{', '.join(valid_concepts)}]

        **STRICT RULES:**
        - If the attribute or object is abstract, grammatical, a function word, a brand, or not visually observable in a real image, RETURN `null` for all negatives.
        - Only change the first occurrence of the specified attribute or object in the caption. Do NOT alter, paraphrase, or swap any other part of the caption.
        - Do NOT use synonyms, reword, or otherwise modify the sentence except for the minimal required change.
        - If no plausible swap is possible (e.g., only one object or only one attribute-object pair), set `swap_negative` to `null`.
        - NEVER remove an attribute or object while swapping (e.g., do not just delete "misty" to make "light").
        - NEVER invent or guess new attribute-object pairs if they do not appear in the caption.
        - All negatives must be grammatically correct and plausible for a real-world image.
        - Do not propose any negative if the result is identical to the original caption.

        **EXAMPLES OF UNACCEPTABLE NEGATIVES:**
        - Changing "site" to "blog" or "website" (not visually observable)
        - Changing "photo" to "image" or "picture" (not visually observable)
        - Changing "Actimel vanilla - 8x100g Brand Price Match..." to anything that only changes the text and not the visible content
        - Proposing a negative that is identical to the input caption

        **HARDNESS**: 
        - Follow the specified hardness level: {hardness}.
        - {hardness_prompt}

        **OUTPUT FORMAT:** (return valid JSON only)
        {{
        "attribute_negative": "...",   // or null
        "object_negative": "...",      // or null
        "swap_negative": "...",        // or null
        "attribute_concept": "..."
        }}
        """



        user_prompt = (
            joint_few_shot.strip() + "\n\n"
            f"Caption: \"{caption}\"\n"
            f"Attribute: {adj}\n"
            f"Object: {obj}\n"
        )
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        for _ in range(n_neg):
            prompt = llm.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            meta.append((caption, adj, obj, concept))

    print(f"Joint generation: {len(prompts)} prompts, batch_size={batch_size}")
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating outputs"):
        batch_prompts = prompts[i:i + batch_size]
        batch_outputs = llm.generate(
            batch_prompts,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            batch_size=batch_size,
            num_return_sequences=1
        )
        outputs.extend(batch_outputs)

    results = [[] for _ in range(len(captions))]
    for i, output in enumerate(outputs):
        # Parse attribute/object negative and concept using regex
        attr_neg = output['attribute_negative']
        obj_neg = output['object_negative']
        swap_neg = output['swap_negative']
        attr_concept = output['attribute_concept']

        orig_caption, adj, obj, concept = meta[i]
        cap_idx = i // n_neg

        res = {}
        if attr_neg and attr_neg.lower() != orig_caption.lower():
            res["attribute_negative"] = attr_neg
        else:
            res["attribute_negative"] = None
        if obj:
            if obj_neg and obj_neg.lower() != orig_caption.lower() and obj.lower() not in obj_neg.lower() and adj.lower() in obj_neg.lower():
                res["object_negative"] = obj_neg
            else:
                res["object_negative"] = None
        else:
            res["object_negative"] = None
        if swap_neg and swap_neg.lower() != orig_caption.lower():
            res["swap_negative"] = swap_neg
        else:
            res["swap_negative"] = None
        res["attribute_concept"] = attr_concept

        results[cap_idx].append(res)
    return results

def apply_negatives_to_caption(caption: str, attribute: str, object_: str, negative: Dict[str, Any]) -> Dict[str, str]:
    output = {}

    # Attribute negative
    attr_neg = negative.get("attribute_negative")
    if attr_neg:
        output["attribute_negative"] = re.sub(rf'\b{re.escape(attribute)}\b', attr_neg, caption, count=1)
    else:
        output["attribute_negative"] = None

    # Object negative
    obj_neg = negative.get("object_negative")
    if obj_neg:
        output["object_negative"] = re.sub(rf'\b{re.escape(object_)}\b', obj_neg, caption, count=1)
    else:
        output["object_negative"] = None

    # Swap negative (only if BOTH exist in the caption)
    swap_neg = negative.get("swap_negative")
    if swap_neg and isinstance(swap_neg, (list, tuple)) and len(swap_neg) == 2:
        word1, word2 = swap_neg
        # Check if both words exist in the caption (case-insensitive, word-boundary)
        match1 = re.search(rf'\b{re.escape(word1)}\b', caption, flags=re.IGNORECASE)
        match2 = re.search(rf'\b{re.escape(word2)}\b', caption, flags=re.IGNORECASE)
        if match1 and match2:
            # Use temporary placeholders to avoid double replacement
            temp1 = "__TEMP1__"
            temp2 = "__TEMP2__"
            temp_caption = re.sub(rf'\b{re.escape(word1)}\b', temp1, caption, count=1, flags=re.IGNORECASE)
            temp_caption = re.sub(rf'\b{re.escape(word2)}\b', temp2, temp_caption, count=1, flags=re.IGNORECASE)
            temp_caption = temp_caption.replace(temp1, word2, 1)
            temp_caption = temp_caption.replace(temp2, word1, 1)
            output["swap_negative"] = temp_caption
        else:
            # If either word is missing, do not perform swap
            output["swap_negative"] = None
    else:
        output["swap_negative"] = None

    output["attribute_concept"] = negative.get("attribute_concept", None)
    return output


def generate_joint_negatives_batched(
    llm,  # LLMWrapper
    captions: List[str],
    adjectives: List[str],
    objects: List[str],
    concepts: List[str],
    n_neg: int = 2,
    batch_size: int = 16,
    hardness: str = "hard"
) -> List[List[Dict[str, Any]]]:
    """
    For each (caption, attribute, object, concept), LLM generates:
      - attribute_negative: the new attribute to swap in (string or null)
      - object_negative: the new object to swap in (string or null)
      - swap_negative: a tuple of two words to swap in the caption (or null)
      - attribute_concept: the concept category
    Returns a list of lists: each inner list is for one caption, each item is a dict of the slot-level negatives and concept.
    """
    # --- Concepts list
    valid_concepts = [
        "color", "size", "shape", "material", "texture", "pattern", "number", "spatial_relation", "orientation", "state", "lighting", "weather", "action", "manner", "pose", "existence", "comparative", "possession", "age", "emotion", "other"
    ]

    if hardness == "easy":
        hardness_prompt = (
            "For easy negatives, replace EITHER the attribute OR the object with something visually very different but still plausible. "
            "The change should be immediately obvious. For example:\n"
            "- ATTRIBUTE change: 'red apple' → 'blue apple' (dramatically different color)\n"
            "- OBJECT change: 'red apple' → 'red car' (completely different object category)\n"
            "- ATTRIBUTE change: 'large dog' → 'tiny dog' (extreme size difference)\n"
            "- OBJECT change: 'large dog' → 'large elephant' (very different animal)\n"
            "- ATTRIBUTE change: 'walking slowly' → 'running fast' (clearly different action/manner)\n"
            "Easy negatives should have high visual contrast - pick attributes or objects that look very different.\n"
            "Change ONLY the attribute OR ONLY the object, not both."
        )
        joint_few_shot = get_fewshot_examples("easy")
    elif hardness == "hard":
        hardness_prompt = (
            "For hard negatives, replace EITHER the attribute OR the object with something visually distinct but conceptually related. "
            "The change should be noticeable but within a similar category. For example:\n"
            "- ATTRIBUTE change: 'red apple' → 'green apple' (different color, same fruit)\n"
            "- OBJECT change: 'red apple' → 'red strawberry' (different fruit, same color)\n"
            "- ATTRIBUTE change: 'wooden chair' → 'metal chair' (different material, same furniture)\n"
            "- OBJECT change: 'wooden chair' → 'wooden stool' (different furniture type, same material)\n"
            "- ATTRIBUTE change: 'walking quickly' → 'walking slowly' (different manner, same action)\n"
            "- OBJECT change: 'small dog' → 'small cat' (similar-sized pets, visually distinguishable)\n"
            "Hard negatives should be visually distinct but conceptually close (dog→cat rather than dog→airplane).\n"
            "AVOID near-synonyms like: interview→news, entry→form, photo→image\n"
            "Change ONLY the attribute OR ONLY the object, not both."
        )
        joint_few_shot = get_fewshot_examples("hard")
    else:  # medium
        hardness_prompt = (
            "For medium negatives, replace EITHER the attribute OR the object with something moderately different. "
            "The change should be clear but not extreme. For example:\n"
            "- ATTRIBUTE change: 'red apple' → 'yellow apple' (noticeable color change)\n"
            "- OBJECT change: 'red apple' → 'red orange' (similar but distinct fruit)\n"
            "- ATTRIBUTE change: 'large dog' → 'small dog' (clear size difference)\n"
            "- OBJECT change: 'large dog' → 'large cat' (related but distinct canine)\n"
            "- ATTRIBUTE change: 'plastic bottle' → 'glass bottle' (different material, same object)\n"
            "- OBJECT change: 'plastic bottle' → 'plastic cup' (different container, same material)\n"
            "Medium negatives balance between similarity and difference (dog→wolf, not dog→cat or dog→airplane).\n"
            "Change ONLY the attribute OR ONLY the object, not both."
        )
        joint_few_shot = get_fewshot_examples("hard")

    prompts = []
    meta = []
    for caption, adj, obj, concept in zip(captions, adjectives, objects, concepts):
        system_prompt = f"""
            You are an expert at generating visually distinct negative captions for image-text compositionality evaluation.

            **TASK**: Given a caption, attribute, and object:
            1. Generate alternative words that would create visually distinguishable changes
            2. Return ONLY the replacement word(s), never full sentences
            3. Classify the attribute into a visual concept category

            **CRITICAL REQUIREMENT**: All changes MUST be visually observable in an image. A person looking at two images should clearly see the difference.

            **OUTPUT OPTIONS**:
            - attribute_negative: A visually different attribute (just the word)
            - object_negative: A visually different object (just the word)
            - swap_negative: Two words from the caption to swap positions [word1, word2]
            - attribute_concept: The visual category of the attribute

            **WHEN TO RETURN NULL**: If the object or attribute is abstract, grammatical, functional, or not visually observable in a real image:
            - Abstract/grammatical words: "of", "for", "by", "with" (unless "with/without" indicates presence)
            - Non-visual attributes: "official", "professional", "exclusive"
            - Generic objects: "thing", "item", "element", "content"
            - Meta-references: "photo", "image", "picture", "view"
            - Near-synonyms that look the same: interview≈meeting, entry≈form, website≈webpage

            **VALID CHANGES MUST BE**:
            - Visually verifiable (not just semantic differences)
            - Grammatically compatible with the sentence
            - Plausible in real images

            **AVOID THESE INVALID CHANGES**:
            - walking → jogging (too similar visually)
            - person → individual (synonym)
            - data entry → data form (indistinguishable)
            - interview → news (both could look identical)

            **GOOD EXAMPLES**:
            - red → green (clear color difference)
            - dog → cat (distinct animals)
            - walking → sitting (different poses)
            - three → eight (countable difference)

            **HARDNESS LEVEL**: 
            {hardness_prompt}

            **OUTPUT FORMAT** (valid JSON only):
            {{
                "attribute_negative": "word",   // or null
                "object_negative": "word",      // or null  
                "swap_negative": ["word1", "word2"],   // or null
                "attribute_concept": "{concept}"  // from: [{', '.join(valid_concepts)}]
            }}
            """


        user_prompt = (
            joint_few_shot.strip() + "\n\n"
            f'Caption: "{caption}"\n'
            f'Attribute: {adj}\n'
            f'Object: {obj}\n'
        )
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        for _ in range(n_neg):
            prompt = llm.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            meta.append((caption, adj, obj, concept))

    print(f"Slot generation: {len(prompts)} prompts, batch_size={batch_size}")
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating outputs"):
        batch_prompts = prompts[i:i + batch_size]
        batch_outputs = llm.generate(
            batch_prompts,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.8,
            batch_size=batch_size,
            num_return_sequences=1
        )
        outputs.extend(batch_outputs)

    results = [[] for _ in range(len(captions))]
    for i, output in enumerate(outputs):
        # Parse the slot-level negatives
        attr_neg = output.get('attribute_negative', None)
        obj_neg = output.get('object_negative', None)
        swap_neg = output.get('swap_negative', None)
        attr_concept = output.get('attribute_concept', None)

        orig_caption, adj, obj, concept = meta[i]
        cap_idx = i // n_neg

        res = {}
        res["attribute_negative"] = attr_neg if attr_neg not in [None, '', "null"] else None
        res["object_negative"] = obj_neg if obj_neg not in [None, '', "null"] else None
        res["swap_negative"] = swap_neg if swap_neg not in [None, '', "null"] else None
        res["attribute_concept"] = attr_concept

        res_processed = apply_negatives_to_caption(
            caption=orig_caption,
            attribute=adj,
            object_=obj,
            negative=res
        )
        results[cap_idx].append(res_processed)

    return results


def generate_lexicon_attribute_neg(
    caption: str, concept: str, adj: str, obj: str, lexicons: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    """Generate negative by flipping adjective using lexicon"""
    if concept not in lexicons:
        return None
    flips = lexicons[concept].get('flips', {})
    lemma = adj.lower()
    candidates = [c for c in flips.get(lemma, []) if c != lemma]
    if not candidates:
        return None
    replacement = random.choice(candidates)
    pattern = rf'\b{re.escape(adj)}\b'
    neg = re.sub(pattern, replacement, caption, count=1)
    return neg if neg != caption else None

def swap_adjectives_same_concept(caption, adj1, obj1, adj2, obj2):
    temp1, temp2 = "__TEMP_ADJ1__", "__TEMP_ADJ2__"
    step1 = re.sub(rf'\b{re.escape(adj1)}\b', temp1, caption, count=1)
    step2 = re.sub(rf'\b{re.escape(adj2)}\b', temp2, step1, count=1)
    step3 = step2.replace(temp1, adj2)
    step4 = step3.replace(temp2, adj1)
    return step4 if step4 != caption else None

def swap_objects_different_concepts(caption, adj1, obj1, adj2, obj2):
    temp1, temp2 = "__TEMP_OBJ1__", "__TEMP_OBJ2__"
    step1 = re.sub(rf'\b{re.escape(obj1)}\b', temp1, caption, count=1)
    step2 = re.sub(rf'\b{re.escape(obj2)}\b', temp2, step1, count=1)
    step3 = step2.replace(temp1, obj2)
    step4 = step3.replace(temp2, obj1)
    return step4 if step4 != caption else None

def perform_concept_based_swapping(
    captions: List[Dict],
    concepts: List[str],
    relevance_lookup: Dict[str, Dict[str, bool]],
    caption_ids: Dict[str, str]
) -> List[Dict]:
    """
    Perform concept-based attribute/object swaps for all captions.
    This is copied almost verbatim from your main script.
    """
    from parsing import parse_caption
    swap_results = []
    for caption_row in captions:
        caption = caption_row['caption']
        cap_url = caption_row.get('url', None)
        cap_id = caption_ids[caption]
        parsed = parse_caption(caption, concepts)
        adj_obj_concept_pairs = []
        for adj in parsed['all_attributes']:
            for concept in concepts:
                if relevance_lookup.get(adj, {}).get(concept, False):
                    for (parsed_adj, obj) in parsed['attributes'][concept]:
                        if parsed_adj == adj and obj:
                            adj_obj_concept_pairs.append((adj, obj, concept))
                            break
        unique_pairs = []
        seen = set()
        for pair in adj_obj_concept_pairs:
            if pair not in seen:
                unique_pairs.append(pair)
                seen.add(pair)
        if len(unique_pairs) >= 2:
            concept_groups = {}
            for adj, obj, concept in unique_pairs:
                if concept not in concept_groups:
                    concept_groups[concept] = []
                concept_groups[concept].append((adj, obj))
            for concept, pairs in concept_groups.items():
                if len(pairs) >= 2:
                    for i in range(len(pairs)):
                        for j in range(i + 1, len(pairs)):
                            adj1, obj1 = pairs[i]
                            adj2, obj2 = pairs[j]
                            swapped_caption = swap_adjectives_same_concept(
                                caption, adj1, obj1, adj2, obj2
                            )
                            if swapped_caption and swapped_caption != caption:
                                swap_results.append({
                                    "caption_id": cap_id,
                                    "caption": caption,
                                    "negative": swapped_caption,
                                    "type": f"swap_{concept}",
                                    "swapped_pairs": [(adj1, obj1), (adj2, obj2)],
                                    "swap_type": "same_concept",
                                    "concept": concept,
                                    "image_url": cap_url
                                })
            concept_list = list(concept_groups.keys())
            if len(concept_list) >= 2:
                for i in range(len(concept_list)):
                    for j in range(i + 1, len(concept_list)):
                        concept1, concept2 = concept_list[i], concept_list[j]
                        pairs1, pairs2 = concept_groups[concept1], concept_groups[concept2]
                        if pairs1 and pairs2:
                            adj1, obj1 = pairs1[0]
                            adj2, obj2 = pairs2[0]
                            swapped_caption = swap_objects_different_concepts(
                                caption, adj1, obj1, adj2, obj2
                            )
                            if swapped_caption and swapped_caption != caption:
                                swap_results.append({
                                    "caption_id": cap_id,
                                    "caption": caption,
                                    "negative": swapped_caption,
                                    "type": "swap_object",
                                    "swapped_pairs": [(adj1, obj1), (adj2, obj2)],
                                    "swap_type": "different_concepts",
                                    "concepts": [concept1, concept2],
                                    "image_url": cap_url
                                })
    return swap_results

def extract_yes_no_from_output(generated_text):
    """Extracts yes/no from LLM output in a robust way."""
    text = generated_text.strip().lower()
    patterns = [
        r'^(yes|no)\b',
        r'\b(yes|no)\b',
        r'answer:\s*(yes|no)',
        r'response:\s*(yes|no)'
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return "no"

def extract_yes_no_from_output(generated_text):
    """Extracts yes/no from LLM output robustly."""
    import re
    text = generated_text.strip().lower()
    patterns = [
        r'^(yes|no)\b',
        r'\b(yes|no)\b',
        r'answer:\s*(yes|no)',
        r'response:\s*(yes|no)'
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return "no"


def generate_positive_captions_batched(
    llm,
    captions: List[str],
    batch_size: int = 16
) -> List[Dict[str, Any]]:
    """
    Extract visual components from captions using LLM and reconstruct them as positive captions.
    
    For a caption like "Red car driving on a motorway and children crossing the street",
    extract components like:
    - "Red car driving"
    - "Motorway" 
    - "Children crossing the street"
    
    Then reconstruct by joining with "and": "Red car driving and motorway and children crossing the street"
    
    Args:
        llm: VLLMWrapper instance
        captions: List of original captions
        batch_size: Batch size for LLM generation
        
    Returns:
        List of dicts with original caption, extracted components, and reconstructed positive caption
    """
    
    few_shot_examples = """
    # Example 1: Multiple visual components
    Input Caption: "Red car driving on a motorway and children crossing the street"
    Output:
    {
        "components": [
            "red car driving",
            "motorway",
            "children crossing the street"
        ],
        "reconstructed_caption": "red car driving and motorway and children crossing the street"
    }

    # Example 2: Simple scene with attributes
    Input Caption: "A large dog and a small cat playing in the garden"
    Output:
    {
        "components": [
            "large dog",
            "small cat",
            "playing in the garden"
        ],
        "reconstructed_caption": "large dog and small cat and playing in the garden"
    }

    # Example 3: Object with multiple attributes
    Input Caption: "Blue and yellow umbrellas line the sunny beach"
    Output:
    {
        "components": [
            "blue umbrellas",
            "yellow umbrellas",
            "sunny beach"
        ],
        "reconstructed_caption": "blue umbrellas and yellow umbrellas and sunny beach"
    }

    # Example 4: Spatial relations
    Input Caption: "A book to the left of a vase on the table"
    Output:
    {
        "components": [
            "book to the left",
            "vase",
            "table"
        ],
        "reconstructed_caption": "book to the left and vase and table"
    }

    # Example 5: Actions and objects
    Input Caption: "Woman in green dress under a red umbrella walking"
    Output:
    {
        "components": [
            "woman in green dress",
            "red umbrella",
            "walking"
        ],
        "reconstructed_caption": "woman in green dress and red umbrella and walking"
    }

# Example 6: Simple object description
Input Caption: "A red apple on a table"
Output:
{
    "components": [
        "red apple",
        "table"
    ],
    "reconstructed_caption": "red apple and table"
}
"""

    system_prompt = """
You are an expert at extracting visually observable components from image captions.

**TASK**: 
1. Break down the caption into its visually distinct components (objects with their attributes, actions, scenes)
2. Extract ONLY components that are visually observable in an image
3. Reconstruct a positive caption by joining the components with "and"

**EXTRACTION RULES**:
- Include objects WITH their visual attributes (color, size, material, etc.)
- Include actions and poses if present
- Include spatial relationships if they define visual layout
- Keep components concise but complete
- DO NOT include abstract, grammatical, or non-visual elements
- DO NOT include redundant information

**VISUAL COMPONENTS TO EXTRACT**:
- Objects with attributes: "red car", "large dog", "wooden chair"
- Actions/activities: "driving", "crossing the street", "playing"
- Scenes/locations: "motorway", "beach", "garden" (if visually distinct)
- Spatial arrangements: "to the left of", "above", "inside"

**NON-VISUAL ELEMENTS TO IGNORE**:
- Articles: "a", "an", "the"
- Prepositions (unless part of spatial relation): "of", "for", "by"
- Abstract concepts: "idea", "concept", "feeling"
- Generic phrases: "photo of", "image of", "view of"

**OUTPUT FORMAT** (valid JSON only):
{
    "components": ["component1", "component2", "component3"],
    "reconstructed_caption": "component1 and component2 and component3"
}

All components should be lowercase and the reconstructed caption should join them with " and ".
"""

    prompts = []
    for caption in captions:
        user_prompt = (
            few_shot_examples.strip() + "\n\n"
            f'Input Caption: "{caption}"\n'
            f'Output:'
        )
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt = llm.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    print(f"Positive generation: {len(prompts)} captions, batch_size={batch_size}")
    
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating positive captions"):
        batch_prompts = prompts[i:i + batch_size]
        batch_outputs = llm.generate(
            batch_prompts,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            batch_size=batch_size,
            num_return_sequences=1,
            use_positive_scheme=True  # Use positive caption schema
        )
        
        for j, output in enumerate(batch_outputs):
            caption_idx = i + j
            original_caption = captions[caption_idx]
            
            try:
                # Parse JSON output
                components = output.get('components', [])
                reconstructed = output.get('reconstructed_caption', '')
                
                # Fallback: if no reconstruction, build it from components
                if not reconstructed and components:
                    reconstructed = ' and '.join(components)
                
                results.append({
                    "original_caption": original_caption,
                    "components": components,
                    "positive_caption": reconstructed,
                    "num_components": len(components)
                })
            except Exception as e:
                print(f"Error parsing output for caption '{original_caption}': {e}")
                results.append({
                    "original_caption": original_caption,
                    "components": [],
                    "positive_caption": original_caption,
                    "num_components": 0,
                    "error": str(e)
                })
    
    return results

def detect_concept_relevance_batched(
    llm,  # <-- LLMWrapper instance
    adjectives,
    concepts,
    batch_size=16
):
    """
    Use LLM to detect if adjectives belong to specific concepts.
    Returns a list of (adjective, concept, is_relevant) tuples.
    """
    if concepts == ['any']:
        # Short-circuit: All adjectives are relevant to "any"
        return [(adj, "any", True) for adj in adjectives]

    concept_examples = {
        "color": "red vs blue, black vs white, dark vs bright, warm vs cool (red, blue, green, yellow, purple, orange, pink, brown, black, white, gray, crimson, azure, emerald)",
        "size": "small vs large, tiny vs huge, mini vs giant (small, large, big, tiny, huge, massive, mini, giant, little, enormous, microscopic, colossal)",
        "relative_size": "larger vs smaller, bigger vs tinier, more vs less (larger, smaller, bigger, tinier, huger, more massive, less significant, greater, lesser)",
        "shape": "round vs square, circular vs rectangular, curved vs straight (round, square, rectangular, circular, triangular, oval, curved, straight, angular, spherical)",
        "material": "wooden vs metallic, soft vs hard, natural vs synthetic (wooden, metallic, plastic, glass, fabric, leather, ceramic, stone, rubber, steel, aluminum)",
        "texture": "smooth vs rough, soft vs hard, fine vs coarse (smooth, rough, soft, hard, bumpy, silky, fuzzy, coarse, fine, textured, grainy, polished)",
        "pattern": "solid vs patterned, uniform vs varied, plain vs decorated (striped, spotted, checkered, floral, geometric, solid, dotted, plaid, patterned, uniform)",
        "spatial_relation": "above vs below, inside vs outside, near vs far (above, below, beside, inside, outside, near, far, next to, on top of, under, behind, in front of)",
        "orientation": "vertical vs horizontal, upright vs tilted, straight vs diagonal (vertically, horizontally, diagonally, upright, sideways, tilted, straight, crooked, perpendicular)",
        "manner": "careful vs careless, gentle vs rough, slow vs fast (quickly, slowly, carefully, roughly, gently, forcefully, smoothly, abruptly, gradually)",
        "speed": "fast vs slow, rapid vs gradual, quick vs leisurely (fast, slow, rapid, swift, sluggish, speedy, leisurely, hurried, rushed, dawdling)"
    }

    few_shot_examples = """
Task: Determine if an adjective belongs to a specific concept category.

Examples:
Adjective: "crimson"
Concept: color
Answer: yes (crimson is a shade of red, which is a color)

Adjective: "enormous"
Concept: size
Answer: yes (enormous describes size)

Adjective: "larger"
Concept: relative_size
Answer: yes (larger is a comparative size descriptor)

Adjective: "vertically"
Concept: orientation
Answer: yes (vertically describes spatial orientation)

Adjective: "quickly"
Concept: manner
Answer: yes (quickly describes manner of action)

Adjective: "wooden"
Concept: color
Answer: no (wooden describes material, not color)

Adjective: "rectangular"
Concept: shape
Answer: yes (rectangular describes shape)

Adjective: "happy"
Concept: size
Answer: no (happy describes emotion, not size)

Adjective: "metallic"
Concept: material
Answer: yes (metallic describes material)

Adjective: "slowly"
Concept: speed
Answer: yes (slowly describes speed)

Adjective: "diagonally"
Concept: spatial_relation
Answer: no (diagonally describes orientation, not spatial relation)

Adjective: "tiny"
Concept: size
Answer: yes (tiny describes size - should be paired with opposite like 'huge' for visual distinction)

Adjective: "smooth"
Concept: texture
Answer: yes (smooth describes texture - should be paired with opposite like 'rough' for visual distinction)
"""

    prompts = []
    meta = []

    for adj in adjectives:
        for concept in concepts:
            examples = concept_examples.get(concept, "")
            system_prompt = (
                "You are an expert at categorizing adjectives by semantic concepts. "
                "Answer with exactly 'yes' or 'no' only."
            )
            user_prompt = (
                few_shot_examples.strip() + "\n\n" +
                f"Adjective: \"{adj}\"\n"
                f"Concept: {concept}\n"
                f"Examples of {concept} adjectives with semantic contrasts: {examples}\n"
                f"Does the adjective '{adj}' belong to the concept '{concept}'?\n"
                f"Consider if '{adj}' can be replaced with a semantically different {concept} adjective for visual distinction.\n"
                f"Answer:"
            )
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompts.append(llm.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
            meta.append((adj, concept))

    # Batched generation (in chunks if needed)
    results = []
    print(f"Concept detection: {len(prompts)} adjective-concept pairs, batch_size={batch_size}")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        outputs = llm.generate(
            batch_prompts,
            max_new_tokens=5,
            do_sample=False,
            num_return_sequences=1
        )
        for j, output in enumerate(outputs):
            content = output.get("generated_text", "")
            answer = extract_yes_no_from_output(content)
            is_relevant = (answer == "yes")
            adj, concept = meta[i + j]
            results.append((adj, concept, is_relevant))

    return results


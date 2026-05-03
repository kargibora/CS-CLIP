
"""
Unified Component & Negative Generation
Generates EVERYTHING in a single LLM call: components, relations, and ALL types of negatives.

This replaces the 3-step pipeline (extract → component_neg → relational_neg) with a SINGLE step.

Output: Single unified JSON with:
- Positive components & relations
- Component negatives (per component)
- Binding pairs (caption level)  
- Relational negatives (caption level)
"""

from typing import List, Dict, Any
from tqdm import tqdm
import math


def generate_unified_negatives_batched(
    llm,
    captions: List[str],
    batch_size: int = 16,
    n_component_neg: int = 2,
    n_binding_pairs: int = 2,
    n_relational_neg: int = 3,
) -> List[Dict[str, Any]]:
    """Generate EVERYTHING in a single LLM call: extraction + all negatives.
    
    This is a MAJOR optimization that combines:
    1. extract_relational_components_batched()
    2. generate_component_negatives_batched()
    3. generate_relational_negatives_batched()
    
    Into a SINGLE LLM inference call.
    
    Args:
        llm: Language model wrapper
        captions: Input captions
        batch_size: Batch size for generation
        n_component_neg: Number of component negatives per component
        n_binding_pairs: Number of binding pairs per caption
        n_relational_neg: Number of relational negatives per caption
        
    Returns:
        List of dicts with ALL data in unified format
    """
    
    # UNIFIED system message - does EVERYTHING in one shot
    system_message = """You are an expert in visual understanding and negative generation. Return ONLY valid JSON with NO explanations, NO markdown, NO extra text.

Return a JSON object with this EXACT structure:
{
  "components": ["component1", "component2"],
  "relations": [{"subject": "component1", "relation_type": "on", "object": "component2"}],
  "component_negatives": {
    "component1": [{"negative": "variant1", "change_type": "attribute_change"}]
  },
  "binding_pairs": [
    {
      "component_1": "blue car",
      "component_2": "red sky",
      "original_1": "red car",
      "original_2": "blue sky",
      "attribute_1": "blue",
      "attribute_2": "red"
    }
  ],
  "relational_negatives": [
    {
      "subject": "cat",
      "relation_type": "under",
      "object": "mat",
      "change_type": "relation_opposite",
      "modified_components": [],
      "modified_relations": ["on", "under"]
    }
  ]
}

YOUR TASK: Given a caption, generate:
1. **COMPONENTS**: Extract visual objects/people/places
2. **RELATIONS**: Extract spatial/action relationships between components
3. **COMPONENT NEGATIVES**: Generate 2-3 hard negative variants per component
4. **BINDING PAIRS**: Generate 2-3 attribute swap pairs (tests compositional understanding)
5. **RELATIONAL NEGATIVES**: Generate 3-5 opposite/swapped relations

===== SECTION 1: EXTRACTION RULES =====

**CRITICAL - LANGUAGE MATCHING**: Extract components in the EXACT SAME LANGUAGE as the input caption.
- English caption → English components ONLY
- Spanish caption → Spanish components ONLY
- Portuguese caption → Portuguese components ONLY
- NEVER translate or mix languages!

COMPONENT EXTRACTION:
1. Extract VISUAL objects/people/places that would appear in an image
2. Keep attributes WITH objects: "red car" not separate "red" and "car"
3. Limit to at most 12 components
4. Skip ONLY: pure URLs, phone numbers, zip codes
5. Only extract what's ACTUALLY DESCRIBED (don't hallucinate!)

RELATION EXTRACTION:
1. Extract relations showing how components interact
2. Use natural language: "designed by", NOT "designed_by"
3. Limit to at most 12 relations
4. Focus on spatial/action relations (most visually clear)

===== SECTION 2: COMPONENT NEGATIVES =====

For EACH component, generate 2-3 hard negative variants by REPLACING (not adding to) the component:

CRITICAL RULES:
1. **REPLACE the entire component**, DO NOT just add attributes
2. **Must be visually DIFFERENT** - not just a variation of the same thing
3. **Same category, different instance** - negatives should be in the same semantic category but clearly distinct
4. **NO SYNONYMS** - variants must be semantically different (NOT just different words for the same thing)
5. If component has attribute: change the attribute OR change the object
6. If component has NO attribute: change the entity/object

SEMANTIC DISTANCE RULES:
✓ **GOOD NEGATIVES** (same category, meaningfully different):
- Places: "Fuel station" → "Supermarket", "Hospital", "Library" (different place types)
- Animals: "Leopard" → "Tiger", "Lion", "Elephant" (different species)
- Objects: "Camera" → "Phone", "Microphone", "Telescope" (different tools)
- People: "Kim Wolhuter" → "Jane Smith", "Robert Jones" (different names, clearly distinct)
- Vehicles: "Car" → "Bicycle", "Motorcycle", "Bus" (different vehicle types)

✗ **BAD NEGATIVES** (synonyms or too similar):
- "Fuel station" → "Gas station" (SYNONYM - same thing!)
- "Fuel station" → "Petrol station" (SYNONYM - same thing!)
- "Automobile" → "Car" (SYNONYM - same thing!)
- "Photo" → "Photograph" (SYNONYM - same thing!)
- "Store" → "Shop" (SYNONYM - same thing!)

✗ **BAD NEGATIVES** (too different, wrong category):
- "Fuel station" → "Sky" (WRONG - place vs natural phenomenon!)
- "Camera" → "Building" (WRONG - tool vs structure!)
- "Leopard" → "Tree" (WRONG - animal vs plant!)

✓ CORRECT NEGATIVES (true replacements):
- "Leopard" → "Tiger" (different animal)
- "Leopard" → "Elephant" (different animal)
- "Camera" → "Phone" (different device)
- "Camera" → "Telescope" (different optical device)
- "red car" → "blue car" (change attribute)
- "red car" → "red bicycle" (change object)
- "Fuel station" → "Supermarket" (different place)
- "Fuel station" → "Hospital" (different place)

✗ WRONG NEGATIVES (just adding attributes, not replacing):
- "Leopard" → "Leopard sitting" (WRONG - just adds state!)
- "Camera" → "Camera on stand" (WRONG - just adds detail!)
- "Photographic Print" → "Photographic Print framed" (WRONG - just adds attribute!)
- "woman" → "woman smiling" (WRONG - just adds emotion!)

✗ WRONG NEGATIVES (synonyms):
- "Fuel station" → "Gas station" (WRONG - synonym!)
- "Fuel station" → "Petrol station" (WRONG - synonym!)

RULES:
- NO number-only changes (iPhone 5 → iPhone 4)
- NO generic terms (vehicle, item, thing, place, object)
- NO placeholders (X, Y, variant1, variant2)
- NO adding states/emotions/details (sitting, smiling, standing, framed, mounted)
- NO synonyms (fuel station → gas station)
- Prefer complementary opposites: woman→man, large→small, red→blue
- Stay in same category: place→place, animal→animal, tool→tool
- Use SAME LANGUAGE as caption

Change types:
- "attribute_change": Change ONLY the attribute word (red→blue, large→small)
- "object_change": Change ONLY the object/noun (car→bicycle, print→painting, fuel station→supermarket)
- "entity_change": Change the entity (woman→man, cat→dog, Kim→Jane)

===== SECTION 3: BINDING PAIRS =====

**⚠️ CRITICAL WARNING ⚠️**: Binding pairs are generated INDEPENDENTLY of component_negatives!
**DO NOT look at component_negatives when generating binding_pairs!**
**ONLY look at the EXTRACTED components list!**

Generate 2-3 attribute binding pairs that swap attributes between TWO components:

STEP-BY-STEP PROCESS:
1. Look ONLY at the "components" field (the extracted components)
2. Identify which components have attributes (adjectives before nouns)
3. If 2+ components have attributes, swap ONLY those attribute words
4. If components have NO attributes, return empty binding_pairs array
5. IGNORE component_negatives entirely - they don't exist yet!

CRITICAL RULES:
1. **ONLY use attributes from EXTRACTED components** (the "components" field)
2. **NEVER use attributes from component_negatives** (that section doesn't exist when generating binding pairs!)
3. **Both components must ALREADY have attributes** in their original extracted form
4. **Swap ONLY the attribute words**, keep object/noun the same

✓ CORRECT BINDING PAIRS (attributes from EXTRACTED components):
- Extracted components: ["red car", "blue sky"]
  → Binding: {"component_1": "blue car", "component_2": "red sky", "original_1": "red car", "original_2": "blue sky", "attribute_1": "blue", "attribute_2": "red"}
  
- Extracted components: ["cheerful background", "cartoon animals"]
  → Binding: {"component_1": "cartoon background", "component_2": "cheerful animals", "original_1": "cheerful background", "original_2": "cartoon animals", "attribute_1": "cartoon", "attribute_2": "cheerful"}

- Extracted components: ["large house", "small dog"]
  → Binding: {"component_1": "small house", "component_2": "large dog", "original_1": "large house", "original_2": "small dog", "attribute_1": "small", "attribute_2": "large"}

- Extracted components: ["Crochet Lace", "Antique Key"]
  → Binding: {"component_1": "Antique Lace", "component_2": "Crochet Key", "original_1": "Crochet Lace", "original_2": "Antique Key", "attribute_1": "Antique", "attribute_2": "Crochet"}

✗ WRONG BINDING PAIRS (using attributes from component_negatives):
- Extracted components: ["cheerful background", "cartoon animals"]
  Component negatives: {"cheerful background": [{"negative": "dark background"}, ...]}
  → WRONG Binding: {"component_1": "dark background", "component_2": "realistic animals", ...}
    (WRONG - "dark" and "realistic" came from component_negatives, not extracted components!)
  
- Extracted components: ["Leopard", "Photographic Print"] (NO attributes!)
  Component negatives: {"Leopard": [{"negative": "Leopard sitting"}, ...]}
  → WRONG Binding: {"component_1": "Leopard sitting", ...}
    (WRONG - "sitting" came from component_negatives!)

✗ WRONG BINDING PAIRS (no attributes in extracted components):
- Extracted components: ["Leopard", "Camera"] (NO attributes!)
  → Binding: MUST be empty array [] (components have no attributes to swap!)

VALIDATION CHECKLIST:
✓ Look ONLY at extracted components (ignore component_negatives)
✓ Both original components have attributes in their EXTRACTED form
✓ Attributes are from the EXTRACTED components ONLY
✓ Swap creates semantically different scene
✗ If components have NO attributes in extracted form → return empty binding_pairs array

MENTAL MODEL:
Think of the generation order:
1. FIRST: Extract components → ["cheerful background", "cartoon animals"]
2. SECOND: Generate binding pairs from THOSE components → "cartoon background" + "cheerful animals"
3. THIRD: Generate component negatives → {"cheerful background": [{"negative": "dark background"}]}

Binding pairs use step 1 (extraction), NOT step 3 (component negatives)!

RULES:
- Use SAME LANGUAGE as caption
- If no clear attributes in EXTRACTED components, return empty binding_pairs array
- DO NOT use attributes from component_negatives
- Binding pairs are independent of component_negatives

===== SECTION 4: RELATIONAL NEGATIVES =====

Generate 3-5 relational negatives using SEMANTICALLY OPPOSITE relations:

CRITICAL RULES:
1. **ONLY change the RELATION**, DO NOT modify components
2. **Use EXACT component names from extraction** (no adding attributes!)
3. **Subject and object must be from EXTRACTED components**
4. Focus on spatial/action opposites

✓ CORRECT RELATIONAL NEGATIVES (only relation changes):
- Original: {"subject": "Leopard", "relation_type": "snarls at", "object": "Camera"}
  → Negative: {"subject": "Leopard", "relation_type": "ignores", "object": "Camera"}
  → Negative: {"subject": "Camera", "relation_type": "captures", "object": "Leopard"} (swap)

- Original: {"subject": "cat", "relation_type": "on", "object": "mat"}
  → Negative: {"subject": "cat", "relation_type": "under", "object": "mat"}
  → Negative: {"subject": "mat", "relation_type": "on", "object": "cat"} (swap)

✗ WRONG RELATIONAL NEGATIVES (modifying components):
- Original: {"subject": "Leopard", "relation_type": "snarls at", "object": "Camera"}
  → WRONG: {"subject": "Leopard sitting", "relation_type": "looks at", "object": "Camera on stand"}
    (WRONG - added "sitting" and "on stand"!)
  
- Original: {"subject": "red car", "relation_type": "near", "object": "house"}
  → WRONG: {"subject": "blue car", "relation_type": "far from", "object": "house"}
    (WRONG - changed "red" to "blue"!)

GOOD RELATION OPPOSITES:
- Spatial: on ↔ under, left of ↔ right of, inside ↔ outside, near ↔ far from
- Actions: snarls at ↔ ignores, holding ↔ dropping, pushing ↔ pulling
- States: created by ↔ destroyed by OR not created by
- Negations: "not X" is valid

VALIDATION CHECKLIST:
✓ Subject is EXACT component name from extraction
✓ Object is EXACT component name from extraction
✓ Only relation_type changes (or subject/object swap)
✗ NO attribute additions (sitting, standing, framed, etc.)
✗ NO component modifications

Change types:
- "relation_opposite": Use TRUE OPPOSITE relation (snarls at→ignores, on→under)
- "subject_object_swap": Swap subject/object (keeps relation or changes it)
- "relation_type_change": Change to DIFFERENT clear relation

REMEMBER: Components in relational_negatives must be IDENTICAL to extracted components!

===== EXAMPLES =====

EXAMPLE 1: Caption with NO attributes in components
"Leopard Snarls at the Camera Photographic Print by Kim Wolhuter"
{
  "components": ["Leopard", "Camera", "Photographic Print", "Kim Wolhuter"],
  "relations": [
    {"subject": "Leopard", "relation_type": "snarls at", "object": "Camera"},
    {"subject": "Photographic Print", "relation_type": "created by", "object": "Kim Wolhuter"}
  ],
  "component_negatives": {
    "Leopard": [
      {"negative": "Tiger", "change_type": "entity_change"},
      {"negative": "Elephant", "change_type": "entity_change"}
    ],
    "Camera": [
      {"negative": "Phone", "change_type": "object_change"},
      {"negative": "Telescope", "change_type": "object_change"}
    ],
    "Photographic Print": [
      {"negative": "Painting", "change_type": "object_change"},
      {"negative": "Sculpture", "change_type": "object_change"}
    ],
    "Kim Wolhuter": [
      {"negative": "Jane Smith", "change_type": "entity_change"}
    ]
  },
  "binding_pairs": [],
  "relational_negatives": [
    {
      "subject": "Leopard",
      "relation_type": "ignores",
      "object": "Camera",
      "change_type": "relation_opposite",
      "modified_components": [],
      "modified_relations": ["snarls at", "ignores"]
    },
    {
      "subject": "Camera",
      "relation_type": "captures",
      "object": "Leopard",
      "change_type": "subject_object_swap",
      "modified_components": ["Leopard", "Camera"],
      "modified_relations": []
    },
    {
      "subject": "Photographic Print",
      "relation_type": "not created by",
      "object": "Kim Wolhuter",
      "change_type": "relation_opposite",
      "modified_components": [],
      "modified_relations": ["created by", "not created by"]
    }
  ]
}

EXAMPLE 2: Caption WITH attributes - BINDING PAIRS DEMONSTRATION
"Cheerful background with cartoon animals vector image"
{
  "components": ["cheerful background", "cartoon animals", "vector image"],
  "relations": [],
  "component_negatives": {
    "cheerful background": [
      {"negative": "dark background", "change_type": "attribute_change"},
      {"negative": "dramatic background", "change_type": "attribute_change"}
    ],
    "cartoon animals": [
      {"negative": "realistic animals", "change_type": "attribute_change"},
      {"negative": "abstract animals", "change_type": "attribute_change"}
    ],
    "vector image": [
      {"negative": "raster image", "change_type": "attribute_change"},
      {"negative": "photograph", "change_type": "object_change"}
    ]
  },
  "binding_pairs": [
    {
      "component_1": "cartoon background",
      "component_2": "cheerful animals",
      "original_1": "cheerful background",
      "original_2": "cartoon animals",
      "attribute_1": "cartoon",
      "attribute_2": "cheerful"
    },
    {
      "component_1": "vector animals",
      "component_2": "cartoon image",
      "original_1": "cartoon animals",
      "original_2": "vector image",
      "attribute_1": "vector",
      "attribute_2": "cartoon"
    }
  ],
  "relational_negatives": []
}
NOTE: Binding pairs use "cheerful" and "cartoon" from EXTRACTED components, NOT "dark" or "realistic" from component_negatives!

EXAMPLE 3: Places and events (avoid synonyms!)
"Fire at Fuel station"
{
  "components": ["Fire", "Fuel station"],
  "relations": [{"subject": "Fire", "relation_type": "at", "object": "Fuel station"}],
  "component_negatives": {
    "Fire": [
      {"negative": "Smoke", "change_type": "object_change"},
      {"negative": "Water", "change_type": "object_change"}
    ],
    "Fuel station": [
      {"negative": "Supermarket", "change_type": "object_change"},
      {"negative": "Hospital", "change_type": "object_change"}
    ]
  },
  "binding_pairs": [],
  "relational_negatives": [
    {
      "subject": "Fire",
      "relation_type": "near",
      "object": "Fuel station",
      "change_type": "relation_opposite",
      "modified_components": [],
      "modified_relations": ["at", "near"]
    },
    {
      "subject": "Fuel station",
      "relation_type": "contains",
      "object": "Fire",
      "change_type": "subject_object_swap",
      "modified_components": ["Fire", "Fuel station"],
      "modified_relations": []
    }
  ]
}

EXAMPLE 4: Caption WITH attributes in components
"Red car parked near blue house"
{
  "components": ["red car", "blue house"],
  "relations": [{"subject": "red car", "relation_type": "parked near", "object": "blue house"}],
  "component_negatives": {
    "red car": [
      {"negative": "blue car", "change_type": "attribute_change"},
      {"negative": "red bicycle", "change_type": "object_change"}
    ],
    "blue house": [
      {"negative": "red house", "change_type": "attribute_change"},
      {"negative": "blue building", "change_type": "object_change"}
    ]
  },
  "binding_pairs": [
    {
      "component_1": "blue car",
      "component_2": "red house",
      "original_1": "red car",
      "original_2": "blue house",
      "attribute_1": "blue",
      "attribute_2": "red"
    }
  ],
  "relational_negatives": [
    {
      "subject": "red car",
      "relation_type": "far from",
      "object": "blue house",
      "change_type": "relation_opposite",
      "modified_components": [],
      "modified_relations": ["parked near", "far from"]
    },
    {
      "subject": "blue house",
      "relation_type": "near",
      "object": "red car",
      "change_type": "subject_object_swap",
      "modified_components": ["red car", "blue house"],
      "modified_relations": []
    }
  ]
}

EXAMPLE 3: Simple caption
"Two white orchids Royalty Free Stock Photos"
{
  "components": ["white orchids", "stock photos"],
  "relations": [],
  "component_negatives": {
    "white orchids": [
      {"negative": "red orchids", "change_type": "attribute_change"},
      {"negative": "white roses", "change_type": "object_change"}
    ],
    "stock photos": [
      {"negative": "paintings", "change_type": "object_change"}
    ]
  },
  "binding_pairs": [],
  "relational_negatives": []
}

EXAMPLE 4: Spanish caption
"Gato sobre la mesa"
{
  "components": ["gato", "mesa"],
  "relations": [{"subject": "gato", "relation_type": "sobre", "object": "mesa"}],
  "component_negatives": {
    "gato": [
      {"negative": "perro", "change_type": "entity_change"}
    ],
    "mesa": [
      {"negative": "silla", "change_type": "object_change"}
    ]
  },
  "binding_pairs": [],
  "relational_negatives": [
    {
      "subject": "gato",
      "relation_type": "debajo de",
      "object": "mesa",
      "change_type": "relation_opposite",
      "modified_components": [],
      "modified_relations": ["sobre", "debajo de"]
    },
    {
      "subject": "mesa",
      "relation_type": "sobre",
      "object": "gato",
      "change_type": "subject_object_swap",
      "modified_components": ["gato", "mesa"],
      "modified_relations": []
    }
  ]
}

REMEMBER:
- Use the SAME LANGUAGE as the input caption for ALL outputs
- Component negatives: REPLACE components, don't add attributes
- Component negatives: NO SYNONYMS (fuel station ≠ gas station, photo ≠ photograph)
- Component negatives: Same category but meaningfully different (place→different place, animal→different animal)
- Binding pairs: ONLY use attributes from EXTRACTED components (NOT from component_negatives!)
- Binding pairs: Generate BEFORE looking at component_negatives (use only "components" field)
- Binding pairs: If extracted components have NO attributes → empty array
- Relational negatives: ONLY change relations, use EXACT component names
- Generate ALL sections in one JSON object
- Empty arrays are OK if no negatives possible

⚠️ CRITICAL: Binding pairs must use attributes from "components", NOT from "component_negatives"!
Example: If components = ["cheerful background", "cartoon animals"], binding must swap "cheerful"↔"cartoon"
DO NOT use "dark" or "realistic" from component_negatives!"""
    
    # Process in batches
    all_results = []
    num_batches = math.ceil(len(captions) / batch_size)
    
    for batch_idx in tqdm(range(num_batches), desc="Unified generation"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(captions))
        batch_captions = captions[start_idx:end_idx]
        
        # Escape braces in captions
        safe_captions = [cap.replace("{", "{{").replace("}", "}}") for cap in batch_captions]
        
        # MINIMAL user prompts - just the caption
        batch_prompts = safe_captions
        
        # Generate with unified scheme
        batch_outputs = llm.generate(
            batch_prompts,
            max_new_tokens=2000,  # Need more tokens for complete unified output
            temperature=0.0,  # Deterministic for structured output
            repetition_penalty=1.0,
            use_unified_negative_scheme=True,
            system_message=system_message,
            stop=["\n\n", "Input:", "Caption:"]
        )
        
        # Process batch outputs
        for i, output in enumerate(batch_outputs):
            if isinstance(output, dict):
                result = {
                    'original_caption': batch_captions[i],
                    'positive_components': output.get('components', []),
                    'relations': output.get('relations', []),
                    'negative_components': output.get('component_negatives', {}),
                    'binding_pairs': output.get('binding_pairs', []),
                    'negative_relations': output.get('relational_negatives', [])
                }
            else:
                # Fallback for failed parsing
                result = {
                    'original_caption': batch_captions[i],
                    'positive_components': [],
                    'relations': [],
                    'negative_components': {},
                    'binding_pairs': [],
                    'negative_relations': []
                }
            
            all_results.append(result)
    
    # Print statistics
    total_components = sum(len(r['positive_components']) for r in all_results)
    total_relations = sum(len(r['relations']) for r in all_results)
    total_comp_negs = sum(
        sum(len(negs) for negs in r['negative_components'].values())
        for r in all_results
    )
    total_binding_pairs = sum(len(r['binding_pairs']) for r in all_results)
    total_rel_negs = sum(len(r['negative_relations']) for r in all_results)
    
    print(f"\n✓ Unified generation complete for {len(all_results)} captions:")
    print(f"  - {total_components} components")
    print(f"  - {total_relations} relations")
    print(f"  - {total_comp_negs} component negatives")
    print(f"  - {total_binding_pairs} binding pairs")
    print(f"  - {total_rel_negs} relational negatives")
    
    return all_results

# llm_utils_vllm.py
import json
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import pydantic

class NegativeScheme(pydantic.BaseModel):
    attribute_negative: Optional[str] 
    object_negative: Optional[str]
    swap_negative: Optional[str]
    attribute_concept: str

class NegativeWordScheme(pydantic.BaseModel):
    attribute_negative: Optional[str]
    object_negative: Optional[str]
    swap_negative: Optional[List[str]]
    attribute_concept: str

class PositiveCaptionScheme(pydantic.BaseModel):
    components: List[str]
    reconstructed_caption: str

class Relation(pydantic.BaseModel):
    subject: str
    relation_type: str
    object: str
    description: Optional[str] = None

class RelationalCaptionScheme(pydantic.BaseModel):
    components: List[str]
    relations: List[Relation]
    reconstructed_caption: Optional[str] = None

class NegativeVariant(pydantic.BaseModel):
    negative: str  # The alternative component (e.g., "blue car" for "red car")
    change_type: str  # Type of change: "color", "object", "size", "material", "shape"

class BindingPair(pydantic.BaseModel):
    """A pair of components with swapped attributes for testing attribute-object binding."""
    component_1: str = pydantic.Field(description="First component with swapped attribute (e.g., 'blue car')")
    component_2: str = pydantic.Field(description="Second component with swapped attribute (e.g., 'red sky')")
    original_1: str = pydantic.Field(description="Original first component (e.g., 'red car')")
    original_2: str = pydantic.Field(description="Original second component (e.g., 'blue sky')")
    attribute_1: str = pydantic.Field(description="Attribute swapped from component 2 to 1 (e.g., 'blue')")
    attribute_2: str = pydantic.Field(description="Attribute swapped from component 1 to 2 (e.g., 'red')")

class ComponentNegativeScheme(pydantic.BaseModel):
    negative_variants: List[NegativeVariant] = pydantic.Field(default_factory=list)


class RelationNegativeVariant(pydantic.BaseModel):
    """A single negative variant for a relation - either a relation change or swap."""
    relation_type: Optional[str] = pydantic.Field(
        default=None,
        description="New relation type (for antonym/negation changes)"
    )
    swap_subject: Optional[str] = pydantic.Field(
        default=None,
        description="New subject (for subject-object swaps)"
    )
    swap_object: Optional[str] = pydantic.Field(
        default=None,
        description="New object (for subject-object swaps)"
    )
    change_type: str = pydantic.Field(
        description="Type of change: 'antonym', 'negation', or 'swap'"
    )


class RelationWithNegatives(pydantic.BaseModel):
    """An original relation with its negative variants."""
    original_relation: str = pydantic.Field(
        description="The original relation type (e.g., 'is sitting on')"
    )
    original_subject: str = pydantic.Field(
        description="The original subject component"
    )
    original_object: str = pydantic.Field(
        description="The original object component"
    )
    negatives: List[RelationNegativeVariant] = pydantic.Field(
        default_factory=list,
        description="List of negative variants for this relation"
    )


class RelationalNegativeScheme(pydantic.BaseModel):
    """Schema for generating negatives per relation."""
    relation_negatives: List[RelationWithNegatives] = pydantic.Field(
        default_factory=list,
        description="List of relations with their negatives"
    )

class ComponentSwap(pydantic.BaseModel):
    """A single component transformation in a swap."""
    original: str = pydantic.Field(description="Original component (e.g., 'wooden table')")
    modified: str = pydantic.Field(description="Modified component with swapped attribute (e.g., 'red table')")

class SwapPair(pydantic.BaseModel):
    """A pair of component swaps between two objects."""
    swaps: List[ComponentSwap] = pydantic.Field(
        description="Exactly 2 component swaps (one for each object in the pair)",
        min_length=2,
        max_length=2
    )

class AttributeBindingScheme(pydantic.BaseModel):
    """Schema for attribute binding negative generation - generates all possible pairwise swaps."""
    swap_pairs: List[SwapPair] = pydantic.Field(
        description="List of all possible pairwise attribute swaps. Can be empty if no valid swaps.",
        default_factory=list
    )

class UnifiedNegativeScheme(pydantic.BaseModel):
    """Unified schema that extracts components, relations, and generates ALL negatives in ONE call."""
    # Positive extraction (what we see in the image)
    components: List[str] = pydantic.Field(
        description="Visual objects/people/places from the caption"
    )
    relations: List[Relation] = pydantic.Field(
        description="Spatial/action relationships between components",
        default_factory=list
    )
    
    # Component-level negatives (per component)
    component_negatives: Dict[str, List[NegativeVariant]] = pydantic.Field(
        description="Map from component to its negative variants (e.g., {'red car': [{'negative': 'blue car', 'change_type': 'attribute_change'}]})",
        default_factory=dict
    )
    
    # Attribute binding pairs (caption level)
    binding_pairs: List[BindingPair] = pydantic.Field(
        description="Pairs of components with swapped attributes for testing attribute-object binding",
        default_factory=list
    )
    
    # Relational negatives (per-relation format)
    relation_negatives: List[RelationWithNegatives] = pydantic.Field(
        description="Each relation with its negative variants (antonyms, negations, swaps)",
        default_factory=list
    )

class VLLMWrapper:
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        batch_size: int = 4,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        enable_prefix_caching: bool = True,  # Enable automatic prefix caching
        **kwargs
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Enable prefix caching for shared system messages/prompts
        if enable_prefix_caching and 'enable_prefix_caching' not in kwargs:
            kwargs['enable_prefix_caching'] = True
        
        self.llm = LLM(model=model_name, tokenizer=tokenizer, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def apply_chat_template(
        self,
        chat: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True
    ) -> str:
        # Same as before, use tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                chat,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        prompt = ""
        for msg in chat:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        if add_generation_prompt:
            prompt += "Assistant:"
        return prompt

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 0.5,
        num_return_sequences: int = 1,
        use_positive_scheme: bool = False,
        use_component_negative_scheme: bool = False,
        use_relational_scheme: bool = False,
        use_relational_negative_scheme: bool = False,
        use_attribute_binding_scheme: bool = False,
        use_unified_negative_scheme: bool = False,  # NEW: Generate everything in one call
        system_message: str = None,
        **gen_kwargs
    ) -> List[str]:
        # Convert prompts to chat format with shared system message for KV cache efficiency
        if system_message:
            # Use chat template to enable prefix caching with shared system message
            formatted_prompts = []
            for p in prompts:
                chat = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": p}
                ]
                formatted_prompts.append(self.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
            prompts = formatted_prompts
        
        # Select schema based on task
        if use_unified_negative_scheme:
            scheme = GuidedDecodingParams(
                json=UnifiedNegativeScheme.model_json_schema()
            )
        elif use_relational_scheme:
            scheme = GuidedDecodingParams(
                json=RelationalCaptionScheme.model_json_schema()
            )
        elif use_relational_negative_scheme:
            scheme = GuidedDecodingParams(
                json=RelationalNegativeScheme.model_json_schema()
            )
        elif use_attribute_binding_scheme:
            scheme = GuidedDecodingParams(
                json=AttributeBindingScheme.model_json_schema()
            )
        elif use_positive_scheme:
            scheme = GuidedDecodingParams(
                json=PositiveCaptionScheme.model_json_schema()
            )
        elif use_component_negative_scheme:
            scheme = GuidedDecodingParams(
                json=ComponentNegativeScheme.model_json_schema()
            )
        else:
            scheme = GuidedDecodingParams(
                json=NegativeWordScheme.model_json_schema()
            )
        
        # vLLM's SamplingParams replaces do_sample, temperature, etc.
        # Note: repetition_penalty can cause degenerate outputs with guided JSON, so use carefully
        sampling_params = SamplingParams(
            n=num_return_sequences,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=gen_kwargs.get("top_p", 0.95),  # Slightly reduced for more focused generation
            top_k=gen_kwargs.get("top_k", -1),
            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.03),  # Disabled - can cause empty outputs with guided decoding
            stop=gen_kwargs.get("stop", None),
            guided_decoding=scheme,
            skip_special_tokens=True,  # Help prevent truncation issues
        )
        outputs = self.llm.generate(prompts, sampling_params)


        # outputs is a list of RequestOutput; extract .outputs[0].text per prompt
        results = []

        # Parse the outputs as json and return a list of dict
        for idx, out in enumerate(outputs):
            raw_text = out.outputs[0].text.strip()
            try:
                result = json.loads(raw_text)
                
                # Debug: Show raw output for first few samples
                if idx < 3 and use_relational_scheme:
                    print(f"DEBUG LLM[{idx}]: Raw output = {raw_text[:200]}")
                    print(f"DEBUG LLM[{idx}]: Parsed components = {result.get('components', [])}")
                
                # Validate and clean the result - reject outputs with placeholder content
                if use_relational_scheme and 'components' in result:
                    original_components = result.get('components', [])
                    # Filter out placeholder/empty components  
                    # Note: We check c.strip() to handle whitespace-only strings
                    valid_components = [
                        c for c in original_components
                        if c and isinstance(c, str) and c.strip() not in ['', '...', '.', ' ']
                    ]
                    
                    # Debug filtering
                    if idx < 3:
                        if len(original_components) != len(valid_components):
                            print(f"DEBUG LLM[{idx}]: Filtered {len(original_components) - len(valid_components)} placeholder components")
                            print(f"DEBUG LLM[{idx}]: Original: {original_components}")
                            print(f"DEBUG LLM[{idx}]: Valid: {valid_components}")
                        
                        # Warn about empty string generation
                        if any(c == '' or (isinstance(c, str) and c.strip() == '') for c in original_components):
                            print(f"⚠ LLM generated EMPTY STRINGS - this indicates sampling issues (temp/penalties)")
                    
                    result['components'] = valid_components
                    
                    # Filter out placeholder relations
                    valid_relations = []
                    for rel in result.get('relations', []):
                        if isinstance(rel, dict):
                            subj = rel.get('subject', '')
                            obj = rel.get('object', '')
                            rel_type = rel.get('relation_type', '')
                            # Check if any field is a placeholder
                            if all(
                                field and isinstance(field, str) and field.strip() not in ['', '...', '.', ' ']
                                for field in [subj, obj, rel_type]
                            ):
                                valid_relations.append(rel)
                    result['relations'] = valid_relations
                    
                    # If all components were placeholders, return empty structure
                    if not valid_components:
                        result = {"components": [], "relations": []}
                
                elif use_relational_negative_scheme and 'relation_negatives' in result:
                    # Validate relation negatives (new nested format)
                    valid_relation_negs = []
                    for rel_neg in result.get('relation_negatives', []):
                        if isinstance(rel_neg, dict):
                            orig_rel = rel_neg.get('original_relation', '')
                            orig_subj = rel_neg.get('original_subject', '')
                            orig_obj = rel_neg.get('original_object', '')
                            negatives = rel_neg.get('negatives', [])
                            
                            # Validate original relation fields
                            if (orig_rel and isinstance(orig_rel, str) and orig_rel.strip() and
                                orig_subj and isinstance(orig_subj, str) and orig_subj.strip() and
                                orig_obj and isinstance(orig_obj, str) and orig_obj.strip()):
                                
                                # Validate each negative variant
                                valid_negatives = []
                                for neg in negatives:
                                    if isinstance(neg, dict):
                                        change_type = neg.get('change_type', '')
                                        if change_type in ['antonym', 'negation', 'swap']:
                                            valid_negatives.append(neg)
                                
                                rel_neg['negatives'] = valid_negatives
                                valid_relation_negs.append(rel_neg)
                    
                    result['relation_negatives'] = valid_relation_negs
                    
                    # Debug output for first few samples
                    if idx < 3:
                        print(f"DEBUG REL_NEG[{idx}]: Raw output = {raw_text[:300]}")
                        print(f"DEBUG REL_NEG[{idx}]: Valid relation_negatives = {len(valid_relation_negs)}")
                
                elif use_attribute_binding_scheme and 'binding_negatives' in result:
                    # Filter out placeholder attribute binding negatives
                    valid_bindings = []
                    for binding in result.get('binding_negatives', []):
                        if isinstance(binding, dict):
                            negative_caption = binding.get('negative_caption', '')
                            change_type = binding.get('change_type', '')
                            swapped_attributes = binding.get('swapped_attributes', [])
                            
                            # Check if all required fields are valid (not placeholders)
                            if (negative_caption and isinstance(negative_caption, str) and negative_caption.strip() not in ['', '...', '.', ' '] and
                                change_type and isinstance(change_type, str) and change_type.strip() not in ['', '...', '.', ' '] and
                                swapped_attributes and isinstance(swapped_attributes, list) and len(swapped_attributes) > 0):
                                
                                # Validate swapped attributes
                                valid_swaps = []
                                for swap in swapped_attributes:
                                    if isinstance(swap, dict):
                                        original = swap.get('original', '')
                                        negative = swap.get('negative', '')
                                        if (original and isinstance(original, str) and original.strip() not in ['', '...', '.', ' '] and
                                            negative and isinstance(negative, str) and negative.strip() not in ['', '...', '.', ' '] and
                                            original != negative):
                                            valid_swaps.append(swap)
                                
                                if valid_swaps:
                                    binding['swapped_attributes'] = valid_swaps
                                    valid_bindings.append(binding)
                    
                    result['binding_negatives'] = valid_bindings
                    
                    # Debug output for first few samples
                    if idx < 3:
                        print(f"DEBUG BINDING[{idx}]: Raw output = {raw_text[:300]}")
                        print(f"DEBUG BINDING[{idx}]: Valid bindings = {len(valid_bindings)}")

                
                # Debug: Check component negatives
                if use_component_negative_scheme:
                    if idx < 3:
                        print(f"DEBUG COMP_NEG[{idx}]: Raw output = {raw_text[:300]}")
                        print(f"DEBUG COMP_NEG[{idx}]: Parsed keys = {list(result.keys())}")
                        print(f"DEBUG COMP_NEG[{idx}]: negative_variants = {result.get('negative_variants', 'KEY_MISSING')}")
                    
                    if 'negative_variants' not in result:
                        print(f"⚠ Component negative output missing 'negative_variants' key!")
                        print(f"  Keys found: {list(result.keys())}")
                        print(f"  Full output: {raw_text[:200]}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from output: {raw_text[:500]}")
                print(f"  Error: {e}")
                # Try to repair incomplete JSON by adding closing braces
                if not raw_text.endswith('}'):
                    repaired = raw_text
                    open_braces = raw_text.count('{')
                    close_braces = raw_text.count('}')
                    open_brackets = raw_text.count('[')
                    close_brackets = raw_text.count(']')
                    
                    missing_braces = open_braces - close_braces
                    missing_brackets = open_brackets - close_brackets
                    
                    if missing_braces > 0 or missing_brackets > 0:
                        # Add missing closing brackets first, then braces
                        repaired = raw_text + (']' * missing_brackets) + ('}' * missing_braces)
                        try:
                            print(f"  Attempting JSON repair (adding {missing_brackets} ']', {missing_braces} '}}')...")
                            result = json.loads(repaired)
                            print("  ✓ JSON repair successful!")
                        except Exception:
                            print("  ✗ JSON repair failed")
                            result = None
                    else:
                        result = None
                else:
                    result = None
                
                # Provide fallback empty structures
                if result is None:
                    if use_relational_scheme:
                        result = {
                            "components": [],
                            "relations": [],
                            "reconstructed_caption": ""
                        }
                    elif use_relational_negative_scheme:
                        result = {
                            "relation_negatives": []
                        }
                    elif use_positive_scheme:
                        result = {
                            "components": [],
                            "reconstructed_caption": ""
                        }
                    elif use_component_negative_scheme:
                        result = {
                            "negative_variants": []
                        }
                    else:
                        result = {
                            "attribute_negative": None,
                            "object_negative": None,
                            "swap_negative": None,
                            "attribute_concept": None
                        }
            results.append(result)

        return results

    def chat(
        self,
        chat_messages: List[List[Dict[str, str]]],
        max_new_tokens: int = 40,
        do_sample: bool = True,
        temperature: float = 0.8,
        num_return_sequences: int = 1,
        add_generation_prompt: bool = True,
        **gen_kwargs
    ) -> List[str]:
        prompts = [
            self.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            ) for chat in chat_messages
        ]
        return self.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            **gen_kwargs
        )

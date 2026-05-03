import json
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import pydantic


class Relation(pydantic.BaseModel):
    subject: str
    relation_type: str
    object: str
    description: Optional[str] = None


class EntityRelationScheme(pydantic.BaseModel):
    entities: List[str]
    relations: List[Relation]


class NegativeVariant(pydantic.BaseModel):
    negative: str
    change_type: str


class EntityNegativeScheme(pydantic.BaseModel):
    negative_variants: List[NegativeVariant] = pydantic.Field(default_factory=list)


class RelationNegativeVariant(pydantic.BaseModel):
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
    original_relation: str = pydantic.Field(
        description="The original relation type (e.g., 'is sitting on')"
    )
    original_subject: str = pydantic.Field(
        description="The original subject entity"
    )
    original_object: str = pydantic.Field(
        description="The original object entity"
    )
    negatives: List[RelationNegativeVariant] = pydantic.Field(
        default_factory=list,
        description="List of negative variants for this relation"
    )


class RelationalNegativeScheme(pydantic.BaseModel):
    relation_negatives: List[RelationWithNegatives] = pydantic.Field(
        default_factory=list,
        description="List of relations with their negatives"
    )


class VLLMWrapper:
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        batch_size: int = 4,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        enable_prefix_caching: bool = True,
        **kwargs
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        
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
        temperature: float = 0.5,
        num_return_sequences: int = 1,
        use_entity_negative_scheme: bool = False,
        use_relational_scheme: bool = False,
        use_relational_negative_scheme: bool = False,
        system_message: str = None,
        **gen_kwargs
    ) -> List[str]:
        if system_message:
            formatted_prompts = []
            for p in prompts:
                chat = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": p}
                ]
                formatted_prompts.append(self.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
            prompts = formatted_prompts
        
        if use_relational_scheme:
            scheme = StructuredOutputsParams(
                json=EntityRelationScheme.model_json_schema()
            )
        elif use_relational_negative_scheme:
            scheme = StructuredOutputsParams(
                json=RelationalNegativeScheme.model_json_schema()
            )
        elif use_entity_negative_scheme:
            scheme = StructuredOutputsParams(
                json=EntityNegativeScheme.model_json_schema()
            )
        else:
            scheme = None
        
        sampling_params = SamplingParams(
            n=num_return_sequences,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=gen_kwargs.get("top_p", 0.95),
            top_k=gen_kwargs.get("top_k", -1),
            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.0),
            stop=gen_kwargs.get("stop", None),
            structured_outputs=scheme,
            skip_special_tokens=True,
        )
        outputs = self.llm.generate(prompts, sampling_params)

        results = []

        for idx, out in enumerate(outputs):
            raw_text = out.outputs[0].text.strip()
            try:
                result = json.loads(raw_text)
                
                if use_relational_scheme:
                    original_entities = result.get('entities', [])
                    valid_entities = [
                        c for c in original_entities
                        if c and isinstance(c, str) and c.strip() not in ['', '...', '.', ' ']
                    ]
                    result['entities'] = valid_entities

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

                    if not valid_entities:
                        result = {"entities": [], "relations": []}
                
                elif use_relational_negative_scheme and 'relation_negatives' in result:
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

                if use_entity_negative_scheme:
                    if 'negative_variants' not in result:
                        result['negative_variants'] = []
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from output: {raw_text[:500]}")
                print(f"  Error: {e}")
                if not raw_text.endswith('}'):
                    repaired = raw_text
                    open_braces = raw_text.count('{')
                    close_braces = raw_text.count('}')
                    open_brackets = raw_text.count('[')
                    close_brackets = raw_text.count(']')
                    
                    missing_braces = open_braces - close_braces
                    missing_brackets = open_brackets - close_brackets
                    
                    if missing_braces > 0 or missing_brackets > 0:
                        repaired = raw_text + (']' * missing_brackets) + ('}' * missing_braces)
                        try:
                            print(f"  Attempting JSON repair (adding {missing_brackets} ']', {missing_braces} '}}')...")
                            result = json.loads(repaired)
                            print("  JSON repair successful")
                        except Exception:
                            print("  JSON repair failed")
                            result = None
                    else:
                        result = None
                else:
                    result = None
                
                if result is None:
                    if use_relational_scheme:
                        result = {
                            "entities": [],
                            "relations": []
                        }
                    elif use_relational_negative_scheme:
                        result = {
                            "relation_negatives": []
                        }
                    elif use_entity_negative_scheme:
                        result = {
                            "negative_variants": []
                        }
                    else:
                        result = {}
            results.append(result)

        return results

    def chat(
        self,
        chat_messages: List[List[Dict[str, str]]],
        max_new_tokens: int = 40,
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
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            **gen_kwargs
        )

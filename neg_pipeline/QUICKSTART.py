"""
Quick Reference: Positive Caption Generation

This module adds positive caption generation to the hard negative pipeline.

WHAT IT DOES:
=============
Extracts visual components from captions and reconstructs them.

Example:
    Input:  "Red car driving on a motorway and children crossing the street"
    Output: "red car driving and motorway and children crossing the street"
    
    Components extracted:
    - red car driving
    - motorway  
    - children crossing the street

HOW TO USE:
===========

1. Basic usage (positives only):
   
   python -m neg_pipeline.main --generate_positives

2. With custom output file:
   
   python -m neg_pipeline.main --generate_positives --positives_output my_positives.json

3. With input file:
   
   python -m neg_pipeline.main --input captions.csv --generate_positives

4. Combined with negatives:
   
   python -m neg_pipeline.main \
       --use_concept_detection \
       --use_object_replacement \
       --generate_positives \
       --output negatives.json \
       --positives_output positives.json

5. With WebDataset:
   
   python -m neg_pipeline.main \
       --shards '/path/to/{00000..00100}.tar' \
       --generate_positives \
       --subset 0 10000

KEY FEATURES:
=============
✓ LLM-based extraction (more reliable than spaCy)
✓ Visual component focus (ignores abstract/grammatical words)
✓ Batched processing for efficiency
✓ JSON schema validation
✓ Works with all input formats (CSV, TSV, TXT, WebDataset)
✓ Preserves metadata (URLs, keys, paths)

OUTPUT FORMAT:
==============
{
    "sample_id": "...",
    "original_caption": "...",
    "components": ["component1", "component2", ...],
    "positive_caption": "component1 and component2 and ...",
    "num_components": 3
}

ARGUMENTS:
==========
--generate_positives       Enable positive generation
--positives_output FILE    Output file (default: positives.json)
--llm_name MODEL          LLM model name
--llm_batch N             Batch size for LLM

See POSITIVE_GENERATION_README.md for full documentation.
"""

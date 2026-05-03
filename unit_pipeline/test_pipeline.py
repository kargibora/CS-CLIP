"""
Quick local smoke test for the generation pipeline.
Mocks the LLM so no GPU or vLLM installation is required.

Run from the repo root:
    python -m neg_pipeline.test_pipeline
"""

import json
import os
import tempfile
import unittest

from unit_pipeline.generation import (
    extract_entities_and_relations_batched,
    generate_entity_negatives_batched,
    generate_relational_negatives_batched,
)
from unit_pipeline.main import (
    build_positive_entries,
    build_structured_entries,
)


class MockLLM:
    """Minimal LLM stub that returns hard-coded valid JSON structures."""

    def generate(self, prompts, **kwargs):
        results = []
        for _ in prompts:
            if kwargs.get("use_relational_scheme"):
                results.append({
                    "entities": ["man", "red umbrella"],
                    "relations": [{"subject": "man", "relation_type": "is holding", "object": "red umbrella"}],
                })
            elif kwargs.get("use_entity_negative_scheme"):
                results.append({
                    "negative_variants": [
                        {"negative": "woman", "change_type": "attribute_change"},
                        {"negative": "blue umbrella", "change_type": "attribute_change"},
                    ]
                })
            elif kwargs.get("use_relational_negative_scheme"):
                results.append({
                    "relation_negatives": [{
                        "original_relation": "is holding",
                        "original_subject": "man",
                        "original_object": "red umbrella",
                        "negatives": [
                            {"relation_type": "is dropping", "change_type": "antonym"},
                            {"relation_type": "is not holding", "change_type": "negation"},
                        ],
                    }]
                })
            else:
                results.append({})
        return results


CAPTIONS = [
    "A man holding a red umbrella",
    "Cat sitting on the wooden table",
]


class TestExtractionStep(unittest.TestCase):
    def setUp(self):
        self.llm = MockLLM()

    def test_extract_entities_and_relations(self):
        results = extract_entities_and_relations_batched(self.llm, CAPTIONS, batch_size=2)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("entities", r)
            self.assertIn("relations", r)
            self.assertIsInstance(r["entities"], list)
            self.assertIsInstance(r["relations"], list)

    def test_entity_negatives(self):
        relational_data = extract_entities_and_relations_batched(self.llm, CAPTIONS, batch_size=2)
        entities_list = [r["entities"] for r in relational_data]
        neg_results = generate_entity_negatives_batched(
            self.llm, entities_list, CAPTIONS, batch_size=2
        )
        self.assertEqual(len(neg_results), 2)
        for r in neg_results:
            self.assertIn("negative_entities", r)
            self.assertIsInstance(r["negative_entities"], dict)

    def test_relational_negatives(self):
        relational_data = extract_entities_and_relations_batched(self.llm, CAPTIONS, batch_size=2)
        rel_neg_results = generate_relational_negatives_batched(
            self.llm, relational_data, batch_size=2
        )
        self.assertIsInstance(rel_neg_results, list)

    def test_full_pipeline(self):
        """End-to-end: extract → entity negs → relation negs → build structured entries."""
        captions_meta = [
            {"caption": cap, "sample_id": f"test_{i}", "image_path": f"images/{i}.jpg"}
            for i, cap in enumerate(CAPTIONS)
        ]
        caption_texts = [c["caption"] for c in captions_meta]

        relational_data = extract_entities_and_relations_batched(self.llm, caption_texts, batch_size=2)
        entity_negatives_data = generate_entity_negatives_batched(
            self.llm,
            entities_list=[r["entities"] for r in relational_data],
            original_captions=caption_texts,
            batch_size=2,
        )
        relational_negatives_data = generate_relational_negatives_batched(
            self.llm, relational_data, batch_size=2
        )

        structured = build_structured_entries(
            captions=captions_meta,
            relational_data=relational_data,
            entity_negatives_data=entity_negatives_data,
            relational_negatives_data=relational_negatives_data,
        )

        self.assertGreater(len(structured), 0)
        sample = structured[0]
        self.assertIn("sample_id", sample)
        self.assertIn("original_caption", sample)
        self.assertIn("entities", sample)
        self.assertIn("negative_entities", sample)
        self.assertIn("relations", sample)
        self.assertIn("image_path", sample)

    def test_output_json_serialisable(self):
        """Ensure the output can be dumped to JSON without errors."""
        captions_meta = [
            {"caption": cap, "sample_id": f"test_{i}", "image_path": f"images/{i}.jpg"}
            for i, cap in enumerate(CAPTIONS)
        ]
        caption_texts = [c["caption"] for c in captions_meta]

        relational_data = extract_entities_and_relations_batched(self.llm, caption_texts, batch_size=2)
        entity_negatives_data = generate_entity_negatives_batched(
            self.llm,
            entities_list=[r["entities"] for r in relational_data],
            original_captions=caption_texts,
            batch_size=2,
        )
        relational_negatives_data = generate_relational_negatives_batched(
            self.llm, relational_data, batch_size=2
        )
        structured = build_structured_entries(
            captions=captions_meta,
            relational_data=relational_data,
            entity_negatives_data=entity_negatives_data,
            relational_negatives_data=relational_negatives_data,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(structured, f, indent=2)
            tmp_path = f.name

        with open(tmp_path) as f:
            loaded = json.load(f)
        os.unlink(tmp_path)

        self.assertEqual(len(loaded), len(structured))


if __name__ == "__main__":
    unittest.main(verbosity=2)

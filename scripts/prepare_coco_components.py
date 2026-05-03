#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script to create COCO component annotations from LLM-parsed data.

This shows how to structure your COCO annotations for use with COCOComponentsDataset.
Adapt this to your LLM parsing pipeline.

Expected input: Your LLM-parsed COCO annotations with components/relations
Expected output: JSON file compatible with COCOComponentsDataset

Usage:
    python scripts/prepare_coco_components.py \\
        --input_json your_llm_parsed_coco.json \\
        --output_json coco_with_components.json \\
        --image_root /path/to/coco/images
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List


def create_example_json():
    """
    Create an example JSON structure showing the expected format.
    """
    example = {
        "images": [
            {
                "image_id": 123456,
                "filepath": "train2014/COCO_train2014_000000123456.jpg",
                "split": "train",  # or "val", "test"
                "captions": [
                    {
                        "caption": "a dog running in the park",
                        "components": ["dog", "park"],
                        "relations": [
                            {
                                "subject": "dog",
                                "relation": "in",
                                "object": "park"
                            },
                            {
                                "subject": "dog",
                                "relation": "running",
                                "object": ""
                            }
                        ]
                    },
                    {
                        "caption": "a brown dog playing outside",
                        "components": ["dog", "outside"],
                        "relations": [
                            {
                                "subject": "dog",
                                "relation": "playing",
                                "object": "outside"
                            }
                        ]
                    }
                ],
                "negative_components": ["cat", "beach", "car", "building"],
                "negative_relations": [
                    {
                        "subject": "dog",
                        "relation": "on",
                        "object": "beach"
                    },
                    {
                        "subject": "cat",
                        "relation": "in",
                        "object": "park"
                    }
                ],
                "swap_negatives": [
                    "a cat running in the park",
                    "a dog running on the beach"
                ]
            },
            {
                "image_id": 654321,
                "filepath": "val2014/COCO_val2014_000000654321.jpg",
                "split": "val",
                "captions": [
                    {
                        "caption": "a person riding a bicycle",
                        "components": ["person", "bicycle"],
                        "relations": [
                            {
                                "subject": "person",
                                "relation": "riding",
                                "object": "bicycle"
                            }
                        ]
                    }
                ],
                "negative_components": ["car", "motorcycle", "bus"],
                "negative_relations": [
                    {
                        "subject": "person",
                        "relation": "driving",
                        "object": "car"
                    }
                ],
                "swap_negatives": [
                    "a person riding a motorcycle"
                ]
            }
        ]
    }
    
    return example


def parse_llm_annotations_to_components(
    llm_json_path: str,
    output_json_path: str,
    image_root: str,
    use_karpathy_splits: bool = True
):
    """
    Convert your LLM-parsed COCO annotations to COCOComponentsDataset format.
    
    Adapt this function to match your LLM output format.
    
    Args:
        llm_json_path: Path to your LLM-parsed annotations
        output_json_path: Path to save formatted annotations
        image_root: Root directory for COCO images
        use_karpathy_splits: Whether to add Karpathy split assignments
    """
    print(f"Loading LLM annotations from {llm_json_path}")
    
    with open(llm_json_path, 'r') as f:
        llm_data = json.load(f)
    
    # Initialize output structure
    output_data = {"images": []}
    
    # Process each image
    # NOTE: Adapt this section to match your LLM output format!
    for img_data in llm_data.get("images", []):
        # Extract image info
        image_id = img_data["image_id"]
        filepath = img_data.get("filepath") or img_data.get("file_name")
        
        # Extract captions with components
        captions_data = []
        for cap_data in img_data.get("captions", []):
            caption_entry = {
                "caption": cap_data["caption"],
                "components": cap_data.get("components", []),
                "relations": cap_data.get("relations", [])
            }
            captions_data.append(caption_entry)
        
        # Extract negatives
        negative_components = img_data.get("negative_components", [])
        negative_relations = img_data.get("negative_relations", [])
        swap_negatives = img_data.get("swap_negatives", [])
        
        # Determine split (if not already specified)
        split = img_data.get("split", "train")
        if use_karpathy_splits:
            # You can implement Karpathy split logic here
            # Or pre-assign splits in your LLM parsing
            pass
        
        # Create formatted entry
        formatted_entry = {
            "image_id": image_id,
            "filepath": filepath,
            "split": split,
            "captions": captions_data,
            "negative_components": negative_components,
            "negative_relations": negative_relations,
            "swap_negatives": swap_negatives
        }
        
        output_data["images"].append(formatted_entry)
    
    # Save formatted annotations
    print(f"Saving formatted annotations to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Processed {len(output_data['images'])} images")
    
    # Print statistics
    total_components = sum(
        len(cap["components"])
        for img in output_data["images"]
        for cap in img["captions"]
    )
    total_relations = sum(
        len(cap["relations"])
        for img in output_data["images"]
        for cap in img["captions"]
    )
    
    print(f"  Total components: {total_components}")
    print(f"  Total relations: {total_relations}")
    print(f"  Avg components per image: {total_components / len(output_data['images']):.2f}")
    print(f"  Avg relations per image: {total_relations / len(output_data['images']):.2f}")


def assign_karpathy_splits(
    json_path: str,
    output_path: str,
    train_size: int = 113287,
    val_size: int = 5000,
    test_size: int = 5000,
    seed: int = 42
):
    """
    Assign Karpathy splits to COCO annotations.
    
    Args:
        json_path: Path to COCO annotations JSON
        output_path: Path to save annotations with splits
        train_size: Number of training images
        val_size: Number of validation images
        test_size: Number of test images
        seed: Random seed
    """
    import random
    random.seed(seed)
    
    print(f"Loading annotations from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Shuffle images
    images = data["images"]
    random.shuffle(images)
    
    # Assign splits
    for i, img in enumerate(images):
        if i < train_size:
            img["split"] = "train"
        elif i < train_size + val_size:
            img["split"] = "val"
        elif i < train_size + val_size + test_size:
            img["split"] = "test"
        else:
            img["split"] = "train"  # Extra images go to train
    
    # Save with splits
    print(f"Saving annotations with splits to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Print split statistics
    split_counts = defaultdict(int)
    for img in images:
        split_counts[img["split"]] += 1
    
    print("\nSplit statistics:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count} images")


def validate_coco_components_json(json_path: str):
    """
    Validate that COCO components JSON has correct structure.
    
    Args:
        json_path: Path to JSON file to validate
    """
    print(f"Validating {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check top-level structure
    assert "images" in data, "Missing 'images' key"
    assert isinstance(data["images"], list), "'images' must be a list"
    
    print(f"✓ Found {len(data['images'])} images")
    
    # Check first image structure
    if len(data["images"]) > 0:
        img = data["images"][0]
        
        required_keys = ["image_id", "filepath", "split", "captions"]
        for key in required_keys:
            assert key in img, f"Missing required key '{key}' in image"
        
        # Check caption structure
        assert isinstance(img["captions"], list), "'captions' must be a list"
        assert len(img["captions"]) > 0, "Each image must have at least one caption"
        
        cap = img["captions"][0]
        assert "caption" in cap, "Caption missing 'caption' text"
        assert "components" in cap, "Caption missing 'components' list"
        assert "relations" in cap, "Caption missing 'relations' list"
        
        print(f"✓ First image structure is valid")
        print(f"  Image ID: {img['image_id']}")
        print(f"  Filepath: {img['filepath']}")
        print(f"  Split: {img['split']}")
        print(f"  Captions: {len(img['captions'])}")
        print(f"  Components: {cap['components']}")
        print(f"  Relations: {len(cap['relations'])}")
    
    print("\n✅ Validation passed!")


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO components annotations")
    parser.add_argument("--action", type=str, required=True,
                       choices=["example", "parse", "split", "validate"],
                       help="Action to perform")
    parser.add_argument("--input_json", type=str, help="Input JSON path")
    parser.add_argument("--output_json", type=str, help="Output JSON path")
    parser.add_argument("--image_root", type=str, help="COCO images root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.action == "example":
        # Create example JSON
        example = create_example_json()
        output_path = args.output_json or "coco_components_example.json"
        with open(output_path, 'w') as f:
            json.dump(example, f, indent=2)
        print(f"✓ Created example JSON: {output_path}")
        print("\nExample structure:")
        print(json.dumps(example, indent=2))
    
    elif args.action == "parse":
        # Parse LLM annotations
        if not args.input_json or not args.output_json:
            print("Error: --input_json and --output_json required for 'parse' action")
            return
        
        parse_llm_annotations_to_components(
            args.input_json,
            args.output_json,
            args.image_root or ""
        )
    
    elif args.action == "split":
        # Assign Karpathy splits
        if not args.input_json or not args.output_json:
            print("Error: --input_json and --output_json required for 'split' action")
            return
        
        assign_karpathy_splits(
            args.input_json,
            args.output_json,
            seed=args.seed
        )
    
    elif args.action == "validate":
        # Validate JSON structure
        if not args.input_json:
            print("Error: --input_json required for 'validate' action")
            return
        
        validate_coco_components_json(args.input_json)


if __name__ == "__main__":
    main()

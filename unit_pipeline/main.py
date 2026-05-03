import argparse
import json
import os

from tqdm import tqdm

from .llm_utils import VLLMWrapper
from .generation import (
    extract_entities_and_relations_batched,
    generate_entity_negatives_batched,
    generate_relational_negatives_batched,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="COCO-only structured entity and relation generation pipeline"
    )
    parser.add_argument(
        "--coco_karpathy",
        type=str,
        required=True,
        help="Path to COCO Karpathy split JSON file",
    )
    parser.add_argument(
        "--coco_images_root",
        type=str,
        required=True,
        help="Root path for COCO images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON for structured negatives",
    )
    parser.add_argument(
        "--positives_output",
        type=str,
        default=None,
        help="Optional output JSON for extracted positive entities/relations",
    )
    parser.add_argument(
        "--coco_split",
        type=str,
        default=None,
        choices=["train", "val", "test", "restval", None],
        help="COCO split to use",
    )
    parser.add_argument(
        "--subset",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Subset the loaded captions with start and end indices",
    )
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen3-14B-AWQ")
    parser.add_argument("--llm_batch", type=int, default=256)
    parser.add_argument(
        "--use_relational_extraction",
        action="store_true",
        default=True,
        help="Extract entity and relation units from captions (default: True)",
    )
    parser.add_argument(
        "--use_relational_negatives",
        action="store_true",
        default=True,
        help="Generate negatives for each relation unit (default: True)",
    )
    parser.add_argument(
        "--n_neg_per_entity",
        type=int,
        default=2,
        help="Number of negative variants to request per entity",
    )
    parser.add_argument(
        "--n_relational_negatives",
        type=int,
        default=3,
        help="Number of negatives to request per relation",
    )
    return parser.parse_args()


def load_coco_captions(coco_json_path, images_root, coco_split):
    print(f"Loading COCO Karpathy split from: {coco_json_path}")
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    captions = []
    for image in tqdm(coco_data["images"], desc="Loading COCO captions"):
        if coco_split is not None and image["split"] != coco_split:
            continue

        image_path = os.path.join(images_root, image["filepath"], image["filename"])
        imgid = image["imgid"]

        for sentence in image["sentences"]:
            sentid = sentence["sentid"]
            captions.append(
                {
                    "caption": sentence["raw"].strip(),
                    "sample_id": f"coco_{imgid}_{sentid}",
                    "image_path": image_path,
                }
            )

    print(f"Loaded {len(captions)} captions from COCO")
    if coco_split is not None:
        print(f"  (filtered to split='{coco_split}')")
    return captions


def apply_subset(captions, subset):
    if not subset:
        return captions
    start, end = subset
    return captions[start:end]


def build_positive_entries(captions, relational_data):
    positive_entries = []
    for caption_row, extracted in zip(captions, relational_data):
        positive_entries.append(
            {
                "sample_id": caption_row["sample_id"],
                "original_caption": caption_row["caption"],
                "entities": extracted.get("entities", []),
                "relations": extracted.get("relations", []),
                "image_path": caption_row["image_path"],
            }
        )
    return positive_entries


def build_structured_entries(
    captions,
    relational_data,
    entity_negatives_data,
    relational_negatives_data,
):
    relational_by_index = {
        entry.get("original_index"): entry for entry in relational_negatives_data
    }

    structured_entries = []
    skipped_empty = 0
    skipped_placeholder = 0

    for index, caption_row in enumerate(captions):
        entities = relational_data[index].get("entities", [])
        if not entities:
            skipped_empty += 1
            continue

        has_placeholder = any(
            entity in ["...", ".", "", " "]
            or (isinstance(entity, str) and entity.strip() in ["", "...", "."])
            for entity in entities
        )
        if has_placeholder:
            skipped_placeholder += 1
            continue

        entry = {
            "sample_id": caption_row["sample_id"],
            "original_caption": caption_row["caption"],
            "entities": entities,
            "negative_entities": entity_negatives_data[index].get("negative_entities", {}),
            "image_path": caption_row["image_path"],
        }

        relation_entry = relational_by_index.get(index)
        if relation_entry and relation_entry.get("relations"):
            entry["relations"] = relation_entry["relations"]
        else:
            entry["relations"] = relational_data[index].get("relations", [])
            for relation in entry["relations"]:
                relation.setdefault("negatives", [])

        structured_entries.append(entry)

    print(
        f"Structured output: {len(structured_entries)} samples "
        f"(skipped {skipped_empty} empty, {skipped_placeholder} placeholder)"
    )
    return structured_entries


def main():
    args = parse_args()

    captions = load_coco_captions(
        coco_json_path=args.coco_karpathy,
        images_root=args.coco_images_root,
        coco_split=args.coco_split,
    )
    captions = apply_subset(captions, args.subset)
    print(f"Using {len(captions)} captions after subset filtering")

    llm = VLLMWrapper(
        model_name=args.llm_name,
        batch_size=args.llm_batch,
    )

    caption_texts = [caption["caption"] for caption in captions]

    print("\n" + "=" * 80)
    print("STEP 1: EXTRACT ENTITIES AND RELATIONS")
    print("=" * 80)
    relational_data = extract_entities_and_relations_batched(
        llm,
        captions=caption_texts,
        batch_size=args.llm_batch,
    )

    print("\n" + "=" * 80)
    print("STEP 2: GENERATE ENTITY NEGATIVES")
    print("=" * 80)
    entity_negatives_data = generate_entity_negatives_batched(
        llm,
        entities_list=[item["entities"] for item in relational_data],
        original_captions=caption_texts,
        batch_size=args.llm_batch,
        n_neg_per_entity=args.n_neg_per_entity,
    )

    relational_negatives_data = []
    if args.use_relational_negatives:
        print("\n" + "=" * 80)
        print("STEP 3: GENERATE RELATIONAL NEGATIVES")
        print("=" * 80)
        relational_negatives_data = generate_relational_negatives_batched(
            llm,
            relational_data=relational_data,
            batch_size=args.llm_batch,
            n_negatives=args.n_relational_negatives,
        )

    structured_entries = build_structured_entries(
        captions=captions,
        relational_data=relational_data,
        entity_negatives_data=entity_negatives_data,
        relational_negatives_data=relational_negatives_data,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(structured_entries, f, indent=2)
    print(f"Saved structured negatives to: {args.output}")

    if args.positives_output:
        positive_entries = build_positive_entries(captions, relational_data)
        os.makedirs(os.path.dirname(args.positives_output) or ".", exist_ok=True)
        with open(args.positives_output, "w") as f:
            json.dump(positive_entries, f, indent=2)
        print(f"Saved positive extraction output to: {args.positives_output}")


if __name__ == "__main__":
    main()

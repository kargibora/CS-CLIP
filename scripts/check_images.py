import json
import os
import argparse
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_valid_image(img_path):
    try:
        with Image.open(img_path) as im:
            im.verify()  # Verify image integrity
        return True
    except Exception as e:
        return False

def main():
    parser = argparse.ArgumentParser(description="Remove JSON samples whose image cannot be opened.")
    parser.add_argument("--input_json", required=True, help="Input JSON file (list of dicts)")
    parser.add_argument("--image_root", required=True, help="Root directory for image paths (will be prepended to image_path if not absolute)")
    parser.add_argument("--output_json", required=True, help="Output JSON with only valid images")
    args = parser.parse_args()

    # Load input data
    with open(args.input_json, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {args.input_json}")

    filtered = []
    for entry in tqdm(data, desc="Checking images"):
        img_path = entry["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(args.image_root, img_path)
        if is_valid_image(img_path):
            filtered.append(entry)
        else:
            print(f"Skipping invalid image: {img_path}")

    print(f"Retained {len(filtered)} valid samples out of {len(data)}.")

    with open(args.output_json, "w") as outf:
        json.dump(filtered, outf, indent=2)
    print(f"Filtered JSON saved to {args.output_json}")

if __name__ == "__main__":
    main()

import os
import json
import requests
import argparse
from urllib.parse import urlparse
from multiprocessing import Pool, cpu_count
import time
import random
from tqdm import tqdm
from collections import defaultdict

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
}

def sanitize_filename(url):
    """Extract and sanitize filename from URL"""
    parsed = urlparse(url)
    fname = os.path.basename(parsed.path)
    
    if not fname:
        import uuid
        fname = str(uuid.uuid4()) + ".jpg"
    elif "." not in fname:
        fname = fname + ".jpg"
    
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        fname = fname.replace(char, '_')
    
    if len(fname) > 255:
        name, ext = os.path.splitext(fname)
        fname = name[:255-len(ext)] + ext
    
    return fname

def resolve_filename_conflicts(caption_to_url, img_dir):
    """Resolve filename conflicts by adding indices"""
    filename_to_captions = defaultdict(list)
    caption_to_filename = {}
    for caption_id, url in caption_to_url.items():
        base_filename = sanitize_filename(url)
        filename_to_captions[base_filename].append(caption_id)
    for base_filename, caption_ids in filename_to_captions.items():
        if len(caption_ids) == 1:
            caption_to_filename[caption_ids[0]] = base_filename
        else:
            name, ext = os.path.splitext(base_filename)
            for i, caption_id in enumerate(caption_ids):
                unique_filename = f"{name}_{i}{ext}"
                caption_to_filename[caption_id] = unique_filename
    return caption_to_filename

def download_one(args):
    caption_id, url, filename, img_dir = args
    local_path = os.path.abspath(os.path.join(img_dir, filename))

    if not os.path.exists(local_path):
        for attempt in range(2):
            try:
                time.sleep(random.uniform(0.5, 1.5))
                r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
                if r.status_code == 200:
                    with open(local_path, "wb") as imgf:
                        imgf.write(r.content)
                    print(f"Downloaded: {caption_id} -> {filename}")
                    break
                elif r.status_code == 403 and attempt == 0:
                    print(f"403 Forbidden for {caption_id}, retrying after delay...")
                    time.sleep(random.uniform(1, 2))
                else:
                    print(f"Failed to download {caption_id}: HTTP {r.status_code}")
                    return (caption_id, None)
            except Exception as e:
                print(f"Exception downloading {caption_id}: {e}")
                return (caption_id, None)

    folder = os.path.basename(os.path.dirname(local_path))
    image_path = os.path.join(folder, filename) if folder else filename
    return (caption_id, image_path)

def normalize_entry_structure(entry, image_path):
    """
    Normalize JSON entry to have all flat keys, just update/add image_path.
    Remove image_url if present.
    """
    normalized = entry.copy()
    normalized["image_path"] = image_path
    if "image_url" in normalized:
        del normalized["image_url"]
    return normalized

def main():
    parser = argparse.ArgumentParser(description="Download images mapped by caption_id and normalize JSON structure (flat format).")
    parser.add_argument("--input_json_files", required=True, nargs='+', help="Input JSON file path containing caption_id and image_url.")
    parser.add_argument("--output_json", required=True, help="Output JSON file path (updated).")
    parser.add_argument("--img_dir", default="images", help="Directory to save downloaded images (default: images)")
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Number of worker processes to use (default: all CPUs)")
    parser.add_argument("--skip_download", action="store_true", help="Skip downloading, only process JSON based on existing images")
    args = parser.parse_args()

    os.makedirs(args.img_dir, exist_ok=True)

    input_json_files = []
    for input_file in args.input_json_files:
        print(f"Checking input file: {input_file}")
        if os.path.isfile(input_file):
            input_json_files.append(input_file)
        else:
            print(f"Warning: Input file {input_file} does not exist, skipping.")
    all_data = []
    for input_file in input_json_files:
        with open(input_file, "r") as f:
            data = json.load(f)
            all_data.extend(data)

    print(f"Loaded {len(all_data)} entries from {args.input_json_files}")
    data = all_data

    caption_to_url = {}
    all_entries = []
    
    for entry in data:
        caption_id = entry.get("caption_id")
        image_url = entry.get("image_url")
        if caption_id and image_url:
            caption_to_url[caption_id] = image_url
        all_entries.append(entry)

    print(f"Found {len(all_entries)} total entries")
    print(f"Found {len(caption_to_url)} unique caption IDs with image URLs")

    caption_to_filename = resolve_filename_conflicts(caption_to_url, args.img_dir)
    caption_to_image_path = {}
    
    if args.skip_download:
        print("Skipping download - only checking existing images...")
        for caption_id, filename in caption_to_filename.items():
            local_path = os.path.abspath(os.path.join(args.img_dir, filename))
            if os.path.exists(local_path):
                folder = os.path.basename(os.path.dirname(local_path))
                image_path = os.path.join(folder, filename) if folder else filename
                caption_to_image_path[caption_id] = image_path
                print(f"Found existing image: {caption_id} -> {filename}")
            else:
                print(f"Missing image: {caption_id} -> {filename}")
        print(f"Found {len(caption_to_image_path)} existing images")
    else:
        download_tasks = [
            (caption_id, caption_to_url[caption_id], filename, args.img_dir)
            for caption_id, filename in caption_to_filename.items()
        ]
        with Pool(args.num_workers) as pool:
            for caption_id, image_path in tqdm(
                pool.imap_unordered(download_one, download_tasks), 
                total=len(download_tasks), 
                desc="Downloading images"
            ):
                if image_path is not None:
                    caption_to_image_path[caption_id] = image_path
        print(f"Successfully downloaded {len(caption_to_image_path)} images")

    results = []
    skipped_count = 0
    
    for entry in all_entries:
        caption_id = entry.get("caption_id")
        if caption_id in caption_to_image_path:
            normalized_entry = normalize_entry_structure(entry, caption_to_image_path[caption_id])
            results.append(normalized_entry)
        else:
            skipped_count += 1
            if args.skip_download:
                print(f"Skipping entry with caption_id {caption_id} - image not found locally")
            else:
                print(f"Skipping entry with caption_id {caption_id} - download failed")

    with open(args.output_json, "w") as outf:
        json.dump(results, outf, indent=2)
    
    operation = "JSON processing" if args.skip_download else "Download and processing"
    print(f"\n{operation} complete!")
    print(f"- Input entries: {len(data)}")
    print(f"- Entries with valid caption_id and image_url: {len(all_entries)}")
    if args.skip_download:
        print(f"- Existing images found: {len(caption_to_image_path)}")
    else:
        print(f"- Unique images downloaded: {len(caption_to_image_path)}")
    print(f"- Output entries: {len(results)}")
    print(f"- Skipped entries: {skipped_count}")
    print(f"- Output saved to: {args.output_json}")
    
    if results:
        print(f"\nExample normalized entry structure:")
        example = results[0]
        print(f"Keys: {list(example.keys())}")
        for k, v in example.items():
            print(f"{k}: {repr(v)}")

if __name__ == "__main__":
    main()

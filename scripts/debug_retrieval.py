#!/usr/bin/env python3
"""
Debug script to understand the retrieval dataset structure
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

# Load a small sample
ds = load_dataset("clip-benchmark/wds_flickr8k", split="test")

print(f"Dataset size: {len(ds)}")
print(f"\nFirst 5 samples:")
print("=" * 80)

for i in range(min(5, len(ds))):
    sample = ds[i]
    print(f"\nSample {i}:")
    print(f"  Keys: {list(sample.keys())}")
    
    # Check all fields
    for key, value in sample.items():
        if key in ['jpg', 'png', 'image']:
            print(f"  {key}: {type(value)}, size: {value.size if hasattr(value, 'size') else 'N/A'}")
        elif key == 'txt':
            print(f"  txt: type={type(value)}")
            if isinstance(value, str):
                print(f"    Content: '{value[:100]}...'")
            elif isinstance(value, list):
                print(f"    List with {len(value)} items:")
                for j, item in enumerate(value[:3]):
                    print(f"      [{j}]: '{item[:80]}...'")
            elif isinstance(value, bytes):
                decoded = value.decode('utf-8') if isinstance(value, bytes) else value
                print(f"    Bytes content: '{decoded[:100]}...'")
        else:
            print(f"  {key}: {type(value)} = {str(value)[:100]}")

print("\n" + "=" * 80)
print("Data structure analysis:")
print("=" * 80)

# Detailed check of first sample
first = ds[0]
print(f"\nFirst sample detailed:")
print(f"  Type: {type(first)}")
print(f"  Keys: {list(first.keys())}")

if 'txt' in first:
    txt_field = first['txt']
    print(f"\n'txt' field analysis:")
    print(f"  Type: {type(txt_field)}")
    print(f"  Value: {txt_field}")
    
    # Try different parsings
    if isinstance(txt_field, str):
        print(f"  It's a string - checking if it contains multiple captions...")
        # Maybe it's newline-separated?
        lines = txt_field.split('\n')
        if len(lines) > 1:
            print(f"    Found {len(lines)} lines (newline-separated)")
            for i, line in enumerate(lines[:3]):
                print(f"      Line {i}: '{line}'")
        
        # Maybe JSON?
        try:
            import json
            parsed = json.loads(txt_field)
            print(f"    JSON parsed: {type(parsed)}")
            print(f"    Content: {parsed}")
        except:
            print(f"    Not JSON")

# Check if there's a pattern in how data is structured
print(f"\n\nChecking all 1000 samples for txt types:")
txt_types = {}
for i in range(len(ds)):
    txt = ds[i].get('txt', None)
    txt_type = type(txt).__name__
    txt_types[txt_type] = txt_types.get(txt_type, 0) + 1

print(f"  txt type distribution: {txt_types}")

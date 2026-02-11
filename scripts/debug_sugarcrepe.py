#!/usr/bin/env python3
"""
Debug script to test SugarCrepe dataset loading
"""

import os
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from data_loading import get_dataset_class
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_sugarcrepe_loading():
    print("🔍 Testing SugarCrepe dataset loading...")
    
    # Configuration
    data_path = "./datasets/SugarCrepe"
    subset_types = ['add_att', 'add_obj', 'replace_att', 'replace_obj', 'replace_rel', 'swap_att', 'swap_obj']
    
    print(f"📁 Data path: {data_path}")
    print(f"📁 Path exists: {os.path.exists(data_path)}")
    
    if os.path.exists(data_path):
        print(f"📁 Contents: {os.listdir(data_path)}")
    
    dataset_class = get_dataset_class('SugarCrepe')
    if dataset_class is None:
        print("❌ Failed to get SugarCrepe dataset class")
        return
    
    print(f"✓ Got dataset class: {dataset_class}")
    
    # Test each subset
    for subset in subset_types:
        print(f"\n🧪 Testing subset: {subset}")
        json_path = os.path.join(data_path, f'{subset}.json')
        print(f"  📄 JSON file: {json_path} (exists: {os.path.exists(json_path)})")
        
        # Check for COCO images
        coco_candidates = [
            data_path,
            os.path.join(data_path, 'coco'),
            os.path.join(data_path, 'val2017'),
            './datasets/COCO'
        ]
        
        coco_root = None
        for candidate in coco_candidates:
            val2017_path = os.path.join(candidate, 'val2017')
            if os.path.exists(val2017_path):
                coco_root = candidate
                print(f"  🖼️  COCO images found at: {coco_root}")
                break
        
        if coco_root is None:
            print(f"  ❌ No COCO val2017 directory found. Tried: {coco_candidates}")
            continue
        
        try:
            dataset = dataset_class(
                data_root=data_path,
                subset_name=subset,
                coco_root=coco_root,
                image_preprocess=None
            )
            print(f"  ✅ Loaded successfully: {len(dataset)} samples")
            
            # Try to get a sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  📝 Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
                
        except Exception as e:
            print(f"  ❌ Failed to load: {e}")

if __name__ == "__main__":
    test_sugarcrepe_loading()

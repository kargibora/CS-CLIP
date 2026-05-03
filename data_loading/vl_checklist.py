import json
import os
import random
import requests
import subprocess
import tarfile
import yaml
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from utils.align import (
        compute_caption_embeddings_intermediate_batch,
        compute_image_embeddings_intermediate_batch,
    )
except ImportError:
    print("Warning: utils.align not found. Evaluation functions may not work.")


class VLCheckListDataset(Dataset):
    """
    VL-CheckList dataset for evaluating vision-language compositional reasoning.
    
    The dataset tests various aspects like attributes (color, size, action, etc.),
    object location/size, and relations. Each sample has positive and negative captions
    for an image.
    
    Dataset structure from VL-CheckList:
    - Each sample has: image_path, positive captions, negative captions
    - Goal: Distinguish between positive and negative text for the same image
    - Supports multiple test types: Attribute, Object, Relation
    
    Available Subsets:
    
    ATTRIBUTE (tests attribute recognition):
    - Attribute/action: vaw_action, vg_action, attribute_action_vaw, attribute_action_vg
    - Attribute/color: vaw_color, vg_color, attribute_color_vaw, attribute_color_vg
    - Attribute/material: vaw_material, vg_material, attribute_material_vaw, attribute_material_vg
    - Attribute/size: vaw_size, vg_size, attribute_size_vaw, attribute_size_vg
    - Attribute/state: vaw_state, vg_state, attribute_state_vaw, attribute_state_vg
    
    OBJECT (tests object location and size):
    - Object/Location/center: object_location_center_[hake|swig_agent|swig_destination|swig_item|swig_tool|vg_obj|vg_subj]
    - Object/Location/margin: object_location_margin_[hake|swig_agent|swig_destination|swig_item|swig_tool|vg_obj|vg_subj]
    - Object/Location/mid: object_location_mid_[hake|swig_agent|swig_destination|swig_item|swig_tool|vg_obj|vg_subj]
    - Object/Size/large: object_size_large_[hake|swig_agent|swig_destination|swig_item|swig_tool|vg_obj|vg_subj]
    - Object/Size/medium: object_size_medium_[hake|swig_agent|swig_destination|swig_item|swig_tool|vg_obj|vg_subj]
    - Object/Size/small: object_size_small_[hake|swig_agent|swig_destination|swig_item|swig_tool|vg_obj|vg_subj]
    
    RELATION (tests relational understanding):
    - Relation/action: hake_action, swig_action, vg_action_relation, relation_action_[hake|swig|vg]
    - Relation/spatial: vg_spatial, relation_spatial_vg
    
    Naming Conventions:
    - Short form: <dataset>_<type> (e.g., "vaw_color", "hake_action")
    - Long form: <category>_<subcategory>_<type>_<dataset> (e.g., "attribute_color_vaw", "relation_action_hake")
    - Object form: object_<location|size>_<subtype>_<dataset> (e.g., "object_location_center_hake")
    
    Returns samples in format compatible with alignment pipeline:
        {
          'image_options': image,  # Single image tensor
          'caption_options': [pos_captions, neg_captions],  # All caption options
          'label': int,  # Index of positive caption in caption_options
          'pos_captions': [pos_captions],  # Just positive captions
          'neg_captions': [neg_captions],  # Just negative captions
        }
    """
    
    # Class-level flag to track if corpus type warning has been shown
    _corpus_warning_shown = set()

    def __init__(
        self,
        data_root: str,
        subset_name: str = "hake_action",
        image_preprocess=None,
        cache_dir: Optional[str] = None,
        download: bool = True,
        verbose: bool = True,
        task: str = "itc",  # 'itc' for image-text contrastive, 'itm' for image-text matching
        version: str = "v1",
        **kwargs
    ):
        self.data_root = data_root
        self.subset_name = subset_name
        self.image_preprocess = image_preprocess
        self.cache_dir = cache_dir or os.path.join(data_root, "cache")
        self.verbose = verbose
        self.task = task
        self.version = version
        
        # VL-CheckList root directory
        self.vl_checklist_root = data_root
        
        # Ensure data directories exist
        os.makedirs(data_root, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if download:
            self._download_and_prepare()
        
        # Load the data
        self.sample_list = self._load_data()
        
        # Build caption vocabulary and mappings
        self._build_caption_mappings()
        
        if verbose:
            print(f"[VLCheckListDataset] Dataset ready with {len(self.sample_list)} examples")
            print(f"[VLCheckListDataset] Unique captions: {len(self.captions)}")
            print(f"[VLCheckListDataset] Subset: {subset_name}")

    def _download_and_prepare(self):
        """Download VL-CheckList dataset and required images if not present."""
        if not os.path.exists(self.vl_checklist_root):
            if self.verbose:
                print(f"[VLCheckListDataset] VL-CheckList not found at {self.vl_checklist_root}")
                print("[VLCheckListDataset] Please download VL-CheckList dataset manually")
                print("[VLCheckListDataset] Git clone: https://github.com/om-ai-lab/VL-CheckList")
            raise FileNotFoundError(
                f"VL-CheckList dataset not found at {self.vl_checklist_root}. "
                "Please download manually or provide the correct path."
            )
        
        # Install required packages for downloading
        self._ensure_download_dependencies()
        
        # Download required images based on subset_name
        self._download_required_images()

    def _ensure_download_dependencies(self):
        """Ensure required packages for downloading are installed."""
        required_packages = {
            'requests': 'requests',
            'gdown': 'gdown',
            'tqdm': 'tqdm'
        }
        
        missing_packages = []
        for package_name, pip_name in required_packages.items():
            try:
                __import__(package_name)
            except ImportError:
                missing_packages.append(pip_name)
        
        if missing_packages:
            if self.verbose:
                print(f"[VLCheckListDataset] Installing missing packages: {', '.join(missing_packages)}")
            
            try:
                import subprocess
                import sys
                for package in missing_packages:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package
                    ], stdout=subprocess.DEVNULL if not self.verbose else None)
                
                if self.verbose:
                    print("[VLCheckListDataset] Successfully installed missing packages")
            except Exception as e:
                if self.verbose:
                    print(f"[VLCheckListDataset] Failed to install packages: {e}")
                    print(f"[VLCheckListDataset] Please install manually: pip install {' '.join(missing_packages)}")

    def _download_required_images(self):
        """Download required images for the specific subset."""
        corpus_type, corpus_name = self._determine_corpus_type(self.subset_name)
        
        if corpus_name == "hake":
            self._download_hake_images()
        elif corpus_name == "swig":
            self._download_swig_images()
        elif corpus_name in ["vg", "vaw"]:
            self._download_vg_images()
        
        if self.verbose:
            print(f"[VLCheckListDataset] Image download preparation completed for {corpus_name}")

    def _download_hake_images(self):
        """Download HAKE dataset images."""
        hake_dir = os.path.join(self.vl_checklist_root, "hake")
        os.makedirs(hake_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "hake_images_20190730", "hake_images_20200614", "hcvrd", 
            "hico_20160224_det", "openimages", "pic", "vcoco"
        ]
        for subdir in subdirs:
            os.makedirs(os.path.join(hake_dir, subdir), exist_ok=True)
        
        # Check if openimages directory has content
        openimages_dir = os.path.join(hake_dir, "openimages")
        if not os.listdir(openimages_dir):
            if self.verbose:
                print("[VLCheckListDataset] HAKE OpenImages directory is empty")
                print("[VLCheckListDataset] For HAKE dataset, please download images manually:")
                print("[VLCheckListDataset] 1. OpenImages (subset): https://drive.google.com/open?id=1XTWYLyL1h-9jJ49dsXmtRCv8GcupVrvM")
                print("[VLCheckListDataset] 2. Or use the official HAKE download script:")
                print("[VLCheckListDataset]    git clone https://github.com/DirtyHarryLYL/HAKE.git")
                print("[VLCheckListDataset]    cd HAKE/Images && sh download_image/download_dataset.sh")
                print("[VLCheckListDataset] 3. HAKE images 20190730: https://drive.google.com/open?id=18R_3Oz7zO1knEjagY6sfUkQ1_6wZf0Ei")
                print("[VLCheckListDataset] 4. HAKE images 20200614: https://drive.google.com/open?id=14K_4FfjviJNDVLJdGM96W2ZLN55dDb2-")
        
        # Try to use gdown for Google Drive downloads if available
        try:
            import gdown
            self._download_hake_with_gdown(hake_dir)
        except ImportError:
            if self.verbose:
                print("[VLCheckListDataset] gdown not available. Install with: pip install gdown")
                print("[VLCheckListDataset] Manual download required for HAKE images")

    def _download_hake_with_gdown(self, hake_dir):
        """Download HAKE images using gdown with improved error handling."""
        try:
            import gdown
            import zipfile
            
            # Download configuration with proper file IDs
            downloads = [
                {
                    "name": "OpenImages subset",
                    "file_id": "1XTWYLyL1h-9jJ49dsXmtRCv8GcupVrvM",
                    "filename": "openimages.zip",
                    "extract_dir": "openimages",
                    "manual_url": "https://drive.google.com/open?id=1XTWYLyL1h-9jJ49dsXmtRCv8GcupVrvM"
                },
                {
                    "name": "HAKE images 2019", 
                    "file_id": "18R_3Oz7zO1knEjagY6sfUkQ1_6wZf0Ei",  # Use the correct ID from manual URL
                    "filename": "hake_2019.zip",
                    "extract_dir": "hake_images_20190730",
                    "manual_url": "https://drive.google.com/open?id=18R_3Oz7zO1knEjagY6sfUkQ1_6wZf0Ei"
                },
                {
                    "name": "HAKE images 2020",
                    "file_id": "14K_4FfjviJNDVLJdGM96W2ZLN55dDb2-",
                    "filename": "hake_2020.zip", 
                    "extract_dir": "hake_images_20200614",
                    "manual_url": "https://drive.google.com/open?id=14K_4FfjviJNDVLJdGM96W2ZLN55dDb2-"
                }
            ]
            
            success_count = 0
            
            for download in downloads:
                extract_path = os.path.join(hake_dir, download["extract_dir"])
                
                # Skip if directory exists and is not empty
                if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
                    if self.verbose:
                        print(f"[VLCheckListDataset] {download['name']} already exists, skipping...")
                    success_count += 1
                    continue
                
                # Create directory if it doesn't exist
                os.makedirs(extract_path, exist_ok=True)
                zip_path = os.path.join(hake_dir, download["filename"])
                
                try:
                    if self.verbose:
                        print(f"[VLCheckListDataset] Downloading {download['name']}...")
                    
                    # Try with fuzzy=True for large files that might need confirmation
                    try:
                        gdown.download(
                            id=download["file_id"], 
                            output=zip_path, 
                            quiet=not self.verbose,
                            fuzzy=True  # Handle files that need confirmation
                        )
                    except Exception:
                        # Fallback to URL method
                        gdown.download(
                            f"https://drive.google.com/uc?id={download['file_id']}", 
                            output=zip_path, 
                            quiet=not self.verbose
                        )
                    
                    # Check if download was successful
                    if not os.path.exists(zip_path):
                        raise Exception("Download failed - file not created")
                    
                    # Check file size (Google Drive error pages are usually small)
                    file_size = os.path.getsize(zip_path)
                    if file_size < 1024 * 1024:  # Less than 1MB is likely an error page
                        raise Exception(f"Downloaded file too small ({file_size} bytes), likely an error page")
                    
                    # Verify it's actually a zip file
                    if not zipfile.is_zipfile(zip_path):
                        raise Exception("Downloaded file is not a valid zip file")
                    
                    # Extract the zip file
                    if self.verbose:
                        print(f"[VLCheckListDataset] Extracting {download['name']}...")
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    
                    # Clean up zip file
                    os.remove(zip_path)
                    success_count += 1
                    
                    if self.verbose:
                        print(f"[VLCheckListDataset] Successfully downloaded and extracted {download['name']}")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"[VLCheckListDataset] Failed to download {download['name']}: {e}")
                        print("[VLCheckListDataset] Google Drive may require manual download for large files")
                        print(f"[VLCheckListDataset] Please download manually from: {download['manual_url']}")
                        print(f"[VLCheckListDataset] Extract to: {extract_path}")
                    
                    # Clean up partial files
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
            
            return success_count > 0
            
        except ImportError:
            if self.verbose:
                print("[VLCheckListDataset] gdown not available. Install with: pip install gdown")
                print("[VLCheckListDataset] Manual download required for HAKE images")
            return False
        except Exception as e:
            if self.verbose:
                print(f"[VLCheckListDataset] gdown download failed: {e}")
            return False

    def _download_swig_images(self):
        """Download SWiG dataset images."""
        swig_dir = os.path.join(self.vl_checklist_root, "swig")
        os.makedirs(swig_dir, exist_ok=True)
        
        if not os.listdir(swig_dir):
            if self.verbose:
                print("[VLCheckListDataset] SWiG images directory is empty")
                print("[VLCheckListDataset] Please download SWiG images manually:")
                print("[VLCheckListDataset] https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip")
                print("[VLCheckListDataset] Extract the zip file to the swig/ directory")
            
            # Try to download automatically
            try:
                import requests
                swig_url = "https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip"
                zip_path = os.path.join(swig_dir, "images_512.zip")
                
                if self.verbose:
                    print("[VLCheckListDataset] Attempting to download SWiG images...")
                
                response = requests.get(swig_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(zip_path, 'wb') as f, tqdm(
                    desc="Downloading SWiG",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    disable=not self.verbose
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
                
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(swig_dir)
                
                # Remove the zip file
                os.remove(zip_path)
                
                if self.verbose:
                    print("[VLCheckListDataset] SWiG images downloaded successfully")
                    
            except Exception as e:
                if self.verbose:
                    print(f"[VLCheckListDataset] Failed to download SWiG images: {e}")
                    print("[VLCheckListDataset] Please download manually")

    def _download_vg_images(self):
        """Download Visual Genome dataset images."""
        vg_dir = os.path.join(self.vl_checklist_root, "vg")
        os.makedirs(vg_dir, exist_ok=True)
        
        # Create subdirectories
        vg_100k_dir = os.path.join(vg_dir, "VG_100K")
        vg_100k_2_dir = os.path.join(vg_dir, "VG_100K_2")
        os.makedirs(vg_100k_dir, exist_ok=True)
        os.makedirs(vg_100k_2_dir, exist_ok=True)
        
        if not os.listdir(vg_100k_dir) or not os.listdir(vg_100k_2_dir):
            if self.verbose:
                print("[VLCheckListDataset] Visual Genome images directories are empty")
                print("[VLCheckListDataset] Please download VG images manually:")
                print("[VLCheckListDataset] Part 1: https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip")
                print("[VLCheckListDataset] Part 2: https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip")
                print("[VLCheckListDataset] Extract part1 to VG_100K/ and part2 to VG_100K_2/")
            
            # Try to download automatically
            try:
                import requests
                
                # Download part 1
                if not os.listdir(vg_100k_dir):
                    if self.verbose:
                        print("[VLCheckListDataset] Downloading VG Part 1...")
                    
                    part1_url = "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
                    zip_path = os.path.join(vg_dir, "images.zip")
                    
                    response = requests.get(part1_url, stream=True)
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get('content-length', 0))
                    with open(zip_path, 'wb') as f, tqdm(
                        desc="Downloading VG Part 1",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        disable=not self.verbose
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            size = f.write(chunk)
                            pbar.update(size)
                    
                    # Extract
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(vg_100k_dir)
                    os.remove(zip_path)
                
                # Download part 2
                if not os.listdir(vg_100k_2_dir):
                    if self.verbose:
                        print("[VLCheckListDataset] Downloading VG Part 2...")
                    
                    part2_url = "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
                    zip_path = os.path.join(vg_dir, "images2.zip")
                    
                    response = requests.get(part2_url, stream=True)
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get('content-length', 0))
                    with open(zip_path, 'wb') as f, tqdm(
                        desc="Downloading VG Part 2",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        disable=not self.verbose
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            size = f.write(chunk)
                            pbar.update(size)
                    
                    # Extract
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(vg_100k_2_dir)
                    os.remove(zip_path)
                
                if self.verbose:
                    print("[VLCheckListDataset] Visual Genome images downloaded successfully")
                    
            except Exception as e:
                if self.verbose:
                    print(f"[VLCheckListDataset] Failed to download VG images: {e}")
                    print("[VLCheckListDataset] Please download manually")

    def _determine_corpus_type(self, subset_name: str) -> tuple:
        """Determine the corpus type and config path from subset name.
        
        Returns:
            tuple: (corpus_path, yaml_filename) where corpus_path is like 
                   "Attribute/action" and yaml_filename is like "vaw"
                   OR (corpus_path, "ALL_DATASETS") for aggregated subsets
                   OR ("ALL_SUBDIRS", base_path) for full type-level aggregation
        """
        # Complete mapping of all VL-CheckList subsets based on corpus structure
        corpus_mappings = {
            # ============ STANDARD PAPER EVALUATION (9 categories) ============
            # Top-level aggregations matching standard VL-CheckList evaluation
            
            # Attribute types - aggregate ALL datasets for each type
            "attr_color": ("ALL_SUBDIRS", "Attribute/color"),
            "attr_material": ("ALL_SUBDIRS", "Attribute/material"),
            "attr_size": ("ALL_SUBDIRS", "Attribute/size"),
            "attr_state": ("ALL_SUBDIRS", "Attribute/state"),
            "attr_action": ("ALL_SUBDIRS", "Attribute/action"),
            
            # Object types - aggregate ALL subtypes AND datasets
            "obj_location": ("ALL_SUBDIRS", "Object/Location"),  # All center+margin+mid × all datasets
            "obj_size": ("ALL_SUBDIRS", "Object/Size"),  # All large+medium+small × all datasets
            
            # Relation types - aggregate ALL datasets
            "rel_action": ("ALL_SUBDIRS", "Relation/action"),
            "rel_spatial": ("ALL_SUBDIRS", "Relation/spatial"),
            
            # ============ ATTRIBUTE TYPES - INDIVIDUAL ============
            # Attribute/action
            "attribute_action_vaw": ("Attribute/action", "vaw"),
            "attribute_action_vg": ("Attribute/action", "vg"),
            "vaw_action": ("Attribute/action", "vaw"),
            "vg_action": ("Attribute/action", "vg"),
            
            # Attribute/color  
            "attribute_color_vaw": ("Attribute/color", "vaw"),
            "attribute_color_vg": ("Attribute/color", "vg"),
            "vaw_color": ("Attribute/color", "vaw"),
            "vg_color": ("Attribute/color", "vg"),
            
            # Attribute/material
            "attribute_material_vaw": ("Attribute/material", "vaw"),
            "attribute_material_vg": ("Attribute/material", "vg"),
            "vaw_material": ("Attribute/material", "vaw"),
            "vg_material": ("Attribute/material", "vg"),
            
            # Attribute/size
            "attribute_size_vaw": ("Attribute/size", "vaw"),
            "attribute_size_vg": ("Attribute/size", "vg"),
            "vaw_size": ("Attribute/size", "vaw"),
            "vg_size": ("Attribute/size", "vg"),
            
            # Attribute/state
            "attribute_state_vaw": ("Attribute/state", "vaw"),
            "attribute_state_vg": ("Attribute/state", "vg"),
            "vaw_state": ("Attribute/state", "vaw"),
            "vg_state": ("Attribute/state", "vg"),
            
            # ============ OBJECT TYPES - AGGREGATED BY SUBTYPE ============
            # Object/Location - aggregated by position (all 7 datasets per position)
            "object_location_center": ("Object/Location/center", "ALL_DATASETS"),
            "object_location_margin": ("Object/Location/margin", "ALL_DATASETS"),
            "object_location_mid": ("Object/Location/mid", "ALL_DATASETS"),
            
            # Object/Size - aggregated by size (all 7 datasets per size)
            "object_size_large": ("Object/Size/large", "ALL_DATASETS"),
            "object_size_medium": ("Object/Size/medium", "ALL_DATASETS"),
            "object_size_small": ("Object/Size/small", "ALL_DATASETS"),
            
            # ============ OBJECT TYPES - INDIVIDUAL ============
            # Object/Location/center
            "object_location_center_hake": ("Object/Location/center", "hake"),
            "object_location_center_swig_agent": ("Object/Location/center", "swig_agent"),
            "object_location_center_swig_destination": ("Object/Location/center", "swig_destination"),
            "object_location_center_swig_item": ("Object/Location/center", "swig_item"),
            "object_location_center_swig_tool": ("Object/Location/center", "swig_tool"),
            "object_location_center_vg_obj": ("Object/Location/center", "vg_obj"),
            "object_location_center_vg_subj": ("Object/Location/center", "vg_subj"),
            
            # Object/Location/margin
            "object_location_margin_hake": ("Object/Location/margin", "hake"),
            "object_location_margin_swig_agent": ("Object/Location/margin", "swig_agent"),
            "object_location_margin_swig_destination": ("Object/Location/margin", "swig_destination"),
            "object_location_margin_swig_item": ("Object/Location/margin", "swig_item"),
            "object_location_margin_swig_tool": ("Object/Location/margin", "swig_tool"),
            "object_location_margin_vg_obj": ("Object/Location/margin", "vg_obj"),
            "object_location_margin_vg_subj": ("Object/Location/margin", "vg_subj"),
            
            # Object/Location/mid
            "object_location_mid_hake": ("Object/Location/mid", "hake"),
            "object_location_mid_swig_agent": ("Object/Location/mid", "swig_agent"),
            "object_location_mid_swig_destination": ("Object/Location/mid", "swig_destination"),
            "object_location_mid_swig_item": ("Object/Location/mid", "swig_item"),
            "object_location_mid_swig_tool": ("Object/Location/mid", "swig_tool"),
            "object_location_mid_vg_obj": ("Object/Location/mid", "vg_obj"),
            "object_location_mid_vg_subj": ("Object/Location/mid", "vg_subj"),
            
            # Object/Size/large
            "object_size_large_hake": ("Object/Size/large", "hake"),
            "object_size_large_swig_agent": ("Object/Size/large", "swig_agent"),
            "object_size_large_swig_destination": ("Object/Size/large", "swig_destination"),
            "object_size_large_swig_item": ("Object/Size/large", "swig_item"),
            "object_size_large_swig_tool": ("Object/Size/large", "swig_tool"),
            "object_size_large_vg_obj": ("Object/Size/large", "vg_obj"),
            "object_size_large_vg_subj": ("Object/Size/large", "vg_subj"),
            
            # Object/Size/medium
            "object_size_medium_hake": ("Object/Size/medium", "hake"),
            "object_size_medium_swig_agent": ("Object/Size/medium", "swig_agent"),
            "object_size_medium_swig_destination": ("Object/Size/medium", "swig_destination"),
            "object_size_medium_swig_item": ("Object/Size/medium", "swig_item"),
            "object_size_medium_swig_tool": ("Object/Size/medium", "swig_tool"),
            "object_size_medium_vg_obj": ("Object/Size/medium", "vg_obj"),
            "object_size_medium_vg_subj": ("Object/Size/medium", "vg_subj"),
            
            # Object/Size/small
            "object_size_small_hake": ("Object/Size/small", "hake"),
            "object_size_small_swig_agent": ("Object/Size/small", "swig_agent"),
            "object_size_small_swig_destination": ("Object/Size/small", "swig_destination"),
            "object_size_small_swig_item": ("Object/Size/small", "swig_item"),
            "object_size_small_swig_tool": ("Object/Size/small", "swig_tool"),
            "object_size_small_vg_obj": ("Object/Size/small", "vg_obj"),
            "object_size_small_vg_subj": ("Object/Size/small", "vg_subj"),
            
            # ============ RELATION TYPES ============
            # Relation/action
            "relation_action_hake": ("Relation/action", "hake"),
            "relation_action_swig": ("Relation/action", "swig"),
            "relation_action_vg": ("Relation/action", "vg"),
            "hake_action": ("Relation/action", "hake"),
            "swig_action": ("Relation/action", "swig"),
            "vg_action_relation": ("Relation/action", "vg"),
            
            # Relation/spatial
            "relation_spatial_vg": ("Relation/spatial", "vg"),
            "vg_spatial": ("Relation/spatial", "vg"),
        }
        
        # Normalize subset name for lookup
        subset_lower = subset_name.lower().replace("-", "_")
        
        # Direct lookup
        if subset_lower in corpus_mappings:
            return corpus_mappings[subset_lower]
        
        # Try original name if normalization didn't match
        if subset_name in corpus_mappings:
            return corpus_mappings[subset_name]
        
        # Intelligent parsing for flexible naming
        parts = subset_lower.split("_")
        
        # Try to identify main category
        if "attribute" in parts or any(attr in parts for attr in ["action", "color", "material", "size", "state"]):
            # Attribute category
            attr_type = None
            dataset = None
            
            for part in parts:
                if part in ["action", "color", "material", "size", "state"]:
                    attr_type = part
                elif part in ["vaw", "vg"]:
                    dataset = part
            
            if attr_type and dataset:
                return (f"Attribute/{attr_type}", dataset)
        
        elif "object" in parts:
            # Object category
            if "location" in parts:
                location_type = None
                dataset = None
                
                for part in parts:
                    if part in ["center", "margin", "mid"]:
                        location_type = part
                    elif part in ["hake", "vg", "swig"] or part.startswith("swig_"):
                        dataset = part
                
                if location_type and dataset:
                    return (f"Object/Location/{location_type}", dataset)
            
            elif "size" in parts:
                size_type = None
                dataset = None
                
                for part in parts:
                    if part in ["large", "medium", "small"]:
                        size_type = part
                    elif part in ["hake", "vg", "swig"] or part.startswith("swig_"):
                        dataset = part
                
                if size_type and dataset:
                    return (f"Object/Size/{size_type}", dataset)
        
        elif "relation" in parts or "hake" in parts or "swig" in parts:
            # Relation category
            if "spatial" in parts:
                return ("Relation/spatial", "vg")
            elif "action" in parts or "hake" in parts or "swig" in parts:
                if "hake" in parts:
                    return ("Relation/action", "hake")
                elif "swig" in parts:
                    return ("Relation/action", "swig")
                elif "vg" in parts:
                    return ("Relation/action", "vg")
        
        # Last resort: try to match based on dataset name alone
        if "hake" in subset_lower:
            return ("Relation/action", "hake")
        elif "swig" in subset_lower:
            if "agent" in subset_lower:
                return ("Object/Location/center", "swig_agent")
            elif "destination" in subset_lower:
                return ("Object/Location/center", "swig_destination")
            elif "item" in subset_lower:
                return ("Object/Location/center", "swig_item")
            elif "tool" in subset_lower:
                return ("Object/Location/center", "swig_tool")
            else:
                return ("Relation/action", "swig")
        elif "vaw" in subset_lower:
            return ("Attribute/color", "vaw")  # Default to color for VAW
        elif "vg" in subset_lower:
            if "spatial" in subset_lower:
                return ("Relation/spatial", "vg")
            else:
                return ("Attribute/color", "vg")  # Default to color for VG
        
        # Ultimate fallback with warning (only show once per subset across all instances)
        if self.verbose and subset_name not in VLCheckListDataset._corpus_warning_shown:
            VLCheckListDataset._corpus_warning_shown.add(subset_name)
            print(f"[VLCheckListDataset] Warning: Could not determine corpus type for '{subset_name}'")
            print("[VLCheckListDataset] Using default: Relation/action/hake")
            print("[VLCheckListDataset] Available formats: attribute_<type>_<dataset>, object_<category>_<type>_<dataset>, relation_<type>_<dataset>")
        
        return ("Relation/action", "hake")

    def _load_all_subdirs(self, base_path: str):
        """Load and aggregate data from ALL subdirectories and corpus files under a base path.
        
        This is used for top-level aggregations like:
        - attr_color: loads Attribute/color/*.yaml (vaw.yaml, vg.yaml)
        - obj_location: loads Object/Location/*/*.yaml (center/*, margin/*, mid/*)
        - obj_size: loads Object/Size/*/*.yaml (large/*, medium/*, small/*)
        
        Args:
            base_path: Base path like "Attribute/color" or "Object/Location"
        
        Returns:
            list: All samples from all corpus files under this path
        """
        corpus_base = os.path.join(self.vl_checklist_root, "corpus", self.version, base_path)
        
        if not os.path.exists(corpus_base):
            raise FileNotFoundError(f"Corpus base directory not found: {corpus_base}")
        
        # Find all yaml files recursively under this path
        yaml_files = []
        for root, dirs, files in os.walk(corpus_base):
            for file in files:
                if file.endswith('.yaml'):
                    yaml_files.append(os.path.join(root, file))
        
        if not yaml_files:
            raise FileNotFoundError(f"No corpus files found under: {corpus_base}")
        
        if self.verbose:
            print(f"[VLCheckListDataset] Loading all subdirs from {base_path}")
            print(f"[VLCheckListDataset] Found {len(yaml_files)} corpus files")
        
        # Aggregate samples from all corpus files
        all_samples = []
        all_missing_images = []
        
        for yaml_path in sorted(yaml_files):  # Sort for consistency
            relative_path = os.path.relpath(yaml_path, corpus_base)
            corpus_name = os.path.splitext(relative_path)[0].replace(os.sep, '_')
            
            try:
                with open(yaml_path, 'r') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                
                # Load annotation data
                anno_path = os.path.join(self.vl_checklist_root, config["ANNO_PATH"])
                if not os.path.exists(anno_path):
                    if self.verbose:
                        print(f"[VLCheckListDataset]   - {corpus_name}: annotation not found, skipping")
                    continue
                
                with open(anno_path, 'r') as f:
                    raw_data = json.load(f)
                
                # Skip empty annotation files
                if not raw_data:
                    if self.verbose:
                        print(f"[VLCheckListDataset]   - {corpus_name}: empty annotation file, skipping")
                    continue
                
                # Image root directory
                img_root = os.path.join(self.vl_checklist_root, config["IMG_ROOT"])
                
                # Process data based on task type
                samples_added = 0
                for item in raw_data:
                    image_path, texts_dict = item
                    full_image_path = os.path.join(img_root, image_path)
                    
                    # Check if image exists
                    if not os.path.exists(full_image_path):
                        all_missing_images.append(full_image_path)
                        continue
                    
                    pos_captions = texts_dict["POS"]
                    neg_captions = texts_dict["NEG"]
                    
                    if self.task == "itc":
                        all_samples.append({
                            "image_path": full_image_path,
                            "pos_captions": pos_captions,
                            "neg_captions": neg_captions,
                            "caption_options": pos_captions + neg_captions,
                            "label": 0,
                        })
                        samples_added += 1
                    elif self.task == "itm":
                        all_samples.append({
                            "image_path": full_image_path,
                            "captions": pos_captions,
                            "label": 1,
                        })
                        all_samples.append({
                            "image_path": full_image_path,
                            "captions": neg_captions,
                            "label": 0,
                        })
                        samples_added += 2
                
                if self.verbose and samples_added > 0:
                    print(f"[VLCheckListDataset]   - {corpus_name}: {samples_added} samples")
                    
            except Exception as e:
                if self.verbose:
                    print(f"[VLCheckListDataset]   - {corpus_name}: Error - {e}")
                continue
        
        if self.verbose:
            print(f"[VLCheckListDataset] Total samples: {len(all_samples)}")
            if all_missing_images:
                unique_missing = list(set([os.path.basename(img) for img in all_missing_images]))
                print(f"[VLCheckListDataset] Warning: {len(all_missing_images)} samples skipped due to {len(unique_missing)} missing images")
        
        return all_samples

    def _load_aggregated_data(self, corpus_type: str):
        """Load and aggregate data from all corpus files in a category.
        
        Args:
            corpus_type: Path to corpus directory (e.g., "Object/Location/center")
        
        Returns:
            tuple: (sample_list, missing_images) containing all samples from all corpus files
        """
        corpus_dir = os.path.join(self.vl_checklist_root, "corpus", self.version, corpus_type)
        
        if not os.path.exists(corpus_dir):
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
        
        # Find all yaml files in this directory
        yaml_files = [f for f in os.listdir(corpus_dir) if f.endswith('.yaml')]
        
        if not yaml_files:
            raise FileNotFoundError(f"No corpus files found in: {corpus_dir}")
        
        if self.verbose:
            print(f"[VLCheckListDataset] Aggregating {len(yaml_files)} corpus files from {corpus_type}")
            print(f"[VLCheckListDataset] Files: {', '.join(yaml_files)}")
        
        # Aggregate samples from all corpus files
        all_samples = []
        all_missing_images = []
        
        for yaml_file in sorted(yaml_files):  # Sort for consistency
            corpus_name = yaml_file.replace('.yaml', '')
            config_path = os.path.join(corpus_dir, yaml_file)
            
            try:
                with open(config_path, 'r') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                
                # Load annotation data
                anno_path = os.path.join(self.vl_checklist_root, config["ANNO_PATH"])
                if not os.path.exists(anno_path):
                    if self.verbose:
                        print(f"[VLCheckListDataset] Warning: Annotation file not found: {anno_path}")
                    continue
                
                with open(anno_path, 'r') as f:
                    raw_data = json.load(f)
                
                # Image root directory
                img_root = os.path.join(self.vl_checklist_root, config["IMG_ROOT"])
                
                # Process data based on task type
                for item in raw_data:
                    image_path, texts_dict = item
                    full_image_path = os.path.join(img_root, image_path)
                    
                    # Check if image exists
                    if not os.path.exists(full_image_path):
                        all_missing_images.append(full_image_path)
                        continue
                    
                    pos_captions = texts_dict["POS"]
                    neg_captions = texts_dict["NEG"]
                    
                    if self.task == "itc":
                        all_samples.append({
                            "image_path": full_image_path,
                            "pos_captions": pos_captions,
                            "neg_captions": neg_captions,
                            "caption_options": pos_captions + neg_captions,
                            "label": 0,
                        })
                    elif self.task == "itm":
                        all_samples.append({
                            "image_path": full_image_path,
                            "captions": pos_captions,
                            "label": 1,
                        })
                        all_samples.append({
                            "image_path": full_image_path,
                            "captions": neg_captions,
                            "label": 0,
                        })
                
                if self.verbose:
                    print(f"[VLCheckListDataset]   - {corpus_name}: {len(raw_data)} raw items")
                    
            except Exception as e:
                if self.verbose:
                    print(f"[VLCheckListDataset] Error loading {corpus_name}: {e}")
                continue
        
        if self.verbose:
            print(f"[VLCheckListDataset] Total aggregated: {len(all_samples)} samples")
            if all_missing_images:
                unique_missing = list(set([os.path.basename(img) for img in all_missing_images]))
                print(f"[VLCheckListDataset] Warning: {len(all_missing_images)} samples skipped due to {len(unique_missing)} missing images")
        
        # Return only the aggregated samples to keep API consistent with _load_data()
        return all_samples

    def _load_data(self):
        """Load data from VL-CheckList corpus."""
        corpus_type, corpus_name = self._determine_corpus_type(self.subset_name)
        
        # Handle full type-level aggregation (e.g., "attr_color" loads from Attribute/color directory)
        if corpus_type == "ALL_SUBDIRS":
            return self._load_all_subdirs(corpus_name)
        
        # Handle aggregated datasets (load all corpus files from this category)
        if corpus_name == "ALL_DATASETS":
            return self._load_aggregated_data(corpus_type)
        
        # Load single dataset configuration
        config_path = os.path.join(
            self.vl_checklist_root, "corpus", self.version, corpus_type, f"{corpus_name}.yaml"
        )
        
        if not os.path.exists(config_path):
            if self.verbose:
                print(f"[VLCheckListDataset] Config not found: {config_path}")
                print("[VLCheckListDataset] Available corpus types might be different")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Load annotation data
        anno_path = os.path.join(self.vl_checklist_root, config["ANNO_PATH"])
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        
        with open(anno_path, 'r') as f:
            raw_data = json.load(f)
        
        # Image root directory
        img_root = os.path.join(self.vl_checklist_root, config["IMG_ROOT"])
        
        # Process data based on task type
        sample_list = []
        missing_images = []
        
        for item in raw_data:
            image_path, texts_dict = item
            full_image_path = os.path.join(img_root, image_path)
            
            # Check if image exists (collect missing images silently)
            if not os.path.exists(full_image_path):
                missing_images.append(full_image_path)
                continue
            
            pos_captions = texts_dict["POS"]
            neg_captions = texts_dict["NEG"]
            
            if self.task == "itc":
                # For image-text contrastive task
                sample_list.append({
                    "image_path": full_image_path,
                    "pos_captions": pos_captions,
                    "neg_captions": neg_captions,
                    "caption_options": pos_captions + neg_captions,
                    "label": 0,  # First caption is always positive in this setup
                })
            elif self.task == "itm":
                # For image-text matching, create separate samples for pos and neg
                sample_list.append({
                    "image_path": full_image_path,
                    "captions": pos_captions,
                    "label": 1,  # Positive
                })
                sample_list.append({
                    "image_path": full_image_path,
                    "captions": neg_captions,
                    "label": 0,  # Negative
                })
        
        if self.verbose:
            print(f"[VLCheckListDataset] Loaded {len(sample_list)} samples from {len(raw_data)} raw items")
            if missing_images:
                # Get unique missing image filenames
                unique_missing = list(set([os.path.basename(img) for img in missing_images]))
                unique_missing.sort()
                
                print(f"[VLCheckListDataset] Warning: {len(missing_images)} samples skipped due to {len(unique_missing)} missing images")
                if len(unique_missing) <= 10:
                    print(f"[VLCheckListDataset] Missing files: {', '.join(unique_missing)}")
                else:
                    print(f"[VLCheckListDataset] Missing files (first 10): {', '.join(unique_missing[:10])}")
                    print(f"[VLCheckListDataset] ... and {len(unique_missing) - 10} more")
                
                # Provide specific download instructions based on missing image paths
                self._provide_download_instructions(missing_images[:5])
        
        return sample_list

    def _provide_download_instructions(self, missing_image_paths):
        """Provide specific download instructions based on missing image paths."""
        if not missing_image_paths:
            return
            
        # Analyze missing paths to determine what needs to be downloaded
        needs_hake_openimages = any("hake/openimages" in path for path in missing_image_paths)
        needs_hake_custom = any("hake/hake_images" in path for path in missing_image_paths)
        needs_swig = any("swig/" in path for path in missing_image_paths)
        needs_vg = any("vg/" in path for path in missing_image_paths)
        
        if self.verbose:
            print("\n[VLCheckListDataset] === DOWNLOAD INSTRUCTIONS ===")
            
            if needs_hake_openimages:
                print("[VLCheckListDataset] Missing HAKE OpenImages. Download from:")
                print("[VLCheckListDataset]   https://drive.google.com/open?id=1XTWYLyL1h-9jJ49dsXmtRCv8GcupVrvM")
                print("[VLCheckListDataset]   Extract to: VL-CheckList/hake/openimages/")
            
            if needs_hake_custom:
                print("[VLCheckListDataset] Missing HAKE custom images. Download from:")
                print("[VLCheckListDataset]   2019: https://drive.google.com/open?id=18R_3Oz7zO1knEjagY6sfUkQ1_6wZf0Ei")
                print("[VLCheckListDataset]   2020: https://drive.google.com/open?id=14K_4FfjviJNDVLJdGM96W2ZLN55dDb2-")
                print("[VLCheckListDataset]   Extract to: VL-CheckList/hake/hake_images_*/")
            
            if needs_swig:
                print("[VLCheckListDataset] Missing SWiG images. Download from:")
                print("[VLCheckListDataset]   https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip")
                print("[VLCheckListDataset]   Extract to: VL-CheckList/swig/")
            
            if needs_vg:
                print("[VLCheckListDataset] Missing Visual Genome images. Download from:")
                print("[VLCheckListDataset]   Part1: https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip")
                print("[VLCheckListDataset]   Part2: https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip") 
                print("[VLCheckListDataset]   Extract Part1 to: VL-CheckList/vg/VG_100K/")
                print("[VLCheckListDataset]   Extract Part2 to: VL-CheckList/vg/VG_100K_2/")
            
            print("[VLCheckListDataset] === END INSTRUCTIONS ===\n")

    def _build_caption_mappings(self):
        """Build caption vocabulary and mappings."""
        caption_set = set()
        
        if self.task == "itc":
            for sample in self.sample_list:
                caption_set.update(sample["caption_options"])
        elif self.task == "itm":
            for sample in self.sample_list:
                caption_set.update(sample["captions"])
        
        self.captions = sorted(caption_set)
        self.caption_to_idx = {cap: i for i, cap in enumerate(self.captions)}

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> Dict:
        idx = int(idx)
        sample = self.sample_list[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert('RGB')
        
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        
        if self.task == "itc":
            # Image-text contrastive task
            return {
                "image_options": image,
                "caption_options": sample["caption_options"],
                "label": sample["label"],
                "pos_captions": sample["pos_captions"],
                "neg_captions": sample["neg_captions"],
                "image_path": sample["image_path"],
            }
        elif self.task == "itm":
            # Image-text matching task
            return {
                "image_options": image,
                "caption_options": sample["captions"],
                "label": sample["label"],
                "image_path": sample["image_path"],
            }

    def get_captions(self) -> List[str]:
        """Return the unique caption vocabulary."""
        return self.captions

    def get_image_paths(self) -> List[str]:
        """Return list of unique image paths."""
        if self.task == "itc":
            return list(set(sample["image_path"] for sample in self.sample_list))
        elif self.task == "itm":
            return list(set(sample["image_path"] for sample in self.sample_list))

    def get_idx_to_ptr(self, idx: int) -> int:
        """Map dataset index -> positive caption pointer."""
        sample = self.sample_list[idx]
        if self.task == "itc":
            # First caption in pos_captions
            caption = sample["pos_captions"][0]
        elif self.task == "itm":
            # For ITM, if label is 1, use first caption, else -1 
            if sample["label"] == 1:
                caption = sample["captions"][0]
            else:
                return -1
        return self.caption_to_idx.get(caption, -1)

    def get_idx_to_candidates_ptr(self, idx: int) -> List[int]:
        """Map dataset index -> list of negative caption indices."""
        sample = self.sample_list[idx]
        if self.task == "itc":
            neg_captions = sample["neg_captions"]
            return [self.caption_to_idx.get(cap) for cap in neg_captions if cap in self.caption_to_idx]
        elif self.task == "itm":
            # For ITM, if label is 0, return the caption indices, else empty
            if sample["label"] == 0:
                captions = sample["captions"]
                return [self.caption_to_idx.get(cap) for cap in captions if cap in self.caption_to_idx]
            else:
                return []

    def evaluate(
        self,
        embedding_model,
        aligning_model=None,
        device="cuda",
        batch_size: int = 64,  # Increased default batch size
        indices: Optional[List[int]] = None,
        intermediate_text_layer_names: List[str] = ["final"],
        intermediate_image_layer_names: List[str] = ["final"],
        precomputed_caption_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        return_embeddings: bool = True,  # NEW: Control embedding return
        max_samples_for_embeddings: int = 5000,  # NEW: Don't return embeddings if dataset is larger
    ):
        """Public wrapper for evaluation with optional pre-computed caption embeddings.
        
        Args:
            ...existing args...
            return_embeddings: If True, return embeddings. If False, only return metrics (saves memory).
            max_samples_for_embeddings: Maximum dataset size to return embeddings. If dataset is larger,
                                        embeddings won't be returned regardless of return_embeddings flag.
        """
        return self._evaluate(
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device,
            batch_size=batch_size,
            indices=indices,
            intermediate_text_layer_names=intermediate_text_layer_names,
            intermediate_image_layer_names=intermediate_image_layer_names,
            precomputed_caption_embeddings=precomputed_caption_embeddings,
            return_embeddings=return_embeddings,
            max_samples_for_embeddings=max_samples_for_embeddings,
        )

    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader to handle variable length caption lists.
        Efficiently batches VL-CheckList samples with proper image stacking and caption grouping.
        
        Args:
            batch: List of dataset samples
            
        Returns:
            dict: Batched data with stacked images and grouped captions
        """
        batch_images = []
        batch_pos_captions = []
        batch_neg_captions = []
        
        for sample in batch:
            # Extract components from each sample using correct keys
            image = sample['image_options']  # Already preprocessed tensor
            pos_captions = sample['pos_captions']  # List of positive captions
            neg_captions = sample['neg_captions']  # List of negative captions
            
            batch_images.append(image)
            batch_pos_captions.append(pos_captions)
            batch_neg_captions.append(neg_captions)
        
        # Stack images into a batch tensor
        batch_images = torch.stack(batch_images)  # [B, C, H, W]
        
        return {
            'images': batch_images,
            'pos_captions': batch_pos_captions,  # List[List[str]] - preserves variable lengths
            'neg_captions': batch_neg_captions   # List[List[str]] - preserves variable lengths
        }

    def _evaluate(
        self,
        embedding_model,
        aligning_model=None,
        device="cuda",
        batch_size: int = 64,  # Increased default batch size
        indices: Optional[List[int]] = None,
        intermediate_text_layer_names: List[str] = ["final"],
        intermediate_image_layer_names: List[str] = ["final"],
        precomputed_caption_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        return_embeddings: bool = True,  # NEW: Control embedding return
        max_samples_for_embeddings: int = 5000,  # NEW: Don't return embeddings if dataset is larger
    ):
        """
        Evaluate VL-CheckList dataset with DataLoader optimization and caching.
        
        For each sample, we have:
        - image, positive captions, negative captions
        - Goal: positive captions should score higher than negative captions for the image
        
        We compute:
        - Accuracy: fraction of samples where max(pos_sim) > max(neg_sim)
        
        Args:
            return_embeddings: If True, store and return all embeddings (memory intensive)
            max_samples_for_embeddings: If dataset size > this value, don't return embeddings
        """
        if 'compute_caption_embeddings_intermediate_batch' not in globals():
            raise ImportError("utils.align functions not available for evaluation")
        
        try:
            from data_loading.embedding_cache import EmbeddingCache
        except ImportError:
            raise ImportError("embedding_cache not available for evaluation")
        
        from torch.utils.data import DataLoader, Subset
        
        if self.task == "itm":
            # For ITM task, use standard binary classification
            return self._evaluate_itm(
                embedding_model, aligning_model, device, batch_size, indices,
                intermediate_text_layer_names, intermediate_image_layer_names
            )
        
        # Create subset dataset if indices provided
        if indices is not None:
            eval_dataset = Subset(self, indices)
        else:
            eval_dataset = self
        
        # Determine if we should return embeddings based on dataset size
        dataset_size = len(eval_dataset)
        should_return_embeddings = return_embeddings and (dataset_size <= max_samples_for_embeddings)
        
        if not should_return_embeddings and dataset_size > max_samples_for_embeddings:
            if self.verbose:
                print(f"[VLCheckListDataset] Dataset size ({dataset_size}) exceeds max_samples_for_embeddings ({max_samples_for_embeddings})")
                print(f"[VLCheckListDataset] Embeddings will NOT be returned to save memory")
            
        # Use DataLoader for efficient multi-threaded batch loading
        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Multi-threaded loading
            pin_memory=True,  # Faster GPU transfer
            collate_fn=self._collate_fn  # Custom collate function
        )
        
        # For ITC task - OPTIMIZED VERSION WITH CACHING
        correct_predictions = []
        image_emb_list = []
        pos_caption_emb_list = []
        neg_caption_emb_by_sample = []  # Store negative captions grouped by sample
        
        # Track cumulative caption counts for proper cache indexing
        cumulative_pos_captions = 0
        cumulative_neg_captions = 0
        
        # Use embedding cache context manager
        with EmbeddingCache(
            dataset_name="VL_CheckList",
            subset_name=self.subset_name,
            embedding_model=embedding_model,
            aligning_model=aligning_model,
            device=device
        ) as cache:
            
            # Now evaluate in batches using DataLoader - simplified approach with proper caching!
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating VL-CheckList")):
                # Batch is already properly formatted by collate_fn
                batch_images = batch['images']  # [B, C, H, W]
                batch_pos_captions = batch['pos_captions']  # List[List[str]]
                batch_neg_captions = batch['neg_captions']  # List[List[str]]
                B = len(batch_pos_captions)
                
                if B == 0:
                    continue
                
                with torch.no_grad():
                    # Collect all positive and negative captions for the batch
                    all_pos_captions = []
                    all_neg_captions = []
                    pos_indices = []  # Track which image each caption belongs to
                    neg_indices = []
                    sample_has_pos = []  # Track which samples have pos captions
                    sample_has_neg = []  # Track which samples have neg captions
                    
                    for i in range(B):
                        pos_captions = batch_pos_captions[i]
                        neg_captions = batch_neg_captions[i]
                        
                        # Track which samples have captions
                        sample_has_pos.append(len(pos_captions) > 0)
                        sample_has_neg.append(len(neg_captions) > 0)
                        
                        # Add all positive captions with their image indices
                        all_pos_captions.extend(pos_captions)
                        pos_indices.extend([i] * len(pos_captions))
                        
                        # Add all negative captions with their image indices
                        all_neg_captions.extend(neg_captions)
                        neg_indices.extend([i] * len(neg_captions))
                    
                    batch_correct = []
                    
                    if all_pos_captions and all_neg_captions:
                        # Standard CLIP path
                        # Get image embeddings (with caching)
                        img_embs = cache.get_or_compute_embeddings(
                            batch_images.to(device),
                            "image",
                            compute_image_embeddings_intermediate_batch,
                            intermediate_image_layer_names,
                            start_idx=batch_idx * batch_size
                        )
                        
                        # Get positive caption embeddings with caching
                        pos_cap_embs = cache.get_or_compute_embeddings(
                            all_pos_captions,
                            "text",
                            compute_caption_embeddings_intermediate_batch,
                            intermediate_text_layer_names,
                            start_idx=cumulative_pos_captions
                        )
                        
                        # Get negative caption embeddings with caching 
                        neg_cap_embs = cache.get_or_compute_embeddings(
                            all_neg_captions,
                            "foil",
                            compute_caption_embeddings_intermediate_batch,
                            intermediate_text_layer_names,
                            start_idx=cumulative_neg_captions
                        )
                        
                        # Update cumulative counters for next batch
                        cumulative_pos_captions += len(all_pos_captions)
                        cumulative_neg_captions += len(all_neg_captions)
                        
                        # Convert indices to tensors for vectorized operations
                        pos_indices_tensor = torch.tensor(pos_indices, device=device)
                        neg_indices_tensor = torch.tensor(neg_indices, device=device)
                        
                        # Vectorized similarity computation for ALL captions vs ALL images
                        pos_sims = torch.matmul(pos_cap_embs, img_embs.T)
                        neg_sims = torch.matmul(neg_cap_embs, img_embs.T)
                        
                        # Extract similarities for correct image-caption pairs
                        pos_sims_matched = pos_sims[torch.arange(len(pos_indices_tensor)), pos_indices_tensor]
                        neg_sims_matched = neg_sims[torch.arange(len(neg_indices_tensor)), neg_indices_tensor]
                        
                        # Find max similarities per image
                        max_pos_sims = torch.full((B,), float('-inf'), device=device)
                        max_neg_sims = torch.full((B,), float('-inf'), device=device)
                        
                        if len(pos_indices_tensor) > 0:
                            max_pos_sims.scatter_reduce_(
                                0, pos_indices_tensor, pos_sims_matched, reduce='amax', include_self=False
                            )
                        
                        if len(neg_indices_tensor) > 0:
                            max_neg_sims.scatter_reduce_(
                                0, neg_indices_tensor, neg_sims_matched, reduce='amax', include_self=False
                            )
                        
                        for i in range(B):
                            if sample_has_pos[i] and sample_has_neg[i]:
                                batch_correct.append((max_pos_sims[i] > max_neg_sims[i]).item())
                    
                    # Extend results
                    correct_predictions.extend(batch_correct)
                    
                    # Store embeddings for analysis only if requested and dataset size allows
                    if should_return_embeddings:
                        # Convert to numpy for consistency
                        # Store image embeddings
                        for img_emb in img_embs.cpu():
                            image_emb_list.append(img_emb.numpy())
                        
                        # Store positive caption embeddings  
                        if all_pos_captions:
                            for i, cap in enumerate(all_pos_captions):
                                pos_caption_emb_list.append(pos_cap_embs[i].cpu().numpy())
                        
                        # Store negative caption embeddings grouped by sample
                        for sample_idx in range(B):
                            sample_neg_embs = []
                            sample_neg_captions = batch_neg_captions[sample_idx]
                            
                            for cap in sample_neg_captions:
                                # Find this caption in all_neg_captions to get the embedding
                                try:
                                    cap_idx = all_neg_captions.index(cap)
                                    sample_neg_embs.append(neg_cap_embs[cap_idx].cpu().numpy())
                                except ValueError:
                                    # Caption not found, skip
                                    continue
                            
                            if sample_neg_embs:
                                # Stack negative embeddings for this sample: [K, D]
                                neg_caption_emb_by_sample.append(np.stack(sample_neg_embs, axis=0))
                            else:
                                # No negative captions for this sample - create empty placeholder
                                # Use first positive caption embedding dimension as reference
                                if pos_caption_emb_list:
                                    embed_dim = pos_caption_emb_list[0].shape[0]
                                    neg_caption_emb_by_sample.append(np.zeros((0, embed_dim)))
                                else:
                                    neg_caption_emb_by_sample.append(np.array([]))
        
        # Calculate final accuracy - convert to float
        if len(correct_predictions) > 0:
            accuracy = float(sum(correct_predictions) / len(correct_predictions))
        else:
            accuracy = 0.0
        
        # Build embeddings dictionary only if embeddings were collected
        if should_return_embeddings:
            # Stack embeddings into numpy arrays like other datasets
            # For negative captions, we need to handle variable number of negatives per sample
            if neg_caption_emb_by_sample and any(len(sample_negs) > 0 for sample_negs in neg_caption_emb_by_sample):
                # Find max number of negative captions across all samples
                max_neg_captions = max(sample_negs.shape[0] for sample_negs in neg_caption_emb_by_sample if len(sample_negs) > 0)
                if max_neg_captions > 0:
                    # Get embedding dimension from first non-empty sample
                    first_valid_sample = next(sample_negs for sample_negs in neg_caption_emb_by_sample if len(sample_negs) > 0)
                    embed_dim = first_valid_sample.shape[1]
                    
                    # Create padded array [N, K, D] where K is max number of negative captions
                    neg_embeddings = np.zeros((len(neg_caption_emb_by_sample), max_neg_captions, embed_dim))
                    for i, sample_negs in enumerate(neg_caption_emb_by_sample):
                        if len(sample_negs) > 0:
                            neg_embeddings[i, :sample_negs.shape[0], :] = sample_negs
                else:
                    neg_embeddings = np.array([])
            else:
                neg_embeddings = np.array([])
            
            embeddings = {
                'image_embeddings': np.stack(image_emb_list, axis=0) if image_emb_list else np.array([]),
                'caption_embeddings': np.stack(pos_caption_emb_list, axis=0) if pos_caption_emb_list else np.array([]),
                'negative_caption_embeddings': neg_embeddings
            }
        else:
            # Return empty embeddings to save memory
            embeddings = {
                'image_embeddings': np.array([]),
                'caption_embeddings': np.array([]),
                'negative_caption_embeddings': np.array([])
            }

        results = {
            'contrastive_accuracy': accuracy,
        }
        return results, embeddings

    def _evaluate_itm(
        self,
        embedding_model,
        aligning_model,
        device,
        batch_size,
        iterate,
        intermediate_text_layer_names,
        intermediate_image_layer_names,
    ):
        """Evaluate for image-text matching task."""
        # For ITM, each sample has binary label (0=negative, 1=positive)
        # We compute similarity and use threshold or ranking
        
        all_similarities = []
        all_labels = []
        
        for start in tqdm(range(0, len(iterate), batch_size), desc="Evaluating VL-CheckList ITM"):
            batch_indices = iterate[start:start + batch_size]
            
            batch_images = []
            batch_captions = []
            batch_labels = []
            
            for idx in batch_indices:
                sample = self[idx]
                batch_images.append(sample['image_options'])
                # For ITM, caption_options is a list of captions
                batch_captions.extend(sample['caption_options'])
                batch_labels.append(sample['label'])
            
            B = len(batch_indices)
            if B == 0:
                continue
            
            with torch.no_grad():
                # Process images
                img_tensors = torch.stack(batch_images).to(device)
                img_feats = compute_image_embeddings_intermediate_batch(
                    img_tensors, embedding_model, device=device,
                    intermediate_layer_names=intermediate_image_layer_names
                )
                
                if aligning_model is not None:
                    img_embs = aligning_model.encode_image(img_feats)
                else:
                    img_embs = img_feats["final"]
                
                img_embs = img_embs.float().to('cuda')
                img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
                
                # Process captions
                cap_feats = compute_caption_embeddings_intermediate_batch(
                    batch_captions, embedding_model, device=device,
                    intermediate_layer_names=intermediate_text_layer_names
                )
                
                if aligning_model is not None:
                    cap_embs = aligning_model.encode_text(cap_feats)
                else:
                    cap_embs = cap_feats["final"]
                
                cap_embs = cap_embs.float().to('cuda')
                cap_embs = cap_embs / cap_embs.norm(dim=-1, keepdim=True)
                
                # Compute similarities (assuming one caption per image for ITM)
                similarities = torch.sum(img_embs * cap_embs, dim=1)
                
                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(batch_labels)
        
        # Compute binary classification accuracy using 0.5 threshold
        predictions = [1 if sim > 0.5 else 0 for sim in all_similarities]
        accuracy = np.mean([pred == label for pred, label in zip(predictions, all_labels)])
        
        results = {
            "binary_accuracy": float(accuracy),
            "accuracy": float(accuracy),
        }
        
        # Create dummy embeddings for compatibility
        embeddings = {
            "image_embeddings": np.array(all_similarities).reshape(-1, 1),
            "caption_embeddings": np.array(all_labels).reshape(-1, 1),
            "negative_caption_embeddings": np.zeros((len(all_labels), 1, 1)),
        }
        
        return results, embeddings

    def split_dataset(
        self,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        seed: int = 42,
        split_type: Literal["random", "object"] = "random",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split dataset into train/val/test.
        """
        rng = random.Random(seed)
        
        if split_type == "random":
            indices = list(range(len(self)))
            rng.shuffle(indices)
            n = len(indices)
            n_test = int(n * test_ratio)
            n_val = int((n - n_test) * val_ratio)
            
            test_idx = np.array(indices[:n_test], dtype=np.int64)
            val_idx = np.array(indices[n_test:n_test + n_val], dtype=np.int64)
            train_idx = np.array(indices[n_test + n_val:], dtype=np.int64)
            
            return {
                "train": {"indices": train_idx},
                "val": {"indices": val_idx},
                "test": {"indices": test_idx},
            }
        
        elif split_type == "object":
            # Group by image path for VL-CheckList
            image_to_indices = defaultdict(list)
            for i, sample in enumerate(self.sample_list):
                image_path = sample["image_path"]
                image_to_indices[image_path].append(i)
            
            image_paths = list(image_to_indices.keys())
            rng.shuffle(image_paths)
            
            n = len(image_paths)
            n_test = int(n * test_ratio)
            n_val = int((n - n_test) * val_ratio)
            
            test_images = image_paths[:n_test]
            val_images = image_paths[n_test:n_test + n_val]
            train_images = image_paths[n_test + n_val:]
            
            test_idx = [i for img in test_images for i in image_to_indices[img]]
            val_idx = [i for img in val_images for i in image_to_indices[img]]
            train_idx = [i for img in train_images for i in image_to_indices[img]]
            
            return {
                "train": {"indices": np.array(train_idx, dtype=np.int64)},
                "val": {"indices": np.array(val_idx, dtype=np.int64)},
                "test": {"indices": np.array(test_idx, dtype=np.int64)},
            }
        
        else:
            raise ValueError(f"Unknown split_type={split_type}")

    @staticmethod
    def list_all_subsets(category: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all available VL-CheckList subsets.
        
        Args:
            category: Optional category filter ('attribute', 'object', 'relation')
                     If None, returns all subsets grouped by category.
        
        Returns:
            Dictionary mapping categories to lists of subset names.
        """
        all_subsets = {
            'Attribute': {
                'action': ['vaw_action', 'vg_action', 'attribute_action_vaw', 'attribute_action_vg'],
                'color': ['vaw_color', 'vg_color', 'attribute_color_vaw', 'attribute_color_vg'],
                'material': ['vaw_material', 'vg_material', 'attribute_material_vaw', 'attribute_material_vg'],
                'size': ['vaw_size', 'vg_size', 'attribute_size_vaw', 'attribute_size_vg'],
                'state': ['vaw_state', 'vg_state', 'attribute_state_vaw', 'attribute_state_vg'],
            },
            'Object': {
                'location_center': [
                    'object_location_center_hake', 'object_location_center_swig_agent',
                    'object_location_center_swig_destination', 'object_location_center_swig_item',
                    'object_location_center_swig_tool', 'object_location_center_vg_obj',
                    'object_location_center_vg_subj'
                ],
                'location_margin': [
                    'object_location_margin_hake', 'object_location_margin_swig_agent',
                    'object_location_margin_swig_destination', 'object_location_margin_swig_item',
                    'object_location_margin_swig_tool', 'object_location_margin_vg_obj',
                    'object_location_margin_vg_subj'
                ],
                'location_mid': [
                    'object_location_mid_hake', 'object_location_mid_swig_agent',
                    'object_location_mid_swig_destination', 'object_location_mid_swig_item',
                    'object_location_mid_swig_tool', 'object_location_mid_vg_obj',
                    'object_location_mid_vg_subj'
                ],
                'size_large': [
                    'object_size_large_hake', 'object_size_large_swig_agent',
                    'object_size_large_swig_destination', 'object_size_large_swig_item',
                    'object_size_large_swig_tool', 'object_size_large_vg_obj',
                    'object_size_large_vg_subj'
                ],
                'size_medium': [
                    'object_size_medium_hake', 'object_size_medium_swig_agent',
                    'object_size_medium_swig_destination', 'object_size_medium_swig_item',
                    'object_size_medium_swig_tool', 'object_size_medium_vg_obj',
                    'object_size_medium_vg_subj'
                ],
                'size_small': [
                    'object_size_small_hake', 'object_size_small_swig_agent',
                    'object_size_small_swig_destination', 'object_size_small_swig_item',
                    'object_size_small_swig_tool', 'object_size_small_vg_obj',
                    'object_size_small_vg_subj'
                ],
            },
            'Relation': {
                'action': ['hake_action', 'swig_action', 'vg_action_relation',
                          'relation_action_hake', 'relation_action_swig', 'relation_action_vg'],
                'spatial': ['vg_spatial', 'relation_spatial_vg'],
            }
        }
        
        if category is None:
            return all_subsets
        
        category_normalized = category.lower().capitalize()
        if category_normalized in all_subsets:
            return {category_normalized: all_subsets[category_normalized]}
        
        return {}

    @staticmethod
    def print_available_subsets():
        """Print all available VL-CheckList subsets in a formatted way."""
        subsets = VLCheckListDataset.list_all_subsets()
        
        print("\n" + "="*70)
        print("VL-CheckList Available Subsets")
        print("="*70)
        
        for category, subcategories in subsets.items():
            print(f"\n📁 {category.upper()}")
            print("-" * 70)
            
            for subcat, subset_list in subcategories.items():
                print(f"\n  {subcat.replace('_', ' ').title()}:")
                # Show only the canonical names (shorter ones)
                canonical = [s for s in subset_list if not s.startswith(category.lower())]
                if not canonical:
                    canonical = subset_list[:len(subset_list)//2]  # Take first half if all are long form
                
                for subset in canonical:
                    print(f"    - {subset}")
        
        print("\n" + "="*70)
        print("Usage:")
        print("  dataset = VLCheckListDataset(")
        print("      data_root='path/to/data',")
        print("      subset_name='vaw_color',  # or any subset from above")
        print("      download=True")
        print("  )")
        print("="*70 + "\n")

    @staticmethod
    def setup_dependencies():
        """Install required dependencies for downloading datasets."""
        required_packages = ["requests", "gdown", "tqdm"]
        try:
            import subprocess
            import sys
            
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    print(f"Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
            print("All dependencies installed successfully!")
            return True
        except Exception as e:
            print(f"Failed to install dependencies: {e}")
            print("Please install manually: pip install requests gdown tqdm")
            return False

    @staticmethod
    def print_setup_guide():
        """Print a complete setup guide for VL-CheckList dataset."""
        print("\n" + "="*60)
        print("VL-CheckList Dataset Setup Guide")
        print("="*60)
        print("1. Clone VL-CheckList repository:")
        print("   git clone https://github.com/om-ai-lab/VL-CheckList.git")
        print("   mv VL-CheckList /path/to/your/data_root/")
        print("")
        print("2. Install required Python packages:")
        print("   pip install requests gdown tqdm")
        print("")
        print("3. Download images for your specific use case:")
        print("")
        print("   For HAKE (Human Action Recognition):")
        print("   - OpenImages subset: https://drive.google.com/open?id=1XTWYLyL1h-9jJ49dsXmtRCv8GcupVrvM")
        print("   - HAKE 2019: https://drive.google.com/open?id=18R_3Oz7zO1knEjagY6sfUkQ1_6wZf0Ei")
        print("   - HAKE 2020: https://drive.google.com/open?id=14K_4FfjviJNDVLJdGM96W2ZLN55dDb2-")
        print("   - Extract to: VL-CheckList/hake/[openimages|hake_images_*/]/")
        print("")
        print("   For SWiG (Situation With Groundings):")
        print("   - Download: https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip")
        print("   - Extract to: VL-CheckList/swig/")
        print("")
        print("   For Visual Genome & VAW:")
        print("   - Part1: https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip")
        print("   - Part2: https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip")
        print("   - Extract Part1 to: VL-CheckList/vg/VG_100K/")
        print("   - Extract Part2 to: VL-CheckList/vg/VG_100K_2/")
        print("")
        print("4. Your final directory structure should look like:")
        print("   data_root/")
        print("   └── VL-CheckList/")
        print("       ├── corpus/")
        print("       ├── data/")
        print("       ├── hake/")
        print("       │   ├── openimages/")
        print("       │   ├── hake_images_20190730/")
        print("       │   └── hake_images_20200614/")
        print("       ├── swig/")
        print("       └── vg/")
        print("           ├── VG_100K/")
        print("           └── VG_100K_2/")
        print("")
        print("5. Use the dataset:")
        print("   dataset = VLCheckListDataset(")
        print("       data_root='your_data_root',")
        print("       subset_name='hake_action',  # or 'swig_action', 'vg_color', etc.")
        print("       download=True  # This will attempt automatic download")
        print("   )")
        print("")
        print("6. For automatic setup, you can also use:")
        print("   from data_loading.vl_checklist import VLCheckListDataset")
        print("   VLCheckListDataset.auto_setup_dataset(")
        print("       data_root='your_data_root',")
        print("       subset_name='hake_action'")
        print("   )")
        print("="*60 + "\n")

    @staticmethod
    def auto_setup_dataset(data_root: str, subset_name: str = "hake_action", verbose: bool = True):
        """Automatically set up the VL-CheckList dataset with images."""
        
        if verbose:
            print("\n🚀 VL-CheckList Automatic Setup")
            print("=" * 40)
        
        # Check if VL-CheckList repo exists
        vl_checklist_path = os.path.join(data_root, "VL-CheckList")
        if not os.path.exists(vl_checklist_path):
            if verbose:
                print("❌ VL-CheckList repository not found")
                print("📥 Attempting to clone repository...")
            
            try:
                import subprocess
                result = subprocess.run([
                    "git", "clone", 
                    "https://github.com/om-ai-lab/VL-CheckList.git",
                    vl_checklist_path
                ], capture_output=True, text=True, cwd=data_root)
                
                if result.returncode == 0:
                    if verbose:
                        print("✅ Successfully cloned VL-CheckList repository")
                else:
                    if verbose:
                        print(f"❌ Failed to clone repository: {result.stderr}")
                        print("Please clone manually:")
                        print(f"  cd {data_root}")
                        print("  git clone https://github.com/om-ai-lab/VL-CheckList.git")
                    return False
            except FileNotFoundError:
                if verbose:
                    print("❌ Git not found. Please install git or clone manually:")
                    print(f"  cd {data_root}")
                    print("  git clone https://github.com/om-ai-lab/VL-CheckList.git")
                return False
        else:
            if verbose:
                print("✅ VL-CheckList repository found")
        
        # Create dataset instance with automatic download
        try:
            if verbose:
                print(f"📦 Setting up dataset: {subset_name}")
            
            dataset = VLCheckListDataset(
                data_root=data_root,
                subset_name=subset_name,
                download=True,
                verbose=verbose
            )
            
            if len(dataset) > 0:
                if verbose:
                    print(f"🎉 Setup completed! Dataset ready with {len(dataset)} samples")
                return True
            else:
                if verbose:
                    print("⚠️ Setup completed but no samples found - check image downloads")
                return False
                
        except Exception as e:
            if verbose:
                print(f"❌ Setup failed: {e}")
            return False

    def check_dataset_integrity(self) -> dict:
        """Check the integrity and completeness of the dataset."""
        stats = {
            "total_annotations": 0,
            "found_images": 0,
            "missing_images": 0,
            "missing_image_paths": [],
            "image_directories": {},
            "status": "unknown"
        }
        
        # Load raw data to check against
        try:
            corpus_type, corpus_name = self._determine_corpus_type(self.subset_name)
            config_path = os.path.join(
                self.vl_checklist_root, "corpus", self.version, corpus_type, f"{corpus_name}.yaml"
            )
            
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            
            anno_path = os.path.join(self.vl_checklist_root, config["ANNO_PATH"])
            with open(anno_path, 'r') as f:
                raw_data = json.load(f)
            
            img_root = os.path.join(self.vl_checklist_root, config["IMG_ROOT"])
            
            stats["total_annotations"] = len(raw_data)
            
            # Check each image
            for item in raw_data:
                image_path, _ = item
                full_image_path = os.path.join(img_root, image_path)
                
                if os.path.exists(full_image_path):
                    stats["found_images"] += 1
                else:
                    stats["missing_images"] += 1
                    stats["missing_image_paths"].append(full_image_path)
                
                # Track directory structure
                dir_name = os.path.dirname(image_path)
                if dir_name not in stats["image_directories"]:
                    stats["image_directories"][dir_name] = {"total": 0, "found": 0}
                stats["image_directories"][dir_name]["total"] += 1
                if os.path.exists(full_image_path):
                    stats["image_directories"][dir_name]["found"] += 1
            
            # Determine status
            if stats["missing_images"] == 0:
                stats["status"] = "complete"
            elif stats["found_images"] > 0:
                stats["status"] = "partial"
            else:
                stats["status"] = "empty"
                
        except Exception as e:
            stats["status"] = "error"
            stats["error"] = str(e)
        
        return stats

    def print_dataset_status(self):
        """Print a detailed status report of the dataset."""
        stats = self.check_dataset_integrity()
        
        print(f"\n📊 VL-CheckList Dataset Status: {self.subset_name}")
        print("=" * 50)
        print(f"Total annotations: {stats['total_annotations']}")
        print(f"Found images: {stats['found_images']}")
        print(f"Missing images: {stats['missing_images']}")
        
        if stats['total_annotations'] > 0:
            completion = (stats['found_images'] / stats['total_annotations']) * 100
            print(f"Completion: {completion:.1f}%")
        
        print(f"Status: {stats['status'].upper()}")
        
        if stats.get('image_directories'):
            print("\nImage directories:")
            for dir_name, dir_stats in stats['image_directories'].items():
                dir_completion = (dir_stats['found'] / dir_stats['total']) * 100 if dir_stats['total'] > 0 else 0
                status_icon = "✅" if dir_completion == 100 else "⚠️" if dir_completion > 0 else "❌"
                print(f"  {status_icon} {dir_name}: {dir_stats['found']}/{dir_stats['total']} ({dir_completion:.1f}%)")
        
        if stats['missing_images'] > 0:
            print(f"\nMissing {stats['missing_images']} images")
            if stats['missing_images'] <= 10:
                print("Missing files:")
                for path in stats['missing_image_paths']:
                    print(f"  - {path}")
            else:
                print("First 10 missing files:")
                for path in stats['missing_image_paths'][:10]:
                    print(f"  - {path}")
                print(f"  ... and {stats['missing_images'] - 10} more")
            
            # Provide download instructions
            self._provide_download_instructions(stats['missing_image_paths'][:5])
        
        print("=" * 50 + "\n")

    def __getstate__(self):
        """Ensure the dataset can be pickled for multiprocessing compatibility."""
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)


class VLCheckListNeg(Dataset):
    """
    Wrapper for VLCheckListDataset to provide tokenized positives and negatives.
    Compatible with other *Neg dataset classes.
    
    For VL-CheckList, we sample negatives from the negative captions.
    """
    
    def __init__(
        self,
        vl_checklist_dataset: VLCheckListDataset,
        indices: List[int],
        num_negatives: int = 1,
    ):
        super().__init__()
        self.dataset = vl_checklist_dataset
        self.num_negatives = num_negatives
        self.idx_to_dataset_idx = {i: idx for i, idx in enumerate(indices)}
        
        print(f"VLCheckListNeg initialized with {self.num_negatives} negatives per sample")
        print("VL-CheckList dataset for compositional vision-language reasoning")

    def __len__(self) -> int:
        return len(self.idx_to_dataset_idx)
    
    def __getitem__(self, idx: int):
        idx = int(idx)
        sample = self.dataset[self.idx_to_dataset_idx[idx]]
        
        # Get positive image and caption
        pos_image = sample['image_options']
        pos_captions = sample['pos_captions']
        neg_captions = sample['neg_captions']
        
        # Sample positive caption
        pos_text = random.choice(pos_captions) if pos_captions else ""
        
        # Sample negative caption
        neg_text = random.choice(neg_captions) if neg_captions else ""
        
        # Tokenize using CLIP
        import clip
        pos_tok = clip.tokenize(pos_text, truncate=True).squeeze(0)
        neg_tok = clip.tokenize(neg_text, truncate=True).squeeze(0)
        
        # For compatibility, create all_neg_tokens with multiple negatives if available
        if self.num_negatives > 1 and len(neg_captions) > 1:
            sampled_negs = random.sample(neg_captions, min(self.num_negatives, len(neg_captions)))
        else:
            sampled_negs = [neg_text]
        
        all_neg_toks = clip.tokenize(sampled_negs, truncate=True)  # Shape: (num_negs, seq_len)
        
        return pos_image, pos_tok, neg_tok, all_neg_toks
    
    def collate_fn(self, batch: List[tuple]) -> dict:
        images, pos_toks, neg_toks, all_neg_toks = zip(*batch)
        
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images, dim=0)
        
        pos_toks = torch.stack(pos_toks, dim=0)
        neg_toks = torch.stack(neg_toks, dim=0)
        
        # Handle variable number of negatives
        max_negs = max(nt.shape[0] for nt in all_neg_toks)
        padded_neg_toks = []
        for nt in all_neg_toks:
            if nt.shape[0] < max_negs:
                # Pad with the last token repeated
                padding = nt[-1:].repeat(max_negs - nt.shape[0], 1)
                nt = torch.cat([nt, padding], dim=0)
            padded_neg_toks.append(nt)
        
        all_neg_toks = torch.stack(padded_neg_toks, dim=0)
        
        return {
            'images': images,
            'pos_tokens': pos_toks,
            'neg_token': neg_toks,
            'all_neg_tokens': all_neg_toks,
        }

if __name__ == "__main__":
    dataset_root = 'datasets/'
    VLCheckListDataset(
        data_root=dataset_root,
        subset_name="hake_action",
        download=True,
        verbose=True
    )
#!/usr/bin/env python3
"""
ELEVATER Benchmark Evaluation Script

This script evaluates models on the ELEVATER Image Classification benchmark.
It uses the same model loading mechanism as batch_evaluate_checkpoints.py
and provides a compatible interface with the ELEVATER toolkit.

Usage:
    # Evaluate a single model
    python scripts/evaluate_elevater.py --config configs/elevater_eval.yaml

    # Evaluate with command-line arguments
    python scripts/evaluate_elevater.py \
        --checkpoint_type openclip \
        --checkpoint_path "ViT-B/32" \
        --output_dir ./elevater_results

    # Evaluate on specific datasets
    python scripts/evaluate_elevater.py \
        --config configs/elevater_eval.yaml \
        --datasets cifar10 cifar100 caltech101

Prerequisites:
    1. Clone the ELEVATER toolkit:
       git clone https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC.git
       cd Elevater_Toolkit_IC
       pip install -e .

    2. Create config file or use command-line arguments

Example Config (configs/elevater_eval.yaml):
    name: "My Model"
    checkpoint_type: "huggingface"  # Options: openclip, huggingface, local, tripletclip, projection
    checkpoint_path: "username/model-name"
    base_model: "ViT-B/32"
    output_dir: "./elevater_results"
    datasets:  # Optional, defaults to all 20 ELEVATER datasets
      - cifar10
      - cifar100
"""

import os
import sys
import argparse
import logging
import yaml
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.clip_wrapper import load_clip_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# =============================================================================
# ELEVATER Dataset Configuration
# =============================================================================

ELEVATER_DATASETS = [
    "caltech101", "cifar10", "cifar100", "country211", "dtd",
    "eurosat", "fer2013", "fgvc_aircraft", "food101", "gtsrb",
    "hateful_memes", "kitti_distance", "mnist", "flowers102",
    "oxford_pets", "patch_camelyon", "sst2", "resisc45",
    "stanford_cars", "voc2007"
]

# Mapping from dataset name to ELEVATER config name
DATASET_CONFIG_MAP = {
    "caltech101": "caltech-101",
    "cifar10": "cifar-10",
    "cifar100": "cifar-100",
    "country211": "country211",
    "dtd": "dtd",
    "eurosat": "eurosat",
    "fer2013": "fer-2013",
    "fgvc_aircraft": "fgvc-aircraft-2013b-variants102",
    "food101": "food-101",
    "gtsrb": "gtsrb",
    "hateful_memes": "hateful-memes",
    "kitti_distance": "kitti-distance",
    "mnist": "mnist",
    "flowers102": "oxford-flower-102",
    "oxford_pets": "oxford-iiit-pets",
    "patch_camelyon": "patch-camelyon",
    "sst2": "rendered-sst2",
    "resisc45": "resisc45",
    "stanford_cars": "stanford-cars",
    "voc2007": "voc-2007-classification",
}

# =============================================================================
# Model Wrapper Classes (from batch_evaluate_checkpoints.py)
# =============================================================================

class TripletCLIPWrapper(nn.Module):
    """
    Wrapper for TripletCLIP models with separate vision and text encoders.
    Provides the same interface as OpenAI CLIP for evaluation.
    """
    
    def __init__(
        self, 
        vision_encoder: nn.Module, 
        text_encoder: nn.Module, 
        image_processor,
        tokenizer,
        device: torch.device,
        logit_scale: float = 100.0
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.device = device
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale).log())
        
        # ELEVATER expects float32 dtype
        self.dtype = torch.float32
        
        # Create preprocess function
        self.preprocess = self._create_preprocess()
    
    def _create_preprocess(self):
        """Create image preprocessing function."""
        from PIL import Image
        
        def preprocess(image):
            if isinstance(image, Image.Image):
                inputs = self.image_processor(images=image, return_tensors="pt")
                return inputs['pixel_values'].squeeze(0)
            elif isinstance(image, torch.Tensor):
                return image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        
        return preprocess
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors."""
        with torch.no_grad():
            images = images.to(self.device)
            
            try:
                # TripletCLIP-style forward
                features = self.vision_encoder({'pixel_values': images})
                if isinstance(features, torch.Tensor):
                    pass
                elif hasattr(features, 'image_embeds'):
                    features = features.image_embeds
                elif hasattr(features, 'pooler_output'):
                    features = features.pooler_output
                else:
                    features = features.last_hidden_state[:, 0, :]
            except Exception:
                # Fallback for different interfaces
                if hasattr(self.vision_encoder, 'model'):
                    if hasattr(self.vision_encoder.model, 'get_image_features'):
                        features = self.vision_encoder.model.get_image_features(pixel_values=images)
                    else:
                        outputs = self.vision_encoder.model(pixel_values=images)
                        if hasattr(outputs, 'image_embeds'):
                            features = outputs.image_embeds
                        elif hasattr(outputs, 'pooler_output'):
                            features = outputs.pooler_output
                        else:
                            features = outputs.last_hidden_state[:, 0, :]
                else:
                    if hasattr(self.vision_encoder, 'get_image_features'):
                        features = self.vision_encoder.get_image_features(pixel_values=images)
                    else:
                        outputs = self.vision_encoder(pixel_values=images)
                        if hasattr(outputs, 'image_embeds'):
                            features = outputs.image_embeds
                        elif hasattr(outputs, 'pooler_output'):
                            features = outputs.pooler_output
                        else:
                            features = outputs.last_hidden_state[:, 0, :]
            
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
            return features
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to feature vectors."""
        with torch.no_grad():
            text = text.to(self.device)
            attention_mask = (text != 0).long().to(self.device)
            
            # Handle different text encoder interfaces
            if hasattr(self.text_encoder, 'model'):
                outputs = self.text_encoder(
                    input_ids=text, 
                    attention_mask=attention_mask
                )
                if hasattr(outputs, 'text_embeds'):
                    features = outputs.text_embeds
                elif hasattr(outputs, 'pooler_output'):
                    features = outputs.pooler_output
                elif isinstance(outputs, torch.Tensor):
                    features = outputs
                else:
                    features = outputs.last_hidden_state[:, 0, :]
            else:
                if hasattr(self.text_encoder, 'get_text_features'):
                    features = self.text_encoder.get_text_features(
                        input_ids=text, 
                        attention_mask=attention_mask
                    )
                else:
                    outputs = self.text_encoder(
                        input_ids=text, 
                        attention_mask=attention_mask
                    )
                    if hasattr(outputs, 'text_embeds'):
                        features = outputs.text_embeds
                    elif hasattr(outputs, 'pooler_output'):
                        features = outputs.pooler_output
                    elif isinstance(outputs, torch.Tensor):
                        features = outputs
                    else:
                        features = outputs.last_hidden_state[:, 0, :]
            
            # Normalize features
            features = features / features.norm(dim=-1, keepdim=True)
            return features
    
    def forward(
        self, 
        images: torch.Tensor, 
        text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        self.vision_encoder = self.vision_encoder.to(device)
        self.text_encoder = self.text_encoder.to(device)
        return super().to(device)


class CLIPWithTextProjection(nn.Module):
    """
    Wrapper for CLIP models with an additional text projection layer.
    """
    
    def __init__(
        self, 
        clip_model: nn.Module, 
        projection_layer: nn.Linear,
        device: torch.device,
        preprocess: callable = None
    ):
        super().__init__()
        self.clip_model = clip_model
        self.projection = projection_layer
        self.device = device
        self.preprocess = preprocess
        
        # Get dtype from clip_model
        self.dtype = getattr(clip_model, 'dtype', torch.float32)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images."""
        with torch.no_grad():
            images = images.to(self.device)
            if hasattr(self.clip_model, 'encode_image'):
                features = self.clip_model.encode_image(images)
            else:
                features = self.clip_model.visual(images)
            
            features = features / features.norm(dim=-1, keepdim=True)
            return features
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text with projection layer."""
        with torch.no_grad():
            text = text.to(self.device)
            
            if hasattr(self.clip_model, 'encode_text'):
                base_features = self.clip_model.encode_text(text)
            else:
                base_features = self.clip_model.text_encoder(text)
            
            # Cast to projection layer dtype
            original_dtype = base_features.dtype
            base_features = base_features.to(self.projection.weight.dtype)
            
            # Apply projection
            features = self.projection(base_features)
            features = features.to(original_dtype)
            
            features = features / features.norm(dim=-1, keepdim=True)
            return features
    
    def forward(
        self, 
        images: torch.Tensor, 
        text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)
        
        if hasattr(self.clip_model, 'logit_scale'):
            logit_scale = self.clip_model.logit_scale.exp()
        else:
            logit_scale = 100.0
        
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        self.clip_model = self.clip_model.to(device)
        self.projection = self.projection.to(device)
        return super().to(device)


# =============================================================================
# Model Loading Functions
# =============================================================================

@dataclass
class ElevaterConfig:
    """Configuration for ELEVATER evaluation."""
    name: str = "CLIP Model"
    checkpoint_type: str = "openclip"  # openclip, huggingface, local, tripletclip, projection, external
    checkpoint_path: str = "ViT-B/32"
    base_model: str = "ViT-B/32"
    output_dir: str = "./elevater_results"
    datasets: List[str] = field(default_factory=lambda: ELEVATER_DATASETS)
    use_fp32: bool = True
    batch_size: int = 64
    num_workers: int = 4
    force_openclip: bool = False
    # For projection checkpoints
    projection_path: Optional[str] = None
    # For CLOVE models
    clove_weight: float = 0.6
    pretrained: str = "openai"


def load_config(config_path: str) -> ElevaterConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return ElevaterConfig(**config_dict)


def load_model(config: ElevaterConfig, device: torch.device) -> Tuple[nn.Module, callable]:
    """
    Load model based on configuration.
    
    Returns:
        Tuple of (model, preprocess_function)
    """
    logging.info(f"Loading model: {config.name}")
    logging.info(f"  Type: {config.checkpoint_type}")
    logging.info(f"  Path: {config.checkpoint_path}")
    
    if config.checkpoint_type == "tripletclip":
        return _load_tripletclip_model(config, device)
    elif config.checkpoint_type == "huggingface":
        return _load_huggingface_model(config, device)
    elif config.checkpoint_type == "openclip":
        return _load_openclip_model(config, device)
    elif config.checkpoint_type == "local":
        return _load_local_checkpoint(config, device)
    elif config.checkpoint_type == "external":
        return _load_external_checkpoint(config, device)
    elif config.checkpoint_type == "projection":
        return _load_projection_checkpoint(config, device)
    elif config.checkpoint_type == "clove":
        return _load_clove_checkpoint(config, device)
    else:
        raise ValueError(f"Unknown checkpoint type: {config.checkpoint_type}")


def _load_tripletclip_model(
    config: ElevaterConfig, 
    device: torch.device
) -> Tuple[nn.Module, callable]:
    """Load TripletCLIP model from HuggingFace."""
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, CLIPProcessor
    from huggingface_hub import hf_hub_download
    import importlib.util
    
    model_id = config.checkpoint_path
    logging.info(f"  Loading TripletCLIP model: {model_id}")
    
    # Try to load utils.py from the model repo
    try:
        utils_path = hf_hub_download(repo_id=model_id, filename="utils.py")
        spec = importlib.util.spec_from_file_location("tripletclip_utils", utils_path)
        tripletclip_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tripletclip_utils)
        logging.info("  ✓ Loaded TripletCLIP utils module")
    except Exception as e:
        logging.warning(f"  Could not load utils.py: {e}")
    
    # Load vision encoder
    logging.info("  Loading vision encoder...")
    vision_encoder = AutoModel.from_pretrained(
        model_id,
        subfolder="vision-encoder",
        trust_remote_code=True
    ).to(device)
    vision_encoder.eval()
    
    # Load text encoder
    logging.info("  Loading text encoder...")
    text_encoder = AutoModel.from_pretrained(
        model_id,
        subfolder="text-encoder",
        trust_remote_code=True
    ).to(device)
    text_encoder.eval()
    
    # Load image processor
    try:
        image_processor = AutoImageProcessor.from_pretrained(
            model_id,
            subfolder="vision-encoder",
            trust_remote_code=True
        )
    except Exception:
        image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").image_processor
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            subfolder="text-encoder",
            trust_remote_code=True
        )
    except Exception:
        tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer
    
    # Create wrapper
    model = TripletCLIPWrapper(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        image_processor=image_processor,
        tokenizer=tokenizer,
        device=device
    )
    model.eval()
    
    logging.info("  ✓ TripletCLIP model loaded successfully")
    return model, model.preprocess


def _load_huggingface_model(
    config: ElevaterConfig, 
    device: torch.device
) -> Tuple[nn.Module, callable]:
    """Load model from HuggingFace Hub."""
    model_name = config.checkpoint_path
    if not model_name.startswith('hf-hub:'):
        model_name = f'hf-hub:{model_name}'
    
    logging.info(f"  Loading from HuggingFace Hub: {model_name}")
    
    model, preprocess = load_clip_model(
        model_name, 
        device, 
        force_openclip=config.force_openclip
    )
    model.eval()
    
    logging.info("  ✓ HuggingFace model loaded successfully")
    return model, preprocess


def _load_openclip_model(
    config: ElevaterConfig, 
    device: torch.device
) -> Tuple[nn.Module, callable]:
    """Load OpenCLIP model."""
    model_name = config.checkpoint_path
    logging.info(f"  Loading OpenCLIP model: {model_name}")
    
    model, preprocess = load_clip_model(model_name, device, force_openclip=True)
    model.eval()
    
    logging.info("  ✓ OpenCLIP model loaded successfully")
    return model, preprocess


def _load_local_checkpoint(
    config: ElevaterConfig, 
    device: torch.device
) -> Tuple[nn.Module, callable]:
    """Load local checkpoint with associated config."""
    checkpoint_path = config.checkpoint_path
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logging.info(f"  Loading local checkpoint: {checkpoint_path}")
    
    # Load base CLIP model
    model, preprocess = load_clip_model(
        config.base_model, 
        device, 
        force_openclip=config.force_openclip
    )
    
    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        elif k.startswith('clip_model.'):
            new_state_dict[k[11:]] = v
        else:
            new_state_dict[k] = v
    
    # Load state dict
    if hasattr(model, 'model'):
        model.model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    logging.info("  ✓ Local checkpoint loaded successfully")
    return model, preprocess


def _load_external_checkpoint(
    config: ElevaterConfig, 
    device: torch.device
) -> Tuple[nn.Module, callable]:
    """Load external checkpoint (state dict only, no config)."""
    checkpoint_path = config.checkpoint_path
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logging.info(f"  Loading external checkpoint: {checkpoint_path}")
    logging.info(f"  Base model: {config.base_model}")
    
    # Load base CLIP model
    model, preprocess = load_clip_model(
        config.base_model, 
        device, 
        force_openclip=config.force_openclip
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    else:
        state_dict = checkpoint
    
    # Clean up state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k
        for prefix in ['module.', 'clip_model.', 'model.']:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        new_state_dict[clean_key] = v
    
    # Load weights
    if hasattr(model, 'model'):
        model.model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    logging.info("  ✓ External checkpoint loaded successfully")
    return model, preprocess


def _load_projection_checkpoint(
    config: ElevaterConfig, 
    device: torch.device
) -> Tuple[nn.Module, callable]:
    """Load CLIP model with text projection layer."""
    checkpoint_path = config.checkpoint_path
    
    logging.info(f"  Loading projection checkpoint: {checkpoint_path}")
    logging.info(f"  Base model: {config.base_model}")
    
    # Load base CLIP model
    clip_model, preprocess = load_clip_model(
        config.base_model, 
        device, 
        force_openclip=config.force_openclip
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    else:
        state_dict = checkpoint
    
    # Find projection layer
    projection_keys = [k for k in state_dict.keys() if 'projection' in k.lower()]
    
    if projection_keys:
        # Get projection layer dimensions
        proj_weight_key = [k for k in projection_keys if 'weight' in k][0]
        proj_weight = state_dict[proj_weight_key]
        out_features, in_features = proj_weight.shape
        
        projection_layer = nn.Linear(in_features, out_features, bias=True)
        projection_layer.weight.data = proj_weight
        
        bias_key = proj_weight_key.replace('weight', 'bias')
        if bias_key in state_dict:
            projection_layer.bias.data = state_dict[bias_key]
        else:
            projection_layer.bias.data.zero_()
        
        projection_layer = projection_layer.to(device)
    else:
        raise ValueError("No projection layer found in checkpoint")
    
    # Create wrapped model
    model = CLIPWithTextProjection(
        clip_model=clip_model.model if hasattr(clip_model, 'model') else clip_model,
        projection_layer=projection_layer,
        device=device,
        preprocess=preprocess
    )
    model.eval()
    
    logging.info("  ✓ Projection checkpoint loaded successfully")
    return model, preprocess


def _load_clove_checkpoint(
    config: ElevaterConfig, 
    device: torch.device
) -> Tuple[nn.Module, callable]:
    """Load CLOVE model with weight interpolation."""
    import open_clip
    
    checkpoint_path = config.checkpoint_path
    clove_weight = config.clove_weight
    pretrained = config.pretrained
    
    logging.info(f"  Loading CLOVE checkpoint: {checkpoint_path}")
    logging.info(f"  Base model: {config.base_model}")
    logging.info(f"  Weight interpolation: {clove_weight}")
    
    # Parse model name
    model_name = config.base_model
    if '@' in model_name:
        model_arch, _ = model_name.split('@', 1)
    else:
        model_arch = model_name.replace('/', '-')
    
    # Load base model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_arch,
        pretrained=pretrained,
        device=device
    )
    
    # Load checkpoint
    if checkpoint_path.startswith('http'):
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    else:
        state_dict = checkpoint
    
    # Interpolate weights
    base_state_dict = model.state_dict()
    interpolated_state_dict = {}
    
    for key in base_state_dict.keys():
        if key in state_dict:
            interpolated_state_dict[key] = (
                (1 - clove_weight) * base_state_dict[key] + 
                clove_weight * state_dict[key]
            )
        else:
            interpolated_state_dict[key] = base_state_dict[key]
    
    model.load_state_dict(interpolated_state_dict)
    model.eval()
    
    # Wrap in UniversalCLIPWrapper
    from utils.clip_wrapper import UniversalCLIPWrapper
    wrapped_model = UniversalCLIPWrapper(
        model=model,
        preprocess=preprocess,
        model_type="openclip",
        model_name=f"{model_arch}+CLOVE"
    )
    
    logging.info("  ✓ CLOVE checkpoint loaded successfully")
    return wrapped_model, preprocess


# =============================================================================
# ELEVATER Evaluation Functions
# =============================================================================

def run_elevater_evaluation(
    model: nn.Module,
    preprocess: callable,
    config: ElevaterConfig,
    device: torch.device
) -> Dict[str, float]:
    """
    Run ELEVATER zero-shot evaluation.
    
    This function attempts to use the ELEVATER toolkit directly.
    If not installed, it falls back to a standalone implementation.
    """
    try:
        # Try using ELEVATER toolkit
        return _run_with_elevater_toolkit(model, preprocess, config, device)
    except ImportError as e:
        logging.warning(f"ELEVATER toolkit not installed: {e}")
        logging.info("Falling back to standalone evaluation...")
        return _run_standalone_evaluation(model, preprocess, config, device)


def _run_with_elevater_toolkit(
    model: nn.Module,
    preprocess: callable,
    config: ElevaterConfig,
    device: torch.device
) -> Dict[str, float]:
    """Run evaluation using ELEVATER toolkit."""
    from vision_benchmark.evaluation import extract_features, extract_text_features, clip_zeroshot_evaluator
    from vision_benchmark.datasets import SimpleTokenizer
    
    results = {}
    
    for dataset in tqdm(config.datasets, desc="Evaluating datasets"):
        logging.info(f"\nEvaluating on {dataset}...")
        
        try:
            # Create ELEVATER config for this dataset
            elevater_cfg = _create_elevater_config(dataset, config, device)
            
            # Extract features
            tokenizer = SimpleTokenizer()
            image_features, image_labels = extract_features(
                elevater_cfg, 
                model=model,
                test_split_only=True
            )
            text_features = extract_text_features(
                elevater_cfg, 
                tokenizer, 
                model=model
            )
            
            # Evaluate
            result, _, metric = clip_zeroshot_evaluator(
                image_features, 
                text_features, 
                image_labels, 
                elevater_cfg
            )
            
            results[dataset] = result * 100
            logging.info(f"  {dataset}: {result * 100:.2f}%")
            
        except Exception as e:
            logging.error(f"  Error evaluating {dataset}: {e}")
            results[dataset] = None
    
    return results


def _run_standalone_evaluation(
    model: nn.Module,
    preprocess: callable,
    config: ElevaterConfig,
    device: torch.device
) -> Dict[str, float]:
    """
    Standalone evaluation without ELEVATER toolkit.
    Uses torchvision datasets and CLIP-style prompts.
    """
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    
    try:
        import clip
        tokenize = clip.tokenize
    except ImportError:
        import open_clip
        tokenize = open_clip.get_tokenizer('ViT-B-32')
    
    results = {}
    
    # Dataset loaders
    dataset_configs = {
        'cifar10': {
            'cls': datasets.CIFAR10,
            'kwargs': {'train': False, 'download': True},
            'templates': ['a photo of a {}.'],
            'classes': None  # Use dataset classes
        },
        'cifar100': {
            'cls': datasets.CIFAR100,
            'kwargs': {'train': False, 'download': True},
            'templates': ['a photo of a {}.'],
            'classes': None
        },
        'mnist': {
            'cls': datasets.MNIST,
            'kwargs': {'train': False, 'download': True},
            'templates': ['a photo of the number: "{}".'],
            'classes': [str(i) for i in range(10)]
        },
        'food101': {
            'cls': datasets.Food101,
            'kwargs': {'split': 'test', 'download': True},
            'templates': ['a photo of {}, a type of food.'],
            'classes': None
        },
        'flowers102': {
            'cls': datasets.Flowers102,
            'kwargs': {'split': 'test', 'download': True},
            'templates': ['a photo of a {}, a type of flower.'],
            'classes': None
        },
        'dtd': {
            'cls': datasets.DTD,
            'kwargs': {'split': 'test', 'download': True},
            'templates': ['{} texture.'],
            'classes': None
        },
        'oxford_pets': {
            'cls': datasets.OxfordIIITPet,
            'kwargs': {'split': 'test', 'download': True},
            'templates': ['a photo of a {}, a type of pet.'],
            'classes': None
        },
        'stanford_cars': {
            'cls': datasets.StanfordCars,
            'kwargs': {'split': 'test', 'download': True},
            'templates': ['a photo of a {}.'],
            'classes': None
        },
        'fgvc_aircraft': {
            'cls': datasets.FGVCAircraft,
            'kwargs': {'split': 'test', 'download': True},
            'templates': ['a photo of a {}, a type of aircraft.'],
            'classes': None
        },
        'caltech101': {
            'cls': datasets.Caltech101,
            'kwargs': {'download': True},
            'templates': ['a photo of a {}.'],
            'classes': None
        },
        'eurosat': {
            'cls': datasets.EuroSAT,
            'kwargs': {'download': True},
            'templates': ['a centered satellite photo of {}.'],
            'classes': None
        },
        'country211': {
            'cls': datasets.Country211,
            'kwargs': {'split': 'test', 'download': True},
            'templates': ['a photo taken in {}.'],
            'classes': None
        },
        'gtsrb': {
            'cls': datasets.GTSRB,
            'kwargs': {'split': 'test', 'download': True},
            'templates': ['a photo of a "{}" traffic sign.'],
            'classes': None
        },
    }
    
    data_root = os.path.join(config.output_dir, 'datasets')
    os.makedirs(data_root, exist_ok=True)
    
    for dataset_name in tqdm(config.datasets, desc="Evaluating datasets"):
        if dataset_name not in dataset_configs:
            logging.warning(f"  {dataset_name} not supported in standalone mode, skipping...")
            continue
        
        logging.info(f"\nEvaluating on {dataset_name}...")
        
        try:
            ds_config = dataset_configs[dataset_name]
            
            # Load dataset
            dataset = ds_config['cls'](
                root=data_root,
                transform=preprocess,
                **ds_config['kwargs']
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True
            )
            
            # Get class names
            if ds_config['classes'] is not None:
                class_names = ds_config['classes']
            elif hasattr(dataset, 'classes'):
                class_names = dataset.classes
            else:
                class_names = [str(i) for i in range(len(set([y for _, y in dataset])))]
            
            # Create text features
            templates = ds_config['templates']
            text_features = []
            
            for classname in class_names:
                texts = [template.format(classname) for template in templates]
                texts_tokenized = tokenize(texts).to(device)
                
                with torch.no_grad():
                    class_embeddings = model.encode_text(texts_tokenized)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    text_features.append(class_embedding)
            
            text_features = torch.stack(text_features, dim=1)
            
            # Extract image features and evaluate
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    logits = 100.0 * image_features @ text_features
                    predictions = logits.argmax(dim=-1)
                    
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            
            accuracy = 100.0 * correct / total
            results[dataset_name] = accuracy
            logging.info(f"  {dataset_name}: {accuracy:.2f}%")
            
        except Exception as e:
            logging.error(f"  Error evaluating {dataset_name}: {e}")
            results[dataset_name] = None
    
    return results


def _create_elevater_config(dataset: str, config: ElevaterConfig, device: torch.device):
    """Create ELEVATER-style config for a dataset."""
    from omegaconf import OmegaConf
    
    elevater_name = DATASET_CONFIG_MAP.get(dataset, dataset)
    
    cfg = OmegaConf.create({
        'DATASET': {
            'DATASET': elevater_name,
            'ROOT': os.path.join(config.output_dir, 'datasets'),
        },
        'MODEL': {
            'NAME': config.name,
            'CLIP_FP32': config.use_fp32,
            'STATS': {},
        },
        'TEST': {
            'METRIC': 'accuracy',
            'BATCH_SIZE': config.batch_size,
        },
        'KNOWLEDGE': {
            'WIKITIONARY': {'USE_DEFINITION': False},
            'WORDNET': {'USE_HIERARCHY': False, 'USE_DEFINITION': False},
            'GPT3': {'USE_GPT3': False},
            'AGGREGATION': {'MEHTOD': 'mean', 'NUM_GPT3_ITEMS': 0},
        },
        'OUTPUT_DIR': config.output_dir,
    })
    
    return cfg


def save_results(results: Dict[str, float], config: ElevaterConfig):
    """Save evaluation results to JSON file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute average (excluding None values)
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        average = sum(valid_results.values()) / len(valid_results)
    else:
        average = 0.0
    
    output = {
        'model_name': config.name,
        'checkpoint_type': config.checkpoint_type,
        'checkpoint_path': config.checkpoint_path,
        'results': results,
        'average': average,
        'num_datasets': len(valid_results),
    }
    
    output_file = output_dir / f"elevater_results_{config.name.replace('/', '_').replace(' ', '_')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    logging.info(f"\nResults saved to: {output_file}")
    logging.info(f"Average accuracy: {average:.2f}% ({len(valid_results)}/{len(results)} datasets)")
    
    return output_file


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='ELEVATER Benchmark Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Config file
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML config file'
    )
    
    # Model arguments
    parser.add_argument(
        '--checkpoint_type',
        type=str,
        choices=['openclip', 'huggingface', 'local', 'tripletclip', 'external', 'projection', 'clove'],
        default='openclip',
        help='Type of checkpoint to load'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='ViT-B/32',
        help='Path or identifier for the checkpoint'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='ViT-B/32',
        help='Base CLIP model architecture'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='CLIP Model',
        help='Name for this model (used in output files)'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./elevater_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=None,
        help='Datasets to evaluate on (default: all ELEVATER datasets)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--use_fp16',
        action='store_true',
        help='Use FP16 instead of FP32'
    )
    parser.add_argument(
        '--force_openclip',
        action='store_true',
        help='Force using OpenCLIP for model loading'
    )
    
    # Projection checkpoint arguments
    parser.add_argument(
        '--projection_path',
        type=str,
        default=None,
        help='Path to projection layer checkpoint (for projection type)'
    )
    
    # CLOVE arguments
    parser.add_argument(
        '--clove_weight',
        type=float,
        default=0.6,
        help='Weight interpolation for CLOVE models'
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        default='openai',
        help='Pretrained weights for base model (for CLOVE)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config from file or command-line arguments
    if args.config:
        config = load_config(args.config)
        # Override with command-line arguments if provided
        if args.datasets:
            config.datasets = args.datasets
        if args.output_dir != './elevater_results':
            config.output_dir = args.output_dir
    else:
        config = ElevaterConfig(
            name=args.name,
            checkpoint_type=args.checkpoint_type,
            checkpoint_path=args.checkpoint_path,
            base_model=args.base_model,
            output_dir=args.output_dir,
            datasets=args.datasets or ELEVATER_DATASETS,
            use_fp32=not args.use_fp16,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force_openclip=args.force_openclip,
            projection_path=args.projection_path,
            clove_weight=args.clove_weight,
            pretrained=args.pretrained,
        )
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model
    model, preprocess = load_model(config, device)
    
    # Move to device and set dtype
    model = model.to(device)
    if config.use_fp32:
        model = model.float()
    else:
        model = model.half()
    
    # Run evaluation
    results = run_elevater_evaluation(model, preprocess, config, device)
    
    # Save results
    output_file = save_results(results, config)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ELEVATER Evaluation Complete")
    print("=" * 60)
    print(f"Model: {config.name}")
    print(f"Results: {output_file}")
    print("\nPer-dataset results:")
    for dataset, acc in sorted(results.items()):
        if acc is not None:
            print(f"  {dataset}: {acc:.2f}%")
        else:
            print(f"  {dataset}: FAILED")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        print(f"\nAverage: {sum(valid_results.values()) / len(valid_results):.2f}%")


if __name__ == '__main__':
    main()

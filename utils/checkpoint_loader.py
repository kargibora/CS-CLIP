"""
Unified Checkpoint Loader for CLIP Models

This module provides a unified interface for loading CLIP models from various sources:
- OpenCLIP (standard CLIP models)
- HuggingFace Hub (e.g., READCLIP, CLIC, etc.)
- External checkpoints (.pt files without config)
- CLOVE checkpoints (with weight interpolation)
- TripletCLIP (separate vision/text encoders)
- DAC checkpoints (with LoRA merging)
- Local checkpoints (with config)

Usage:
    from utils.checkpoint_loader import load_checkpoint_model

    # OpenCLIP baseline
    model, preprocess, tokenize = load_checkpoint_model(
        checkpoint_type="openclip",
        checkpoint_path="ViT-B/32",
        device=device
    )
    
    # HuggingFace model (e.g., READCLIP)
    model, preprocess, tokenize = load_checkpoint_model(
        checkpoint_type="huggingface",
        checkpoint_path="hf-hub:Mayfull/READ-CLIP",
        device=device,
        base_model="ViT-B/32"
    )
    
    # External checkpoint
    model, preprocess, tokenize = load_checkpoint_model(
        checkpoint_type="external",
        checkpoint_path="/path/to/checkpoint.pt",
        device=device,
        base_model="ViT-B/32"
    )
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable
from dataclasses import dataclass


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CheckpointConfig:
    """Configuration for loading a checkpoint."""
    
    # Type: 'openclip', 'huggingface', 'external', 'clove', 'tripletclip', 'dac', 'local', 'projection'
    checkpoint_type: str
    
    # Path/identifier for the checkpoint
    # - For 'openclip': model name (e.g., 'ViT-B/32')
    # - For 'huggingface': HuggingFace Hub model ID (e.g., 'hf-hub:Mayfull/READ-CLIP')
    # - For 'external'/'clove'/'dac'/'local'/'projection': path to .pt file
    checkpoint_path: str
    
    # Base CLIP model architecture (for loading external checkpoints)
    base_model: str = "ViT-B/32"
    
    # Force OpenCLIP instead of OpenAI CLIP
    force_openclip: bool = False
    
    # CLOVE-specific: weight for state_dict interpolation
    clove_weight: float = 0.6
    
    # CLOVE-specific: pretrained weights for base model
    pretrained: str = "openai"


# =============================================================================
# Model Wrappers
# =============================================================================

class TripletCLIPWrapper(nn.Module):
    """
    Wrapper for TripletCLIP models with separate vision and text encoders.
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
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale).log())
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
                if hasattr(self.vision_encoder, 'model'):
                    if hasattr(self.vision_encoder.model, 'get_image_features'):
                        features = self.vision_encoder.model.get_image_features(pixel_values=images)
                    else:
                        outputs = self.vision_encoder.model(pixel_values=images)
                        features = getattr(outputs, 'image_embeds', 
                                          getattr(outputs, 'pooler_output', 
                                                 outputs.last_hidden_state[:, 0, :]))
                else:
                    if hasattr(self.vision_encoder, 'get_image_features'):
                        features = self.vision_encoder.get_image_features(pixel_values=images)
                    else:
                        outputs = self.vision_encoder(pixel_values=images)
                        features = getattr(outputs, 'image_embeds',
                                          getattr(outputs, 'pooler_output',
                                                 outputs.last_hidden_state[:, 0, :]))
            
            return features / features.norm(dim=-1, keepdim=True)
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to feature vectors."""
        with torch.no_grad():
            text = text.to(self.device)
            attention_mask = (text != 0).long().to(self.device)
            
            if hasattr(self.text_encoder, 'model'):
                outputs = self.text_encoder(input_ids=text, attention_mask=attention_mask)
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
                    features = self.text_encoder.get_text_features(input_ids=text, attention_mask=attention_mask)
                else:
                    outputs = self.text_encoder(input_ids=text, attention_mask=attention_mask)
                    if hasattr(outputs, 'text_embeds'):
                        features = outputs.text_embeds
                    elif hasattr(outputs, 'pooler_output'):
                        features = outputs.pooler_output
                    elif isinstance(outputs, torch.Tensor):
                        features = outputs
                    else:
                        features = outputs.last_hidden_state[:, 0, :]
            
            return features / features.norm(dim=-1, keepdim=True)
    
    def forward(self, images: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits."""
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


class CLOVEModelWrapper(nn.Module):
    """
    Wrapper for CLOVE models to provide consistent interface.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        self.device = device
        self.preprocess = getattr(model, 'preprocess', None)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors."""
        with torch.no_grad():
            images = images.to(self.device)
            if hasattr(self.model, 'encode_image'):
                features = self.model.encode_image(images)
            else:
                output = self.model(images, None)
                features = output['image_features'] if isinstance(output, dict) else output[0]
            return features / features.norm(dim=-1, keepdim=True)
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to feature vectors."""
        with torch.no_grad():
            text = text.to(self.device)
            if hasattr(self.model, 'encode_text'):
                features = self.model.encode_text(text)
            else:
                output = self.model(None, text)
                features = output['text_features'] if isinstance(output, dict) else output[1]
            return features / features.norm(dim=-1, keepdim=True)
    
    def forward(self, images: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits."""
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)
        
        if hasattr(self.model, 'logit_scale'):
            logit_scale = self.model.logit_scale.exp()
        else:
            logit_scale = 100.0
        
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        return super().to(device)


class CLIPWithTextProjection(nn.Module):
    """
    Wrapper for CLIP models with an additional text projection layer.
    
    This wrapper applies a linear projection layer after the CLIP text encoder.
    The projection layer transforms text features before similarity computation.
    
    Structure: text_tokens -> CLIP_text_encoder -> projection_layer -> text_features
    """
    
    def __init__(
        self, 
        clip_model: nn.Module, 
        projection_layer: nn.Linear,
        device: torch.device,
        preprocess: Callable = None
    ):
        super().__init__()
        self.clip_model = clip_model
        self.projection = projection_layer
        self.device = device
        self.preprocess = preprocess
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors (unchanged from base CLIP)."""
        with torch.no_grad():
            images = images.to(self.device)
            if hasattr(self.clip_model, 'encode_image'):
                features = self.clip_model.encode_image(images)
            else:
                features = self.clip_model.visual(images)
            
            features = features / features.norm(dim=-1, keepdim=True)
            return features
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to feature vectors with projection layer applied."""
        with torch.no_grad():
            text = text.to(self.device)
            
            # Get base text features from CLIP
            if hasattr(self.clip_model, 'encode_text'):
                base_features = self.clip_model.encode_text(text)
            else:
                base_features = self.clip_model.text_encoder(text)
            
            # Store original dtype for potential conversion back
            original_dtype = base_features.dtype
            
            # Cast to projection layer dtype to avoid dtype mismatch
            base_features = base_features.to(self.projection.weight.dtype)
            
            # Apply projection layer
            features = self.projection(base_features)
            
            # Cast back to original dtype if needed
            features = features.to(original_dtype)
            
            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)
            return features
    
    def forward(
        self, 
        images: torch.Tensor, 
        text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits per image and per text."""
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)
        
        # Get logit_scale from base CLIP model
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
# Helper Functions
# =============================================================================

def clean_state_dict(state_dict: dict) -> dict:
    """
    Clean state dict by removing wrapper prefixes.
    
    Handles various wrapper patterns:
    - module.* (DDP wrapper)
    - model.module.* (nested DDP)
    - _orig_mod.* (torch.compile)
    - model.model.* (double wrapped)
    - model.* (single model prefix - e.g., from training pipeline)
    - clip_model.* (CLIP wrapper prefix)
    
    The goal is to get keys that match the base CLIP model:
    - visual.*, transformer.*, positional_embedding, etc.
    """
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        
        # Handle different wrapper patterns (order matters!)
        if new_key.startswith('module.'):
            # DDP wrapper: module.model.visual.* -> model.visual.*
            new_key = new_key[len('module.'):]
        
        if new_key.startswith('model.module.'):
            # Nested DDP: model.module.visual.* -> visual.*
            new_key = new_key[len('model.module.'):]
        
        if new_key.startswith('_orig_mod.'):
            # torch.compile wrapper
            new_key = new_key[len('_orig_mod.'):]
        
        # Check for double model prefix (model.model.*)
        if new_key.startswith('model.model.'):
            # Remove only ONE 'model.' prefix
            new_key = new_key[len('model.'):]
        
        # Handle single model. prefix
        if new_key.startswith('model.'):
            new_key = new_key[len('model.'):]
        
        # Handle clip_model. prefix
        if new_key.startswith('clip_model.'):
            new_key = new_key[len('clip_model.'):]
        
        cleaned[new_key] = value
    return cleaned


def merge_lora_weights(state_dict: dict) -> dict:
    """
    Merge LoRA A/B matrices into base weights (for DAC checkpoints).
    
    DAC uses LoRA-style decomposition: W' = W + B @ A
    """
    merged_state_dict = {}
    lora_keys_processed = set()
    
    # Identify LoRA pairs
    lora_pairs = {}
    for key in state_dict.keys():
        if key.endswith('.lora_A') or key.endswith('_lora_A'):
            if key.endswith('.lora_A'):
                base_key = key[:-len('.lora_A')]
                lora_b_key = base_key + '.lora_B'
            else:
                base_key = key[:-len('_lora_A')]
                lora_b_key = base_key + '_lora_B'
            
            if lora_b_key in state_dict:
                lora_pairs[base_key] = (key, lora_b_key)
                lora_keys_processed.add(key)
                lora_keys_processed.add(lora_b_key)
    
    logger.info(f"Found {len(lora_pairs)} LoRA weight pairs to merge")
    
    # Merge weights
    for key, value in state_dict.items():
        if key in lora_keys_processed:
            continue
        
        clean_key = key
        if clean_key.startswith('module.'):
            clean_key = clean_key[len('module.'):]
        
        original_base_key = key[len('module.'):] if key.startswith('module.') else key
        
        if key in lora_pairs or original_base_key in lora_pairs:
            pair_key = key if key in lora_pairs else original_base_key
            lora_a_key, lora_b_key = lora_pairs[pair_key]
            
            lora_a = state_dict[lora_a_key]
            lora_b = state_dict[lora_b_key]
            
            try:
                if len(value.shape) == 2:
                    delta = lora_b @ lora_a
                elif len(value.shape) == 4:
                    delta = (lora_b @ lora_a).view(value.shape)
                elif len(value.shape) == 1:
                    delta = (lora_b @ lora_a).squeeze()
                else:
                    delta = (lora_b @ lora_a).view(value.shape)
                
                merged_weight = value + delta
            except Exception as e:
                logger.warning(f"Failed to merge LoRA for {clean_key}: {e}")
                merged_weight = value
        else:
            merged_weight = value
        
        merged_state_dict[clean_key] = merged_weight
    
    return merged_state_dict


def interpolate_pos_embed(pos_embed: torch.Tensor, target_length: int) -> torch.Tensor:
    """Interpolate positional embeddings to match target length."""
    orig_length, embed_dim = pos_embed.shape
    
    if orig_length == target_length:
        return pos_embed
    
    pos_embed_reshaped = pos_embed.T.unsqueeze(0)
    interpolated = F.interpolate(
        pos_embed_reshaped,
        size=target_length,
        mode='linear',
        align_corners=False
    )
    return interpolated.squeeze(0).T


def get_tokenizer(model_name: str, model_type: str = "openai") -> Callable:
    """Get tokenizer for the model."""
    if model_type == "openai":
        try:
            import clip
            return clip.tokenize
        except ImportError:
            pass
    
    try:
        import open_clip
        return open_clip.get_tokenizer(model_name)
    except Exception:
        pass
    
    # Fallback to simple tokenizer
    try:
        import clip
        return clip.tokenize
    except ImportError:
        raise RuntimeError("No tokenizer available. Install 'clip' or 'open_clip_torch'.")


# =============================================================================
# Main Loading Functions
# =============================================================================

def load_openclip_model(
    model_name: str,
    device: torch.device,
    pretrained: str = "openai",
    force_openclip: bool = False
) -> Tuple[nn.Module, Callable, Callable]:
    """Load OpenCLIP or OpenAI CLIP model."""
    from utils.clip_wrapper import load_clip_model
    
    model, preprocess = load_clip_model(model_name, device, force_openclip=force_openclip)
    model.eval()
    
    # Get tokenizer
    tokenize = get_tokenizer(model_name, "openclip" if force_openclip else "openai")
    
    return model, preprocess, tokenize


def load_huggingface_model(
    model_path: str,
    device: torch.device,
    base_model: str = "ViT-B/32"
) -> Tuple[nn.Module, Callable, Callable]:
    """Load HuggingFace Hub CLIP model."""
    from utils.clip_wrapper import load_clip_model
    
    # Ensure hf-hub: prefix
    if not model_path.startswith('hf-hub:'):
        model_path = f'hf-hub:{model_path}'
    
    logger.info(f"Loading HuggingFace model: {model_path}")
    
    model, preprocess = load_clip_model(model_path, device, force_openclip=True)
    model.eval()
    
    # Get tokenizer
    tokenize = get_tokenizer(base_model, "openclip")
    
    return model, preprocess, tokenize


def load_tripletclip_model(
    model_id: str,
    device: torch.device
) -> Tuple[nn.Module, Callable, Callable]:
    """Load TripletCLIP model with separate encoders."""
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, CLIPProcessor
    from huggingface_hub import hf_hub_download
    import importlib.util
    
    logger.info(f"Loading TripletCLIP model: {model_id}")
    
    # Load utils.py from HuggingFace
    try:
        utils_path = hf_hub_download(repo_id=model_id, filename="utils.py")
        spec = importlib.util.spec_from_file_location("tripletclip_utils", utils_path)
        tripletclip_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tripletclip_utils)
    except Exception as e:
        logger.warning(f"Could not load utils.py: {e}")
    
    # Load vision encoder
    vision_encoder = AutoModel.from_pretrained(
        model_id,
        subfolder="vision-encoder",
        trust_remote_code=True
    ).to(device).eval()
    
    # Load text encoder
    text_encoder = AutoModel.from_pretrained(
        model_id,
        subfolder="text-encoder",
        trust_remote_code=True
    ).to(device).eval()
    
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
    wrapped_model = TripletCLIPWrapper(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        image_processor=image_processor,
        tokenizer=tokenizer,
        device=device
    )
    wrapped_model.eval()
    
    # Create tokenize function
    def tokenize(texts, context_length=77):
        if isinstance(texts, str):
            texts = [texts]
        return tokenizer(
            texts,
            padding='max_length',
            max_length=context_length,
            truncation=True,
            return_tensors='pt'
        )['input_ids']
    
    return wrapped_model, wrapped_model.preprocess, tokenize


def load_external_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    base_model: str = "ViT-B/32",
    force_openclip: bool = False
) -> Tuple[nn.Module, Callable, Callable]:
    """Load external checkpoint (.pt file) into base CLIP model.
    
    Automatically detects and merges LoRA weights if present (e.g., TSVLC, DAC).
    """
    from utils.clip_wrapper import load_clip_model
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading external checkpoint: {checkpoint_path}")
    logger.info(f"Base model: {base_model}")
    
    # Load base model
    model, preprocess = load_clip_model(base_model, device, force_openclip=force_openclip)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Clean state dict (remove wrapper prefixes)
    state_dict = clean_state_dict(state_dict)
    
    # Check if checkpoint contains LoRA weights and merge them if so
    lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
    if lora_keys:
        logger.info(f"Detected {len(lora_keys)} LoRA keys in checkpoint, merging...")
        state_dict = merge_lora_weights(state_dict)
        logger.info(f"LoRA weights merged, state dict now has {len(state_dict)} keys")
    
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.info("Loaded checkpoint with strict=True")
    except Exception as e:
        logger.warning(f"Could not load with strict=True: {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)}")
        logger.info("Loaded checkpoint with strict=False")
    
    model.eval()
    tokenize = get_tokenizer(base_model, "openclip" if force_openclip else "openai")
    
    return model, preprocess, tokenize


def load_dac_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    base_model: str = "ViT-B/32",
    force_openclip: bool = False
) -> Tuple[nn.Module, Callable, Callable]:
    """Load DAC checkpoint with LoRA merging."""
    from utils.clip_wrapper import load_clip_model
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading DAC checkpoint: {checkpoint_path}")
    
    # Load base model
    model, preprocess = load_clip_model(base_model, device, force_openclip=force_openclip)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        for key in ['state_dict', 'model_state_dict', 'model']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Merge LoRA weights
    merged_state_dict = merge_lora_weights(state_dict)
    
    # Load merged weights
    try:
        model.load_state_dict(merged_state_dict, strict=True)
        logger.info("Loaded merged checkpoint with strict=True")
    except Exception as e:
        logger.warning(f"Could not load with strict=True: {e}")
        missing, unexpected = model.load_state_dict(merged_state_dict, strict=False)
        logger.info("Loaded merged checkpoint with strict=False")
    
    model.eval()
    tokenize = get_tokenizer(base_model, "openclip" if force_openclip else "openai")
    
    return model, preprocess, tokenize


def load_clove_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    base_model: str = "ViT-B/32",
    pretrained: str = "openai",
    clove_weight: float = 0.6
) -> Tuple[nn.Module, Callable, Callable]:
    """Load CLOVE checkpoint with weight interpolation."""
    import open_clip
    
    logger.info(f"Loading CLOVE checkpoint: {checkpoint_path}")
    logger.info(f"Base model: {base_model}, pretrained: {pretrained}, weight: {clove_weight}")
    
    # Convert model name format
    openclip_model_name = base_model.replace('/', '-')
    
    # Create base model
    model, _, preprocess = open_clip.create_model_and_transforms(
        openclip_model_name,
        pretrained=pretrained,
        device=device
    )
    model.eval()
    
    # Load checkpoint
    if checkpoint_path.startswith('http://') or checkpoint_path.startswith('https://'):
        logger.info("Downloading checkpoint from URL...")
        try:
            from open_clip.factory import pt_load
            state_dict = pt_load(checkpoint_path, map_location=device)
        except ImportError:
            import urllib.request
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
                urllib.request.urlretrieve(checkpoint_path, tmp_file.name)
                state_dict = torch.load(tmp_file.name, map_location=device, weights_only=False)
                os.unlink(tmp_file.name)
    else:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state dict if nested
    if isinstance(state_dict, dict):
        for key in ['state_dict', 'model_state_dict', 'model']:
            if key in state_dict:
                state_dict = state_dict[key]
                break
    
    # Try CLOVE's patch_model if available
    try:
        from training.utils import get_state_dict, patch_model
        processed_state_dict = get_state_dict(state_dict, model)
        patch_model(model, processed_state_dict, weight_for_state_dict=clove_weight)
        logger.info(f"Applied CLOVE patch_model with weight={clove_weight}")
    except ImportError:
        logger.warning("CLOVE training.utils not available, using manual patching")
        _manual_clove_patch(model, state_dict, clove_weight)
    
    model.eval()
    
    # Wrap model
    wrapped_model = CLOVEModelWrapper(model, device)
    
    # Get tokenizer
    tokenize = open_clip.get_tokenizer(openclip_model_name)
    
    return wrapped_model, preprocess, tokenize


def _manual_clove_patch(model: nn.Module, state_dict: dict, weight: float):
    """Manually patch model weights with CLOVE checkpoint."""
    model_state = model.state_dict()
    
    # Clean state dict
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[len('module.'):] if key.startswith('module.') else key
        cleaned_state_dict[new_key] = value
    
    patched_count = 0
    interpolated_count = 0
    
    for key, base_value in model_state.items():
        if key not in cleaned_state_dict:
            continue
        
        ckpt_value = cleaned_state_dict[key]
        
        # Handle shape mismatch
        if base_value.shape != ckpt_value.shape:
            if 'positional_embedding' in key:
                ckpt_value = interpolate_pos_embed(ckpt_value, base_value.shape[0])
                interpolated_count += 1
            else:
                continue
        
        # Interpolate: new = (1 - weight) * base + weight * checkpoint
        new_value = (1 - weight) * base_value + weight * ckpt_value.to(base_value.device)
        model_state[key] = new_value
        patched_count += 1
    
    model.load_state_dict(model_state)
    logger.info(f"Manual CLOVE patching: {patched_count} patched, {interpolated_count} interpolated")


def load_projection_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    base_model: str = "ViT-B/32",
    force_openclip: bool = False
) -> Tuple[nn.Module, Callable, Callable]:
    """
    Load checkpoint with text projection layer.
    
    This handles checkpoints that contain only a linear projection layer to be
    applied after the CLIP text encoder. The checkpoint format should be:
        {"linear.weight": Tensor, ...} or {"weight": Tensor, ...}
    
    Args:
        checkpoint_path: Path to projection checkpoint file
        device: Target device
        base_model: Base CLIP model architecture
        force_openclip: Force OpenCLIP instead of OpenAI CLIP
        
    Returns:
        Tuple of (model, preprocess, tokenize)
    """
    import os
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading projection checkpoint: {checkpoint_path}")
    logger.info(f"Base model: {base_model}")
    
    # Load base CLIP model
    from utils.clip_wrapper import load_clip_model
    clip_model, preprocess = load_clip_model(base_model, device, force_openclip=force_openclip)
    logger.info("Base CLIP model loaded")
    
    # Load projection checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info("Projection checkpoint file loaded")
    
    # Extract projection weights
    weight = None
    bias = None
    
    if isinstance(checkpoint, dict):
        # Try different key formats
        if 'linear.weight' in checkpoint:
            weight = checkpoint['linear.weight']
            bias = checkpoint.get('linear.bias', None)
        elif 'weight' in checkpoint:
            weight = checkpoint['weight']
            bias = checkpoint.get('bias', None)
        elif 'projection.weight' in checkpoint:
            weight = checkpoint['projection.weight']
            bias = checkpoint.get('projection.bias', None)
        else:
            # Check for state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                weight = state_dict.get('linear.weight', state_dict.get('weight'))
                bias = state_dict.get('linear.bias', state_dict.get('bias', None))
            else:
                raise ValueError(
                    f"Could not find projection weights in checkpoint. "
                    f"Available keys: {list(checkpoint.keys())}"
                )
    else:
        # Assume checkpoint is the weight tensor directly
        weight = checkpoint
        bias = None
    
    if weight is None:
        raise ValueError("Could not extract projection weight from checkpoint")
    
    # Infer dimensions from weight shape (Linear weight is [out_features, in_features])
    out_features, in_features = weight.shape
    logger.info(f"Projection layer: {in_features} -> {out_features}")
    
    # Create projection layer
    projection = nn.Linear(in_features, out_features, bias=bias is not None)
    projection.weight.data = weight
    if bias is not None:
        projection.bias.data = bias
    projection = projection.to(device)
    projection.eval()
    
    logger.info("Projection layer created")
    
    # Create wrapped model
    wrapped_model = CLIPWithTextProjection(
        clip_model=clip_model,
        projection_layer=projection,
        device=device,
        preprocess=preprocess
    )
    wrapped_model.eval()
    
    # Get tokenizer
    try:
        import open_clip
        openclip_model_name = base_model.replace('/', '-')
        tokenize = open_clip.get_tokenizer(openclip_model_name)
    except Exception:
        try:
            import clip
            tokenize = clip.tokenize
        except ImportError:
            tokenize = None
    
    logger.info("Projection checkpoint loaded successfully")
    
    return wrapped_model, preprocess, tokenize


# =============================================================================
# Main Entry Point
# =============================================================================

def load_checkpoint_model(
    checkpoint_type: str,
    checkpoint_path: str,
    device: torch.device,
    base_model: str = "ViT-B/32",
    force_openclip: bool = False,
    pretrained: str = "openai",
    clove_weight: float = 0.6
) -> Tuple[nn.Module, Callable, Callable]:
    """
    Universal function to load CLIP models from various sources.
    
    Args:
        checkpoint_type: Type of checkpoint:
            - 'openclip': Standard OpenCLIP/OpenAI CLIP model
            - 'huggingface': HuggingFace Hub model
            - 'tripletclip': TripletCLIP with separate encoders
            - 'external': External .pt checkpoint
            - 'dac': DAC checkpoint with LoRA
            - 'clove': CLOVE checkpoint with interpolation
            - 'projection': Projection layer applied after CLIP text encoder
        checkpoint_path: Path or model identifier
        device: Target device
        base_model: Base CLIP architecture (for external checkpoints)
        force_openclip: Force OpenCLIP instead of OpenAI CLIP
        pretrained: Pretrained weights for CLOVE
        clove_weight: Interpolation weight for CLOVE
        
    Returns:
        Tuple of (model, preprocess, tokenize)
    """
    logger.info(f"Loading checkpoint: type={checkpoint_type}, path={checkpoint_path}")
    
    if checkpoint_type == 'openclip':
        return load_openclip_model(checkpoint_path, device, pretrained, force_openclip)
    
    elif checkpoint_type == 'huggingface':
        return load_huggingface_model(checkpoint_path, device, base_model)
    
    elif checkpoint_type == 'tripletclip':
        return load_tripletclip_model(checkpoint_path, device)
    
    elif checkpoint_type == 'external':
        return load_external_checkpoint(checkpoint_path, device, base_model, force_openclip)
    
    elif checkpoint_type == 'dac':
        return load_dac_checkpoint(checkpoint_path, device, base_model, force_openclip)
    
    elif checkpoint_type == 'clove':
        return load_clove_checkpoint(checkpoint_path, device, base_model, pretrained, clove_weight)
    
    elif checkpoint_type == 'projection':
        return load_projection_checkpoint(checkpoint_path, device, base_model, force_openclip)
    
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")

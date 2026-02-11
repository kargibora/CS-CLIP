"""
Batch Checkpoint Evaluation Script

Evaluates multiple checkpoints (local, HuggingFace, or OpenAI-OpenCLIP models) and saves
their scores to separate CSV files.

Usage:
    # Evaluate from config file
    python scripts/batch_evaluate_checkpoints.py --config configs/eval_checkpoints.yaml
    
    # Evaluate with command-line arguments
    python scripts/batch_evaluate_checkpoints.py \
        --checkpoint_type openclip \
        --checkpoint_path "ViT-B/32" \
        --csv_filename baseline.csv \
        --name "OpenAI CLIP Baseline"
    
    # Multi-GPU evaluation (8 GPUs)
    torchrun --nproc_per_node=8 scripts/batch_evaluate_checkpoints.py --config configs/eval_checkpoints.yaml
"""

import os
import sys
import logging
import torch
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignment.simple_dataset_evaluation import setup_dataset_evaluation
from utils.clip_wrapper import load_clip_model
from utils.head_init import create_model_from_config
from models import CLIPEndToEndPipeline, CLIPFeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


# ============================================================================
# TripletCLIP Wrapper Classes
# ============================================================================

class TripletCLIPWrapper(torch.nn.Module):
    """
    Wrapper for TripletCLIP models with separate vision and text encoders.
    Provides the same interface as OpenAI CLIP for evaluation (encode_image, encode_text).
    
    TripletCLIP models from HuggingFace have:
    - vision-encoder/: CLIPVisionEncoderOnly or similar
    - text-encoder/: CLIPTextEncoderOnly or CustomTextEncoderOnly
    
    Note: TripletCLIP uses max_length=64 for text, not 77 like standard CLIP.
    """
    
    def __init__(
        self, 
        vision_encoder: torch.nn.Module, 
        text_encoder: torch.nn.Module, 
        image_processor,
        tokenizer,
        device: torch.device,
        logit_scale: float = 100.0,
        max_text_length: int = 64  # TripletCLIP uses 64, not 77
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.device = device
        self.max_text_length = max_text_length
        
        # Learnable temperature parameter (initialize with typical CLIP value)
        self.logit_scale = torch.nn.Parameter(torch.tensor(logit_scale).log())
        
        # Create preprocess function compatible with CLIP evaluation
        self.preprocess = self._create_preprocess()
    
    def _create_preprocess(self):
        """Create image preprocessing function compatible with CLIP evaluation."""
        from PIL import Image
        
        def preprocess(image):
            if isinstance(image, Image.Image):
                # Use HuggingFace image processor for PIL images
                inputs = self.image_processor(images=image, return_tensors="pt")
                return inputs['pixel_values'].squeeze(0)
            elif isinstance(image, torch.Tensor):
                # Already a tensor - assume it's preprocessed
                return image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        
        return preprocess
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature vectors.
        
        Args:
            images: Tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        with torch.no_grad():
            # Ensure images are on the correct device
            images = images.to(self.device)
            
            # TripletCLIP's CLIPVisionEncoderOnly expects a dict and returns image_embeds directly
            # See: https://huggingface.co/TripletCLIP/CC12M_TripletCLIP_ViTB12/blob/main/utils.py
            try:
                # Try TripletCLIP-style forward (expects dict, returns tensor directly)
                features = self.vision_encoder({'pixel_values': images})
                if isinstance(features, torch.Tensor):
                    pass  # Already have features
                elif hasattr(features, 'image_embeds'):
                    features = features.image_embeds
                elif hasattr(features, 'pooler_output'):
                    features = features.pooler_output
                else:
                    features = features.last_hidden_state[:, 0, :]
            except Exception:
                # Fallback: try different interfaces
                if hasattr(self.vision_encoder, 'model'):
                    # CLIPVisionEncoderOnly wraps the model
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
                    # Direct model call
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
        """
        Encode text to feature vectors.
        
        Args:
            text: Tensor of token IDs with shape (batch_size, seq_length)
                  Note: If seq_length > max_text_length (64 for TripletCLIP), 
                  it will be truncated.
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        with torch.no_grad():
            # Ensure text is on the correct device
            text = text.to(self.device)
            
            # Truncate/pad to max_text_length if needed (TripletCLIP uses 64, not 77)
            if text.shape[1] > self.max_text_length:
                text = text[:, :self.max_text_length]
            elif text.shape[1] < self.max_text_length:
                # Pad with zeros (padding token)
                padding = torch.zeros(
                    text.shape[0], 
                    self.max_text_length - text.shape[1],
                    dtype=text.dtype,
                    device=text.device
                )
                text = torch.cat([text, padding], dim=1)
            
            # Create attention mask
            attention_mask = (text != 0).long().to(self.device)
            
            # Handle different text encoder interfaces
            if hasattr(self.text_encoder, 'model'):
                # CLIPTextEncoderOnly or CustomTextEncoderOnly wraps the model
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
                # Direct model call
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
        """
        Forward pass returning logits per image and per text.
        
        Args:
            images: Tensor of images
            text: Tensor of token IDs
            
        Returns:
            Tuple of (logits_per_image, logits_per_text)
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)
        
        # Compute similarity with learned temperature
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


class CLOVEModelWrapper(torch.nn.Module):
    """
    Wrapper for CLOVE models to provide consistent interface with OpenAI CLIP.
    
    CLOVE models from open_clip have a slightly different interface than OpenAI CLIP.
    This wrapper provides encode_image, encode_text methods compatible with our evaluation.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        self.device = device
        
        # Get preprocess from the model if available
        self.preprocess = getattr(model, 'preprocess', None)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors."""
        with torch.no_grad():
            images = images.to(self.device)
            # OpenCLIP models use encode_image
            if hasattr(self.model, 'encode_image'):
                features = self.model.encode_image(images)
            else:
                # Fallback for different model interfaces
                output = self.model(images, None)
                features = output['image_features'] if isinstance(output, dict) else output[0]
            
            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)
            return features
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to feature vectors."""
        with torch.no_grad():
            text = text.to(self.device)
            # OpenCLIP models use encode_text
            if hasattr(self.model, 'encode_text'):
                features = self.model.encode_text(text)
            else:
                # Fallback for different model interfaces
                output = self.model(None, text)
                features = output['text_features'] if isinstance(output, dict) else output[1]
            
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
        
        # Get logit_scale from model
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


class CLIPWithTextProjection(torch.nn.Module):
    """
    Wrapper for CLIP models with an additional text projection layer.
    
    This wrapper applies a linear projection layer after the CLIP text encoder.
    The projection layer transforms text features before similarity computation.
    
    Structure: text_tokens -> CLIP_text_encoder -> projection_layer -> text_features
    """
    
    def __init__(
        self, 
        clip_model: torch.nn.Module, 
        projection_layer: torch.nn.Linear,
        device: torch.device,
        preprocess: callable = None
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
                # OpenAI CLIP uses visual.forward
                features = self.clip_model.visual(images)
            
            # Normalize
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
                # Fallback for different interfaces
                base_features = self.clip_model.text_encoder(text)
            
            # Store original dtype for potential conversion back
            original_dtype = base_features.dtype
            
            # Cast to projection layer dtype to avoid dtype mismatch
            # (CLIP may output float16, but projection weights are float32)
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


@dataclass
class CheckpointConfig:
    """Configuration for a checkpoint to evaluate."""
    
    # Name for this checkpoint (used in logs and CSV filename)
    name: str
    
    # Output CSV file name (without directory)
    csv_filename: str
    
    # Type: 'local', 'huggingface', 'openclip', 'external', 'clove', 'tripletclip', or 'projection'
    checkpoint_type: str
    
    # Path/identifier for the checkpoint
    # - For 'local': path to .pt file (with config)
    # - For 'external': path to .pt file (without config, loads into base CLIP)
    # - For 'clove': path to .pt file or URL (uses patch_model with weight interpolation)
    # - For 'huggingface': HuggingFace Hub model ID (e.g., 'username/model-name')
    # - For 'openclip': OpenCLIP model name (e.g., 'ViT-B-32@openai')
    checkpoint_path: str
    
    # Base CLIP model to use (for local checkpoints)
    # For HF/OpenCLIP, this is usually inferred from the model
    base_model: str = "ViT-B/32"
    
    # Whether this is a fine-tuned model (FT) or feature-based (LabCLIP/HNB)
    # Set to None for auto-detection
    is_finetuned: Optional[bool] = None
    
    # Optional: Path to config.json file for local checkpoints
    # This is used to load model configuration from training
    base_config_path: Optional[str] = None
    
    # Optional: Force loading from OpenCLIP instead of OpenAI CLIP
    # Default is False (use OpenAI CLIP when possible)
    force_openclip: bool = False
    
    # Optional: specific datasets to evaluate (None = all default datasets)
    datasets: Optional[List[str]] = None
    
    # Optional: custom config overrides (as dict)
    config_overrides: Optional[Dict] = None
    
    # CLOVE-specific: weight for state_dict interpolation (default 0.6)
    # Final weights = (1 - clove_weight) * base_weights + clove_weight * checkpoint_weights
    clove_weight: float = 0.6
    
    # CLOVE-specific: pretrained weights for base model (e.g., "openai", "laion2b_s34b_b79k")
    pretrained: str = "openai"


class BatchCheckpointEvaluator:
    """Evaluates multiple checkpoints in batch with multi-GPU support."""
    
    def __init__(
        self,
        checkpoints: List[CheckpointConfig],
        output_dir: str = "evaluation_results",
        default_datasets: Optional[List[str]] = None,
        base_config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        world_size: int = 1,
        rank: int = 0,
        parallel_mode: str = "checkpoints",
        replicate_to_all_steps: bool = False,
        replicate_steps: Optional[List[int]] = None
    ):
        """
        Initialize batch evaluator.
        
        Args:
            checkpoints: List of checkpoint configurations
            output_dir: Directory to save CSV results
            default_datasets: Default datasets to evaluate (if not specified per checkpoint)
            base_config_path: Path to base Hydra config file (optional, only needed for local checkpoints)
            device: Device to use (defaults to CUDA if available)
            world_size: Number of distributed processes
            rank: Rank of current process
            parallel_mode: How to parallelize across GPUs:
                - "checkpoints": Distribute different checkpoints to different GPUs (default)
                - "datasets": Distribute different datasets for same checkpoint across GPUs
            replicate_to_all_steps: If True, replicate results to all existing step values in the CSV
            replicate_steps: Explicit list of step values to replicate to (overrides CSV-based detection)
        """
        self.checkpoints = checkpoints
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.world_size = world_size
        self.rank = rank
        self.parallel_mode = parallel_mode
        self.is_distributed = world_size > 1
        self.replicate_to_all_steps = replicate_to_all_steps
        self.replicate_steps = replicate_steps
        
        self.default_datasets = default_datasets or [
            'VALSE', 'Winoground', 'SugarCrepe', 'VL_CheckList',
            'COCO_Order', 'VG_Relation', 'VG_Attribution',
            'ColorSwap', 'SVO_Probes'
        ]
        
        # Set device based on rank
        if device is None:
            if torch.cuda.is_available():
                if self.is_distributed:
                    self.device = torch.device(f'cuda:{rank}')
                else:
                    self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Load base config (only if provided or if path exists)
        if base_config_path and os.path.exists(base_config_path):
            self.base_cfg = self._load_base_config(base_config_path)
        else:
            # Use minimal default config for OpenCLIP/HuggingFace models
            self.base_cfg = OmegaConf.create({
                'model': {
                    'clip_model': 'ViT-B/32',
                    'embedding_dim': 512,
                    'image_layer_names': [],
                    'text_layer_names': [],
                },
                'alignment': {
                    'alignment_type': 'FT',
                    'ft_image': False,
                    'ft_text': False,
                },
                'evaluation': {
                    'dataset_eval_datasets': self.default_datasets,
                }
            })
            if self.rank == 0 and base_config_path:
                logging.info(f"Config file not found at {base_config_path}, using default config")
        
        if self.rank == 0:
            logging.info("Initialized BatchCheckpointEvaluator")
            logging.info(f"  Output directory: {self.output_dir}")
            logging.info(f"  Device: {self.device}")
            logging.info(f"  World size: {world_size}")
            logging.info(f"  Rank: {rank}")
            logging.info(f"  Number of checkpoints: {len(checkpoints)}")
            logging.info(f"  Default datasets: {', '.join(self.default_datasets)}")
    
    def _load_base_config(self, config_path: str) -> DictConfig:
        """Load base Hydra configuration."""
        try:
            # Try to load as YAML file
            if os.path.exists(config_path):
                cfg = OmegaConf.load(config_path)
                if self.rank == 0:
                    logging.info(f"Loaded config from: {config_path}")
            else:
                # Create minimal default config (used for OpenCLIP/HF models that don't need it)
                cfg = OmegaConf.create({
                    'model': {
                        'clip_model': 'ViT-B/32',
                        'embedding_dim': 512,
                        'image_layer_names': [],
                        'text_layer_names': [],
                    },
                    'alignment': {
                        'alignment_type': 'FT',
                        'ft_image': False,
                        'ft_text': False,
                    },
                    'evaluation': {
                        'dataset_eval_datasets': self.default_datasets,
                    }
                })
                if self.rank == 0:
                    logging.info(f"Config file not found at {config_path}, using default config (this is normal for OpenCLIP/HuggingFace models)")
            
            return cfg
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            raise
    
    def _load_checkpoint_model(
        self,
        checkpoint_cfg: CheckpointConfig
    ) -> Tuple[torch.nn.Module, torch.nn.Module, callable, str]:
        """
        Load a checkpoint model.
        
        Returns:
            Tuple of (model, clip_model, preprocess, alignment_type)
        """
        logging.info(f"\nLoading checkpoint: {checkpoint_cfg.name}")
        logging.info(f"  Type: {checkpoint_cfg.checkpoint_type}")
        logging.info(f"  Path: {checkpoint_cfg.checkpoint_path}")
        
        # Update config with checkpoint-specific settings
        cfg = OmegaConf.merge(self.base_cfg, checkpoint_cfg.config_overrides or {})
        cfg.model.clip_model = checkpoint_cfg.base_model
        
        if checkpoint_cfg.checkpoint_type == 'tripletclip':
            return self._load_tripletclip_model(checkpoint_cfg, cfg)
        elif checkpoint_cfg.checkpoint_type == 'huggingface':
            return self._load_huggingface_model(checkpoint_cfg, cfg)
        elif checkpoint_cfg.checkpoint_type == 'openclip':
            return self._load_openclip_model(checkpoint_cfg, cfg)
        elif checkpoint_cfg.checkpoint_type == 'external':
            return self._load_external_checkpoint(checkpoint_cfg, cfg)
        elif checkpoint_cfg.checkpoint_type == 'dac':
            return self._load_dac_checkpoint(checkpoint_cfg, cfg)
        elif checkpoint_cfg.checkpoint_type == 'clove':
            return self._load_clove_checkpoint(checkpoint_cfg, cfg)
        elif checkpoint_cfg.checkpoint_type == 'projection':
            return self._load_projection_checkpoint(checkpoint_cfg, cfg)
        elif checkpoint_cfg.checkpoint_type == 'local':
            return self._load_local_checkpoint(checkpoint_cfg, cfg)
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_cfg.checkpoint_type}")
    
    def _load_tripletclip_model(
        self,
        checkpoint_cfg: CheckpointConfig,
        cfg: DictConfig
    ) -> Tuple[torch.nn.Module, torch.nn.Module, callable, str]:
        """
        Load TripletCLIP model with separate vision and text encoders.
        
        TripletCLIP models from HuggingFace have separate encoder subfolders:
        - vision-encoder/: CLIPVisionEncoderOnly
        - text-encoder/: CLIPTextEncoderOnly or CustomTextEncoderOnly
        
        The model uses custom HuggingFace model classes defined in utils.py.
        """
        from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, CLIPProcessor
        from huggingface_hub import hf_hub_download
        import importlib.util
        
        model_id = checkpoint_cfg.checkpoint_path
        logging.info(f"  Loading TripletCLIP model: {model_id}")
        
        # Download and load the utils.py module from HuggingFace
        try:
            utils_path = hf_hub_download(repo_id=model_id, filename="utils.py")
            spec = importlib.util.spec_from_file_location("tripletclip_utils", utils_path)
            tripletclip_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tripletclip_utils)
            logging.info("  ✓ Loaded TripletCLIP utils module")
        except Exception as e:
            logging.warning(f"  Could not load utils.py: {e}")
            tripletclip_utils = None
        
        # Load vision encoder
        logging.info("  Loading vision encoder...")
        try:
            vision_encoder = AutoModel.from_pretrained(
                model_id,
                subfolder="vision-encoder",
                trust_remote_code=True
            ).to(self.device)
            vision_encoder.eval()
            logging.info("  ✓ Vision encoder loaded")
        except Exception as e:
            logging.error(f"  Failed to load vision encoder: {e}")
            raise
        
        # Load text encoder
        logging.info("  Loading text encoder...")
        try:
            text_encoder = AutoModel.from_pretrained(
                model_id,
                subfolder="text-encoder",
                trust_remote_code=True
            ).to(self.device)
            text_encoder.eval()
            logging.info("  ✓ Text encoder loaded")
        except Exception as e:
            logging.error(f"  Failed to load text encoder: {e}")
            raise
        
        # Load image processor
        logging.info("  Loading image processor...")
        try:
            image_processor = AutoImageProcessor.from_pretrained(
                model_id,
                subfolder="vision-encoder",
                trust_remote_code=True
            )
        except Exception:
            logging.info("  Using default CLIP image processor")
            image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").image_processor
        
        # Load tokenizer
        logging.info("  Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                subfolder="text-encoder",
                trust_remote_code=True
            )
        except Exception:
            logging.info("  Using default CLIP tokenizer")
            tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer
        
        # Create wrapper model
        wrapped_model = TripletCLIPWrapper(
            vision_encoder=vision_encoder,
            text_encoder=text_encoder,
            image_processor=image_processor,
            tokenizer=tokenizer,
            device=self.device
        )
        wrapped_model.eval()
        
        logging.info("  ✓ TripletCLIP model loaded successfully")
        
        # Return wrapper as both model and clip_model (same interface for FT-style evaluation)
        return wrapped_model, wrapped_model, wrapped_model.preprocess, "TripletCLIP"
    
    def _load_huggingface_model(
        self,
        checkpoint_cfg: CheckpointConfig,
        cfg: DictConfig
    ) -> Tuple[torch.nn.Module, torch.nn.Module, callable, str]:
        """Load HuggingFace Hub model."""
        
        # Add hf-hub prefix if not present
        model_name = checkpoint_cfg.checkpoint_path
        if not model_name.startswith('hf-hub:'):
            model_name = f'hf-hub:{model_name}'
        
        logging.info(f"  Loading from HuggingFace Hub: {model_name}")
        
        # Load using universal CLIP wrapper with force_openclip flag
        clip_model, preprocess = load_clip_model(
            model_name, 
            self.device, 
            force_openclip=checkpoint_cfg.force_openclip
        )
        
        logging.info("  ✓ HuggingFace model loaded successfully")
        
        # HuggingFace models are typically used as-is (zero-shot style)
        return clip_model, clip_model, preprocess, "FT"
    
    def _load_openclip_model(
        self,
        checkpoint_cfg: CheckpointConfig,
        cfg: DictConfig
    ) -> Tuple[torch.nn.Module, torch.nn.Module, callable, str]:
        """Load OpenCLIP model."""
        
        model_name = checkpoint_cfg.checkpoint_path
        logging.info(f"  Loading OpenCLIP model: {model_name}")
        
        # Load using universal CLIP wrapper (it handles OpenCLIP)
        clip_model, preprocess = load_clip_model(model_name, self.device, force_openclip=True)
        
        logging.info(f"  ✓ OpenCLIP model loaded successfully")
        
        return clip_model, clip_model, preprocess, "OpenCLIP"
    
    def _load_external_checkpoint(
        self,
        checkpoint_cfg: CheckpointConfig,
        cfg: DictConfig
    ) -> Tuple[torch.nn.Module, torch.nn.Module, callable, str]:
        """
        Load external checkpoint (.pt file without config).
        
        This is for models like CE-CLIP, FSC-CLIP that are just state dicts
        to be loaded into a base CLIP model.
        """
        
        checkpoint_path = checkpoint_cfg.checkpoint_path
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logging.info(f"  Loading external checkpoint: {checkpoint_path}")
        logging.info(f"  Base model: {checkpoint_cfg.base_model}")
        
        # Load base CLIP model with force_openclip flag
        clip_model, preprocess = load_clip_model(
            checkpoint_cfg.base_model, 
            self.device, 
            force_openclip=checkpoint_cfg.force_openclip
        )
        logging.info("  ✓ Base CLIP model loaded")
        
        # Load checkpoint
        # Note: weights_only=False is required for external checkpoints that may contain
        # numpy objects or other non-tensor types. Only use with trusted checkpoints.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        logging.info(f"  ✓ Checkpoint file loaded")
        
        # Extract state dict (handle different checkpoint formats)
        if isinstance(checkpoint, dict):
            # Try different keys
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Assume the checkpoint itself is the state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        logging.info(f"  State dict has {len(state_dict)} keys")
        
        # Clean state dict
        state_dict = self._clean_state_dict(state_dict)
        
        # Check if checkpoint contains LoRA weights and merge them if so
        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
        if lora_keys:
            logging.info(f"  Detected {len(lora_keys)} LoRA keys in checkpoint, merging...")
            state_dict = self._merge_lora_weights(state_dict)
            logging.info(f"  ✓ LoRA weights merged, state dict now has {len(state_dict)} keys")
        
        # Try to load into CLIP model with strict=True first, then strict=False
        try:
            clip_model.load_state_dict(state_dict, strict=True)
            logging.info(f"  ✓ Loaded checkpoint with strict=True")
        except Exception as e:
            logging.warning(f"  Could not load with strict=True: {e}")
            logging.info(f"  Trying with strict=False...")
            
            missing_keys, unexpected_keys = clip_model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logging.warning(f"  Missing keys: {len(missing_keys)} keys")
                if len(missing_keys) <= 10:
                    for key in missing_keys:
                        logging.warning(f"    - {key}")
            
            if unexpected_keys:
                logging.warning(f"  Unexpected keys: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 10:
                    for key in unexpected_keys:
                        logging.warning(f"    - {key}")
            
            logging.info(f"  ✓ Loaded checkpoint with strict=False")
        
        clip_model.eval()
        
        logging.info(f"  ✓ External checkpoint loaded successfully")
        
        return clip_model, clip_model, preprocess, "External"
    
    def _merge_lora_weights(self, state_dict: dict) -> dict:
        """
        Merge LoRA A/B matrices into base weights.
        
        DAC (Decomposed Attribute Contrast) checkpoints use LoRA-style weight decomposition:
            W' = W + B @ A
        where A is [rank, in_features] and B is [out_features, rank].
        
        This function merges the LoRA deltas into the base weights and returns
        a clean state dict compatible with standard CLIP models.
        
        Args:
            state_dict: State dict containing base weights and LoRA A/B matrices
            
        Returns:
            Merged state dict with LoRA weights applied to base weights
        """
        merged_state_dict = {}
        lora_keys_processed = set()
        
        # First pass: identify all LoRA key pairs
        lora_pairs = {}  # base_key -> (lora_a_key, lora_b_key)
        
        for key in state_dict.keys():
            if key.endswith('.lora_A') or key.endswith('_lora_A'):
                # Find corresponding base weight and lora_B
                if key.endswith('.lora_A'):
                    base_key = key[:-len('.lora_A')]
                    lora_b_key = base_key + '.lora_B'
                else:  # ends with _lora_A (e.g., in_proj_weight_lora_A)
                    base_key = key[:-len('_lora_A')]
                    lora_b_key = base_key + '_lora_B'
                
                if lora_b_key in state_dict:
                    lora_pairs[base_key] = (key, lora_b_key)
                    lora_keys_processed.add(key)
                    lora_keys_processed.add(lora_b_key)
        
        logging.info(f"  Found {len(lora_pairs)} LoRA weight pairs to merge")
        
        # Second pass: merge weights
        for key, value in state_dict.items():
            # Skip LoRA keys - they're processed with their base weights
            if key in lora_keys_processed:
                continue
            
            # Clean key (remove module. prefix)
            clean_key = key
            if clean_key.startswith('module.'):
                clean_key = clean_key[len('module.'):]
            
            # Check if this key has LoRA weights to merge
            # Need to check both original and cleaned key
            original_base_key = key[len('module.'):] if key.startswith('module.') else key
            
            if key in lora_pairs or original_base_key in lora_pairs:
                pair_key = key if key in lora_pairs else original_base_key
                lora_a_key, lora_b_key = lora_pairs[pair_key]
                
                lora_a = state_dict[lora_a_key]  # [rank, in_features] or similar
                lora_b = state_dict[lora_b_key]  # [out_features, rank] or similar
                
                # Compute delta = B @ A and add to base weight
                try:
                    if len(value.shape) == 2:
                        # Linear layer: W is [out, in], A is [r, in], B is [out, r]
                        # Delta = B @ A gives [out, in]
                        delta = lora_b @ lora_a
                    elif len(value.shape) == 4:
                        # Conv2d: W is [out, in, h, w]
                        # Reshape delta to match
                        delta = (lora_b @ lora_a).view(value.shape)
                    elif len(value.shape) == 1:
                        # Bias or 1D tensor - shouldn't have LoRA, but handle gracefully
                        delta = (lora_b @ lora_a).squeeze()
                    else:
                        # Default: try matrix multiply and reshape
                        delta = (lora_b @ lora_a).view(value.shape)
                    
                    merged_weight = value + delta
                    logging.debug(f"    Merged LoRA into: {clean_key}")
                except Exception as e:
                    logging.warning(f"  Failed to merge LoRA for {clean_key}: {e}")
                    logging.warning(f"    Base shape: {value.shape}, A shape: {lora_a.shape}, B shape: {lora_b.shape}")
                    merged_weight = value
            else:
                merged_weight = value
            
            merged_state_dict[clean_key] = merged_weight
        
        logging.info(f"  Merged state dict has {len(merged_state_dict)} keys")
        return merged_state_dict
    
    def _load_dac_checkpoint(
        self,
        checkpoint_cfg: CheckpointConfig,
        cfg: DictConfig
    ) -> Tuple[torch.nn.Module, torch.nn.Module, callable, str]:
        """
        Load DAC (Decomposed Attribute Contrast) checkpoint with LoRA merging.
        
        DAC checkpoints contain LoRA weights (lora_A, lora_B) that need to be
        merged into the base CLIP weights before loading.
        
        Paper: https://arxiv.org/abs/2310.11563
        """
        checkpoint_path = checkpoint_cfg.checkpoint_path
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logging.info(f"  Loading DAC checkpoint with LoRA merging: {checkpoint_path}")
        logging.info(f"  Base model: {checkpoint_cfg.base_model}")
        
        # Load base CLIP model
        clip_model, preprocess = load_clip_model(
            checkpoint_cfg.base_model, 
            self.device, 
            force_openclip=checkpoint_cfg.force_openclip
        )
        logging.info("  ✓ Base CLIP model loaded")
        
        # Load checkpoint (weights_only=False required for DAC checkpoints with numpy objects)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        logging.info("  ✓ Checkpoint file loaded")
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Count LoRA keys
        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
        logging.info(f"  Found {len(lora_keys)} LoRA keys in checkpoint")
        
        # Merge LoRA weights into base weights
        merged_state_dict = self._merge_lora_weights(state_dict)
        
        # Load merged weights into model
        try:
            clip_model.load_state_dict(merged_state_dict, strict=True)
            logging.info("  ✓ Loaded merged checkpoint with strict=True")
        except Exception as e:
            logging.warning(f"  Could not load with strict=True: {e}")
            logging.info("  Trying with strict=False...")
            
            missing_keys, unexpected_keys = clip_model.load_state_dict(merged_state_dict, strict=False)
            
            if missing_keys:
                logging.warning(f"  Missing keys: {len(missing_keys)}")
                if len(missing_keys) <= 10:
                    for key in missing_keys:
                        logging.warning(f"    - {key}")
            
            if unexpected_keys:
                logging.warning(f"  Unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 10:
                    for key in unexpected_keys:
                        logging.warning(f"    - {key}")
            
            logging.info("  ✓ Loaded merged checkpoint with strict=False")
        
        clip_model.eval()
        
        logging.info("  ✓ DAC checkpoint loaded successfully (LoRA weights merged)")
        
        return clip_model, clip_model, preprocess, "DAC"
    
    def _load_clove_checkpoint(
        self,
        checkpoint_cfg: CheckpointConfig,
        cfg: DictConfig
    ) -> Tuple[torch.nn.Module, torch.nn.Module, callable, str]:
        """
        Load CLOVE (Contrastive Language-Object Visual Encoding) checkpoint.
        
        CLOVE checkpoints require special handling:
        1. They use patch_model with weight interpolation (default weight=0.6)
        2. They may have positional_embedding size mismatch (64 vs 77) requiring interpolation
        
        Paper/GitHub: https://github.com/netflix/clove
        
        Example checkpoint URL:
        https://github.com/Netflix/clove/releases/download/pretrained/clove_without_patching.pt
        """
        import open_clip
        
        checkpoint_path = checkpoint_cfg.checkpoint_path
        clove_weight = checkpoint_cfg.clove_weight
        pretrained = checkpoint_cfg.pretrained
        
        logging.info(f"  Loading CLOVE checkpoint: {checkpoint_path}")
        logging.info(f"  Base model: {checkpoint_cfg.base_model}")
        logging.info(f"  Pretrained: {pretrained}")
        logging.info(f"  CLOVE weight (for interpolation): {clove_weight}")
        
        # Convert base model format (ViT-B/32 -> ViT-B-32 for open_clip)
        openclip_model_name = checkpoint_cfg.base_model.replace('/', '-')
        
        # Create base model using open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            openclip_model_name, 
            pretrained=pretrained,
            device=self.device
        )
        model.eval()
        logging.info("  ✓ Base OpenCLIP model loaded")
        
        # Load checkpoint - handle both local files and URLs
        if checkpoint_path.startswith('http://') or checkpoint_path.startswith('https://'):
            logging.info("  Downloading checkpoint from URL...")
            try:
                # Try using open_clip's pt_load if available
                from open_clip.factory import pt_load
                state_dict = pt_load(checkpoint_path, map_location=self.device)
            except ImportError:
                # Fallback: download manually
                import urllib.request
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
                    urllib.request.urlretrieve(checkpoint_path, tmp_file.name)
                    state_dict = torch.load(tmp_file.name, map_location=self.device, weights_only=False)
                    os.unlink(tmp_file.name)
        else:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        logging.info("  ✓ Checkpoint file loaded")
        
        # Extract state dict if nested
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        
        # Try to use CLOVE's patch_model if available
        try:
            from training.utils import get_state_dict, patch_model
            
            # get_state_dict handles key mapping between checkpoint and model
            processed_state_dict = get_state_dict(state_dict, model)
            
            # patch_model interpolates: new_weights = (1-weight)*base + weight*checkpoint
            patch_model(model, processed_state_dict, weight_for_state_dict=clove_weight)
            logging.info(f"  ✓ Applied CLOVE patch_model with weight={clove_weight}")
            
        except ImportError:
            logging.warning("  CLOVE training.utils not available, using manual patching")
            self._manual_clove_patch(model, state_dict, clove_weight)
        
        model.eval()
        logging.info("  ✓ CLOVE checkpoint loaded successfully")
        
        # Create a wrapper that's compatible with our evaluation interface
        wrapped_model = CLOVEModelWrapper(model, self.device)
        
        return wrapped_model, wrapped_model, preprocess, "CLOVE"
    
    def _manual_clove_patch(
        self, 
        model: torch.nn.Module, 
        state_dict: dict, 
        weight: float
    ):
        """
        Manually patch model weights with CLOVE checkpoint using interpolation.
        
        This handles the positional_embedding size mismatch (64 vs 77) by interpolating
        the checkpoint's smaller positional embeddings to match the model's expected size.
        
        Args:
            model: Base CLIP model
            state_dict: CLOVE checkpoint state dict
            weight: Interpolation weight (0.6 = 60% checkpoint, 40% base)
        """
        model_state = model.state_dict()
        
        # Clean state dict keys
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
            cleaned_state_dict[new_key] = value
        
        patched_count = 0
        skipped_count = 0
        interpolated_count = 0
        
        for key, base_value in model_state.items():
            if key not in cleaned_state_dict:
                skipped_count += 1
                continue
            
            ckpt_value = cleaned_state_dict[key]
            
            # Handle shape mismatch (especially for positional_embedding)
            if base_value.shape != ckpt_value.shape:
                if 'positional_embedding' in key:
                    # Interpolate positional embeddings
                    ckpt_value = self._interpolate_pos_embed(ckpt_value, base_value.shape[0])
                    interpolated_count += 1
                    logging.info(f"    Interpolated {key}: {cleaned_state_dict[key].shape} -> {base_value.shape}")
                else:
                    logging.warning(f"    Shape mismatch for {key}: checkpoint {ckpt_value.shape} vs model {base_value.shape}, skipping")
                    skipped_count += 1
                    continue
            
            # Interpolate: new = (1 - weight) * base + weight * checkpoint
            new_value = (1 - weight) * base_value + weight * ckpt_value.to(base_value.device)
            model_state[key] = new_value
            patched_count += 1
        
        # Load patched state dict
        model.load_state_dict(model_state)
        logging.info(f"  ✓ Manual CLOVE patching complete: {patched_count} patched, {interpolated_count} interpolated, {skipped_count} skipped")
    
    def _interpolate_pos_embed(
        self, 
        pos_embed: torch.Tensor, 
        target_length: int
    ) -> torch.Tensor:
        """
        Interpolate positional embeddings to match target length.
        
        CLOVE checkpoints may have 64 positional embeddings while the model expects 77.
        We use linear interpolation to resize.
        
        Args:
            pos_embed: Original positional embeddings [orig_len, embed_dim]
            target_length: Target number of positions (e.g., 77)
            
        Returns:
            Interpolated positional embeddings [target_length, embed_dim]
        """
        orig_length, embed_dim = pos_embed.shape
        
        if orig_length == target_length:
            return pos_embed
        
        # Use linear interpolation
        # Reshape to [1, embed_dim, orig_length] for interpolation
        pos_embed_reshaped = pos_embed.T.unsqueeze(0)  # [1, embed_dim, orig_length]
        
        # Interpolate
        interpolated = torch.nn.functional.interpolate(
            pos_embed_reshaped,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        
        # Reshape back to [target_length, embed_dim]
        return interpolated.squeeze(0).T

    def _load_projection_checkpoint(
        self,
        checkpoint_cfg: CheckpointConfig,
        cfg: DictConfig
    ) -> Tuple[torch.nn.Module, torch.nn.Module, callable, str]:
        """
        Load checkpoint with text projection layer.
        
        This handles checkpoints that contain only a linear projection layer to be
        applied after the CLIP text encoder. The checkpoint format should be:
            {"linear.weight": Tensor, ...} or {"weight": Tensor, ...}
        
        The projection transforms text features: text_features = projection(clip_text_features)
        """
        
        checkpoint_path = checkpoint_cfg.checkpoint_path
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logging.info(f"  Loading projection checkpoint: {checkpoint_path}")
        logging.info(f"  Base model: {checkpoint_cfg.base_model}")
        
        # Load base CLIP model
        clip_model, preprocess = load_clip_model(
            checkpoint_cfg.base_model, 
            self.device, 
            force_openclip=checkpoint_cfg.force_openclip
        )
        logging.info("  ✓ Base CLIP model loaded")
        
        # Load projection checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        logging.info("  ✓ Projection checkpoint file loaded")
        
        # Extract projection weights
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
                # Assume the checkpoint is the weight tensor or contains state_dict
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
        
        # Infer dimensions from weight shape
        # Linear weight is [out_features, in_features]
        out_features, in_features = weight.shape
        logging.info(f"  Projection layer: {in_features} -> {out_features}")
        
        # Create projection layer
        projection = torch.nn.Linear(in_features, out_features, bias=bias is not None)
        projection.weight.data = weight
        if bias is not None:
            projection.bias.data = bias
        projection = projection.to(self.device)
        projection.eval()
        
        logging.info("  ✓ Projection layer created")
        
        # Create wrapped model
        wrapped_model = CLIPWithTextProjection(
            clip_model=clip_model,
            projection_layer=projection,
            device=self.device,
            preprocess=preprocess
        )
        wrapped_model.eval()
        
        logging.info("  ✓ Projection checkpoint loaded successfully")
        
        return wrapped_model, wrapped_model, preprocess, "Projection"

    def _load_local_checkpoint(
        self,
        checkpoint_cfg: CheckpointConfig,
        cfg: DictConfig
    ) -> Tuple[torch.nn.Module, torch.nn.Module, callable, str]:
        """Load local checkpoint file."""
        
        checkpoint_path = checkpoint_cfg.checkpoint_path
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logging.info(f"  Loading local checkpoint: {checkpoint_path}")
        
        # Load config from base_config_path if provided
        if checkpoint_cfg.base_config_path:
            config_path = checkpoint_cfg.base_config_path
            if os.path.exists(config_path):
                logging.info(f"  Loading config from: {config_path}")
                import json
                with open(config_path, 'r') as f:
                    ckpt_config = json.load(f)
                # Merge checkpoint config with base config
                cfg = OmegaConf.merge(cfg, OmegaConf.create(ckpt_config))
                logging.info(f"  ✓ Config loaded and merged")
            else:
                logging.warning(f"  Config file not found: {config_path}, using default config")
        
        # Load base CLIP model with force_openclip flag
        clip_model, preprocess = load_clip_model(
            checkpoint_cfg.base_model, 
            self.device, 
            force_openclip=checkpoint_cfg.force_openclip
        )
        
        # Load checkpoint
        # Note: weights_only=False is required for checkpoints that may contain
        # config dicts and other non-tensor types. Only use with trusted checkpoints.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Determine checkpoint type
        if checkpoint_cfg.is_finetuned is not None:
            is_finetuned = checkpoint_cfg.is_finetuned
        else:
            is_finetuned = self._infer_is_finetuned(checkpoint)
        
        alignment_type = "FT" if is_finetuned else "LabCLIP"
        logging.info(f"  Detected alignment type: {alignment_type}")
        
        if is_finetuned:
            # Load as fine-tuning checkpoint
            model = self._load_ft_checkpoint(clip_model, checkpoint, cfg)
            logging.info(f"  ✓ Fine-tuning checkpoint loaded")
            return model, clip_model, preprocess, alignment_type
        else:
            # Load as LabCLIP/HNB checkpoint (head only)
            model = self._load_labclip_checkpoint(checkpoint, cfg)
            logging.info(f"  ✓ LabCLIP/HNB checkpoint loaded")
            return model, clip_model, preprocess, alignment_type
    
    def _infer_is_finetuned(self, checkpoint: dict) -> bool:
        """Infer if checkpoint is fine-tuned or feature-based."""
        
        # Check config in checkpoint
        if 'config' in checkpoint:
            cfg = checkpoint['config']
            if isinstance(cfg, dict):
                alignment = cfg.get('alignment', {})
                if isinstance(alignment, dict):
                    align_type = alignment.get('alignment_type', '')
                    return align_type == 'FT'
        
        # Check state dict keys
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        keys = list(state_dict.keys())
        
        # FT checkpoints have CLIP encoder parameters
        has_encoder_params = any('visual' in k or 'text' in k or 'token_embedding' in k for k in keys)
        
        # LabCLIP checkpoints only have head parameters
        has_only_head_params = any('head' in k or 'mlp' in k or 'proj' in k for k in keys)
        
        return has_encoder_params and not has_only_head_params
    
    def _load_ft_checkpoint(
        self,
        clip_model: torch.nn.Module,
        checkpoint: dict,
        cfg: DictConfig
    ) -> torch.nn.Module:
        """Load fine-tuning checkpoint."""
        
        # Get config from checkpoint if available
        if 'config' in checkpoint:
            ckpt_cfg = OmegaConf.create(checkpoint['config'])
            cfg = OmegaConf.merge(cfg, ckpt_cfg)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        # Clean state dict - remove common prefixes (module., model., etc.)
        state_dict = self._clean_state_dict(state_dict)
        
        # Get layer names from config
        image_layer_names = ['final'] + list(getattr(cfg.model, 'image_layer_names', []))
        text_layer_names = ['final'] + list(getattr(cfg.model, 'text_layer_names', []))
        
        # Infer dimensions from CLIP model
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_txt = torch.zeros(1, 77, dtype=torch.long).to(self.device)
            
            img_features = clip_model.encode_image(dummy_img)
            txt_features = clip_model.encode_text(dummy_txt)
            
            image_layer_dims = {'final': img_features.shape[-1]}
            text_layer_dims = {'final': txt_features.shape[-1]}
        
        # Create head
        head, _, _, _, _ = create_model_from_config(
            cfg=cfg,
            image_layer_dims=image_layer_dims,
            text_layer_dims=text_layer_dims,
            image_layer_names=image_layer_names,
            text_layer_names=text_layer_names,
            dtype=torch.float32,
            is_ft=True,
        )
        
        # Create pipeline
        model = CLIPEndToEndPipeline(
            model=clip_model,
            head=head,
            image_layer_names=image_layer_names,
            text_layer_names=text_layer_names,
            ft_image_encoder=getattr(cfg.alignment, 'ft_image', False),
            ft_text_encoder=getattr(cfg.alignment, 'ft_text', False),
            assume_inputs_on_device=False,
        ).to(self.device)
        
        # Load state dict with strict=False to allow missing/unexpected keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logging.warning(f"  Missing keys in checkpoint: {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                logging.warning(f"    {missing_keys}")
        
        if unexpected_keys:
            logging.warning(f"  Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                logging.warning(f"    {unexpected_keys}")
        
        model.eval()
        
        return model
    
    def _clean_state_dict(self, state_dict: dict) -> dict:
        """
        Clean state dict by removing wrapper prefixes.
        
        Handles various wrapper patterns:
        - module.* (DDP wrapper)
        - model.module.* (nested DDP)
        - _orig_mod.* (torch.compile)
        - model.model.* (double wrapped)
        - model.* (single model prefix - e.g., from training pipeline)
        
        The goal is to get keys that match the base CLIP model:
        - visual.*, transformer.*, positional_embedding, etc.
        """
        cleaned = {}
        
        for key, value in state_dict.items():
            new_key = key
            
            # Handle different wrapper patterns
            if new_key.startswith('module.'):
                # DDP wrapper: module.model.visual.* -> model.visual.*
                new_key = new_key[len('module.'):]
            elif new_key.startswith('model.module.'):
                # Nested DDP: model.module.model.visual.* -> model.model.visual.* -> model.visual.*
                new_key = new_key[len('model.module.'):]
            elif new_key.startswith('_orig_mod.'):
                # torch.compile wrapper
                new_key = new_key[len('_orig_mod.'):]
            
            # Check for double model prefix (model.model.*)
            # This happens when CLIPEndToEndPipeline is wrapped again
            if new_key.startswith('model.model.'):
                # Remove only ONE 'model.' prefix: model.model.visual.* -> model.visual.*
                new_key = new_key[len('model.'):]
            
            # Handle single model. prefix (from training pipeline)
            # model.visual.* -> visual.*, model.transformer.* -> transformer.*
            if new_key.startswith('model.'):
                new_key = new_key[len('model.'):]
            
            cleaned[new_key] = value
        
        return cleaned
    
    def _load_labclip_checkpoint(
        self,
        checkpoint: dict,
        cfg: DictConfig
    ) -> torch.nn.Module:
        """Load LabCLIP/HNB checkpoint (head only)."""
        
        # Get config from checkpoint if available
        if 'config' in checkpoint:
            ckpt_cfg = OmegaConf.create(checkpoint['config'])
            cfg = OmegaConf.merge(cfg, ckpt_cfg)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        # Clean state dict - remove common prefixes
        state_dict = self._clean_state_dict(state_dict)
        
        # Infer dimensions from state dict
        image_layer_dims = {}
        text_layer_dims = {}
        
        for key in state_dict.keys():
            if 'image' in key and 'weight' in key:
                if state_dict[key].dim() > 1:
                    layer_name = key.split('.')[1] if '.' in key else 'final'
                    image_layer_dims[layer_name] = state_dict[key].shape[1]
            elif 'text' in key and 'weight' in key:
                if state_dict[key].dim() > 1:
                    layer_name = key.split('.')[1] if '.' in key else 'final'
                    text_layer_dims[layer_name] = state_dict[key].shape[1]
        
        # Fallback to config dimensions
        if not image_layer_dims:
            image_layer_dims = {'final': getattr(cfg.model, 'embedding_dim', 512)}
        if not text_layer_dims:
            text_layer_dims = {'final': getattr(cfg.model, 'embedding_dim', 512)}
        
        image_layer_names = list(image_layer_dims.keys())
        text_layer_names = list(text_layer_dims.keys())
        
        # Create head
        head, _, _, _, _ = create_model_from_config(
            cfg=cfg,
            image_layer_dims=image_layer_dims,
            text_layer_dims=text_layer_dims,
            image_layer_names=image_layer_names,
            text_layer_names=text_layer_names,
            dtype=torch.float32,
            is_ft=False,
        )
        
        # Wrap in feature pipeline
        model = CLIPFeaturePipeline(head=head).to(self.device)
        
        # Load state dict with strict=False
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logging.warning(f"  Missing keys in checkpoint: {len(missing_keys)} keys")
        
        if unexpected_keys:
            logging.warning(f"  Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        
        model.eval()
        
        return model
    
    def evaluate_checkpoint(
        self,
        checkpoint_cfg: CheckpointConfig
    ) -> Dict:
        """Evaluate a single checkpoint."""
        
        output_csv = self.output_dir / checkpoint_cfg.csv_filename
        
        logging.info(f"\n{'='*80}")
        logging.info(f"EVALUATING: {checkpoint_cfg.name}")
        logging.info(f"{'='*80}")
        
        try:
            # Load model
            model, clip_model, preprocess, alignment_type = self._load_checkpoint_model(checkpoint_cfg)
            
            # Get datasets to evaluate
            datasets = checkpoint_cfg.datasets or self.default_datasets
            
            # Setup dataset evaluator
            logging.info(f"\nSetting up dataset evaluation...")
            logging.info(f"  Datasets: {', '.join(datasets)}")
            logging.info(f"  Output CSV: {output_csv}")
            
            dataset_evaluator = setup_dataset_evaluation(
                datasets=datasets,
                csv_path=str(output_csv),
                enable_visualization=False,
                device=self.device,
                is_ft=(alignment_type == "FT")
            )
            
            # Run evaluation
            logging.info(f"\n{'='*80}")
            logging.info(f"RUNNING EVALUATION: {checkpoint_cfg.name}")
            logging.info(f"{'='*80}\n")
            
            results = dataset_evaluator.evaluate_all(
                model=model,
                clip_model=clip_model,
                preprocess=preprocess,
                step=0,
                epoch=None,
                alignment_type=alignment_type,
                wandb_log=False,
                is_initial_eval=True
            )
            
            # If replicate_to_all_steps is enabled, replicate results to other steps
            if self.replicate_to_all_steps:
                self._replicate_results_to_all_steps(output_csv, results)
            
            # Print summary
            self._print_checkpoint_summary(checkpoint_cfg, results, alignment_type, output_csv)
            
            return {
                'status': 'success',
                'checkpoint': checkpoint_cfg.name,
                'results': results,
                'csv_path': str(output_csv)
            }
        
        except Exception as e:
            logging.error(f"Failed to evaluate {checkpoint_cfg.name}: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'status': 'failed',
                'checkpoint': checkpoint_cfg.name,
                'error': str(e)
            }
    
    def evaluate_checkpoint_dataset_parallel(
        self,
        checkpoint_cfg: CheckpointConfig
    ) -> Dict:
        """
        Evaluate a single checkpoint with datasets distributed across GPUs.
        
        Each GPU evaluates a subset of datasets, then results are gathered.
        Use this when you have one checkpoint but want to speed up evaluation
        by running different datasets on different GPUs.
        """
        import json
        
        output_csv = self.output_dir / checkpoint_cfg.csv_filename
        
        if self.rank == 0:
            logging.info(f"\n{'='*80}")
            logging.info(f"EVALUATING (Dataset-Parallel): {checkpoint_cfg.name}")
            logging.info(f"{'='*80}")
        
        try:
            # Load model on each GPU
            model, clip_model, preprocess, alignment_type = self._load_checkpoint_model(checkpoint_cfg)
            
            # Get all datasets to evaluate
            all_datasets = checkpoint_cfg.datasets or self.default_datasets
            
            # Distribute datasets across GPUs (round-robin)
            my_datasets = []
            for i, dataset in enumerate(all_datasets):
                if i % self.world_size == self.rank:
                    my_datasets.append(dataset)
            
            if self.rank == 0:
                logging.info(f"\nDataset distribution across {self.world_size} GPUs:")
                for r in range(self.world_size):
                    r_datasets = [d for i, d in enumerate(all_datasets) if i % self.world_size == r]
                    logging.info(f"  GPU {r}: {r_datasets}")
            
            logging.info(f"\n[GPU {self.rank}] Evaluating datasets: {my_datasets}")
            
            # Create temporary output file for this GPU's results
            temp_csv = self.output_dir / f".tmp_gpu{self.rank}_{checkpoint_cfg.csv_filename}"
            
            my_results = {}
            
            if my_datasets:
                # Setup dataset evaluator for this GPU's datasets only
                dataset_evaluator = setup_dataset_evaluation(
                    datasets=my_datasets,
                    csv_path=str(temp_csv),
                    enable_visualization=False,
                    device=self.device,
                    is_ft=(alignment_type in ["FT", "TripletCLIP", "External", "DAC", "OpenCLIP"])
                )
                
                # Run evaluation
                my_results = dataset_evaluator.evaluate_all(
                    model=model,
                    clip_model=clip_model,
                    preprocess=preprocess,
                    step=0,
                    epoch=None,
                    alignment_type=alignment_type,
                    wandb_log=False,
                    is_initial_eval=True
                )
                
                logging.info(f"[GPU {self.rank}] Completed {len(my_datasets)} datasets")
            
            # Save partial results to temp file for gathering
            results_file = self.output_dir / f".tmp_results_gpu{self.rank}.json"
            with open(results_file, 'w') as f:
                # Convert tensor values to Python types for JSON serialization
                serializable_results = {}
                for k, v in my_results.items():
                    if hasattr(v, 'item'):
                        serializable_results[k] = v.item()
                    elif isinstance(v, (int, float, str, bool, type(None))):
                        serializable_results[k] = v
                    else:
                        serializable_results[k] = str(v)
                json.dump(serializable_results, f)
            
            # Synchronize - wait for all GPUs to finish
            if self.is_distributed:
                torch.distributed.barrier()
            
            # Gather results on main process
            all_results = {}
            if self.rank == 0:
                for r in range(self.world_size):
                    r_results_file = self.output_dir / f".tmp_results_gpu{r}.json"
                    if r_results_file.exists():
                        with open(r_results_file, 'r') as f:
                            r_results = json.load(f)
                            all_results.update(r_results)
                        # Clean up temp file
                        r_results_file.unlink()
                
                # Clean up temp CSV files
                for r in range(self.world_size):
                    temp_csv_r = self.output_dir / f".tmp_gpu{r}_{checkpoint_cfg.csv_filename}"
                    if temp_csv_r.exists():
                        temp_csv_r.unlink()
                
                # Write combined results to final CSV
                self._write_combined_results_csv(output_csv, all_results, checkpoint_cfg)
                
                # Print summary
                self._print_checkpoint_summary(checkpoint_cfg, all_results, alignment_type, output_csv)
                
                return {
                    'status': 'success',
                    'checkpoint': checkpoint_cfg.name,
                    'results': all_results,
                    'csv_path': str(output_csv)
                }
            else:
                # Non-main processes just return empty result
                return {
                    'status': 'success',
                    'checkpoint': checkpoint_cfg.name,
                    'results': my_results,
                    'csv_path': str(temp_csv)
                }
        
        except Exception as e:
            logging.error(f"[GPU {self.rank}] Failed to evaluate {checkpoint_cfg.name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Synchronize even on failure
            if self.is_distributed:
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            
            return {
                'status': 'failed',
                'checkpoint': checkpoint_cfg.name,
                'error': str(e)
            }
    
    def _write_combined_results_csv(
        self,
        output_csv: Path,
        results: Dict,
        checkpoint_cfg: CheckpointConfig
    ):
        """
        Write combined results from all GPUs to a single CSV file.
        
        Uses the same format as DatasetEvaluator for consistency:
        timestamp, step, epoch, dataset, subset, metric, value, total_samples
        
        Keys in results dict have format: eval/{dataset}/{subset}/{metric}
        
        If self.replicate_to_all_steps is True and the file exists, the results will
        be written for ALL unique step values found in the existing CSV file.
        """
        import csv
        import pandas as pd
        from datetime import datetime
        
        # Group results by dataset/subset to get total_samples for each group
        grouped_results = {}  # (dataset, subset) -> {metric: value, ...}
        
        for key, value in results.items():
            if not isinstance(value, (int, float)):
                continue
                
            # Parse key format: eval/{dataset}/{subset}/{metric}
            parts = key.split('/')
            if len(parts) >= 4 and parts[0] == 'eval':
                dataset = parts[1]
                subset = parts[2]
                metric = '/'.join(parts[3:])  # Handle metrics with '/' in name
                
                group_key = (dataset, subset)
                if group_key not in grouped_results:
                    grouped_results[group_key] = {}
                grouped_results[group_key][metric] = value
        
        # Get list of steps to write to
        steps_to_write = [0]  # Default: just step 0
        epochs_for_steps = {0: ''}  # Default: empty epoch for step 0
        
        if self.replicate_to_all_steps:
            # Priority 1: Use explicitly provided steps
            if self.replicate_steps:
                steps_to_write = sorted(self.replicate_steps)
                epochs_for_steps = {step: '' for step in steps_to_write}
                logging.info(f"  → Replicating results to {len(steps_to_write)} explicit steps: {steps_to_write}")
            # Priority 2: Read from existing CSV
            elif output_csv.exists():
                try:
                    existing_df = pd.read_csv(output_csv)
                    if 'step' in existing_df.columns:
                        unique_steps = existing_df['step'].dropna().unique().tolist()
                        if unique_steps:
                            steps_to_write = sorted([int(s) for s in unique_steps if pd.notna(s)])
                            # Also get corresponding epochs for each step
                            for step in steps_to_write:
                                step_rows = existing_df[existing_df['step'] == step]
                                if not step_rows.empty and 'epoch' in step_rows.columns:
                                    epoch_val = step_rows['epoch'].iloc[0]
                                    epochs_for_steps[step] = epoch_val if pd.notna(epoch_val) else ''
                                else:
                                    epochs_for_steps[step] = ''
                            logging.info(f"  → Replicating results to {len(steps_to_write)} existing steps from CSV: {steps_to_write}")
                except Exception as e:
                    logging.warning(f"  Could not read existing CSV for step replication: {e}")
                    steps_to_write = [0]
                    epochs_for_steps = {0: ''}
            else:
                logging.warning(f"  --replicate_to_all_steps set but CSV doesn't exist and no --replicate_steps provided. Using step 0 only.")
        
        # Build rows in the same format as DatasetEvaluator
        rows = []
        timestamp = datetime.now().isoformat()
        
        for step in steps_to_write:
            epoch = epochs_for_steps.get(step, '')
            for (dataset, subset), metrics in sorted(grouped_results.items()):
                # Get total_samples for this group (default to 0)
                total_samples = metrics.get('total_samples', 0)
                
                for metric, value in sorted(metrics.items()):
                    # Skip num_samples and total_samples metrics to avoid redundancy
                    if metric in ('num_samples', 'total_samples'):
                        continue
                        
                    rows.append([
                        timestamp,
                        step,
                        epoch,
                        dataset,
                        subset,
                        metric,
                        value,
                        total_samples
                    ])
        
        # Write CSV with same headers as DatasetEvaluator
        # If file exists, append without header; otherwise create with header
        if rows:
            file_exists = output_csv.exists()
            mode = 'a' if file_exists else 'w'
            
            with open(output_csv, mode, newline='') as f:
                writer = csv.writer(f)
                # Only write header if creating new file
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'step', 'epoch', 'dataset', 'subset',
                        'metric', 'value', 'total_samples'
                    ])
                writer.writerows(rows)
            
            if file_exists:
                if self.replicate_to_all_steps and len(steps_to_write) > 1:
                    logging.info(f"  ✓ Results replicated to {len(steps_to_write)} steps and appended to: {output_csv}")
                else:
                    logging.info(f"  ✓ Results appended to existing file: {output_csv}")
            else:
                logging.info(f"  ✓ Combined results written to: {output_csv}")
    
    def _replicate_results_to_all_steps(
        self,
        output_csv: Path,
        results: Dict
    ):
        """
        Replicate results (already written with step=0) to all other steps in the CSV.
        
        This is used when --replicate_to_all_steps is enabled and we want to copy
        the baseline results to match all existing step values in the CSV file.
        """
        import csv
        import pandas as pd
        from datetime import datetime
        
        if not output_csv.exists():
            logging.warning("  Cannot replicate: CSV file doesn't exist yet")
            return
        
        # Determine which steps to replicate to
        steps_to_replicate = []
        epochs_for_steps = {}
        
        # Priority 1: Use explicitly provided steps
        if self.replicate_steps:
            steps_to_replicate = [s for s in self.replicate_steps if s != 0]  # Skip 0, already written
            epochs_for_steps = {step: '' for step in steps_to_replicate}
            if steps_to_replicate:
                logging.info(f"  → Replicating to {len(steps_to_replicate)} explicit steps (excluding 0): {steps_to_replicate}")
        else:
            # Priority 2: Read from existing CSV
            try:
                existing_df = pd.read_csv(output_csv)
                if 'step' in existing_df.columns:
                    unique_steps = existing_df['step'].dropna().unique().tolist()
                    if unique_steps:
                        all_steps = sorted([int(s) for s in unique_steps if pd.notna(s)])
                        steps_to_replicate = [s for s in all_steps if s != 0]  # Skip 0, already written
                        # Get corresponding epochs for each step
                        for step in steps_to_replicate:
                            step_rows = existing_df[existing_df['step'] == step]
                            if not step_rows.empty and 'epoch' in step_rows.columns:
                                epoch_val = step_rows['epoch'].iloc[0]
                                epochs_for_steps[step] = epoch_val if pd.notna(epoch_val) else ''
                            else:
                                epochs_for_steps[step] = ''
                        if steps_to_replicate:
                            logging.info(f"  → Replicating to {len(steps_to_replicate)} steps from CSV (excluding 0): {steps_to_replicate}")
            except Exception as e:
                logging.warning(f"  Could not read existing CSV for step replication: {e}")
                return
        
        if not steps_to_replicate:
            logging.info("  → No additional steps to replicate to (only step 0 exists)")
            return
        
        # Parse results and build rows for replication
        # Results dict has format: eval/{dataset}/{subset}/{metric} -> value
        grouped_results = {}
        for key, value in results.items():
            if not isinstance(value, (int, float)):
                continue
            
            parts = key.split('/')
            if len(parts) >= 4 and parts[0] == 'eval':
                dataset = parts[1]
                subset = parts[2]
                metric = '/'.join(parts[3:])
                
                group_key = (dataset, subset)
                if group_key not in grouped_results:
                    grouped_results[group_key] = {}
                grouped_results[group_key][metric] = value
        
        # Build rows for additional steps
        rows = []
        timestamp = datetime.now().isoformat()
        
        for step in steps_to_replicate:
            epoch = epochs_for_steps.get(step, '')
            for (dataset, subset), metrics in sorted(grouped_results.items()):
                total_samples = metrics.get('total_samples', 0)
                
                for metric, value in sorted(metrics.items()):
                    if metric in ('num_samples', 'total_samples'):
                        continue
                    
                    rows.append([
                        timestamp,
                        step,
                        epoch,
                        dataset,
                        subset,
                        metric,
                        value,
                        total_samples
                    ])
        
        # Append replicated rows to CSV
        if rows:
            with open(output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            logging.info(f"  ✓ Results replicated to {len(steps_to_replicate)} additional steps")

    def _print_checkpoint_summary(
        self,
        checkpoint_cfg: CheckpointConfig,
        results: Dict,
        alignment_type: str,
        output_csv: Path
    ):
        """Print evaluation summary for a checkpoint."""
        
        logging.info(f"\n{'='*80}")
        logging.info(f"EVALUATION COMPLETE: {checkpoint_cfg.name}")
        logging.info(f"{'='*80}")
        logging.info(f"\nCheckpoint Type: {checkpoint_cfg.checkpoint_type}")
        logging.info(f"Alignment Type: {alignment_type}")
        logging.info(f"Output CSV: {output_csv}")
        logging.info(f"\nResults Summary:")
        logging.info(f"{'-'*80}")
        
        # Aggregate results by dataset
        dataset_results = {}
        for key, value in results.items():
            if isinstance(value, (int, float)):
                parts = key.split('/')
                if len(parts) >= 2:
                    dataset = parts[0]
                    metric = '/'.join(parts[1:])
                    
                    if dataset not in dataset_results:
                        dataset_results[dataset] = {}
                    dataset_results[dataset][metric] = value
        
        # Print per-dataset results
        for dataset, metrics in sorted(dataset_results.items()):
            logging.info(f"\n{dataset}:")
            for metric, value in sorted(metrics.items()):
                if 'accuracy' in metric.lower():
                    logging.info(f"  {metric:40s}: {value:.4f} ({value*100:.2f}%)")
                else:
                    logging.info(f"  {metric:40s}: {value:.4f}")
        
        logging.info(f"\n{'='*80}\n")
    
    def evaluate_all(self) -> List[Dict]:
        """Evaluate all checkpoints with multi-GPU support."""
        
        if self.rank == 0:
            logging.info(f"\n{'#'*80}")
            logging.info("BATCH EVALUATION STARTING")
            logging.info(f"{'#'*80}")
            logging.info(f"Total checkpoints to evaluate: {len(self.checkpoints)}")
            logging.info(f"Output directory: {self.output_dir}")
            logging.info(f"Parallel mode: {self.parallel_mode}")
            if self.is_distributed:
                logging.info(f"Distributed evaluation across {self.world_size} GPUs")
            logging.info(f"{'#'*80}\n")
        
        # Choose parallelization strategy
        if self.parallel_mode == "datasets" and self.is_distributed:
            # Dataset-parallel: All GPUs evaluate same checkpoint, different datasets
            return self._evaluate_all_dataset_parallel()
        else:
            # Checkpoint-parallel (default): Different GPUs evaluate different checkpoints
            return self._evaluate_all_checkpoint_parallel()
    
    def _evaluate_all_checkpoint_parallel(self) -> List[Dict]:
        """Evaluate checkpoints in parallel across GPUs (original behavior)."""
        all_results = []
        
        # Distribute checkpoints across GPUs
        checkpoints_to_process = []
        for i, checkpoint_cfg in enumerate(self.checkpoints):
            # Assign checkpoint to GPU based on round-robin
            if i % self.world_size == self.rank:
                checkpoints_to_process.append((i, checkpoint_cfg))
        
        if self.rank == 0:
            logging.info("Checkpoint distribution:")
            for rank in range(self.world_size):
                assigned = [i for i, _ in enumerate(self.checkpoints) if i % self.world_size == rank]
                logging.info(f"  GPU {rank}: Checkpoints {assigned}")
        
        # Synchronize before starting (quick barrier, all GPUs should be ready)
        if self.is_distributed:
            torch.distributed.barrier()
        
        # Process assigned checkpoints
        for idx, checkpoint_cfg in checkpoints_to_process:
            logging.info(f"\n[GPU {self.rank}] [{idx+1}/{len(self.checkpoints)}] Starting evaluation of: {checkpoint_cfg.name}")
            
            result = self.evaluate_checkpoint(checkpoint_cfg)
            all_results.append(result)
            
            # Clear CUDA cache between evaluations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Log completion for this GPU (no barrier needed - each GPU finishes independently)
        logging.info(f"\n[GPU {self.rank}] Completed all assigned checkpoints ({len(checkpoints_to_process)} total)")
        
        # Optional: Final barrier only if we need to synchronize for summary
        # Skip this barrier to avoid timeout when GPUs have very different workloads
        # The results are already saved to CSV, so no data is lost
        # if self.is_distributed:
        #     torch.distributed.barrier()
        
        # Print final summary on main process (only shows results from this GPU)
        if self.rank == 0:
            self._print_final_summary(all_results)
        
        return all_results
    
    def _evaluate_all_dataset_parallel(self) -> List[Dict]:
        """
        Evaluate checkpoints with datasets distributed across GPUs.
        
        For each checkpoint, all GPUs work together to evaluate different datasets
        in parallel. This is useful when you have 1 checkpoint but many datasets.
        """
        all_results = []
        
        for idx, checkpoint_cfg in enumerate(self.checkpoints):
            if self.rank == 0:
                logging.info(f"\n[{idx+1}/{len(self.checkpoints)}] Starting dataset-parallel evaluation of: {checkpoint_cfg.name}")
            
            result = self.evaluate_checkpoint_dataset_parallel(checkpoint_cfg)
            
            # Only main process has full results
            if self.rank == 0:
                all_results.append(result)
            
            # Clear CUDA cache between checkpoints
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Print final summary on main process
        if self.rank == 0:
            self._print_final_summary(all_results)
        
        return all_results
    
    def _print_final_summary(self, all_results: List[Dict]):
        """Print final summary of all evaluations."""
        
        logging.info(f"\n{'#'*80}")
        logging.info(f"BATCH EVALUATION COMPLETE")
        logging.info(f"{'#'*80}\n")
        
        successful = [r for r in all_results if r['status'] == 'success']
        failed = [r for r in all_results if r['status'] == 'failed']
        
        logging.info(f"Total checkpoints: {len(all_results)}")
        logging.info(f"✓ Successful: {len(successful)}")
        logging.info(f"✗ Failed: {len(failed)}")
        
        if successful:
            logging.info(f"\n✓ Successfully evaluated:")
            for result in successful:
                logging.info(f"  - {result['checkpoint']}")
                logging.info(f"    CSV: {result['csv_path']}")
        
        if failed:
            logging.info(f"\n✗ Failed evaluations:")
            for result in failed:
                logging.info(f"  - {result['checkpoint']}")
                logging.info(f"    Error: {result['error']}")
        
        logging.info(f"\n{'#'*80}\n")


# ============================================================================
# Helper Functions
# ============================================================================

def load_checkpoints_from_yaml(config_path: str) -> Tuple[List[CheckpointConfig], Optional[List[str]]]:
    """
    Load checkpoint configurations from YAML file.
    
    Returns:
        Tuple of (checkpoints, default_datasets)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoints = []
    for ckpt_dict in config.get('checkpoints', []):
        checkpoints.append(CheckpointConfig(**ckpt_dict))
    
    # Get default_datasets from config if provided
    default_datasets = config.get('default_datasets', None)
    
    return checkpoints, default_datasets


def parse_args():
    """Parse command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch evaluate multiple CLIP checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate from YAML config
    python scripts/batch_evaluate_checkpoints.py --config configs/eval_checkpoints.yaml
    
    # Single checkpoint evaluation
    python scripts/batch_evaluate_checkpoints.py \\
        --checkpoint_type openclip \\
        --checkpoint_path "ViT-B/32" \\
        --csv_filename baseline.csv \\
        --name "OpenAI CLIP Baseline"
    
    # Multi-GPU evaluation (8 GPUs)
    torchrun --nproc_per_node=8 scripts/batch_evaluate_checkpoints.py \\
        --config configs/eval_checkpoints.yaml
        """
    )
    
    # Config file or individual checkpoint arguments
    parser.add_argument('--config', type=str, help='Path to YAML config file with checkpoint list')
    
    # Individual checkpoint arguments (used if --config not provided)
    parser.add_argument('--name', type=str, help='Name for this checkpoint')
    parser.add_argument('--csv_filename', type=str, help='Output CSV filename')
    parser.add_argument('--checkpoint_type', type=str, 
                       choices=['local', 'external', 'huggingface', 'openclip', 'tripletclip'],
                       help='Type of checkpoint: local (with config), external (.pt only), huggingface, openclip, tripletclip (separate encoders)')
    parser.add_argument('--checkpoint_path', type=str, help='Path or identifier for checkpoint')
    parser.add_argument('--base_model', type=str, default='ViT-B/32', help='Base CLIP model')
    parser.add_argument('--is_finetuned', type=lambda x: None if x == 'auto' else x == 'true',
                       default=None, help='Whether checkpoint is finetuned (true/false/auto)')
    parser.add_argument('--datasets', type=str, nargs='+', help='Specific datasets to evaluate')
    
    # General arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for CSV files')
    parser.add_argument('--base_config', type=str, default=None,
                       help='Base config file path (only needed for local checkpoints, optional)')
    
    # Parallelization mode
    parser.add_argument('--parallel_mode', type=str, default='checkpoints',
                       choices=['checkpoints', 'datasets'],
                       help='How to parallelize across GPUs: "checkpoints" (default) distributes different '
                            'checkpoints to different GPUs, "datasets" distributes different datasets '
                            'for the SAME checkpoint across GPUs (use for single checkpoint eval)')
    
    # Replicate results to all existing steps in CSV
    parser.add_argument('--replicate_to_all_steps', action='store_true',
                       help='If set, replicate the evaluation results to ALL existing step values '
                            'found in the CSV file. Useful when adding a new dataset (like VL_CheckList) '
                            'to CSVs that already have results for multiple training steps.')
    parser.add_argument('--replicate_steps', type=str, default=None,
                       help='Comma-separated list of step values to replicate results to. '
                            'Use with --replicate_to_all_steps. If not provided, steps are read from existing CSV. '
                            'Example: --replicate_steps "0,1000,2000,3000"')
    
    # Distributed training arguments (auto-detected from environment if using torchrun)
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    return parser.parse_args()


def setup_distributed_environment():
    """Setup distributed environment for multi-GPU evaluation."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Using torchrun or similar launcher
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group with extended timeout (6 hours)
        # This prevents timeout when GPUs have uneven workloads
        import datetime
        timeout = datetime.timedelta(hours=6)
        
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timeout
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU or CPU
        return 0, 1, 0


# ============================================================================
# CONFIGURATION: Define your checkpoints here (used if no --config provided)
# ============================================================================

DEFAULT_CHECKPOINTS_TO_EVALUATE = [
    # ============================================================================
    # Zero-shot Baselines
    # ============================================================================
    
    # Example 1: Zero-shot OpenAI CLIP baseline
    CheckpointConfig(
        name="OpenAI CLIP ViT-B/32 (Zero-shot)",
        csv_filename="00_baseline_openai_vitb32.csv",
        checkpoint_type="openclip",
        checkpoint_path="ViT-B/32",
        base_model="ViT-B/32",
    ),
    
    # Example 2: OpenCLIP model with specific pretrained weights
    CheckpointConfig(
        name="OpenCLIP ViT-L/14 OpenAI weights",
        csv_filename="01_openclip_vitl14_openai.csv",
        checkpoint_type="openclip",
        checkpoint_path="ViT-L-14@openai",
        base_model="ViT-L/14",
    ),
    
    # ============================================================================
    # External CLIP Variants
    # ============================================================================
    
    # Example 3: TripletCLIP from HuggingFace (separate vision/text encoders)
    # CheckpointConfig(
    #     name="TripletCLIP CC12M ViT-B/12",
    #     csv_filename="tripletclip_cc12m_vitb12.csv",
    #     checkpoint_type="tripletclip",
    #     checkpoint_path="TripletCLIP/CC12M_TripletCLIP_ViTB12",
    #     base_model="ViT-B/32",  # Base model (for reference, TripletCLIP has its own encoders)
    # ),
    
    # Example 4: CE-CLIP from HuggingFace
    # CheckpointConfig(
    #     name="CE-CLIP ViT-B/32",
    #     csv_filename="ce_clip_vitb32.csv",
    #     checkpoint_type="huggingface",
    #     checkpoint_path="le723z/CE_CLIP",
    #     base_model="ViT-B/32",
    # ),
    
    # Example 4: FSC-CLIP external checkpoint
    # CheckpointConfig(
    #     name="FSC-CLIP ViT-B/32 (LaiOnCoco)",
    #     csv_filename="fsc_clip_laioncoco_vitb32.csv",
    #     checkpoint_type="external",
    #     checkpoint_path="external_checkpoints/fsc-clip/laioncoco_fsc-clip-ViT-B-32.pt",
    #     base_model="ViT-B/32",
    # ),
    
    # ============================================================================
    # HuggingFace Models
    # ============================================================================
    
    # Example 5: HuggingFace Hub model
    # CheckpointConfig(
    #     name="NegCLIP from HuggingFace",
    #     csv_filename="02_negclip_huggingface.csv",
    #     checkpoint_type="huggingface",
    #     checkpoint_path="meronym/negclip-base",  # or "hf-hub:meronym/negclip-base"
    #     base_model="ViT-B/32",
    # ),
    
    # ============================================================================
    # Local Checkpoints
    # ============================================================================
    
    # Example 6: Local fine-tuned checkpoint
    # CheckpointConfig(
    #     name="My LabCLIP FT Model",
    #     csv_filename="03_labclip_ft_model.csv",
    #     checkpoint_type="local",
    #     checkpoint_path="/path/to/your/checkpoint.pt",
    #     base_model="ViT-B/32",
    #     is_finetuned=True,  # Set to True for FT, False for LabCLIP/HNB, None for auto-detect
    # ),
    
    # Example 7: Local LabCLIP/HNB checkpoint (head only)
    # CheckpointConfig(
    #     name="My LabCLIP Head Model",
    #     csv_filename="04_labclip_head_model.csv",
    #     checkpoint_type="local",
    #     checkpoint_path="/path/to/your/head_checkpoint.pt",
    #     base_model="ViT-B/32",
    #     is_finetuned=False,
    # ),
    
    # Example 8: Evaluate only specific datasets
    # CheckpointConfig(
    #     name="Model on specific datasets",
    #     csv_filename="05_specific_datasets.csv",
    #     checkpoint_type="openclip",
    #     checkpoint_path="ViT-B/16",
    #     base_model="ViT-B/16",
    #     datasets=['VALSE', 'Winoground', 'SugarCrepe'],  # Only these datasets
    # ),
    
    # ============================================================================
    # Projection Checkpoints (text projection layer applied after CLIP encoder)
    # ============================================================================
    
    # Example 9: Text projection checkpoint
    # This type loads a linear projection layer to be applied after CLIP's text encoder.
    # Checkpoint format: {"linear.weight": Tensor, "linear.bias": Tensor (optional)}
    # CheckpointConfig(
    #     name="COCO Alignment Projection",
    #     csv_filename="coco_alignment_projection.csv",
    #     checkpoint_type="projection",
    #     checkpoint_path="/path/to/projection_checkpoint.pt",
    #     base_model="ViT-B/32",  # Base CLIP model to use
    # ),
]


def main():
    """Main entry point with command-line support and multi-GPU."""
    
    # Parse command-line arguments
    args = parse_args()
    
    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed_environment()
    
    # Configure logging (only main process prints to console)
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Determine checkpoints to evaluate
    default_datasets = None
    
    if args.config:
        # Load from YAML config
        if rank == 0:
            logging.info(f"Loading checkpoints from config: {args.config}")
        checkpoints, default_datasets = load_checkpoints_from_yaml(args.config)
    elif args.checkpoint_type and args.checkpoint_path and args.csv_filename:
        # Single checkpoint from command line
        if rank == 0:
            logging.info("Using checkpoint from command-line arguments")
        checkpoints = [
            CheckpointConfig(
                name=args.name or f"{args.checkpoint_type}_{args.checkpoint_path}",
                csv_filename=args.csv_filename,
                checkpoint_type=args.checkpoint_type,
                checkpoint_path=args.checkpoint_path,
                base_model=args.base_model,
                is_finetuned=args.is_finetuned,
                datasets=args.datasets,
            )
        ]
    else:
        # Use default checkpoints defined in script
        if rank == 0:
            logging.info("Using default checkpoints from script")
        checkpoints = DEFAULT_CHECKPOINTS_TO_EVALUATE
    
    # Parse replicate_steps if provided
    replicate_steps = None
    if args.replicate_steps:
        replicate_steps = [int(s.strip()) for s in args.replicate_steps.split(',')]
        if rank == 0:
            logging.info(f"Explicit replicate steps: {replicate_steps}")
    
    if rank == 0:
        logging.info(f"Total checkpoints to evaluate: {len(checkpoints)}")
        if world_size > 1:
            logging.info(f"Using distributed evaluation across {world_size} GPUs")
            logging.info(f"Parallel mode: {args.parallel_mode}")
        if args.replicate_to_all_steps:
            if replicate_steps:
                logging.info(f"Replicate to all steps: ENABLED (will use explicit steps: {replicate_steps})")
            else:
                logging.info("Replicate to all steps: ENABLED (results will be written for all existing step values in CSV)")
    
    # Initialize batch evaluator with distributed settings
    evaluator = BatchCheckpointEvaluator(
        checkpoints=checkpoints,
        output_dir=args.output_dir,
        default_datasets=default_datasets,  # Use from config if provided
        base_config_path=args.base_config,
        world_size=world_size,
        rank=rank,
        parallel_mode=args.parallel_mode,
        replicate_to_all_steps=args.replicate_to_all_steps,
        replicate_steps=replicate_steps,
    )
    
    # Run evaluation on all checkpoints
    evaluator.evaluate_all()
    
    # Cleanup distributed
    if world_size > 1:
        torch.distributed.destroy_process_group()
    
    if rank == 0:
        logging.info("✓ All evaluations complete!")


if __name__ == "__main__":
    main()

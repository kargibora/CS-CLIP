import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, Dict, Any
import logging


class TransformersCLIPWrapper(nn.Module):
    """
    Wrapper for HuggingFace transformers CLIP models.
    Provides interface compatible with OpenAI CLIP.
    """
    def __init__(self, model, processor, model_name: str = None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.model_name = model_name
        self.model_type = "transformers"
        
        # Create attributes to match OpenAI CLIP interface
        self.visual = model.vision_model
        self.logit_scale = model.logit_scale
        self.dtype = next(model.parameters()).dtype
        
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images"""
        # If image is already preprocessed tensor
        outputs = self.model.get_image_features(pixel_values=image)
        return outputs
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text - text should be tokenized input_ids"""
        outputs = self.model.get_text_features(input_ids=text)
        return outputs
    
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        return image_features, text_features
    
    def get_temperature(self) -> torch.Tensor:
        """Get temperature parameter"""
        return 1.0 / self.logit_scale.exp()
    
    def train(self, mode: bool = True):
        self.model.train(mode)
        return super().train(mode)
    
    def eval(self):
        self.model.eval()
        return super().eval()
    
    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def parameters(self):
        return self.model.parameters()
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)


class UniversalCLIPWrapper(nn.Module):
    """
    Universal wrapper for both OpenAI CLIP and OpenCLIP models.
    Provides consistent interface and methods for both implementations.
    """
    def __init__(self, model, preprocess, model_type: str = "openai", model_name: str = None):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.model_type = model_type  # "openai" or "openclip"
        self.model_name = model_name
        
        # Store original methods for compatibility
        self._setup_attributes()
        
    def _setup_attributes(self):
        """Setup attributes to match OpenAI CLIP interface"""
        if self.model_type == "openai":
            # Already has the right interface
            self.visual = self.model.visual
            self.transformer = self.model.transformer
            self.token_embedding = self.model.token_embedding
            self.positional_embedding = self.model.positional_embedding
            self.ln_final = self.model.ln_final
            self.text_projection = self.model.text_projection
            self.logit_scale = self.model.logit_scale
            self.dtype = self.model.dtype
            
        elif self.model_type == "openclip":
            # Map OpenCLIP to OpenAI CLIP interface
            # Handle different model architectures (standard CLIP vs SigLIP/CustomTextCLIP)
            self.visual = self.model.visual
            self.logit_scale = self.model.logit_scale
            self.dtype = self.model.dtype if hasattr(self.model, 'dtype') else torch.float32
            
            # Standard CLIP architecture has 'transformer', SigLIP has 'text'
            if hasattr(self.model, 'transformer'):
                self.transformer = self.model.transformer
                self.token_embedding = self.model.token_embedding
                self.positional_embedding = self.model.positional_embedding
                self.ln_final = self.model.ln_final
                self.text_projection = self.model.text_projection
            elif hasattr(self.model, 'text'):
                # SigLIP/CustomTextCLIP architecture - use text encoder directly
                self.text_encoder = self.model.text
                # Set other attributes to None or equivalent if available
                self.transformer = getattr(self.model.text, 'transformer', None)
                self.token_embedding = getattr(self.model.text, 'token_embedding', None)
                self.positional_embedding = getattr(self.model.text, 'positional_embedding', None)
                self.ln_final = getattr(self.model.text, 'ln_final', None)
                self.text_projection = getattr(self.model.text, 'text_projection', 
                                               getattr(self.model, 'text_projection', None))
            else:
                # Fallback: set to None to avoid AttributeError
                logging.warning(f"Unknown OpenCLIP architecture for {self.model_name}, some attributes may be None")
                self.transformer = None
                self.token_embedding = None
                self.positional_embedding = None
                self.ln_final = None
                self.text_projection = getattr(self.model, 'text_projection', None)
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images - compatible with both CLIP versions"""
        return self.model.encode_image(image)
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text - compatible with both CLIP versions"""
        return self.model.encode_text(text)
    
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - compatible with both CLIP versions"""
        if hasattr(self.model, 'forward') and callable(self.model.forward):
            return self.model(image, text)
        else:
            # Fallback: encode separately
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)
            return image_features, text_features
    
    def get_temperature(self) -> torch.Tensor:
        """Get temperature parameter"""
        if hasattr(self.logit_scale, 'exp'):
            return 1.0 / self.logit_scale.exp()
        else:
            return torch.tensor(0.07)  # Default CLIP temperature
    
    def train(self, mode: bool = True):
        """Set training mode"""
        self.model.train(mode)
        return super().train(mode)
    
    def eval(self):
        """Set evaluation mode"""
        self.model.eval()
        return super().eval()
    
    def to(self, *args, **kwargs):
        """Move model to device/dtype"""
        self.model = self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def parameters(self):
        """Return model parameters"""
        return self.model.parameters()
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        """Return named parameters"""
        return self.model.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    
    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        """Return named modules"""
        return self.model.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)
    
    def state_dict(self, *args, **kwargs):
        """Return state dict"""
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict"""
        return self.model.load_state_dict(state_dict, strict=strict)


def _load_transformers_clip(model_name: str, device: torch.device) -> Tuple[TransformersCLIPWrapper, Any]:
    """
    Load a CLIP model using HuggingFace transformers library.
    
    Args:
        model_name: HuggingFace model name (e.g., 'openai/clip-vit-base-patch32', 'Mayfull/READ-CLIP')
        device: Target device
        
    Returns:
        (wrapped_model, preprocess_function)
    """
    try:
        from transformers import CLIPModel, CLIPProcessor, AutoConfig
        import json
        from huggingface_hub import hf_hub_download, list_repo_files
        
        logging.info(f"Loading HuggingFace transformers CLIP model: {model_name}")
        
        model = CLIPModel.from_pretrained(model_name)
        
        # Try to load processor from the model repo, fall back to base model if not available
        try:
            # Check if preprocessor_config.json exists in the repo
            repo_files = list_repo_files(model_name)
            if 'preprocessor_config.json' in repo_files:
                processor = CLIPProcessor.from_pretrained(model_name)
            else:
                raise FileNotFoundError("No preprocessor_config.json")
        except Exception as e:
            logging.warning(f"Processor not found in {model_name}: {e}")
            # Try to get base model from config
            try:
                config_path = hf_hub_download(model_name, "config.json")
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Check for _name_or_path which often contains the base model
                base_model = config.get('_name_or_path')
                if base_model is None:
                    raise ValueError(f"No '_name_or_path' found in config.json for {model_name}. Cannot determine base model for processor.")
                logging.info(f"Using processor from base model: {base_model}")
                processor = CLIPProcessor.from_pretrained(base_model)
            except Exception as e2:
                # Do NOT fallback - raise error instead
                raise RuntimeError(
                    f"Failed to load processor for {model_name}. "
                    f"No preprocessor_config.json found and could not determine base model from config.json. "
                    f"Error: {e2}"
                ) from e2
        
        model = model.to(device)
        
        # Create a preprocess function compatible with the rest of the codebase
        def preprocess(image):
            """Preprocess image using CLIP processor"""
            from PIL import Image
            if isinstance(image, torch.Tensor):
                # Convert tensor to PIL Image
                if image.dim() == 4:
                    image = image[0]  # Remove batch dim
                image = image.permute(1, 2, 0).cpu().numpy()
                image = (image * 255).astype('uint8')
                image = Image.fromarray(image)
            
            inputs = processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
        
        wrapped_model = TransformersCLIPWrapper(
            model=model,
            processor=processor,
            model_name=model_name
        )
        
        logging.info(f"Successfully loaded HuggingFace transformers model: {model_name}")
        return wrapped_model, preprocess
        
    except ImportError:
        logging.error("transformers library not installed. Install with: pip install transformers")
        raise
    except Exception as e:
        logging.error(f"Failed to load transformers CLIP model {model_name}: {e}")
        raise


def load_clip_model(model_name: str, device: torch.device, download_root: str = None, 
                   force_openclip: bool = False) -> Tuple[UniversalCLIPWrapper, Any]:
    """
    Universal CLIP model loader that supports OpenAI CLIP, OpenCLIP, and HuggingFace Hub models.
    
    Args:
        model_name: Model name. Supports:
            - OpenAI CLIP: 'ViT-B/32', 'ViT-L/14', etc.
            - OpenCLIP: 'ViT-L-14', 'convnext_large_d_320', etc.
            - HuggingFace Hub: 'hf-hub:username/model-name' or 'username/model-name'
        device: Target device
        download_root: Download directory
        force_openclip: Force using OpenCLIP even for OpenAI model names
    
    Returns:
        (wrapped_model, preprocess_function)
    
    Examples:
        # OpenAI CLIP
        model, preprocess = load_clip_model('ViT-B/32', device)
        
        # OpenCLIP with specific pretrained weights
        model, preprocess = load_clip_model('ViT-L-14@openai', device)
        
        # HuggingFace Hub model
        model, preprocess = load_clip_model('hf-hub:nmndeep/CLIC-ViT-B-32-224-CogVLM', device)
        model, preprocess = load_clip_model('nmndeep/CLIC-ViT-B-32-224-CogVLM', device)
    """
    # Check if this is a HuggingFace Hub model
    is_hf_hub = 'hf-hub:' in model_name or '/' in model_name
    
    # If it looks like a HuggingFace model but doesn't have the prefix, add it
    if is_hf_hub and not model_name.startswith('hf-hub:'):
        # Check if it's actually a HF model (has username/model format)
        if '/' in model_name and not any(x in model_name for x in ['ViT-', '@']):
            model_name = f'hf-hub:{model_name}'
            logging.info(f"Detected HuggingFace model, using: {model_name}")
    
    # OpenCLIP model names that don't exist in OpenAI CLIP
    openclip_exclusive = [
        'convnext', 'coca', 'EVA', 'ViT-H-14', 'ViT-g-14', 'ViT-bigG-14',
        'ViT-L-14-336', 'ViT-B-16-plus', 'roberta', 'xlm-roberta', 'hf-hub:'
    ]
    
    # Check if this is definitely an OpenCLIP model
    is_openclip_model = force_openclip or any(name in model_name for name in openclip_exclusive)
    
    if is_openclip_model:
        try:
            import open_clip
            logging.info(f"Loading OpenCLIP model: {model_name}")
            
            # Handle HuggingFace Hub models
            if model_name.startswith('hf-hub:'):
                # HuggingFace Hub model (e.g., "hf-hub:username/model-name")
                logging.info(f"Loading from HuggingFace Hub: {model_name}")
                try:
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        model_name,
                        device=device,
                        cache_dir=download_root
                    )
                    
                    wrapped_model = UniversalCLIPWrapper(
                        model=model, 
                        preprocess=preprocess, 
                        model_type="openclip",
                        model_name=model_name
                    )
                    
                    logging.info(f"Successfully loaded HuggingFace model via OpenCLIP: {model_name}")
                    return wrapped_model, preprocess
                    
                except Exception as e:
                    # OpenCLIP failed, try transformers library
                    logging.warning(f"OpenCLIP failed to load HF model: {e}")
                    logging.info("Trying HuggingFace transformers library...")
                    return _load_transformers_clip(model_name.replace('hf-hub:', ''), device)
            
            # Handle different OpenCLIP model specifications
            elif '@' in model_name:
                # Format: model@pretrained (e.g., "ViT-L-14@openai")
                model_arch, pretrained = model_name.split('@', 1)
            else:
                # Try common pretraining datasets (updated order based on availability)
                model_arch = model_name
                
                # Special case for SigLIP models - they only support 'webli'
                if 'SigLIP' in model_arch or 'siglip' in model_arch.lower():
                    pretrained = 'webli'
                    logging.info(f"SigLIP model detected, using 'webli' pretrained weights")
                else:
                    pretrained_options = ['laion400m_e32', 'openai', 'laion2b_s34b_b79k', 'datacomp_xl_s13b_b90k']
                    
                    pretrained = None
                    for pt in pretrained_options:
                        try:
                            # Test if this pretrained version exists
                            open_clip.create_model_and_transforms(model_arch, pretrained=pt)
                            pretrained = pt
                            break
                        except Exception:
                            continue
                    
                    if pretrained is None:
                        # Fallback to default (use available checkpoints)
                        pretrained = 'openai' if 'ViT' in model_arch else 'laion400m_e32'
            
            # Load standard OpenCLIP model (not from HF Hub)
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_arch, 
                pretrained=pretrained,
                device=device,
                cache_dir=download_root
            )
            
            wrapped_model = UniversalCLIPWrapper(
                model=model, 
                preprocess=preprocess, 
                model_type="openclip",
                model_name=f"{model_arch}@{pretrained}"
            )
            
            logging.info(f"Successfully loaded OpenCLIP model: {model_arch}@{pretrained}")
            return wrapped_model, preprocess
            
        except ImportError:
            logging.error("OpenCLIP not installed. Install with: pip install open_clip_torch")
            raise
        except Exception as e:
            logging.error(f"Failed to load OpenCLIP model {model_name}: {e}")
            raise
    
    else:
        # Use OpenAI CLIP
        try:
            import clip
            logging.info(f"Loading OpenAI CLIP model: {model_name}")
            
            model, preprocess = clip.load(
                model_name,
                device=device,
                download_root=download_root
            )
            
            wrapped_model = UniversalCLIPWrapper(
                model=model,
                preprocess=preprocess,
                model_type="openai",
                model_name=model_name
            )
            
            logging.info(f"Successfully loaded OpenAI CLIP model: {model_name}")
            return wrapped_model, preprocess
            
        except Exception as e:
            logging.error(f"Failed to load OpenAI CLIP model {model_name}: {e}")
            # Fallback to OpenCLIP if OpenAI CLIP fails
            logging.info("Falling back to OpenCLIP...")
            return load_clip_model(model_name, device, download_root, force_openclip=True)

def get_available_models() -> Dict[str, list]:
    """Get available models from both CLIP implementations"""
    models = {"openai": [], "openclip": []}
    
    try:
        import clip
        models["openai"] = clip.available_models()
    except ImportError:
        logging.warning("OpenAI CLIP not available")
    
    try:
        import open_clip
        openclip_models = open_clip.list_models()
        models["openclip"] = openclip_models
    except ImportError:
        logging.warning("OpenCLIP not available")
    
    return models
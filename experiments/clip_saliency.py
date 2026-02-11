#!/usr/bin/env python3
"""
CLIP Attention/Saliency Visualization (Corrected)

Standard CLIP doesn't have cross-attention between image and text.
Instead, we use these methods to visualize "where CLIP looks" for a text:

1. **Patch-Text Similarity Maps** (CLIP Surgery style)
   - Compute similarity between each image patch embedding and the text embedding
   - This shows which regions are semantically similar to the text
   
2. **Gradient-based Attribution (GradCAM)**
   - Compute gradients of image-text similarity w.r.t. visual features
   - Shows which regions most influence the similarity score

3. **Attention Rollout**
   - Properly aggregate self-attention across ViT layers
   - Shows general attention patterns (not text-specific)

Usage:
    python experiments/clip_saliency.py \
        --json_folder swap_pos_json/coco_train/ \
        --image_root . \
        --output_dir saliency_results \
        --num_samples 20
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'figure.dpi': 150,
})


# =============================================================================
# CLIP Feature Extractor with Patch Embeddings
# =============================================================================

class CLIPSaliencyExtractor:
    """
    Extracts patch-level features from CLIP for saliency visualization.
    
    Key insight: CLIP's visual encoder outputs patch embeddings before 
    pooling to [CLS]. We can compute similarity between each patch 
    and the text to create a saliency map.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        checkpoint_type: str = "openclip",
        force_openclip: bool = False,
        pretrained: str = "openai",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model {model_name} on {self.device}")
        
        from utils.checkpoint_loader import load_checkpoint_model
        
        effective_path = model_name if checkpoint_type == "openclip" else checkpoint_path
        
        self.model, self.preprocess, self.tokenize = load_checkpoint_model(
            checkpoint_type=checkpoint_type,
            checkpoint_path=effective_path,
            device=self.device,
            base_model=model_name,
            force_openclip=force_openclip,
            pretrained=pretrained,
        )
        
        self.model.eval()
        self.model_name = model_name
        
        # Determine patch size and grid size from model name
        if "32" in model_name:
            self.patch_size = 32
        elif "16" in model_name:
            self.patch_size = 16
        elif "14" in model_name:
            self.patch_size = 14
        else:
            self.patch_size = 32  # default
        
        # For 224x224 input
        self.grid_size = 224 // self.patch_size
        
        # Hook to capture intermediate features
        self._patch_features = None
        
        # Get the visual projection matrix (projects from visual dim to shared dim)
        self._proj = None
        self._get_projection_matrix()
        
        self._register_hooks()
        
        logger.info(f"Model loaded. Patch size: {self.patch_size}, Grid: {self.grid_size}x{self.grid_size}")
        
        # Test that hooks are working
        self._test_hooks()
    
    def _test_hooks(self):
        """Test that the hooks are capturing patch features."""
        try:
            # Create a dummy image
            dummy_image = Image.new('RGB', (224, 224), color='white')
            patch_embs, cls_emb = self.get_patch_embeddings(dummy_image)
            
            if patch_embs is not None:
                expected_patches = self.grid_size * self.grid_size
                logger.info(f"Hook test PASSED: Got {patch_embs.shape[0]} patches (expected {expected_patches}), dim={patch_embs.shape[1]}")
            else:
                logger.warning("Hook test FAILED: No patch embeddings captured!")
        except Exception as e:
            logger.warning(f"Hook test failed with error: {e}")
    
    def _get_projection_matrix(self):
        """Get the projection matrix that maps visual features to shared embedding space."""
        visual = getattr(self.model, 'visual', None)
        if visual is None:
            visual = getattr(self.model, 'vision_model', None)
        
        if visual is None:
            return
        
        # OpenCLIP / OpenAI CLIP style
        if hasattr(visual, 'proj'):
            proj = visual.proj
            if proj is not None:
                self._proj = proj.detach().clone()
                logger.info(f"Found visual projection: {self._proj.shape}")
                return
        
        # Some models have output_proj
        if hasattr(visual, 'output_proj'):
            proj = visual.output_proj
            if proj is not None:
                if hasattr(proj, 'weight'):
                    self._proj = proj.weight.T.detach().clone()  # Linear layer
                else:
                    self._proj = proj.detach().clone()
                logger.info(f"Found output projection: {self._proj.shape}")
                return
        
        logger.warning("Could not find visual projection matrix - dimensions may not match")
    
    def _register_hooks(self):
        """
        Register hooks to capture patch embeddings BEFORE CLS extraction.
        
        In OpenCLIP/OpenAI CLIP ViT:
        - Transformer outputs (seq_len, batch, dim) with all patch tokens + CLS
        - Then CLS is extracted: x = x[:, 0, :] or x[0, :, :]  
        - Then ln_post is applied to CLS only
        - Then projection is applied
        
        We need to hook on the transformer output (before CLS extraction).
        """
        visual = getattr(self.model, 'visual', None)
        if visual is None:
            visual = getattr(self.model, 'vision_model', None)
        
        if visual is None:
            logger.warning("Could not find visual encoder")
            return
        
        logger.info(f"Visual encoder type: {type(visual).__name__}")
        
        hook_registered = False
        
        # Priority 1: Hook on transformer output (before CLS extraction)
        transformer = getattr(visual, 'transformer', None)
        if transformer is not None:
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self._patch_features = output[0].detach()
                else:
                    self._patch_features = output.detach()
            transformer.register_forward_hook(hook)
            logger.info("Registered hook on transformer output")
            hook_registered = True
        
        # Priority 2: Hook on last resblock
        if not hook_registered:
            if transformer is not None:
                resblocks = getattr(transformer, 'resblocks', None)
                if resblocks is not None and len(resblocks) > 0:
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            self._patch_features = output[0].detach()
                        else:
                            self._patch_features = output.detach()
                    resblocks[-1].register_forward_hook(hook)
                    logger.info(f"Registered hook on last resblock (of {len(resblocks)})")
                    hook_registered = True
        
        # Priority 3: Try blocks (timm-style ViT)
        if not hook_registered:
            blocks = getattr(visual, 'blocks', None)
            if blocks is not None and len(blocks) > 0:
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        self._patch_features = output[0].detach()
                    else:
                        self._patch_features = output.detach()
                blocks[-1].register_forward_hook(hook)
                logger.info(f"Registered hook on last block (of {len(blocks)})")
                hook_registered = True
        
        # Priority 4: Try encoder.layers (HuggingFace style)
        if not hook_registered:
            encoder = getattr(visual, 'encoder', None)
            if encoder is not None:
                layers = getattr(encoder, 'layers', None)
                if layers is not None and len(layers) > 0:
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            self._patch_features = output[0].detach()
                        else:
                            self._patch_features = output.detach()
                    layers[-1].register_forward_hook(hook)
                    logger.info("Registered hook on encoder last layer")
                    hook_registered = True
        
        # NOTE: We do NOT hook on ln_post because it's applied AFTER CLS extraction
        # and only contains the CLS token, not all patches
        
        if not hook_registered:
            logger.warning("Could not register hook on any layer! Patch embeddings will not be available.")
    
    @torch.no_grad()
    def get_patch_embeddings(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get patch embeddings and CLS embedding for an image.
        
        Returns:
            patch_embs: (num_patches, embed_dim) - embeddings for each image patch (projected to text space)
            cls_emb: (embed_dim,) - final image embedding
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Reset patch features before forward pass
        self._patch_features = None
        
        # Forward pass (hooks will capture patch features)
        image_features = self.model.encode_image(image_input)
        cls_emb = F.normalize(image_features, dim=-1).cpu().numpy()[0]
        
        if self._patch_features is not None:
            # patch_features shape: (1, num_patches+1, dim) or (num_patches+1, 1, dim) or (num_patches+1, dim)
            feats = self._patch_features
            logger.debug(f"Captured patch features shape: {feats.shape}")
            
            if len(feats.shape) == 3:
                if feats.shape[0] == 1:
                    feats = feats[0]  # (num_patches+1, dim)
                elif feats.shape[1] == 1:
                    feats = feats[:, 0, :]  # (num_patches+1, dim)
            
            # Skip CLS token (first token)
            if len(feats.shape) == 2 and feats.shape[0] > 1:
                patch_feats = feats[1:]  # (num_patches, visual_dim)
            else:
                logger.warning(f"Unexpected feature shape after processing: {feats.shape}")
                return None, cls_emb
            
            # Apply projection to match text embedding dimension
            if self._proj is not None:
                # proj shape: (visual_dim, embed_dim) - e.g., (768, 512)
                patch_feats = patch_feats @ self._proj.to(self.device)  # (num_patches, embed_dim)
            
            patch_embs = patch_feats.cpu().numpy()
            
            # Normalize
            patch_embs = patch_embs / (np.linalg.norm(patch_embs, axis=1, keepdims=True) + 1e-8)
            
            return patch_embs, cls_emb
        else:
            logger.warning("No patch features captured! Hook may not be registered correctly.")
        
        return None, cls_emb
    
    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to normalized embedding."""
        try:
            text_input = self.tokenize([text]).to(self.device)
        except (RuntimeError, ValueError, TypeError):
            text_input = self.tokenize([text], truncate=True).to(self.device)
        
        features = self.model.encode_text(text_input)
        features = F.normalize(features, dim=-1)
        return features.cpu().numpy()[0]
    
    def compute_patch_text_similarity(
        self,
        image: Image.Image,
        text: str,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute similarity between each image patch and the text.
        
        This is the main saliency method - shows which image regions
        are semantically similar to the text description.
        
        Returns:
            saliency_map: (grid_size, grid_size) - similarity for each patch
            overall_sim: float - overall image-text similarity
        """
        patch_embs, cls_emb = self.get_patch_embeddings(image)
        text_emb = self.encode_text(text)
        
        overall_sim = float(np.dot(cls_emb, text_emb))
        
        if patch_embs is None:
            return None, overall_sim
        
        # Compute similarity for each patch
        patch_sims = np.dot(patch_embs, text_emb)  # (num_patches,)
        
        # Reshape to grid
        n_patches = patch_sims.shape[0]
        grid_size = int(np.sqrt(n_patches))
        
        if grid_size * grid_size != n_patches:
            logger.warning(f"Non-square grid: {n_patches} patches")
            return None, overall_sim
        
        saliency_map = patch_sims.reshape(grid_size, grid_size)
        
        return saliency_map, overall_sim
    
    def compute_gradient_saliency(
        self,
        image: Image.Image,
        text: str,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Compute gradient-based saliency (GradCAM-style).
        
        Computes gradients of image-text similarity w.r.t. image pixels,
        then aggregates to show important regions.
        """
        self.model.zero_grad()
        
        # Prepare inputs
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        image_input.requires_grad = True
        
        try:
            text_input = self.tokenize([text]).to(self.device)
        except Exception:
            text_input = self.tokenize([text], truncate=True).to(self.device)
        
        # Forward pass
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_input)
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        similarity = (image_features * text_features).sum()
        
        # Backward pass
        similarity.backward()
        
        overall_sim = similarity.item()
        
        if image_input.grad is not None:
            # Get gradients
            grads = image_input.grad.detach().cpu().numpy()[0]  # (C, H, W)
            
            # Take absolute value and average across channels
            saliency = np.abs(grads).mean(axis=0)  # (H, W)
            
            # Smooth with gaussian
            try:
                from scipy.ndimage import gaussian_filter
                saliency = gaussian_filter(saliency, sigma=3)
            except ImportError:
                pass
            
            # Normalize to [0, 1]
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            return saliency, overall_sim
        
        return None, overall_sim

    # =========================================================================
    # NEW: Token-level Text Embeddings
    # =========================================================================
    
    @torch.no_grad()
    def get_token_embeddings(self, text: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Get per-token embeddings from the text transformer.
        
        Returns:
            token_embs: (num_tokens, embed_dim) - embedding for each token
            tokens: List[str] - decoded token strings  
            pooled_emb: (embed_dim,) - final pooled text embedding
        """
        try:
            text_input = self.tokenize([text]).to(self.device)
        except (RuntimeError, ValueError, TypeError):
            text_input = self.tokenize([text], truncate=True).to(self.device)
        
        # Get the text encoder
        text_encoder = getattr(self.model, 'transformer', None)
        if text_encoder is None:
            text_encoder = getattr(self.model, 'text', None)
        if text_encoder is None:
            text_encoder = getattr(self.model, 'text_model', None)
        
        # Hook to capture token embeddings before final projection
        token_features = None
        
        def hook_fn(module, input, output):
            nonlocal token_features
            if isinstance(output, tuple):
                token_features = output[0].detach()
            else:
                token_features = output.detach()
        
        # Try different locations for the hook
        hook_handle = None
        
        # OpenCLIP style: look for ln_final on the text transformer
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'resblocks'):
            # Get the last resblock
            last_block = self.model.transformer.resblocks[-1]
            hook_handle = last_block.register_forward_hook(hook_fn)
        elif hasattr(self.model, 'text') and hasattr(self.model.text, 'transformer'):
            last_block = self.model.text.transformer.resblocks[-1]
            hook_handle = last_block.register_forward_hook(hook_fn)
        
        # Forward pass
        pooled_features = self.model.encode_text(text_input)
        pooled_emb = F.normalize(pooled_features, dim=-1).cpu().numpy()[0]
        
        if hook_handle is not None:
            hook_handle.remove()
        
        if token_features is None:
            logger.warning("Could not capture token embeddings")
            return None, [], pooled_emb
        
        # token_features shape: (seq_len, batch, dim) or (batch, seq_len, dim)
        if len(token_features.shape) == 3:
            if token_features.shape[1] == 1:
                token_features = token_features[:, 0, :]  # (seq_len, dim)
            elif token_features.shape[0] == 1:
                token_features = token_features[0]  # (seq_len, dim)
        
        # Get text projection matrix
        text_proj = getattr(self.model, 'text_projection', None)
        if text_proj is None and hasattr(self.model, 'text'):
            text_proj = getattr(self.model.text, 'proj', None)
        
        # Apply projection if available
        if text_proj is not None:
            token_features = token_features @ text_proj.to(self.device)
        
        # Normalize
        token_embs = F.normalize(token_features, dim=-1).cpu().numpy()
        
        # Decode tokens
        tokens = self._decode_tokens(text_input, text)
        
        return token_embs, tokens, pooled_emb
    
    def _decode_tokens(self, token_ids: torch.Tensor, original_text: str) -> List[str]:
        """Decode token IDs to strings (approximate)."""
        # Simple word-based fallback
        words = original_text.replace(',', ' ,').replace('.', ' .').split()
        
        # Pad with special tokens
        tokens = ['[SOS]'] + words + ['[EOS]']
        
        # Pad to match token_ids length
        seq_len = token_ids.shape[1] if len(token_ids.shape) > 1 else len(token_ids)
        while len(tokens) < seq_len:
            tokens.append('[PAD]')
        
        return tokens[:seq_len]
    
    @torch.no_grad()
    def compute_patch_token_similarity(
        self,
        image: Image.Image,
        text: str,
    ) -> Tuple[Optional[np.ndarray], List[str], float]:
        """
        Compute patch × token similarity matrix.
        
        S_{i,j} = V_i · T_j gives similarity between patch i and token j.
        
        Returns:
            sim_matrix: (num_patches, num_tokens) - patch-token similarities
            tokens: List[str] - token strings
            overall_sim: float - overall image-text similarity
        """
        patch_embs, cls_emb = self.get_patch_embeddings(image)
        token_embs, tokens, text_emb = self.get_token_embeddings(text)
        
        overall_sim = float(np.dot(cls_emb, text_emb))
        
        if patch_embs is None or token_embs is None:
            return None, tokens, overall_sim
        
        # Compute similarity matrix: (num_patches, num_tokens)
        sim_matrix = np.dot(patch_embs, token_embs.T)
        
        return sim_matrix, tokens, overall_sim
    
    def get_token_heatmap(
        self,
        image: Image.Image,
        text: str,
        token_idx: int = None,
        token_word: str = None,
    ) -> Tuple[Optional[np.ndarray], str, float]:
        """
        Get heatmap for a specific token.
        
        Args:
            token_idx: Index of token to visualize
            token_word: Word to find and visualize (uses first match)
            
        Returns:
            heatmap: (grid_size, grid_size) - similarity map for the token
            token_str: The token being visualized
            overall_sim: Overall image-text similarity
        """
        sim_matrix, tokens, overall_sim = self.compute_patch_token_similarity(image, text)
        
        if sim_matrix is None:
            return None, "", overall_sim
        
        # Find token index
        if token_word is not None:
            token_idx = None
            for i, tok in enumerate(tokens):
                if token_word.lower() in tok.lower():
                    token_idx = i
                    break
            if token_idx is None:
                logger.warning(f"Token '{token_word}' not found in: {tokens}")
                return None, "", overall_sim
        
        if token_idx is None or token_idx >= sim_matrix.shape[1]:
            return None, "", overall_sim
        
        # Get similarities for this token
        token_sims = sim_matrix[:, token_idx]
        
        # Reshape to grid
        n_patches = token_sims.shape[0]
        grid_size = int(np.sqrt(n_patches))
        
        if grid_size * grid_size != n_patches:
            return None, tokens[token_idx], overall_sim
        
        heatmap = token_sims.reshape(grid_size, grid_size)
        
        return heatmap, tokens[token_idx], overall_sim

    # =========================================================================
    # NEW: ViT-Native Grad-CAM (gradients w.r.t. patch tokens)
    # =========================================================================
    
    def compute_vit_gradcam(
        self,
        image: Image.Image,
        text: str,
        layer_idx: int = -2,  # Penultimate layer by default
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Compute ViT-native Grad-CAM using gradients w.r.t. patch tokens.
        
        This is more principled than pixel-level gradients for ViT models.
        
        Recipe:
        1. Capture activations A = patch tokens at layer l (shape N×D)
        2. Compute gradients ∂sim/∂A
        3. Compute weights per channel: w = mean_over_tokens(∂sim/∂A) (shape D)
        4. Score per patch: m_i = ReLU(A_i · w)
        
        Returns:
            saliency_map: (grid_size, grid_size) - Grad-CAM scores
            overall_sim: float - image-text similarity
        """
        self.model.zero_grad()
        
        # Storage for activations and gradients
        activations = None
        gradients = None
        
        def forward_hook(module, input, output):
            nonlocal activations
            if isinstance(output, tuple):
                activations = output[0]
            else:
                activations = output
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            if isinstance(grad_output, tuple):
                gradients = grad_output[0]
            else:
                gradients = grad_output
        
        # Find the target layer
        visual = getattr(self.model, 'visual', None)
        if visual is None:
            visual = getattr(self.model, 'vision_model', None)
        
        if visual is None:
            logger.warning("Could not find visual encoder for Grad-CAM")
            return None, 0.0
        
        # Get transformer blocks
        transformer = getattr(visual, 'transformer', None)
        if transformer is None:
            transformer = getattr(visual, 'encoder', None)
        
        if transformer is None or not hasattr(transformer, 'resblocks'):
            # Try alternative structure
            if hasattr(visual, 'blocks'):
                blocks = visual.blocks
            else:
                logger.warning("Could not find transformer blocks")
                return None, 0.0
        else:
            blocks = transformer.resblocks
        
        # Register hooks on target layer
        target_layer = blocks[layer_idx]
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)
        
        try:
            # Prepare inputs
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            try:
                text_input = self.tokenize([text]).to(self.device)
            except Exception:
                text_input = self.tokenize([text], truncate=True).to(self.device)
            
            # Forward pass
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
            
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity
            similarity = (image_features * text_features).sum()
            overall_sim = similarity.item()
            
            # Backward pass
            similarity.backward()
            
        finally:
            fwd_handle.remove()
            bwd_handle.remove()
        
        if activations is None or gradients is None:
            logger.warning("Failed to capture activations/gradients")
            return None, overall_sim
        
        # activations shape: (seq_len, batch, dim) or (batch, seq_len, dim)
        acts = activations.detach()
        grads = gradients.detach()
        
        # Normalize shape to (seq_len, dim)
        if len(acts.shape) == 3:
            if acts.shape[1] == 1:
                acts = acts[:, 0, :]
                grads = grads[:, 0, :]
            elif acts.shape[0] == 1:
                acts = acts[0]
                grads = grads[0]
        
        # Skip CLS token
        patch_acts = acts[1:]  # (num_patches, dim)
        patch_grads = grads[1:]  # (num_patches, dim)
        
        # Compute channel weights: w = mean_over_patches(grads)
        weights = patch_grads.mean(dim=0)  # (dim,)
        
        # Compute weighted activation: m_i = ReLU(A_i · w)
        cam = torch.relu((patch_acts * weights).sum(dim=-1))  # (num_patches,)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Reshape to grid
        n_patches = cam.shape[0]
        grid_size = int(np.sqrt(n_patches))
        
        if grid_size * grid_size != n_patches:
            logger.warning(f"Non-square grid for Grad-CAM: {n_patches} patches")
            return None, overall_sim
        
        saliency_map = cam.reshape(grid_size, grid_size)
        
        return saliency_map, overall_sim

    # =========================================================================
    # NEW: Relation Analysis Methods
    # =========================================================================
    
    def compute_relation_contrast_maps(
        self,
        image: Image.Image,
        text_a_on_b: str,  # e.g., "cat on table"
        text_b_on_a: str,  # e.g., "table on cat"
        text_a_and_b: str = None,  # e.g., "cat and table"
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Compute contrast maps for relation analysis.
        
        Returns maps for:
        - base: patch similarity for "A on B"
        - swapped: patch similarity for "B on A"  
        - swap_contrast: map(A on B) - map(B on A)
        - cooccurrence: patch similarity for "A and B" (if provided)
        - relation_contrast: map(A on B) - map(A and B) (if cooccurrence provided)
        """
        results = {}
        
        # Base relation
        smap_base, sim_base = self.compute_patch_text_similarity(image, text_a_on_b)
        results['base'] = (smap_base, sim_base)
        results['base_text'] = text_a_on_b
        
        # Swapped relation
        smap_swap, sim_swap = self.compute_patch_text_similarity(image, text_b_on_a)
        results['swapped'] = (smap_swap, sim_swap)
        results['swapped_text'] = text_b_on_a
        
        # Swap contrast
        if smap_base is not None and smap_swap is not None:
            results['swap_contrast'] = (smap_base - smap_swap, sim_base - sim_swap)
        
        # Co-occurrence baseline
        if text_a_and_b:
            smap_and, sim_and = self.compute_patch_text_similarity(image, text_a_and_b)
            results['cooccurrence'] = (smap_and, sim_and)
            results['cooccurrence_text'] = text_a_and_b
            
            # Relation-specific signal
            if smap_base is not None and smap_and is not None:
                results['relation_contrast'] = (smap_base - smap_and, sim_base - sim_and)
        
        return results
    
    def compute_partial_truth_analysis(
        self,
        image: Image.Image,
        full_caption: str,  # e.g., "a cat in the park"
        true_part: str,     # e.g., "cat"
        false_part: str,    # e.g., "park" (if image has no park)
    ) -> Dict[str, any]:
        """
        Analyze partial truth / distractor vulnerability.
        
        Checks whether model improperly attributes high similarity to 
        the full caption when only part of it is true in the image.
        """
        results = {}
        
        # Full caption
        smap_full, sim_full = self.compute_patch_text_similarity(image, full_caption)
        results['full'] = {'map': smap_full, 'sim': sim_full, 'text': full_caption}
        
        # True part
        smap_true, sim_true = self.compute_patch_text_similarity(image, true_part)
        results['true_part'] = {'map': smap_true, 'sim': sim_true, 'text': true_part}
        
        # False part
        smap_false, sim_false = self.compute_patch_text_similarity(image, false_part)
        results['false_part'] = {'map': smap_false, 'sim': sim_false, 'text': false_part}
        
        # Analysis
        results['analysis'] = {
            'sim_drop_full_vs_true': sim_full - sim_true,
            'false_part_sim': sim_false,
            'bag_of_words_score': (sim_true + sim_false) / 2,  # Rough BoW estimate
        }
        
        # Token-level analysis for false part
        if smap_false is not None:
            # Check if false part has coherent localization
            false_max = smap_false.max()
            false_std = smap_false.std()
            
            results['analysis']['false_part_localized'] = false_std > 0.1  # Rough threshold
            results['analysis']['false_part_max_activation'] = float(false_max)
        
        return results


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class Sample:
    """Sample for visualization."""
    image_path: str
    sample_id: str
    original_caption: str
    components: List[str]
    component_negatives: Dict[str, List[Dict[str, str]]]
    relations: List[Dict[str, str]]
    binding_negatives: List[Dict[str, str]]


def load_samples(
    json_folder: str,
    image_root: str,
    max_samples: int = 20,
    seed: int = 42,
) -> List[Sample]:
    """Load samples from JSON files."""
    random.seed(seed)
    np.random.seed(seed)
    
    json_folder = Path(json_folder)
    samples = []
    
    json_files = sorted(json_folder.glob("*.json"))
    if not json_files:
        if json_folder.is_file():
            json_files = [json_folder]
        else:
            raise ValueError(f"No JSON files found in {json_folder}")
    
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
    
    random.shuffle(all_data)
    
    for item in all_data:
        if len(samples) >= max_samples:
            break
        
        image_path = item.get('image_path', '')
        if image_path and image_root and not image_path.startswith('/'):
            image_path = os.path.join(image_root, image_path)
        
        if not image_path or not os.path.exists(image_path):
            continue
        
        components = item.get('positive_components', [])
        if not components:
            continue
        
        samples.append(Sample(
            image_path=image_path,
            sample_id=item.get('sample_id', f"sample_{len(samples)}"),
            original_caption=item.get('caption', item.get('original_caption', '')),
            components=components,
            component_negatives=item.get('negative_components', {}),
            relations=item.get('relations', []),
            binding_negatives=item.get('binding_negatives', []),
        ))
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


# =============================================================================
# Visualization Functions
# =============================================================================

def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap: str = 'jet',
) -> np.ndarray:
    """Overlay heatmap on image."""
    # Resize heatmap to image size
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    ) / 255.0
    
    # Apply colormap
    cm = plt.get_cmap(cmap)
    heatmap_colored = cm(heatmap_resized)[:, :, :3]
    
    # Blend with image
    image_array = np.array(image) / 255.0
    blended = (1 - alpha) * image_array + alpha * heatmap_colored
    
    return (blended * 255).astype(np.uint8)


def normalize_saliency(smap: np.ndarray) -> np.ndarray:
    """Normalize saliency map to [0, 1]."""
    if smap is None:
        return None
    if not isinstance(smap, np.ndarray):
        return None
    if smap.size == 0:
        return None
    smap = smap.copy()
    smin, smax = smap.min(), smap.max()
    if smax - smin < 1e-10:
        return np.zeros_like(smap)
    smap = (smap - smin) / (smax - smin + 1e-8)
    return smap


def plot_token_heatmaps(
    extractor: CLIPSaliencyExtractor,
    image: Image.Image,
    text: str,
    output_path: str,
    tokens_to_show: List[str] = None,
    max_tokens: int = 8,
):
    """
    Plot per-token heatmaps showing which patches match which words.
    
    Args:
        tokens_to_show: Specific tokens/words to visualize. If None, shows all.
        max_tokens: Maximum number of tokens to display.
    """
    sim_matrix, tokens, overall_sim = extractor.compute_patch_token_similarity(image, text)
    
    if sim_matrix is None:
        logger.warning("Could not compute token similarity matrix")
        return
    
    # Filter tokens
    if tokens_to_show:
        # Find indices of requested tokens
        indices = []
        selected_tokens = []
        for word in tokens_to_show:
            for i, tok in enumerate(tokens):
                if word.lower() in tok.lower() and i not in indices:
                    indices.append(i)
                    selected_tokens.append(tok)
                    break
    else:
        # Skip special tokens
        indices = []
        selected_tokens = []
        for i, tok in enumerate(tokens):
            if tok not in ['[SOS]', '[EOS]', '[PAD]', '<|startoftext|>', '<|endoftext|>']:
                indices.append(i)
                selected_tokens.append(tok)
    
    # Limit number of tokens
    indices = indices[:max_tokens]
    selected_tokens = selected_tokens[:max_tokens]
    
    if not indices:
        logger.warning("No valid tokens to visualize")
        return
    
    # Create figure
    n_tokens = len(indices)
    n_cols = min(4, n_tokens + 1)
    n_rows = (n_tokens + n_cols) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Original\nSim: {overall_sim:.3f}", fontsize=10)
    axes[0, 0].axis('off')
    
    # Grid size
    n_patches = sim_matrix.shape[0]
    grid_size = int(np.sqrt(n_patches))
    
    # Plot token heatmaps
    for idx, (tok_idx, tok_str) in enumerate(zip(indices, selected_tokens)):
        row = (idx + 1) // n_cols
        col = (idx + 1) % n_cols
        
        if row >= n_rows or col >= n_cols:
            break
        
        # Get heatmap for this token
        token_sims = sim_matrix[:, tok_idx]
        heatmap = token_sims.reshape(grid_size, grid_size)
        heatmap_norm = normalize_saliency(heatmap)
        
        if heatmap_norm is not None:
            overlay = overlay_heatmap(image, heatmap_norm, alpha=0.6)
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f"'{tok_str}'\nmax: {heatmap.max():.3f}", fontsize=9)
        else:
            axes[row, col].imshow(image)
            axes[row, col].set_title(f"'{tok_str}' (no data)", fontsize=9)
        
        axes[row, col].axis('off')
    
    # Hide unused axes
    for row in range(n_rows):
        for col in range(n_cols):
            if row * n_cols + col > n_tokens:
                axes[row, col].axis('off')
                axes[row, col].set_visible(False)
    
    plt.suptitle(f"Token-level Heatmaps: \"{text[:60]}...\"", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved token heatmap to {output_path}")


def plot_relation_contrast(
    extractor: CLIPSaliencyExtractor,
    image: Image.Image,
    subject: str,
    relation: str,
    obj: str,
    output_path: str,
):
    """
    Plot relation contrast analysis.
    
    Shows:
    - "A relation B" map
    - "B relation A" map (swapped)
    - "A and B" map (co-occurrence baseline)
    - Contrast maps
    """
    text_a_on_b = f"{subject} {relation} {obj}"
    text_b_on_a = f"{obj} {relation} {subject}"
    text_a_and_b = f"{subject} and {obj}"
    
    results = extractor.compute_relation_contrast_maps(
        image, text_a_on_b, text_b_on_a, text_a_and_b
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Base maps
    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=10)
    axes[0, 0].axis('off')
    
    # A on B
    smap, sim = results['base']
    if smap is not None:
        overlay = overlay_heatmap(image, normalize_saliency(smap), alpha=0.6)
        axes[0, 1].imshow(overlay)
    else:
        axes[0, 1].imshow(image)
    axes[0, 1].set_title(f"'{text_a_on_b}'\nSim: {sim:.3f}", fontsize=10)
    axes[0, 1].axis('off')
    
    # B on A (swapped)
    smap, sim = results['swapped']
    if smap is not None:
        overlay = overlay_heatmap(image, normalize_saliency(smap), alpha=0.6)
        axes[0, 2].imshow(overlay)
    else:
        axes[0, 2].imshow(image)
    axes[0, 2].set_title(f"'{text_b_on_a}' (swapped)\nSim: {sim:.3f}", fontsize=10)
    axes[0, 2].axis('off')
    
    # Row 2: Contrasts
    # A and B (co-occurrence)
    if 'cooccurrence' in results:
        smap, sim = results['cooccurrence']
        if smap is not None:
            overlay = overlay_heatmap(image, normalize_saliency(smap), alpha=0.6)
            axes[1, 0].imshow(overlay)
        else:
            axes[1, 0].imshow(image)
        axes[1, 0].set_title(f"'{text_a_and_b}'\nSim: {sim:.3f}", fontsize=10)
    else:
        axes[1, 0].axis('off')
        axes[1, 0].set_visible(False)
    axes[1, 0].axis('off')
    
    # Swap contrast
    if 'swap_contrast' in results:
        smap, delta_sim = results['swap_contrast']
        if smap is not None and smap.size > 0:
            # Use diverging colormap (blue = B>A, red = A>B)
            vmax = max(abs(smap.min()), abs(smap.max()), 1e-8)
            smap_resized = np.array(
                Image.fromarray(((smap + vmax) / (2*vmax) * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
            ) / 255.0
            
            axes[1, 1].imshow(image)
            im = axes[1, 1].imshow(smap_resized * 2 - 1, cmap='RdBu_r', alpha=0.6, vmin=-1, vmax=1)
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        axes[1, 1].set_title(f"Swap Contrast (A-B)\nΔSim: {delta_sim:.3f}", fontsize=10)
    else:
        axes[1, 1].set_visible(False)
    axes[1, 1].axis('off')
    
    # Relation contrast
    if 'relation_contrast' in results:
        smap, delta_sim = results['relation_contrast']
        if smap is not None and smap.size > 0:
            vmax = max(abs(smap.min()), abs(smap.max()), 1e-8)
            smap_resized = np.array(
                Image.fromarray(((smap + vmax) / (2*vmax) * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
            ) / 255.0
            
            axes[1, 2].imshow(image)
            im = axes[1, 2].imshow(smap_resized * 2 - 1, cmap='RdBu_r', alpha=0.6, vmin=-1, vmax=1)
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        axes[1, 2].set_title(f"Relation vs Co-occur\nΔSim: {delta_sim:.3f}", fontsize=10)
    else:
        axes[1, 2].set_visible(False)
    axes[1, 2].axis('off')
    
    plt.suptitle(f"Relation Analysis: {subject} {relation} {obj}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved relation contrast to {output_path}")


def plot_method_comparison(
    extractor: CLIPSaliencyExtractor,
    image: Image.Image,
    text: str,
    output_path: str,
):
    """
    Compare different saliency methods side by side.
    
    Shows:
    - Patch-text similarity (fast, direct)
    - ViT Grad-CAM (gradient-based, sharper)
    - Pixel gradient (legacy, noisier)
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis('off')
    
    # Patch-text similarity
    smap_patch, sim = extractor.compute_patch_text_similarity(image, text)
    if smap_patch is not None:
        overlay = overlay_heatmap(image, normalize_saliency(smap_patch), alpha=0.6)
        axes[1].imshow(overlay)
        axes[1].set_title(f"Patch-Text Similarity\nSim: {sim:.3f}", fontsize=10)
    else:
        axes[1].imshow(image)
        axes[1].set_title("Patch-Text (failed)", fontsize=10)
    axes[1].axis('off')
    
    # ViT Grad-CAM
    smap_gradcam, _ = extractor.compute_vit_gradcam(image, text)
    if smap_gradcam is not None:
        overlay = overlay_heatmap(image, normalize_saliency(smap_gradcam), alpha=0.6)
        axes[2].imshow(overlay)
        axes[2].set_title("ViT Grad-CAM\n(patch token gradients)", fontsize=10)
    else:
        axes[2].imshow(image)
        axes[2].set_title("ViT Grad-CAM (failed)", fontsize=10)
    axes[2].axis('off')
    
    # Pixel gradient
    smap_pixel, _ = extractor.compute_gradient_saliency(image, text)
    if smap_pixel is not None:
        overlay = overlay_heatmap(image, normalize_saliency(smap_pixel), alpha=0.6)
        axes[3].imshow(overlay)
        axes[3].set_title("Pixel Gradients\n(legacy, noisier)", fontsize=10)
    else:
        axes[3].imshow(image)
        axes[3].set_title("Pixel Gradients (failed)", fontsize=10)
    axes[3].axis('off')
    
    plt.suptitle(f"Saliency Method Comparison: \"{text[:50]}...\"", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved method comparison to {output_path}")


def plot_patch_similarity_comparison(
    extractor: CLIPSaliencyExtractor,
    sample: Sample,
    output_path: str,
    method: str = "patch",  # "patch" or "gradient"
):
    """
    Compare patch-text similarity for positive vs negative texts.
    
    Shows how the model's attention differs between correct and incorrect descriptions.
    """
    try:
        image = Image.open(sample.image_path).convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to load image: {e}")
        return
    
    # Collect texts to visualize
    texts = []
    labels = []
    is_positive = []
    
    # Add components and their negatives
    for comp in sample.components[:2]:
        texts.append(comp)
        labels.append(f"✓ {comp[:25]}...")
        is_positive.append(True)
        
        negs = sample.component_negatives.get(comp, [])
        for neg in negs[:1]:
            neg_text = neg.get('negative', '') if isinstance(neg, dict) else neg
            change_type = neg.get('change_type', '') if isinstance(neg, dict) else ''
            if neg_text:
                texts.append(neg_text)
                labels.append(f"✗ {neg_text[:20]}... ({change_type})")
                is_positive.append(False)
    
    # Add relations
    for rel in sample.relations[:1]:
        subject = rel.get('subject', '')
        relation_type = rel.get('relation_type', '')
        obj = rel.get('object', '')
        
        if subject and relation_type and obj:
            rel_text = f"{subject} {relation_type} {obj}"
            texts.append(rel_text)
            labels.append(f"✓ {rel_text[:25]}...")
            is_positive.append(True)
            
            swapped = f"{obj} {relation_type} {subject}"
            texts.append(swapped)
            labels.append(f"✗ Swapped: {swapped[:20]}...")
            is_positive.append(False)
    
    if len(texts) < 2:
        return
    
    # Compute saliency maps
    saliency_maps = []
    similarities = []
    
    for text in texts:
        if method == "patch":
            smap, sim = extractor.compute_patch_text_similarity(image, text)
        else:
            smap, sim = extractor.compute_gradient_saliency(image, text)
        
        saliency_maps.append(smap)
        similarities.append(sim)
    
    # Create figure
    n_texts = len(texts)
    n_cols = min(4, n_texts + 1)
    n_rows = (n_texts + n_cols) // n_cols
    
    fig = plt.figure(figsize=(4.5 * n_cols, 4 * n_rows + 1))
    gs = GridSpec(n_rows + 1, n_cols, height_ratios=[0.15] + [1] * n_rows, hspace=0.3, wspace=0.2)
    
    # Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.text(0.5, 0.5, f"Patch-Text Similarity Maps: {sample.sample_id[:30]}...",
                 ha='center', va='center', fontsize=14, fontweight='bold')
    ax_title.axis('off')
    
    # Original image
    ax_orig = fig.add_subplot(gs[1, 0])
    ax_orig.imshow(image)
    ax_orig.set_title("Original Image", fontsize=11, fontweight='bold')
    ax_orig.axis('off')
    
    # Saliency maps
    for i, (smap, sim, label, pos) in enumerate(zip(saliency_maps, similarities, labels, is_positive)):
        row = (i + 1) // n_cols + 1
        col = (i + 1) % n_cols
        
        ax = fig.add_subplot(gs[row, col])
        
        if smap is not None and smap.size > 0:
            # Normalize saliency map
            smap_norm = normalize_saliency(smap)
            
            if smap_norm is not None:
                # Overlay on image
                overlay = overlay_heatmap(image, smap_norm, alpha=0.6)
                ax.imshow(overlay)
            else:
                ax.imshow(image)
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha='center', fontsize=12)
        else:
            ax.imshow(image)
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha='center', fontsize=12)
        
        # Color title based on positive/negative
        title_color = 'green' if pos else 'red'
        ax.set_title(f"{label}\nsim={sim:.3f}", fontsize=9, color=title_color)
        ax.axis('off')
    
    # Hide unused axes
    for i in range(len(texts) + 1, n_rows * n_cols):
        row = i // n_cols + 1
        col = i % n_cols
        if row <= n_rows:
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved saliency comparison to {output_path}")


def plot_saliency_difference(
    extractor: CLIPSaliencyExtractor,
    sample: Sample,
    output_path: str,
):
    """
    Show the DIFFERENCE in saliency between correct and incorrect texts.
    
    This reveals which regions the model focuses on MORE for the correct text
    vs the incorrect text.
    """
    try:
        image = Image.open(sample.image_path).convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to load image: {e}")
        return
    
    # Get a positive-negative pair
    if not sample.components:
        return
    
    comp = sample.components[0]
    negs = sample.component_negatives.get(comp, [])
    
    if not negs:
        return
    
    neg_text = negs[0].get('negative', '') if isinstance(negs[0], dict) else negs[0]
    
    if not neg_text:
        return
    
    # Compute saliency maps
    smap_pos, sim_pos = extractor.compute_patch_text_similarity(image, comp)
    smap_neg, sim_neg = extractor.compute_patch_text_similarity(image, neg_text)
    
    if smap_pos is None or smap_neg is None:
        return
    if smap_pos.size == 0 or smap_neg.size == 0:
        return
    
    # Normalize both
    smap_pos_norm = normalize_saliency(smap_pos)
    smap_neg_norm = normalize_saliency(smap_neg)
    
    if smap_pos_norm is None or smap_neg_norm is None:
        return
    
    # Compute difference
    diff = smap_pos_norm - smap_neg_norm  # Positive = model focuses more on this for correct text
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Correct text saliency
    overlay_pos = overlay_heatmap(image, smap_pos_norm, alpha=0.6)
    axes[1].imshow(overlay_pos)
    axes[1].set_title(f"✓ Correct: {comp[:30]}...\nsim={sim_pos:.3f}", fontsize=10, color='green')
    axes[1].axis('off')
    
    # Incorrect text saliency
    overlay_neg = overlay_heatmap(image, smap_neg_norm, alpha=0.6)
    axes[2].imshow(overlay_neg)
    axes[2].set_title(f"✗ Incorrect: {neg_text[:25]}...\nsim={sim_neg:.3f}", fontsize=10, color='red')
    axes[2].axis('off')
    
    # Difference map (use diverging colormap)
    # Resize difference to image size
    diff_resized = np.array(
        Image.fromarray(((diff + 1) / 2 * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    ) / 255.0 * 2 - 1  # Back to [-1, 1]
    
    axes[3].imshow(image)
    im = axes[3].imshow(diff_resized, cmap='RdBu_r', alpha=0.6, vmin=-1, vmax=1)
    axes[3].set_title("Difference (Correct - Incorrect)\nGreen=Correct focus, Red=Incorrect", fontsize=10)
    axes[3].axis('off')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label('Δ Similarity', fontsize=9)
    
    plt.suptitle(f"Saliency Difference Analysis: {sample.sample_id[:40]}...", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved saliency difference to {output_path}")


def plot_component_saliency_grid(
    extractor: CLIPSaliencyExtractor,
    sample: Sample,
    output_path: str,
):
    """
    Grid showing saliency for each component in the image.
    
    Good for showing how the model localizes different objects/attributes.
    """
    try:
        image = Image.open(sample.image_path).convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to load image: {e}")
        return
    
    components = sample.components[:6]  # Max 6 components
    if len(components) < 2:
        return
    
    n_comps = len(components)
    n_cols = min(3, n_comps)
    n_rows = (n_comps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(4 * (n_cols + 1), 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Original image (spanning all rows)
    for row in range(n_rows):
        axes[row, 0].imshow(image)
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
    
    # Component saliencies
    for i, comp in enumerate(components):
        row = i // n_cols
        col = (i % n_cols) + 1
        
        smap, sim = extractor.compute_patch_text_similarity(image, comp)
        
        if smap is not None and smap.size > 0:
            smap_norm = normalize_saliency(smap)
            if smap_norm is not None:
                overlay = overlay_heatmap(image, smap_norm, alpha=0.6)
                axes[row, col].imshow(overlay)
            else:
                axes[row, col].imshow(image)
        else:
            axes[row, col].imshow(image)
        
        axes[row, col].set_title(f"{comp[:30]}...\nsim={sim:.3f}", fontsize=9)
        axes[row, col].axis('off')
    
    # Hide unused
    for i in range(n_comps, n_rows * n_cols):
        row = i // n_cols
        col = (i % n_cols) + 1
        axes[row, col].axis('off')
    
    plt.suptitle(f"Component Saliency Maps: {sample.sample_id[:40]}...", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def compute_saliency_statistics(
    extractor: CLIPSaliencyExtractor,
    samples: List[Sample],
    output_path: str,
):
    """Compute statistics about saliency patterns."""
    stats = {
        'component_pos_mean_sim': [],
        'component_neg_mean_sim': [],
        'relation_correct_mean_sim': [],
        'relation_swapped_mean_sim': [],
        'pos_neg_margin': [],
    }
    
    for sample in tqdm(samples, desc="Computing saliency statistics"):
        try:
            image = Image.open(sample.image_path).convert('RGB')
        except Exception:
            continue
        
        # Components
        for comp in sample.components[:3]:
            _, sim_pos = extractor.compute_patch_text_similarity(image, comp)
            stats['component_pos_mean_sim'].append(sim_pos)
            
            negs = sample.component_negatives.get(comp, [])
            for neg in negs[:2]:
                neg_text = neg.get('negative', '') if isinstance(neg, dict) else neg
                if neg_text:
                    _, sim_neg = extractor.compute_patch_text_similarity(image, neg_text)
                    stats['component_neg_mean_sim'].append(sim_neg)
                    stats['pos_neg_margin'].append(sim_pos - sim_neg)
        
        # Relations
        for rel in sample.relations[:2]:
            subject = rel.get('subject', '')
            relation_type = rel.get('relation_type', '')
            obj = rel.get('object', '')
            
            if subject and relation_type and obj:
                rel_text = f"{subject} {relation_type} {obj}"
                _, sim_correct = extractor.compute_patch_text_similarity(image, rel_text)
                stats['relation_correct_mean_sim'].append(sim_correct)
                
                swapped = f"{obj} {relation_type} {subject}"
                _, sim_swapped = extractor.compute_patch_text_similarity(image, swapped)
                stats['relation_swapped_mean_sim'].append(sim_swapped)
    
    # Compute summary
    summary = {
        'component_accuracy': np.mean([m > 0 for m in stats['pos_neg_margin']]) if stats['pos_neg_margin'] else 0,
        'relation_accuracy': np.mean([c > s for c, s in zip(stats['relation_correct_mean_sim'], 
                                                             stats['relation_swapped_mean_sim'])]) if stats['relation_correct_mean_sim'] else 0,
        'mean_pos_sim': np.mean(stats['component_pos_mean_sim']) if stats['component_pos_mean_sim'] else 0,
        'mean_neg_sim': np.mean(stats['component_neg_mean_sim']) if stats['component_neg_mean_sim'] else 0,
        'mean_margin': np.mean(stats['pos_neg_margin']) if stats['pos_neg_margin'] else 0,
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SALIENCY STATISTICS")
    print("=" * 60)
    print(f"Component Accuracy (pos > neg): {summary['component_accuracy']:.1%}")
    print(f"Relation Accuracy (correct > swapped): {summary['relation_accuracy']:.1%}")
    print(f"Mean Positive Similarity: {summary['mean_pos_sim']:.3f}")
    print(f"Mean Negative Similarity: {summary['mean_neg_sim']:.3f}")
    print(f"Mean Margin: {summary['mean_margin']:.3f}")
    print("=" * 60)
    
    return summary


# =============================================================================
# Multi-Model Comparison
# =============================================================================

def plot_model_comparison(
    extractors: Dict[str, CLIPSaliencyExtractor],
    image: Image.Image,
    text: str,
    output_path: str,
    title: str = None,
):
    """
    Compare saliency maps across multiple models side-by-side.
    
    Args:
        extractors: Dict mapping model name to extractor
        image: PIL Image
        text: Text query for saliency
        output_path: Where to save the figure
        title: Optional title
    """
    n_models = len(extractors)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Each model's saliency
    for i, (model_name, extractor) in enumerate(extractors.items()):
        smap, sim = extractor.compute_patch_text_similarity(image, text)
        
        if smap is not None and smap.size > 0:
            smap_norm = normalize_saliency(smap)
            if smap_norm is not None:
                overlay = overlay_heatmap(image, smap_norm, alpha=0.6)
                axes[i + 1].imshow(overlay)
            else:
                axes[i + 1].imshow(image)
        else:
            axes[i + 1].imshow(image)
            axes[i + 1].text(0.5, 0.5, "N/A", transform=axes[i + 1].transAxes, 
                            ha='center', fontsize=12)
        
        axes[i + 1].set_title(f"{model_name}\nsim={sim:.3f}", fontsize=10)
        axes[i + 1].axis('off')
    
    if title:
        plt.suptitle(f"{title[:80]}...", fontsize=11, fontweight='bold')
    else:
        plt.suptitle(f"\"{text[:60]}...\"", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_model_comparison_grid(
    extractors: Dict[str, CLIPSaliencyExtractor],
    sample: Sample,
    output_path: str,
    method: str = "patch",  # "patch", "vit_gradcam", or "both"
):
    """
    Compare multiple models on positive and negative texts.
    
    Creates a grid: rows = texts (positive/negative), columns = models
    
    Args:
        method: "patch" for patch-text similarity, "vit_gradcam" for gradient-based,
                "both" to show both methods side by side
    """
    try:
        image = Image.open(sample.image_path).convert('RGB')
    except Exception as e:
        logger.warning(f"Failed to load image: {e}")
        return
    
    # Collect texts
    texts = []
    labels = []
    is_positive = []
    
    # Add first component and its negative
    if sample.components:
        comp = sample.components[0]
        texts.append(comp)
        labels.append(f"✓ {comp[:40]}")
        is_positive.append(True)
        
        negs = sample.component_negatives.get(comp, [])
        if negs:
            neg_text = negs[0].get('negative', '') if isinstance(negs[0], dict) else negs[0]
            if neg_text:
                texts.append(neg_text)
                labels.append(f"✗ {neg_text[:40]}")
                is_positive.append(False)
    
    if not texts:
        return
    
    n_texts = len(texts)
    n_models = len(extractors)
    model_names = list(extractors.keys())
    
    # Determine number of columns based on method
    if method == "both":
        # Show both patch and gradcam for each model
        n_method_cols = n_models * 2
        col_labels = []
        for name in model_names:
            col_labels.extend([f"{name}\n(Patch)", f"{name}\n(GradCAM)"])
    else:
        n_method_cols = n_models
        method_suffix = "(Patch)" if method == "patch" else "(GradCAM)"
        col_labels = [f"{name}\n{method_suffix}" for name in model_names]
    
    fig, axes = plt.subplots(n_texts, n_method_cols + 1, figsize=(4 * (n_method_cols + 1), 4 * n_texts))
    
    if n_texts == 1:
        axes = axes.reshape(1, -1)
    
    for row, (text, label, pos) in enumerate(zip(texts, labels, is_positive)):
        # Original image in first column
        axes[row, 0].imshow(image)
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=10, fontweight='bold')
        axes[row, 0].set_ylabel(label, fontsize=9, color='green' if pos else 'red')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        
        # Each model (and method)
        col_idx = 1
        for model_name in model_names:
            extractor = extractors[model_name]
            
            if method == "both":
                # Patch similarity
                smap_patch, sim_patch = extractor.compute_patch_text_similarity(image, text)
                if smap_patch is not None and smap_patch.size > 0:
                    smap_norm = normalize_saliency(smap_patch)
                    if smap_norm is not None:
                        overlay = overlay_heatmap(image, smap_norm, alpha=0.6)
                        axes[row, col_idx].imshow(overlay)
                    else:
                        axes[row, col_idx].imshow(image)
                else:
                    axes[row, col_idx].imshow(image)
                
                if row == 0:
                    axes[row, col_idx].set_title(f"{model_name}\n(Patch)", fontsize=10, fontweight='bold')
                axes[row, col_idx].text(0.02, 0.98, f"sim={sim_patch:.3f}", 
                                       transform=axes[row, col_idx].transAxes,
                                       fontsize=8, va='top', ha='left',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                axes[row, col_idx].axis('off')
                col_idx += 1
                
                # GradCAM
                smap_grad, sim_grad = extractor.compute_vit_gradcam(image, text)
                if smap_grad is not None and smap_grad.size > 0:
                    smap_norm = normalize_saliency(smap_grad)
                    if smap_norm is not None:
                        overlay = overlay_heatmap(image, smap_norm, alpha=0.6)
                        axes[row, col_idx].imshow(overlay)
                    else:
                        axes[row, col_idx].imshow(image)
                else:
                    axes[row, col_idx].imshow(image)
                
                if row == 0:
                    axes[row, col_idx].set_title(f"{model_name}\n(GradCAM)", fontsize=10, fontweight='bold')
                axes[row, col_idx].text(0.02, 0.98, f"sim={sim_grad:.3f}", 
                                       transform=axes[row, col_idx].transAxes,
                                       fontsize=8, va='top', ha='left',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                axes[row, col_idx].axis('off')
                col_idx += 1
            else:
                # Single method
                if method == "vit_gradcam":
                    smap, sim = extractor.compute_vit_gradcam(image, text)
                    method_label = "GradCAM"
                else:
                    smap, sim = extractor.compute_patch_text_similarity(image, text)
                    method_label = "Patch"
                
                if smap is not None and smap.size > 0:
                    smap_norm = normalize_saliency(smap)
                    if smap_norm is not None:
                        overlay = overlay_heatmap(image, smap_norm, alpha=0.6)
                        axes[row, col_idx].imshow(overlay)
                    else:
                        axes[row, col_idx].imshow(image)
                else:
                    axes[row, col_idx].imshow(image)
                
                if row == 0:
                    axes[row, col_idx].set_title(f"{model_name}\n({method_label})", fontsize=10, fontweight='bold')
                
                axes[row, col_idx].text(0.02, 0.98, f"sim={sim:.3f}", 
                                       transform=axes[row, col_idx].transAxes,
                                       fontsize=8, va='top', ha='left',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                axes[row, col_idx].axis('off')
                col_idx += 1
    
    plt.suptitle(f"Model Comparison: {sample.sample_id[:40]}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved model comparison to {output_path}")


def run_multi_model_comparison(
    model_configs: List[Dict],
    json_folder: str,
    image_root: str,
    output_dir: str,
    num_samples: int = 20,
    seed: int = 42,
    method: str = "patch",  # NEW: Add method parameter
):
    """
    Run saliency comparison across multiple models.
    
    Args:
        model_configs: List of dicts with keys:
            - name: Display name for the model
            - checkpoint_path: Path to checkpoint (or None for baseline)
            - checkpoint_type: Type of checkpoint
            - model_name: Base model architecture
        json_folder: Path to JSON files with samples
        image_root: Root path for images
        output_dir: Output directory
        num_samples: Number of samples to visualize
        seed: Random seed
        method: Saliency method - "patch", "vit_gradcam", or "both"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_comparison").mkdir(exist_ok=True)
    
    # Load samples
    samples = load_samples(json_folder, image_root, num_samples, seed)
    if not samples:
        logger.error("No samples loaded!")
        return
    
    # Initialize all extractors
    extractors = {}
    for config in model_configs:
        name = config.get('name', 'Model')
        logger.info(f"Loading model: {name}")
        
        try:
            extractor = CLIPSaliencyExtractor(
                model_name=config.get('model_name', 'ViT-B/32'),
                checkpoint_path=config.get('checkpoint_path'),
                checkpoint_type=config.get('checkpoint_type', 'openclip'),
                force_openclip=config.get('force_openclip', False),
                pretrained=config.get('pretrained', 'openai'),
            )
            extractors[name] = extractor
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            continue
    
    if not extractors:
        logger.error("No models loaded successfully!")
        return
    
    logger.info(f"Loaded {len(extractors)} models: {list(extractors.keys())}")
    logger.info(f"Saliency method: {method}")
    
    # Generate comparisons
    for i, sample in enumerate(tqdm(samples, desc="Generating model comparisons")):
        sample_name = f"{i:02d}_{sample.sample_id[:20]}"
        
        plot_model_comparison_grid(
            extractors, sample,
            str(output_dir / "model_comparison" / f"comparison_{sample_name}.png"),
            method=method,
        )
    
    logger.info(f"✅ Model comparisons saved to {output_dir / 'model_comparison'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CLIP Saliency Visualization")
    
    # Model arguments (single model mode)
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--checkpoint_type", type=str, default="openclip",
                       choices=["openclip", "huggingface", "tripletclip", "external", "dac", "clove"])
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--force_openclip", action="store_true")
    parser.add_argument("--pretrained", type=str, default="openai")
    
    # Multi-model comparison mode
    parser.add_argument("--compare_checkpoints", type=str, nargs="+", default=None,
                       help="Paths to checkpoints to compare. Use 'baseline' for pretrained model.")
    parser.add_argument("--checkpoint_names", type=str, nargs="+", default=None,
                       help="Display names for each checkpoint (must match --compare_checkpoints length)")
    
    # Data arguments
    parser.add_argument("--json_folder", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=".")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="saliency_results")
    
    # Visualization options
    parser.add_argument("--method", type=str, default="patch",
                       choices=["patch", "gradient", "vit_gradcam", "both"],
                       help="Saliency method: 'patch' for patch-text similarity, 'gradient' for pixel GradCAM, 'vit_gradcam' for ViT-native GradCAM, 'both' to show patch and vit_gradcam side-by-side")
    parser.add_argument("--visualizations", type=str, nargs="+",
                       default=["comparisons", "differences", "tokens", "relations", "methods"],
                       choices=["comparisons", "differences", "tokens", "relations", "methods", "components"],
                       help="Which visualizations to generate")
    
    args = parser.parse_args()
    
    # =========================================================================
    # Multi-model comparison mode
    # =========================================================================
    if args.compare_checkpoints:
        logger.info("Running in multi-model comparison mode")
        
        # Build model configs
        model_configs = []
        names = args.checkpoint_names or [f"Model_{i}" for i in range(len(args.compare_checkpoints))]
        
        if len(names) != len(args.compare_checkpoints):
            logger.warning("Checkpoint names don't match checkpoints count, using default names")
            names = [f"Model_{i}" for i in range(len(args.compare_checkpoints))]
        
        for name, ckpt_path in zip(names, args.compare_checkpoints):
            if ckpt_path.lower() == 'baseline':
                # Use pretrained model
                model_configs.append({
                    'name': name,
                    'checkpoint_path': None,
                    'checkpoint_type': 'openclip',
                    'model_name': args.model_name,
                    'pretrained': args.pretrained,
                })
            else:
                model_configs.append({
                    'name': name,
                    'checkpoint_path': ckpt_path,
                    'checkpoint_type': args.checkpoint_type,
                    'model_name': args.model_name,
                    'force_openclip': args.force_openclip,
                })
        
        run_multi_model_comparison(
            model_configs=model_configs,
            json_folder=args.json_folder,
            image_root=args.image_root,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            method=args.method,
        )
        return
    
    # =========================================================================
    # Single model mode (original behavior)
    # =========================================================================
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for viz_type in args.visualizations:
        (output_dir / viz_type).mkdir(exist_ok=True)
    
    # Load samples
    samples = load_samples(
        args.json_folder,
        args.image_root,
        max_samples=args.num_samples,
        seed=args.seed,
    )
    
    if not samples:
        logger.error("No samples loaded!")
        return
    
    # Initialize extractor
    extractor = CLIPSaliencyExtractor(
        model_name=args.model_name,
        checkpoint_type=args.checkpoint_type,
        checkpoint_path=args.checkpoint_path,
        force_openclip=args.force_openclip,
        pretrained=args.pretrained,
    )
    
    # Generate visualizations
    for i, sample in enumerate(tqdm(samples, desc="Generating saliency visualizations")):
        try:
            image = Image.open(sample.image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image: {e}")
            continue
        
        sample_name = f"{i:02d}_{sample.sample_id[:20]}"
        
        # Comparison plots (positive vs negative)
        if "comparisons" in args.visualizations:
            plot_patch_similarity_comparison(
                extractor, sample,
                str(output_dir / "comparisons" / f"comparison_{sample_name}.png"),
                method=args.method,
            )
        
        # Difference plots
        if "differences" in args.visualizations:
            plot_saliency_difference(
                extractor, sample,
                str(output_dir / "differences" / f"difference_{sample_name}.png"),
            )
        
        # Token-level heatmaps (NEW)
        if "tokens" in args.visualizations and sample.components:
            comp = sample.components[0]
            plot_token_heatmaps(
                extractor, image, comp,
                str(output_dir / "tokens" / f"tokens_{sample_name}.png"),
            )
        
        # Relation contrast analysis (NEW)
        if "relations" in args.visualizations and sample.relations:
            rel = sample.relations[0]
            subject = rel.get('subject', '')
            relation_type = rel.get('relation_type', '')
            obj = rel.get('object', '')
            
            if subject and relation_type and obj:
                plot_relation_contrast(
                    extractor, image, subject, relation_type, obj,
                    str(output_dir / "relations" / f"relation_{sample_name}.png"),
                )
        
        # Method comparison (NEW)
        if "methods" in args.visualizations and sample.components:
            comp = sample.components[0]
            plot_method_comparison(
                extractor, image, comp,
                str(output_dir / "methods" / f"methods_{sample_name}.png"),
            )
        
        # Component grids (legacy)
        if "components" in args.visualizations:
            plot_component_saliency_grid(
                extractor, sample,
                str(output_dir / "components" / f"components_{sample_name}.png"),
            )
    
    # Statistics
    compute_saliency_statistics(
        extractor, samples,
        str(output_dir / "statistics.json"),
    )
    
    logger.info(f"\n✅ All visualizations saved to {output_dir}")
    logger.info(f"   Generated: {', '.join(args.visualizations)}")


if __name__ == "__main__":
    main()

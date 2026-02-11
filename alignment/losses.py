import torch
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class ComponentLossConfig:
    """Configuration for modular multi-caption contrastive loss.
    
    The loss is computed as:
        L = λ_full * L_full + λ_comp * L_comp + λ_align * L_align + λ_rank * L_rank + λ_para * L_para
    
    Where:
        - L_full: Full caption contrastive loss (CLIP or NegCLIP)
        - L_comp: Component contrastive loss (CLIP, NegCLIP, or NegCLIP_hard)
        - L_align: Alignment loss (cosine or margin-based)
        - L_rank: Ranking loss (enforces full > components)
        - L_para: Paraphrase alignment loss
    
    Each loss term is independently configurable with its own lambda weight.
    
    Examples:
        # Standard CLIP on full captions only
        config = ComponentLossConfig(lambda_full=1.0)
        
        # NegCLIP on full + component contrastive + alignment
        config = ComponentLossConfig(
            lambda_full=1.0,
            use_negatives_full=True,
            lambda_components=0.5,
            component_loss_type="negclip",
            lambda_alignment=0.25,
            alignment_loss_type="margin",
            alignment_margin=0.1,
        )
        
        # Full spectrum: all losses enabled
        config = ComponentLossConfig(
            lambda_full=1.0,
            use_negatives_full=True,
            lambda_components=0.5,
            component_loss_type="negclip_hard",
            lambda_alignment=0.25,
            alignment_loss_type="margin",
            lambda_rank=0.1,
            lambda_paraphrase=0.1,
        )
    """
    
    # ===== Full Caption Loss =====
    lambda_full: float = 1.0           # Weight for full caption loss
    use_negatives_full: bool = True    # Use NegCLIP (True) or CLIP (False) for full caption
    
    # ===== Component Contrastive Loss =====
    lambda_components: float = 0.0     # Weight for component contrastive loss (0 = disabled)
    component_loss_type: str = "negclip"  # "clip", "negclip", or "negclip_hard"
    
    # ===== Alignment Loss (separate from contrastive) =====
    lambda_alignment: float = 0.0      # Weight for alignment loss (0 = disabled)
    alignment_loss_type: str = "margin"  # "cosine" (1 - cos_sim) or "margin" (hinge loss)
    alignment_margin: float = 0.1      # Margin for alignment_margin loss
    
    # ===== Ranking Loss =====
    lambda_rank: float = 0.0           # Weight for ranking loss (0 = disabled)
    rank_margin: float = 0.1           # Margin for ranking: s(full) > s(comp) + margin
    rank_reduction: str = "mean"       # How to reduce over components: "mean" or "max"
    
    # ===== Text Contrastive Margin Loss =====
    lambda_text_contrastive: float = 0.0  # Weight for text-space contrastive margin loss (0 = disabled)
    text_contrastive_margin: float = 0.1  # Margin for text contrastive: s(t_full, t_comp+) > s(t_full, t_comp-) + margin
    
    # ===== Paraphrase Loss =====
    lambda_paraphrase: float = 0.0     # Weight for sentence alignment loss (0 = disabled)
    
    # ===== Deprecated (for backward compatibility) =====
    hybrid_margin_weight: float = 0.0  # Deprecated: use lambda_alignment instead
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for **kwargs unpacking."""
        return {
            'lambda_full': self.lambda_full,
            'use_negatives_full': self.use_negatives_full,
            'lambda_components': self.lambda_components,
            'component_loss_type': self.component_loss_type,
            'lambda_alignment': self.lambda_alignment,
            'alignment_loss_type': self.alignment_loss_type,
            'alignment_margin': self.alignment_margin,
            'lambda_rank': self.lambda_rank,
            'rank_margin': self.rank_margin,
            'rank_reduction': self.rank_reduction,
            'lambda_text_contrastive': self.lambda_text_contrastive,
            'text_contrastive_margin': self.text_contrastive_margin,
            'lambda_paraphrase': self.lambda_paraphrase,
        }
    
    @classmethod
    def from_alpha(cls, alpha: float, **kwargs) -> 'ComponentLossConfig':
        """
        Create config from old alpha parameter for backward compatibility.
        
        Alpha interpretation:
        - alpha = 1.0: Full caption only (lambda_full=1.0, lambda_components=0.0)
        - alpha = 0.5: Equal weight (lambda_full=1.0, lambda_components=1.0)
        - alpha = 0.0: Components only (lambda_full=0.0, lambda_components=1.0)
        
        Formula: 
        - lambda_full = alpha / (1 - alpha + epsilon) if alpha < 1 else very large
        - lambda_components = (1 - alpha) / (alpha + epsilon)
        
        Simpler approach: Use alpha directly as ratio
        - lambda_full = alpha
        - lambda_components = (1 - alpha)
        
        Args:
            alpha: Weight for full caption (0.0 to 1.0)
            **kwargs: Other config parameters
        
        Returns:
            ComponentLossConfig with converted weights
        """
        return cls(
            lambda_full=alpha,
            lambda_components=(1.0 - alpha),
            **kwargs
        )



def multi_caption_contrastive_loss(
    image_embeddings: torch.Tensor,
    pos_text_embeddings: torch.Tensor,
    neg_text_embeddings: torch.Tensor,
    temperature: torch.Tensor,
    device: torch.device,
    contrastive_mode: str = "without_negatives",
    # ===== Full Caption Loss =====
    lambda_full: float = 1.0,          # Weight for full caption loss
    use_negatives_full: bool = True,   # Use NegCLIP (True) or CLIP (False) for full caption
    # ===== Component Contrastive Loss =====
    lambda_components: float = 0.0,    # Weight for component contrastive loss (0 = disabled)
    component_loss_type: str = "negclip",  # "clip", "negclip", or "negclip_hard"
    # ===== Alignment Loss (separate from contrastive) =====
    lambda_alignment: float = 0.0,     # Weight for alignment loss (0 = disabled)
    alignment_loss_type: str = "margin",  # "cosine" or "margin"
    alignment_margin: float = 0.1,     # Margin for alignment_margin loss
    # ===== Ranking Loss =====
    lambda_rank: float = 0.0,          # Weight for ranking loss (0 = disabled)
    rank_margin: float = 0.1,          # Margin for ranking: s(full) > s(comp) + margin
    rank_reduction: str = "mean",      # How to reduce over components: "mean" or "max"
    # ===== Text Contrastive Margin Loss =====
    lambda_text_contrastive: float = 0.0,  # Weight for text-space contrastive margin loss (0 = disabled)
    text_contrastive_margin: float = 0.1,  # Margin: s(t_full, t_comp+) > s(t_full, t_comp-) + margin
    # ===== Paraphrase Loss =====
    lambda_paraphrase: float = 0.0,    # Weight for sentence alignment loss (0 = disabled)
    # ===== Per-component-caption coverage information =====
    components_per_caption: Optional[torch.Tensor] = None,  # [B, N] binary mask or counts per component
    num_components_available: Optional[torch.Tensor] = None,  # [B] - total components available
    # ===== Validity mask for filtering invalid samples =====
    caption_valid_mask: Optional[torch.Tensor] = None,  # [B, 1+N] - validity mask for each caption
    # ===== Paraphrase embeddings =====
    paraphrase_embeddings: Optional[torch.Tensor] = None,  # [B, D] - paraphrase text embeddings
    has_paraphrase: Optional[torch.Tensor] = None,  # [B] - boolean mask for samples with paraphrases
    # ===== Backward compatibility (deprecated) =====
    alpha: Optional[float] = None,     # Deprecated: use lambda_full/lambda_components
    hybrid_margin_weight: float = 0.0, # Deprecated: use lambda_alignment instead
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Modular multi-caption contrastive loss with independent loss terms.
    
    Total Loss:
        L = λ_full * L_full + λ_comp * L_comp + λ_align * L_align + λ_rank * L_rank + λ_para * L_para
    
    Loss Components:
        1. L_full: Full caption contrastive loss
           - CLIP or NegCLIP based on use_negatives_full flag
           
        2. L_comp: Component contrastive loss (independent from alignment)
           - "clip": Standard CLIP loss
           - "negclip": NegCLIP with batch negatives (B+B scores)
           - "negclip_hard": NegCLIP with per-sample hard negatives only (B+1 scores)
           
        3. L_align: Alignment loss (can be used WITH component loss)
           - "cosine": Direct alignment L = 1 - cos(I, t_k)
           - "margin": Pairwise margin L = max(0, m - s_pos + s_neg)
           
        4. L_rank: Ranking loss (enforces s(full) > s(comp) + margin)
        
        5. L_para: Paraphrase alignment loss
    
    Args:
        image_embeddings: [B, D] - normalized image embeddings
        pos_text_embeddings: [B, 1+N, D] - text embeddings where:
            - [:, 0, :] = full caption embeddings
            - [:, 1:, :] = component caption embeddings (N components)
        neg_text_embeddings: [B, 1+N, D] or [B, D] - negative caption embeddings
        temperature: scalar or tensor - temperature for scaling
        device: torch device
        contrastive_mode: Mode for enabling component processing:
            - "without_negatives": standard CLIP loss on full captions only
            - "with_negatives": NegCLIP-style loss on full captions with negatives
            - "with_components": Enable component processing (no negatives)
            - "with_components_negatives": Enable component processing (with negatives)
        lambda_full: Weight for full caption loss
        use_negatives_full: Use NegCLIP (True) or CLIP (False) for full caption
        lambda_components: Weight for component contrastive loss
        component_loss_type: Type of contrastive loss for components
        lambda_alignment: Weight for alignment loss (separate from contrastive)
        alignment_loss_type: Type of alignment loss ("cosine" or "margin")
        alignment_margin: Margin for margin-based alignment loss
        lambda_rank: Weight for ranking loss
        rank_margin: Margin for ranking loss
        lambda_paraphrase: Weight for paraphrase alignment loss
        
    Returns:
        Dict with all loss components and metrics
    
    Examples:
        # Standard NegCLIP on full captions only
        result = multi_caption_contrastive_loss(
            ..., 
            lambda_full=1.0, 
            use_negatives_full=True
        )
        
        # NegCLIP full + component contrastive + alignment (replaces hybrid)
        result = multi_caption_contrastive_loss(
            ...,
            lambda_full=1.0,
            use_negatives_full=True,
            lambda_components=0.5,
            component_loss_type="negclip",
            lambda_alignment=0.25,
            alignment_loss_type="margin",
            alignment_margin=0.1,
        )
        
        # Full spectrum with all losses
        result = multi_caption_contrastive_loss(
            ...,
            lambda_full=1.0,
            lambda_components=0.5,
            lambda_alignment=0.25,
            lambda_rank=0.1,
            lambda_paraphrase=0.1,
        )
    """
    # Backward compatibility: convert alpha to lambdas
    if alpha is not None:
        lambda_full = alpha
        lambda_components = (1.0 - alpha)
    
    # Backward compatibility: convert hybrid to lambda_alignment
    if hybrid_margin_weight > 0 and lambda_alignment == 0:
        # Old hybrid mode: split component weight between contrastive and alignment
        lambda_alignment = lambda_components * hybrid_margin_weight
        lambda_components = lambda_components * (1.0 - hybrid_margin_weight)
        alignment_loss_type = "margin"
    
    # ================================================================
    # FP32 NUMERICAL STABILITY FOR AMP
    # ================================================================
    # When using AMP (mixed precision), the forward pass runs in FP16
    # but loss computation needs FP32 to avoid overflow/underflow.
    # This is critical for:
    # - Temperature scaling (1/0.07 ≈ 14.3 amplifies values)
    # - Softmax with large logits (exp overflow in FP16)
    # - Margin loss subtraction (precision loss in FP16)
    # 
    # We disable autocast here and convert to FP32 explicitly.
    # The gradients will be automatically scaled back by AMP's GradScaler.
    # ================================================================
    
    # Disable autocast for loss computation (compute in FP32)
    with torch.amp.autocast('cuda', enabled=False):
        # Convert all embeddings to FP32 for numerical stability
        image_embeddings = image_embeddings.float().to(device)
        pos_text_embeddings = pos_text_embeddings.float().to(device)
        if neg_text_embeddings is not None:
            neg_text_embeddings = neg_text_embeddings.float().to(device)
        
        if caption_valid_mask is not None:
            caption_valid_mask = caption_valid_mask.to(device)

        if isinstance(temperature, torch.Tensor):
            temperature = temperature.float().to(device)
        else:
            temperature = torch.tensor(temperature, dtype=torch.float32, device=device)
        
        num_positive_captions = pos_text_embeddings.shape[1] if pos_text_embeddings.dim() == 3 else 1
    
        # Ensure pos_text_embeddings is [B, 1+N, D]
        if pos_text_embeddings.dim() == 2:
            pos_text_embeddings = pos_text_embeddings.unsqueeze(1)  # [B, 1, D]
    
        # Extract full captions and components
        full_captions = pos_text_embeddings[:, 0, :]  # [B, D]
        component_captions = pos_text_embeddings[:, 1:, :] if num_positive_captions > 1 else None  # [B, N, D]
        
        # Handle neg_text_embeddings: can be [B, 1+N, D] (new format) or [B, D] (legacy)
        # New format: paired negatives for each caption type
        # Legacy format: shared negative for all captions
        use_negatives = (contrastive_mode == "with_negatives" or contrastive_mode == "with_components_negatives")
        
        if neg_text_embeddings.dim() == 3 and neg_text_embeddings.shape[1] == num_positive_captions:
            # New format: [B, 1+N, D] - extract full caption negative and component negatives
            full_caption_neg = neg_text_embeddings[:, 0, :]  # [B, D]
            component_negatives = neg_text_embeddings[:, 1:, :] if num_positive_captions > 1 else None  # [B, N, D]
        else:
            # Legacy format: [B, D] - use as shared negative
            full_caption_neg = neg_text_embeddings  # [B, D]
            component_negatives = neg_text_embeddings  # [B, D] - will be broadcast in component loss
        
        # Extract validity masks for full caption and components
        if caption_valid_mask is not None:
            full_caption_valid = caption_valid_mask[:, 0]  # [B]
            component_valid_mask = caption_valid_mask[:, 1:] if caption_valid_mask.shape[1] > 1 else None  # [B, N]
        else:
            full_caption_valid = None
            component_valid_mask = None

        # ================================================================
        # Step 1: Full Caption Loss (CLIP or NegCLIP)
        # ================================================================
        # Determine whether to use negatives for full caption
        # - use_negatives_full=True: NegCLIP-style (adds negatives to denominator)
        # - use_negatives_full=False: Standard CLIP (no negatives)
        # Note: contrastive_mode still controls whether negatives are available
        use_neg_for_full = use_negatives_full and (
            contrastive_mode == "with_negatives" or 
            contrastive_mode == "with_components_negatives"
        )
        
        full_result = _compute_clip_loss_for_caption(
            image_embeddings,
            full_captions,
            full_caption_neg if use_neg_for_full else None,
            temperature,
            device,
            valid_mask=full_caption_valid
        )
        full_loss = full_result['loss']
        full_accuracy = full_result['accuracy']
        
        # ================================================================
        # Step 2: Component Contrastive Loss (CLIP, NegCLIP, or NegCLIP_hard)
        # ================================================================
        has_components = (component_captions is not None and component_captions.shape[1] > 0)
        compute_components = (contrastive_mode == "with_components" or contrastive_mode == "with_components_negatives")
        use_negatives = (contrastive_mode == "with_negatives" or contrastive_mode == "with_components_negatives")
        
        if compute_components and has_components and lambda_components > 0:
            # Compute component contrastive losses (CLIP, NegCLIP, or NegCLIP_hard)
            component_loss, component_accuracy = _compute_component_losses(
                image_embeddings=image_embeddings,
                component_captions=component_captions,
                neg_text_embeddings=component_negatives if use_negatives else None,
                temperature=temperature,
                device=device,
                component_loss_type=component_loss_type,
                components_per_caption=components_per_caption,
                num_components_available=num_components_available,
                component_valid_mask=component_valid_mask,
                alignment_margin=alignment_margin,  # For backward compat with hybrid
                hybrid_margin_weight=0.0,  # Disable hybrid - use separate alignment loss
            )
        else:
            component_loss = torch.tensor(0.0, device=device)
            component_accuracy = 0.0
        
        # ================================================================
        # Step 3: Alignment Loss (cosine or margin-based)
        # ================================================================
        # Separate from component contrastive loss - can be used together
        if compute_components and has_components and lambda_alignment > 0:
            if alignment_loss_type == "margin":
                # Pairwise margin loss: max(0, m - s_pos + s_neg)
                has_paired_negatives = (
                    component_negatives is not None 
                    and component_negatives.dim() == 3 
                    and component_negatives.shape[1] == component_captions.shape[1]
                )
                if has_paired_negatives:
                    alignment_loss, alignment_accuracy = _compute_alignment_margin_loss_for_components(
                        image_embeddings=image_embeddings,
                        component_captions=component_captions,
                        neg_text_embeddings=component_negatives,
                        num_components=component_captions.shape[1],
                        components_per_caption=components_per_caption,
                        component_valid_mask=component_valid_mask,
                        margin=alignment_margin,
                    )
                else:
                    # Fallback to cosine alignment if no paired negatives
                    alignment_loss, alignment_accuracy = _compute_alignment_loss_for_components(
                        image_embeddings=image_embeddings,
                        component_captions=component_captions,
                        valid_component_mask=None,
                        num_components=component_captions.shape[1],
                        components_per_caption=components_per_caption,
                        loss_type="cosine",
                    )
            else:
                # Cosine alignment: 1 - cos(I, t_k)
                alignment_loss, alignment_accuracy = _compute_alignment_loss_for_components(
                    image_embeddings=image_embeddings,
                    component_captions=component_captions,
                    valid_component_mask=None,
                    num_components=component_captions.shape[1],
                    components_per_caption=components_per_caption,
                    loss_type="cosine",
                )
        else:
            alignment_loss = torch.tensor(0.0, device=device)
            alignment_accuracy = 0.0
        
        # ================================================================
        # Step 4: Paraphrase/Sentence Alignment Loss
        # ================================================================
        if paraphrase_embeddings is not None and lambda_paraphrase > 0:
            paraphrase_result = _compute_sentence_alignment_loss(
                text_embeddings=full_captions,  # Original captions [B, D]
                paraphrase_embeddings=paraphrase_embeddings.float().to(device),  # Paraphrase embeddings [B, D]
                temperature=temperature,
                device=device,
                valid_mask=has_paraphrase,  # Only compute for samples with paraphrases
            )
            paraphrase_loss = paraphrase_result['loss']
            paraphrase_accuracy = paraphrase_result['accuracy']
        else:
            paraphrase_loss = torch.tensor(0.0, device=device)
            paraphrase_accuracy = 0.0
        
        # ================================================================
        # Step 5: Ranking Loss (enforces s(full) > s(comp) + margin)
        # ================================================================
        if lambda_rank > 0 and has_components:
            rank_result = _compute_ranking_loss(
                image_embeddings=image_embeddings,
                full_caption_embeddings=full_captions,
                component_captions=component_captions,
                margin=rank_margin,
                reduction=rank_reduction,
                component_valid_mask=component_valid_mask,
            )
            ranking_loss = rank_result['loss']
            rank_violation_rate = rank_result['violation_rate']
            rank_avg_margin = rank_result['avg_margin']
        else:
            ranking_loss = torch.tensor(0.0, device=device)
            rank_violation_rate = 0.0
            rank_avg_margin = 0.0
        
        # ================================================================
        # Step 5b: Text Contrastive Margin Loss 
        # (enforces s(t_full, t_comp+) > s(t_full, t_comp-) + margin)
        # ================================================================
        if lambda_text_contrastive > 0 and has_components:
            # Check for paired negatives
            has_paired_negatives = (
                component_negatives is not None 
                and component_negatives.dim() == 3 
                and component_negatives.shape[1] == component_captions.shape[1]
            )
            if has_paired_negatives:
                text_contrastive_result = _compute_text_contrastive_margin_loss_for_components(
                    full_caption_embeddings=full_captions,
                    component_captions=component_captions,
                    neg_text_embeddings=component_negatives,
                    num_components=component_captions.shape[1],
                    components_per_caption=components_per_caption,
                    component_valid_mask=component_valid_mask,
                    margin=text_contrastive_margin,
                )
                text_contrastive_loss = text_contrastive_result[0]
                text_contrastive_accuracy = text_contrastive_result[1]
            else:
                text_contrastive_loss = torch.tensor(0.0, device=device)
                text_contrastive_accuracy = 0.0
        else:
            text_contrastive_loss = torch.tensor(0.0, device=device)
            text_contrastive_accuracy = 0.0
        
        # ================================================================
        # Step 6: Combine All Losses
        # ================================================================
        # L = λ_full * L_full + λ_comp * L_comp + λ_align * L_align + λ_rank * L_rank + λ_text * L_text + λ_para * L_para
        total_loss = (
            lambda_full * full_loss 
            + lambda_components * component_loss 
            + lambda_alignment * alignment_loss
            + lambda_rank * ranking_loss
            + lambda_text_contrastive * text_contrastive_loss
            + lambda_paraphrase * paraphrase_loss
        )
        
        # ================================================================
        # Step 7: Compute Weighted Overall Accuracy
        # ================================================================
        lambda_sum = lambda_full + lambda_components + lambda_alignment
        if lambda_sum > 0 and (component_accuracy > 0 or alignment_accuracy > 0):
            accuracy = (
                lambda_full * full_accuracy 
                + lambda_components * component_accuracy
                + lambda_alignment * alignment_accuracy
            ) / lambda_sum
        else:
            accuracy = full_accuracy
        
        # ================================================================
        # Step 8: Compute Mask Statistics for Logging
        # ================================================================
        B = image_embeddings.shape[0]
        N = component_captions.shape[1] if has_components else 0
        
        # Full caption mask stats
        if full_caption_valid is not None:
            num_valid_full = int(full_caption_valid.sum().item())
            num_masked_full = B - num_valid_full
        else:
            num_valid_full = B
            num_masked_full = 0
        
        # Component mask stats
        if component_valid_mask is not None and N > 0:
            num_valid_components = int(component_valid_mask.sum().item())
            num_total_components = B * N
            num_masked_components = num_total_components - num_valid_components
        else:
            num_total_components = B * N
            num_valid_components = num_total_components
            num_masked_components = 0
        
        # Paraphrase mask stats
        if has_paraphrase is not None:
            num_valid_paraphrase = int(has_paraphrase.sum().item())
        else:
            num_valid_paraphrase = B if paraphrase_embeddings is not None else 0
        
        # ================================================================
        # Return Unified Result Dictionary
        # ================================================================
        result = {
            'loss': total_loss,
            'accuracy': accuracy,
            # Individual loss terms
            'full_loss': full_loss,
            'component_loss': component_loss,
            'alignment_loss': alignment_loss,
            'ranking_loss': ranking_loss,
            'text_contrastive_loss': text_contrastive_loss,
            'paraphrase_loss': paraphrase_loss,
            # Accuracies
            'full_accuracy': full_accuracy,
            'component_accuracy': component_accuracy,
            'alignment_accuracy': alignment_accuracy,
            'text_contrastive_accuracy': text_contrastive_accuracy,
            'paraphrase_accuracy': paraphrase_accuracy,
            # Ranking loss statistics
            'rank_violation_rate': rank_violation_rate,  # Fraction of pairs violating s(full) > s(comp)
            'rank_avg_margin': rank_avg_margin,          # Avg gap between full and component scores
            # Mask statistics
            'num_valid_full': num_valid_full,
            'num_masked_full': num_masked_full,
            'num_valid_components': num_valid_components,
            'num_masked_components': num_masked_components,
            'num_total_components': num_total_components,
            'num_valid_paraphrase': num_valid_paraphrase,
            'mask_rate_full': num_masked_full / B if B > 0 else 0.0,
            'mask_rate_components': num_masked_components / num_total_components if num_total_components > 0 else 0.0,
        }
            
        return result


def _compute_clip_loss_for_caption(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    neg_text_embeddings: Optional[torch.Tensor],
    temperature: torch.Tensor,
    device: torch.device,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Helper function to compute CLIP loss for a single caption type (full or component).
    
    Args:
        image_embeddings: [B, D] image features
        text_embeddings: [B, D] or [num_valid, D] text features
        neg_text_embeddings: [B, D] negative text features (optional)
        temperature: scalar temperature
        device: torch device
        valid_mask: [B] boolean mask for valid samples (None = all valid)
    
    Returns:
        Dict with 'loss', 'accuracy', 'acc_i2t', 'acc_t2i'
    """
    B = image_embeddings.shape[0]
    
    # Disable autocast and force float32 for numerical stability with AMP
    with torch.amp.autocast('cuda', enabled=False):
        image_embeddings = image_embeddings.float()
        text_embeddings = text_embeddings.float()
        if neg_text_embeddings is not None:
            neg_text_embeddings = neg_text_embeddings.float()
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.float()
        
        # Apply mask if provided
        if valid_mask is not None and valid_mask.sum() < B:
            num_valid = int(valid_mask.sum().item())
            if num_valid == 0:
                return {
                    'loss': torch.tensor(0.0, device=device),
                    'accuracy': 0.0,
                    'acc_i2t': torch.tensor(0.0, device=device),
                    'acc_t2i': torch.tensor(0.0, device=device),
                }
            
            # Extract valid samples and create mini-batch
            valid_images = image_embeddings[valid_mask]  # [num_valid, D]
            valid_text = text_embeddings[valid_mask]      # [num_valid, D]
            
            # CRITICAL: Reindex labels to [0..num_valid-1] for the compressed mini-batch
            # Each valid sample i's positive text is now at position i (not original index)
            valid_labels = torch.arange(num_valid, device=device, dtype=torch.long)  # [0, 1, 2, ..., num_valid-1]
            
            # i2t: valid images vs valid texts (+ negatives if provided)
            if neg_text_embeddings is not None:
                # With negatives: [num_valid, num_valid + B]
                # Columns 0..num_valid-1 are valid texts, num_valid..num_valid+B-1 are negatives
                all_text = torch.cat([valid_text, neg_text_embeddings], dim=0)  # [num_valid + B, D]
            else:
                # Without negatives: [num_valid, num_valid]
                all_text = valid_text
            
            logits_i2t = (valid_images @ all_text.T) / temperature  # [num_valid, num_valid (+B)]
            loss_i2t = F.cross_entropy(logits_i2t, valid_labels)
            acc_i2t = (torch.argmax(logits_i2t, dim=1) == valid_labels).float().mean()
            
            # t2i: valid texts vs valid images
            # Each text i should match image i in the compressed mini-batch
            logits_t2i = (valid_text @ valid_images.T) / temperature  # [num_valid, num_valid]
            loss_t2i = F.cross_entropy(logits_t2i, valid_labels)
            acc_t2i = (torch.argmax(logits_t2i, dim=1) == valid_labels).float().mean()
            
        else:
            # No masking - standard computation with full batch
            # Labels are simply [0, 1, 2, ..., B-1] since all pairs are valid
            labels = torch.arange(B, device=device, dtype=torch.long)
            
            if neg_text_embeddings is not None:
                # With negatives: [B, 2B]
                all_text = torch.cat([text_embeddings, neg_text_embeddings], dim=0)
            else:
                # Without negatives: [B, B]
                all_text = text_embeddings
            
            logits_i2t = (image_embeddings @ all_text.T) / temperature
            loss_i2t = F.cross_entropy(logits_i2t, labels)
            acc_i2t = (torch.argmax(logits_i2t, dim=1) == labels).float().mean()
            
            logits_t2i = (text_embeddings @ image_embeddings.T) / temperature
            loss_t2i = F.cross_entropy(logits_t2i, labels)
            acc_t2i = (torch.argmax(logits_t2i, dim=1) == labels).float().mean()
        
        loss = (loss_i2t + loss_t2i) / 2
        accuracy = ((acc_i2t + acc_t2i) / 2).item()
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'acc_i2t': acc_i2t,
        'acc_t2i': acc_t2i,
    }


def _compute_sentence_alignment_loss(
    text_embeddings: torch.Tensor,
    paraphrase_embeddings: torch.Tensor,
    temperature: torch.Tensor,
    device: torch.device,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute sentence-level alignment loss between original captions and their paraphrases.
    
    This implements the paraphrase loss from READ-CLIP:
    L_sentence = -1/B * sum_i log( exp(sim(T_i, T'_i)/τ) / sum_j exp(sim(T_i, T'_j)/τ) )
    
    For each caption T_i, its paraphrase T'_i is the positive, and paraphrases
    from other samples in the batch (T'_j, j≠i) serve as negatives.
    
    This encourages the encoder to embed paraphrased captions close together,
    helping capture semantic relationships between different phrasings of the same meaning.
    
    Args:
        text_embeddings: [B, D] - original caption embeddings (full captions)
        paraphrase_embeddings: [B, D] - paraphrase caption embeddings
        temperature: scalar temperature for softmax
        device: torch device
        valid_mask: [B] boolean mask for samples with valid paraphrases (None = all valid)
    
    Returns:
        Dict with 'loss', 'accuracy'
    """
    B = text_embeddings.shape[0]
    
    # Disable autocast and force float32 for numerical stability with AMP
    with torch.amp.autocast('cuda', enabled=False):
        text_embeddings = text_embeddings.float()
        paraphrase_embeddings = paraphrase_embeddings.float()
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.float()
        
        # Apply mask if provided - extract only valid samples
        if valid_mask is not None and valid_mask.sum() < B:
            num_valid = int(valid_mask.sum().item())
            if num_valid == 0:
                return {
                    'loss': torch.tensor(0.0, device=device),
                    'accuracy': 0.0,
                }
            
            # Extract valid samples
            valid_text = text_embeddings[valid_mask]  # [num_valid, D]
            valid_paraphrase = paraphrase_embeddings[valid_mask]  # [num_valid, D]
            
            # Labels for the compressed mini-batch: each text i should match paraphrase i
            valid_labels = torch.arange(num_valid, device=device, dtype=torch.long)
            
            # Compute similarity: text_i vs all paraphrases
            # logits[i, j] = sim(T_i, T'_j) / temperature
            logits = (valid_text @ valid_paraphrase.T) / temperature  # [num_valid, num_valid]
            
            # Cross-entropy loss: maximize sim(T_i, T'_i) relative to all T'_j
            loss = F.cross_entropy(logits, valid_labels)
            
            # Accuracy: fraction of samples where argmax similarity is the correct paraphrase
            acc = (torch.argmax(logits, dim=1) == valid_labels).float().mean()
            
        else:
            # No masking - standard computation with full batch
            labels = torch.arange(B, device=device, dtype=torch.long)
            
            # Compute similarity matrix: text_i vs all paraphrases
            logits = (text_embeddings @ paraphrase_embeddings.T) / temperature  # [B, B]
            
            # Cross-entropy loss
            loss = F.cross_entropy(logits, labels)
            
            # Accuracy
            acc = (torch.argmax(logits, dim=1) == labels).float().mean()
    
    return {
        'loss': loss,
        'accuracy': acc.item(),
    }


def _compute_ranking_loss(
    image_embeddings: torch.Tensor,
    full_caption_embeddings: torch.Tensor,
    component_captions: torch.Tensor,
    margin: float = 0.1,
    reduction: str = "mean",
    component_valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute ranking loss to enforce: s(I, t+) > s(I, t_k+) + margin
    
    This creates a semantic hierarchy where the complete caption scores higher
    than its component captions (individual objects, attributes, relations).
    
    Intuition: "A cat sitting on a red mat" should score higher with the image
    than just "cat" or "red mat" individually. The full caption provides more
    complete information about the scene.
    
    Loss formulation:
        L_rank = (1/K) * sum_k max(0, margin + s(I, t_k+) - s(I, t+))
    
    This is a margin ranking loss where:
    - s(I, t+) = cosine similarity between image and full caption
    - s(I, t_k+) = cosine similarity between image and k-th component caption
    - margin = desired gap between full and component scores
    
    The loss is 0 when s(I, t+) >= s(I, t_k+) + margin for all components.
    
    Args:
        image_embeddings: [B, D] - normalized image embeddings
        full_caption_embeddings: [B, D] - normalized full caption embeddings
        component_captions: [B, N, D] - normalized component caption embeddings
        margin: Desired gap between full and component scores (default: 0.1)
        reduction: How to reduce over components - "mean" or "max" (default: "mean")
            - "mean": Average margin violation across all components
            - "max": Only penalize the worst violating component
        component_valid_mask: [B, N] - boolean mask for valid components (None = all valid)
    
    Returns:
        Dict with:
            - 'loss': The ranking loss value
            - 'violation_rate': Fraction of pairs where s(comp) > s(full)
            - 'avg_margin': Average (s(full) - s(comp)) across all valid pairs
    """
    B, N, D = component_captions.shape
    device = image_embeddings.device
    
    # Disable autocast and force float32 for numerical stability
    with torch.amp.autocast('cuda', enabled=False):
        image_embeddings = image_embeddings.float()
        full_caption_embeddings = full_caption_embeddings.float()
        component_captions = component_captions.float()
        
        # Compute similarity scores
        # s_full[i] = cos(I_i, t+_i)
        s_full = (image_embeddings * full_caption_embeddings).sum(dim=-1)  # [B]
        
        # s_comp[i, k] = cos(I_i, t_k+_i)
        # Expand image embeddings for broadcasting: [B, 1, D] @ [B, D, N] -> [B, N]
        s_comp = torch.bmm(
            image_embeddings.unsqueeze(1),  # [B, 1, D]
            component_captions.transpose(1, 2)  # [B, D, N]
        ).squeeze(1)  # [B, N]
        
        # Compute margin violations: max(0, margin + s_comp - s_full)
        # We want s_full > s_comp + margin, so violation = margin + s_comp - s_full
        s_full_expanded = s_full.unsqueeze(1)  # [B, 1]
        violations = torch.clamp(margin + s_comp - s_full_expanded, min=0.0)  # [B, N]
        
        # Apply component validity mask if provided
        if component_valid_mask is not None:
            component_valid_mask = component_valid_mask.to(device)
            # Mask out invalid components (set violations to 0)
            violations = violations * component_valid_mask.float()
            valid_counts = component_valid_mask.sum(dim=1).clamp(min=1)  # [B]
        else:
            valid_counts = torch.full((B,), N, device=device, dtype=torch.float32)
        
        # Reduce across components per sample
        if reduction == "max":
            # Only penalize the worst violating component per sample
            per_sample_loss = violations.max(dim=1)[0]  # [B]
        else:  # "mean"
            # Average violation across components
            per_sample_loss = violations.sum(dim=1) / valid_counts  # [B]
        
        # Average across batch
        loss = per_sample_loss.mean()
        
        # Compute statistics for logging
        # Violation rate: fraction of pairs where s_comp >= s_full (ignoring margin)
        actual_violations = (s_comp >= s_full_expanded)  # [B, N]
        if component_valid_mask is not None:
            actual_violations = actual_violations & component_valid_mask
            total_valid = component_valid_mask.sum().item()
        else:
            total_valid = B * N
        
        violation_rate = actual_violations.sum().item() / max(total_valid, 1)
        
        # Average margin: mean(s_full - s_comp) across valid pairs
        gaps = s_full_expanded - s_comp  # [B, N]
        if component_valid_mask is not None:
            gaps = gaps * component_valid_mask.float()
            avg_margin = gaps.sum().item() / max(total_valid, 1)
        else:
            avg_margin = gaps.mean().item()
    
    return {
        'loss': loss,
        'violation_rate': violation_rate,
        'avg_margin': avg_margin,
    }


def _compute_component_losses(
    image_embeddings: torch.Tensor,
    component_captions: torch.Tensor,
    neg_text_embeddings: Optional[torch.Tensor],
    temperature: torch.Tensor,
    device: torch.device,
    component_loss_type: str = "negclip",
    components_per_caption: Optional[torch.Tensor] = None,
    num_components_available: Optional[torch.Tensor] = None,
    component_valid_mask: Optional[torch.Tensor] = None,
    alignment_margin: float = 0.1,
    hybrid_margin_weight: float = 0.3,
) -> Tuple[torch.Tensor, float]:
    """
    Compute loss for component captions.
    
    Supports five loss types:
    - "clip": Standard CLIP loss (no negatives)
    - "negclip": NegCLIP-style loss (with negatives) [default]
    - "alignment": Direct embedding alignment (cosine or L2, no negatives)
    - "alignment_margin": Pairwise margin loss between pos/neg for each component
    - "hybrid": Combines negclip + alignment_margin with configurable weighting
               L = (1 - w) * L_negclip + w * L_margin
    
    Args:
        image_embeddings: [B, D] normalized image embeddings
        component_captions: [B, N, D] normalized component embeddings
        neg_text_embeddings: [B, N, D] or [B, D] - paired negative embeddings per component
            - If [B, N, D]: use neg_text_embeddings[:, k, :] for component k
            - If [B, D]: use as shared negative (legacy)
        temperature: scalar temperature
        device: torch device
        component_loss_type: Type of loss ("clip", "negclip", "alignment", "alignment_margin", or "hybrid")
        components_per_caption: [B, N] binary mask indicating which components are present
        num_components_available: [B] total components available per sample
        component_valid_mask: [B, N] validity mask for each component (True = valid pair)
        alignment_margin: Margin for alignment_margin and hybrid loss (default: 0.1)
        hybrid_margin_weight: Weight for margin loss in hybrid mode (default: 0.3)
            0.0 = pure negclip, 1.0 = pure alignment_margin
    
    Returns:
        Tuple of (component_loss, component_accuracy)
    """
    num_components = component_captions.shape[1]
    
    # Check if we should compute component losses
    # Skip if all samples have <= 1 component available
    if num_components_available is not None:
        valid_component_mask = (num_components_available > 1).to(device=device)
        num_valid_samples = int(valid_component_mask.sum().item())
        if num_valid_samples == 0:
            return torch.tensor(0.0, device=device), 0.0
    else:
        # If not provided, assume multiple available by default
        valid_component_mask = torch.ones((image_embeddings.shape[0],), dtype=torch.bool, device=device)
    
    # Compute losses based on component_loss_type
    if component_loss_type == "alignment":
        # Direct embedding alignment: L = 1 - cos(f(I), g(t^k))
        # For alignment loss, don't use component availability masks - compute for all samples

        component_loss_avg, component_accuracy = _compute_alignment_loss_for_components(
            image_embeddings=image_embeddings,
            component_captions=component_captions,
            valid_component_mask=None,  # Don't mask for alignment
            num_components=num_components,
            components_per_caption=components_per_caption,
            loss_type="cosine",  # Use cosine distance by default
        )
    
    elif component_loss_type == "alignment_margin":
        # Pairwise margin loss: L = max(0, margin - s_pos + s_neg)
        # where s_pos = cos(f(I), g(t_k+)) and s_neg = cos(f(I), g(t_k-))
        # This enforces s_pos >= s_neg + margin for each (image, component) pair
        
        # Check that we have paired negatives in the correct format
        has_paired_negatives = (
            neg_text_embeddings is not None 
            and neg_text_embeddings.dim() == 3 
            and neg_text_embeddings.shape[1] == num_components
        )
        
        if not has_paired_negatives:
            # Fallback: if we don't have paired negatives (e.g., during evaluation),
            # use pure alignment loss instead of raising an error
            # This allows training with alignment_margin but evaluating with legacy data
            component_loss_avg, component_accuracy = _compute_alignment_loss_for_components(
                image_embeddings=image_embeddings,
                component_captions=component_captions,
                valid_component_mask=None,
                num_components=num_components,
                components_per_caption=components_per_caption,
                loss_type="cosine",
            )
        else:
            component_loss_avg, component_accuracy = _compute_alignment_margin_loss_for_components(
                image_embeddings=image_embeddings,
                component_captions=component_captions,
                neg_text_embeddings=neg_text_embeddings,
                num_components=num_components,
                components_per_caption=components_per_caption,
                component_valid_mask=component_valid_mask,
                margin=alignment_margin,
            )
    
    elif component_loss_type == "negclip_hard":
        # NegCLIP with per-sample hard negatives only (not batch negatives)
        # Each sample's denominator has B+1 scores: batch positives + 1 hard negative
        # Unlike "negclip" which has B+B scores (batch positives + batch negatives)
        
        # Check that we have paired negatives in the correct format
        has_paired_negatives = (
            neg_text_embeddings is not None 
            and neg_text_embeddings.dim() == 3 
            and neg_text_embeddings.shape[1] == num_components
        )
        
        if not has_paired_negatives:
            # Fallback to standard CLIP if no paired negatives
            component_loss_avg, component_accuracy = _compute_alignment_loss_for_components(
                image_embeddings=image_embeddings,
                component_captions=component_captions,
                valid_component_mask=None,
                num_components=num_components,
                components_per_caption=components_per_caption,
                loss_type="cosine",
            )
        else:
            component_loss_avg, component_accuracy = _compute_negclip_hard_loss_for_components(
                image_embeddings=image_embeddings,
                component_captions=component_captions,
                neg_text_embeddings=neg_text_embeddings,
                temperature=temperature,
                device=device,
                num_components=num_components,
                components_per_caption=components_per_caption,
                component_valid_mask=component_valid_mask,
            )
    
    elif component_loss_type == "hybrid":
        # Hybrid loss: combines NegCLIP (batch contrastive) with alignment_margin (pairwise margin)
        # L_hybrid = (1 - hybrid_margin_weight) * L_negclip + hybrid_margin_weight * L_margin
        #
        # Benefits:
        # - NegCLIP provides batch-level discrimination (good for general retrieval)
        # - Margin loss provides fine-grained pairwise separation (good for compositionality)
        # - Weighted combination balances both objectives
        
        # Check that we have paired negatives in the correct format
        has_paired_negatives = (
            neg_text_embeddings is not None 
            and neg_text_embeddings.dim() == 3 
            and neg_text_embeddings.shape[1] == num_components
        )
        
        if not has_paired_negatives:
            # Fallback to pure alignment loss if no paired negatives
            component_loss_avg, component_accuracy = _compute_alignment_loss_for_components(
                image_embeddings=image_embeddings,
                component_captions=component_captions,
                valid_component_mask=None,
                num_components=num_components,
                components_per_caption=components_per_caption,
                loss_type="cosine",
            )
        else:
            # Compute NegCLIP loss (batch contrastive with negatives)
            negclip_results = []
            for k in range(num_components):
                comp_k = component_captions[:, k, :]  # [B, D]
                comp_neg_k = neg_text_embeddings[:, k, :]  # [B, D]
                
                # Build per-component valid mask
                if components_per_caption is not None:
                    per_comp_mask = (components_per_caption[:, k] > 0).to(device=device)
                    valid_mask = (valid_component_mask & per_comp_mask)
                else:
                    valid_mask = valid_component_mask
                
                # Apply component_valid_mask if provided
                if component_valid_mask is not None:
                    comp_valid_k = component_valid_mask[:, k].to(device=device)
                    valid_mask = valid_mask & comp_valid_k
                
                comp_result = _compute_clip_loss_for_caption(
                    image_embeddings, comp_k, comp_neg_k,
                    temperature, device, valid_mask=valid_mask
                )
                negclip_results.append(comp_result)
            
            negclip_losses = torch.stack([r['loss'] for r in negclip_results])
            negclip_loss = negclip_losses.mean()
            negclip_accuracies = [r['accuracy'] for r in negclip_results]
            negclip_accuracy = sum(negclip_accuracies) / len(negclip_accuracies)
            
            # Compute alignment margin loss (pairwise hinge)
            margin_loss, margin_accuracy = _compute_alignment_margin_loss_for_components(
                image_embeddings=image_embeddings,
                component_captions=component_captions,
                neg_text_embeddings=neg_text_embeddings,
                num_components=num_components,
                components_per_caption=components_per_caption,
                component_valid_mask=component_valid_mask,
                margin=alignment_margin,
            )
            
            # Combine losses with weighting
            # hybrid_margin_weight=0.0 -> pure negclip
            # hybrid_margin_weight=1.0 -> pure alignment_margin
            # hybrid_margin_weight=0.3 -> 70% negclip + 30% margin (default)
            component_loss_avg = (1.0 - hybrid_margin_weight) * negclip_loss + hybrid_margin_weight * margin_loss
            
            # Use negclip accuracy as the primary metric (since margin accuracy is binary correct/incorrect)
            component_accuracy = negclip_accuracy
        
    else:
        # CLIP or NegCLIP loss for components
        # Determine whether to use negatives based on component_loss_type
        use_negatives_for_components = (component_loss_type == "negclip")
        
        # Check neg_text_embeddings format: [B, N, D] (paired) or [B, D] (shared)
        has_paired_negatives = (
            neg_text_embeddings is not None 
            and neg_text_embeddings.dim() == 3 
            and neg_text_embeddings.shape[1] == num_components
        )
        
        component_results = []
        for k in range(num_components):
            comp_k = component_captions[:, k, :]  # [B, D]
            
            # Get negative for this component
            if use_negatives_for_components and neg_text_embeddings is not None:
                if has_paired_negatives:
                    # Per-component paired negative [B, N, D] -> [B, D]
                    comp_neg_k = neg_text_embeddings[:, k, :]
                else:
                    # Shared negative [B, D]
                    comp_neg_k = neg_text_embeddings
            else:
                comp_neg_k = None
            
            # Build per-component valid mask combining:
            # 1. Sample has num_components_available > 1 (from valid_component_mask)
            # 2. This specific component is present: components_per_caption[i, k] > 0
            # 3. This component-negative pair is valid: component_valid_mask[i, k]
            if components_per_caption is not None:
                per_comp_mask = (components_per_caption[:, k] > 0).to(device=device)
                valid_mask = (valid_component_mask & per_comp_mask)
            else:
                valid_mask = valid_component_mask
            
            # Also apply caption_valid_mask for this component if provided
            if component_valid_mask is not None:
                comp_valid_k = component_valid_mask[:, k].to(device=device)
                valid_mask = valid_mask & comp_valid_k

            comp_result = _compute_clip_loss_for_caption(
                image_embeddings, comp_k, comp_neg_k,
                temperature, device, valid_mask=valid_mask
            )
            component_results.append(comp_result)
        
        # Average losses and accuracies
        component_losses = torch.stack([r['loss'] for r in component_results])
        component_loss_avg = component_losses.mean()
        
        component_accuracies = [r['accuracy'] for r in component_results]
        component_accuracy = sum(component_accuracies) / len(component_accuracies)
    
    return component_loss_avg, component_accuracy


def _compute_alignment_loss_for_components(
    image_embeddings: torch.Tensor,
    component_captions: torch.Tensor,
    valid_component_mask: Optional[torch.Tensor],
    num_components: int,
    components_per_caption: Optional[torch.Tensor] = None,
    loss_type: str = "cosine",
) -> Tuple[torch.Tensor, float]:
    """
    Compute direct alignment loss between image embeddings and component captions.
    
    L_comp-align(I, t^k) = ||f(I) - g(t^k)||^2  or  1 - cos(f(I), g(t^k))
    
    Instead of contrastive learning, this directly minimizes the distance between
    image and component embeddings.
    
    Args:
        image_embeddings: [B, D] normalized image embeddings
        component_captions: [B, N, D] normalized component embeddings
        valid_component_mask: [B] boolean mask for samples with valid components
        num_components: N - number of component captions
        components_per_caption: [B, 1+N] or [B, N] counts - component presence info
        loss_type: "cosine" (1 - cos_sim) or "mse" (L2 distance)
    
    Returns:
        Tuple of (average_loss, average_accuracy)
        - average_loss: scalar tensor, mean alignment loss across all components
        - average_accuracy: float, placeholder accuracy (0.0 for alignment loss)
    """
    device = image_embeddings.device
    B = image_embeddings.shape[0]
    
    # If no components, return zero loss
    if num_components == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    
    # Force float32 for numerical stability (prevents CUBLAS errors with AMP)
    image_embeddings = image_embeddings.float()
    component_captions = component_captions.float()
    
    # Check for NaN/Inf in inputs
    if torch.isnan(image_embeddings).any() or torch.isinf(image_embeddings).any():
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    if torch.isnan(component_captions).any() or torch.isinf(component_captions).any():
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    
    # ===== VECTORIZED COMPUTATION =====
    # image_embeddings: [B, D] -> [B, 1, D] for broadcasting
    # component_captions: [B, N, D]
    img_expanded = image_embeddings.unsqueeze(1)  # [B, 1, D]
    
    if loss_type == "cosine":
        # Cosine similarity: [B, 1, D] * [B, N, D] -> sum over D -> [B, N]
        cos_sim = (img_expanded * component_captions).sum(dim=-1)  # [B, N]
        # Clamp for numerical stability
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        per_sample_loss = 1.0 - cos_sim  # [B, N]
        
    elif loss_type == "mse":
        # L2 distance: ||image - component||^2
        diff = img_expanded - component_captions  # [B, N, D]
        per_sample_loss = (diff ** 2).sum(dim=-1)  # [B, N]
        
    else:
        raise ValueError(f"Unknown loss_type for alignment: {loss_type}")
    
    # ===== BUILD VALIDITY MASK =====
    # components_per_caption may be [B, 1+N] (includes full caption) or [B, N] (only components)
    if components_per_caption is not None:
        if components_per_caption.shape[1] == num_components + 1:
            # Slice to exclude full caption: [B, 1+N] -> [B, N]
            comp_mask = components_per_caption[:, 1:]
        else:
            comp_mask = components_per_caption
        comp_mask = (comp_mask > 0).to(device=device, dtype=torch.float)  # [B, N]
    else:
        comp_mask = torch.ones((B, num_components), device=device, dtype=torch.float)
    
    # Also apply valid_component_mask if provided (per-sample mask)
    if valid_component_mask is not None:
        # valid_component_mask is [B], broadcast to [B, N]
        sample_mask = valid_component_mask.float().unsqueeze(1)  # [B, 1]
        comp_mask = comp_mask * sample_mask  # [B, N]
    
    # ===== COMPUTE MASKED MEAN =====
    num_valid = comp_mask.sum()
    if num_valid > 0:
        avg_loss = (per_sample_loss * comp_mask).sum() / num_valid
    else:
        avg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Alignment loss doesn't have a natural accuracy metric
    return avg_loss, 0.0


def _compute_alignment_margin_loss_for_components(
    image_embeddings: torch.Tensor,
    component_captions: torch.Tensor,
    neg_text_embeddings: torch.Tensor,
    num_components: int,
    components_per_caption: Optional[torch.Tensor] = None,
    component_valid_mask: Optional[torch.Tensor] = None,
    margin: float = 0.1,
) -> Tuple[torch.Tensor, float]:
    """
    Compute pairwise margin-based alignment loss for component caption pairs.
    
    For each image I and component caption pair (t_k+, t_k-):
        s_pos = cos(f(I), g(t_k+))   # positive similarity
        s_neg = cos(f(I), g(t_k-))   # negative similarity
        
    We want: s_pos >= s_neg + margin
    
    Hinge loss: L = max(0, margin - s_pos + s_neg)
    
    This is a pairwise, local loss (not in-batch contrastive) that enforces
    the image to be closer to the positive component than the negative component
    by at least `margin`.
    
    Args:
        image_embeddings: [B, D] normalized image embeddings
        component_captions: [B, N, D] normalized positive component embeddings
        neg_text_embeddings: [B, N, D] normalized paired negative embeddings
        num_components: N - number of component captions
        components_per_caption: [B, 1+N] or [B, N] component presence info
        component_valid_mask: [B, N] validity mask for each component pair (True = valid)
        margin: Margin value m > 0 (default: 0.1)
    
    Returns:
        Tuple of (average_loss, accuracy)
        - average_loss: scalar tensor, mean margin loss across all valid pairs
        - accuracy: float, fraction of pairs where s_pos > s_neg (no margin)
    """
    device = image_embeddings.device
    B = image_embeddings.shape[0]
    
    # If no components, return zero loss
    if num_components == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    
    # Disable autocast and force float32 for numerical stability (prevents CUBLAS errors with AMP)
    with torch.amp.autocast('cuda', enabled=False):
        image_embeddings = image_embeddings.float()
        component_captions = component_captions.float()
        neg_text_embeddings = neg_text_embeddings.float()
        
        # Check for NaN/Inf in inputs
        if (torch.isnan(image_embeddings).any() or torch.isinf(image_embeddings).any() or
            torch.isnan(component_captions).any() or torch.isinf(component_captions).any() or
            torch.isnan(neg_text_embeddings).any() or torch.isinf(neg_text_embeddings).any()):
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0
        
        # ===== VECTORIZED COMPUTATION =====
        # image_embeddings: [B, D] -> [B, 1, D] for broadcasting
        img_expanded = image_embeddings.unsqueeze(1)  # [B, 1, D]
        
        # Cosine similarities (embeddings are normalized)
        s_pos = (img_expanded * component_captions).sum(dim=-1)  # [B, N]
        s_neg = (img_expanded * neg_text_embeddings).sum(dim=-1)  # [B, N]
        
        # Clamp for numerical stability
        s_pos = torch.clamp(s_pos, -1.0, 1.0)
        s_neg = torch.clamp(s_neg, -1.0, 1.0)
        
        # Hinge loss: max(0, margin - s_pos + s_neg)
        margin_violation = margin - s_pos + s_neg  # [B, N]
        hinge_loss = F.relu(margin_violation)  # [B, N]
        
        # ===== BUILD VALIDITY MASK =====
        # components_per_caption may be [B, 1+N] (includes full caption) or [B, N]
        if components_per_caption is not None and component_valid_mask is not None:
            if components_per_caption.shape[1] == num_components + 1:
                comp_per_cap = components_per_caption[:, 1:]
            else:
                comp_per_cap = components_per_caption
            valid_mask = (comp_per_cap > 0) & component_valid_mask  # [B, N]
        elif components_per_caption is not None:
            if components_per_caption.shape[1] == num_components + 1:
                comp_per_cap = components_per_caption[:, 1:]
            else:
                comp_per_cap = components_per_caption
            valid_mask = (comp_per_cap > 0)  # [B, N]
        elif component_valid_mask is not None:
            valid_mask = component_valid_mask  # [B, N]
        else:
            valid_mask = torch.ones((B, num_components), dtype=torch.bool, device=device)
        
        valid_mask = valid_mask.to(device=device, dtype=torch.float32)  # [B, N] - explicit float32
        
        # ===== COMPUTE MASKED MEAN =====
        num_valid = valid_mask.sum()
        if num_valid > 0:
            avg_loss = (hinge_loss * valid_mask).sum() / num_valid
        else:
            avg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute accuracy: fraction of valid pairs where s_pos > s_neg
        correct = (s_pos > s_neg).float()  # [B, N]
        if num_valid > 0:
            accuracy = ((correct * valid_mask).sum() / num_valid).item()
        else:
            accuracy = 0.0
    
    return avg_loss, accuracy


def _compute_text_contrastive_margin_loss_for_components(
    full_caption_embeddings: torch.Tensor,
    component_captions: torch.Tensor,
    neg_text_embeddings: torch.Tensor,
    num_components: int,
    components_per_caption: Optional[torch.Tensor] = None,
    component_valid_mask: Optional[torch.Tensor] = None,
    margin: float = 0.1,
) -> Tuple[torch.Tensor, float]:
    """
    Compute text-space contrastive margin loss between positive and negative components.
    
    This loss enforces a margin between the similarity of the full caption to its
    positive component versus its negative component:
    
        s_pos = cos(t_full, t_k+)   # full caption to positive component
        s_neg = cos(t_full, t_k-)   # full caption to negative component
        
    We want: s_pos >= s_neg + margin
    
    Hinge loss: L = max(0, margin - s_pos + s_neg)
    
    Intuition:
    - The full caption "a cat on a red mat" should be more similar to 
      its true component "red mat" than to the negative "blue mat"
    - This creates text-space structure that helps with compositionality
    - Unlike image-text alignment, this operates purely in text embedding space
    
    Use case:
    - When you want to regularize the text encoder to maintain semantic structure
    - Helps prevent the encoder from collapsing positive/negative components
    - Complementary to image-text alignment losses
    
    Args:
        full_caption_embeddings: [B, D] normalized full caption embeddings
        component_captions: [B, N, D] normalized positive component embeddings
        neg_text_embeddings: [B, N, D] normalized paired negative embeddings
        num_components: N - number of component captions
        components_per_caption: [B, 1+N] or [B, N] component presence info
        component_valid_mask: [B, N] validity mask for each component pair (True = valid)
        margin: Margin value m > 0 (default: 0.1)
    
    Returns:
        Tuple of (average_loss, accuracy)
        - average_loss: scalar tensor, mean margin loss across all valid pairs
        - accuracy: float, fraction of pairs where s_pos > s_neg (no margin)
    """
    device = full_caption_embeddings.device
    B = full_caption_embeddings.shape[0]
    
    # If no components, return zero loss
    if num_components == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    
    # Disable autocast and force float32 for numerical stability
    with torch.amp.autocast('cuda', enabled=False):
        full_caption_embeddings = full_caption_embeddings.float()
        component_captions = component_captions.float()
        neg_text_embeddings = neg_text_embeddings.float()
        
        # Check for NaN/Inf in inputs
        if (torch.isnan(full_caption_embeddings).any() or torch.isinf(full_caption_embeddings).any() or
            torch.isnan(component_captions).any() or torch.isinf(component_captions).any() or
            torch.isnan(neg_text_embeddings).any() or torch.isinf(neg_text_embeddings).any()):
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0
        
        # ===== VECTORIZED COMPUTATION =====
        # full_caption_embeddings: [B, D] -> [B, 1, D] for broadcasting
        text_expanded = full_caption_embeddings.unsqueeze(1)  # [B, 1, D]
        
        # Cosine similarities in TEXT SPACE (embeddings are normalized)
        # s_pos[i, k] = cos(t_full_i, t_k+_i)
        s_pos = (text_expanded * component_captions).sum(dim=-1)  # [B, N]
        # s_neg[i, k] = cos(t_full_i, t_k-_i)
        s_neg = (text_expanded * neg_text_embeddings).sum(dim=-1)  # [B, N]
        
        # Clamp for numerical stability
        s_pos = torch.clamp(s_pos, -1.0, 1.0)
        s_neg = torch.clamp(s_neg, -1.0, 1.0)
        
        # Hinge loss: max(0, margin - s_pos + s_neg)
        # We want s_pos >= s_neg + margin
        margin_violation = margin - s_pos + s_neg  # [B, N]
        hinge_loss = F.relu(margin_violation)  # [B, N]
        
        # ===== BUILD VALIDITY MASK =====
        if components_per_caption is not None and component_valid_mask is not None:
            if components_per_caption.shape[1] == num_components + 1:
                comp_per_cap = components_per_caption[:, 1:]
            else:
                comp_per_cap = components_per_caption
            valid_mask = (comp_per_cap > 0) & component_valid_mask  # [B, N]
        elif components_per_caption is not None:
            if components_per_caption.shape[1] == num_components + 1:
                comp_per_cap = components_per_caption[:, 1:]
            else:
                comp_per_cap = components_per_caption
            valid_mask = (comp_per_cap > 0)  # [B, N]
        elif component_valid_mask is not None:
            valid_mask = component_valid_mask  # [B, N]
        else:
            valid_mask = torch.ones((B, num_components), dtype=torch.bool, device=device)
        
        valid_mask = valid_mask.to(device=device, dtype=torch.float32)
        
        # ===== COMPUTE MASKED MEAN =====
        num_valid = valid_mask.sum()
        if num_valid > 0:
            avg_loss = (hinge_loss * valid_mask).sum() / num_valid
        else:
            avg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute accuracy: fraction of valid pairs where s_pos > s_neg
        correct = (s_pos > s_neg).float()  # [B, N]
        if num_valid > 0:
            accuracy = ((correct * valid_mask).sum() / num_valid).item()
        else:
            accuracy = 0.0
    
    return avg_loss, accuracy


def _compute_negclip_hard_loss_for_components(
    image_embeddings: torch.Tensor,
    component_captions: torch.Tensor,
    neg_text_embeddings: torch.Tensor,
    temperature: torch.Tensor,
    device: torch.device,
    num_components: int,
    components_per_caption: Optional[torch.Tensor] = None,
    component_valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Compute NegCLIP-style loss with per-sample hard negatives (not batch negatives).
    
    Unlike standard NegCLIP which adds all B negatives to the denominator (B+B scores),
    this variant adds only the sample's own hard negative (+1 score per sample).
    
    For each image I_i and component k:
        Numerator:   exp(s(I_i, t_k+) / τ)
        Denominator: sum_j exp(s(I_i, t_j+) / τ) + exp(s(I_i, t_k-) / τ)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^
                     B batch positives (standard CLIP)   +1 hard negative
    
    This creates a loss where the model learns to:
    1. Match images to their positive components (standard contrastive)
    2. Distinguish positives from their specific hard negatives
    
    The key difference from "negclip":
    - "negclip": denominator = B positives + B negatives (batch hard negatives)
    - "negclip_hard": denominator = B positives + 1 per-sample hard negative
    
    Args:
        image_embeddings: [B, D] normalized image embeddings
        component_captions: [B, N, D] normalized positive component embeddings
        neg_text_embeddings: [B, N, D] normalized paired negative embeddings
        temperature: scalar temperature for scaling
        device: torch device
        num_components: N - number of component captions
        components_per_caption: [B, 1+N] or [B, N] component presence info
        component_valid_mask: [B, N] validity mask for each component pair (True = valid)
    
    Returns:
        Tuple of (average_loss, accuracy)
        - average_loss: scalar tensor, mean loss across all valid components
        - accuracy: float, fraction of samples where positive ranks highest
    """
    B = image_embeddings.shape[0]
    
    # If no components, return zero loss
    if num_components == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    
    # Disable autocast and force float32 for numerical stability
    with torch.amp.autocast('cuda', enabled=False):
        image_embeddings = image_embeddings.float()
        component_captions = component_captions.float()
        neg_text_embeddings = neg_text_embeddings.float()
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.float()
        
        # Check for NaN/Inf in inputs
        if (torch.isnan(image_embeddings).any() or torch.isinf(image_embeddings).any() or
            torch.isnan(component_captions).any() or torch.isinf(component_captions).any() or
            torch.isnan(neg_text_embeddings).any() or torch.isinf(neg_text_embeddings).any()):
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0
        
        # Process each component separately
        component_losses = []
        component_accs = []
        
        for k in range(num_components):
            # Extract component k's positive and negative embeddings
            comp_k_pos = component_captions[:, k, :]  # [B, D]
            comp_k_neg = neg_text_embeddings[:, k, :]  # [B, D]
            
            # Build validity mask for this component
            if components_per_caption is not None and component_valid_mask is not None:
                if components_per_caption.shape[1] == num_components + 1:
                    comp_per_cap = components_per_caption[:, k + 1]  # Skip full caption column
                else:
                    comp_per_cap = components_per_caption[:, k]
                valid_mask = (comp_per_cap > 0) & component_valid_mask[:, k]  # [B]
            elif components_per_caption is not None:
                if components_per_caption.shape[1] == num_components + 1:
                    comp_per_cap = components_per_caption[:, k + 1]
                else:
                    comp_per_cap = components_per_caption[:, k]
                valid_mask = (comp_per_cap > 0)  # [B]
            elif component_valid_mask is not None:
                valid_mask = component_valid_mask[:, k]  # [B]
            else:
                valid_mask = torch.ones(B, dtype=torch.bool, device=device)
            
            num_valid = int(valid_mask.sum().item())
            if num_valid == 0:
                component_losses.append(torch.tensor(0.0, device=device))
                component_accs.append(0.0)
                continue
            
            # Extract valid samples
            valid_images = image_embeddings[valid_mask]  # [num_valid, D]
            valid_pos = comp_k_pos[valid_mask]  # [num_valid, D]
            valid_neg = comp_k_neg[valid_mask]  # [num_valid, D]
            
            # Compute i2t logits: [num_valid, num_valid + 1]
            # Columns 0..num_valid-1: similarities to all batch positives
            # Column num_valid+i: similarity to sample i's hard negative (diagonal pattern)
            
            # Standard batch logits: valid_images @ valid_pos.T
            batch_logits = (valid_images @ valid_pos.T) / temperature  # [num_valid, num_valid]
            
            # Per-sample hard negative similarity: s(I_i, t_k-_i)
            # This is a diagonal: each sample only sees its own hard negative
            hard_neg_sims = (valid_images * valid_neg).sum(dim=-1, keepdim=True) / temperature  # [num_valid, 1]
            
            # Append hard negative to logits for each sample
            # For sample i: logits_i = [s(I_i, t_0+), ..., s(I_i, t_B-1+), s(I_i, t_i-)]
            # The positive is at position i, hard negative is at position num_valid
            logits_i2t = torch.cat([batch_logits, hard_neg_sims], dim=1)  # [num_valid, num_valid + 1]
            
            # Labels: each sample's positive is at its own index
            labels = torch.arange(num_valid, device=device, dtype=torch.long)
            
            # Cross-entropy loss
            loss_i2t = F.cross_entropy(logits_i2t, labels)
            
            # Accuracy: check if positive (index i) has highest score
            acc_i2t = (torch.argmax(logits_i2t, dim=1) == labels).float().mean()
            
            # t2i direction: text to image matching
            # logits_t2i: valid_pos @ valid_images.T (no hard negative needed for t2i)
            logits_t2i = (valid_pos @ valid_images.T) / temperature  # [num_valid, num_valid]
            loss_t2i = F.cross_entropy(logits_t2i, labels)
            acc_t2i = (torch.argmax(logits_t2i, dim=1) == labels).float().mean()
            
            # Average i2t and t2i
            loss_k = (loss_i2t + loss_t2i) / 2
            acc_k = ((acc_i2t + acc_t2i) / 2).item()
            
            component_losses.append(loss_k)
            component_accs.append(acc_k)
        
        # Average across components
        if len(component_losses) > 0:
            avg_loss = torch.stack(component_losses).mean()
            avg_acc = sum(component_accs) / len(component_accs)
        else:
            avg_loss = torch.tensor(0.0, device=device, requires_grad=True)
            avg_acc = 0.0
    
    return avg_loss, avg_acc

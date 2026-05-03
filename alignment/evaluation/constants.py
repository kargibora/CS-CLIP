"""Constants for evaluation metrics."""


class MetricKeys:
    """Standard metric key names."""
    VAL_LOSS = 'val_loss'
    VAL_ACCURACY = 'val_accuracy'
    CONTRASTIVE_ACCURACY = 'contrastive_accuracy'
    POS_GREATER_THAN_NEG = 'pos_greater_than_neg'
    POS_GREATER_THAN_ALL_NEGS = 'pos_greater_than_all_negs'
    NEG_TEXT_SIMILARITY = 'neg_text_similarity'
    NEG_TEXT_POS_IMAGE_SIMILARITY = 'neg_text_pos_image_similarity'
    POS_SIMILARITY = 'pos_similarity'
    PER_NEG_ACCURACY = 'per_neg_accuracy'
    PER_NEG_TEXT_SIMILARITY = 'per_neg_text_similarity'
    PER_NEG_TEXT_POS_IMAGE_SIMILARITY = 'per_neg_text_pos_image_similarity'
    POS_NEG_SIMILARITY_GAP = 'pos_neg_similarity_gap'
    
    # Component-based metrics (multi-caption mode)
    COMPONENT_IMAGE_SIMILARITY = 'component_image_similarity'
    COMPONENT_FULL_CAPTION_SIMILARITY = 'component_full_caption_similarity'
    NEG_FULL_CAPTION_SIMILARITY = 'neg_full_caption_similarity'
    COMPONENT_NEG_SIMILARITY = 'component_neg_similarity'
    FULL_VS_COMPONENT_ACCURACY = 'full_vs_component_accuracy'
    FULL_VS_NEG_ACCURACY = 'full_vs_neg_accuracy'
    

class ModelType:
    """Model type identifiers."""
    FLEXIBLE_CLIP_MULTILAYER = "FlexibleCLIPMultiLayerAlignment"
    CLIP_MULTILAYER_FT = "CLIPMultiLayerFTAlignment"

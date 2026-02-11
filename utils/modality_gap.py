import torch

def get_modality_gap_and_variance(image_embeddings, text_embeddings):
    """
    Returns mean and variance of L2 distances between image and text embeddings.
    """
    distances = torch.norm(image_embeddings - text_embeddings, dim=1)
    return distances.mean(), distances.var()

def get_modality_gap_per_dimension_and_variance(image_embeddings, text_embeddings):
    """
    Returns mean and variance of the absolute difference per embedding dimension.
    """
    gap = (image_embeddings - text_embeddings).abs()
    return gap.mean(dim=0), gap.var(dim=0)

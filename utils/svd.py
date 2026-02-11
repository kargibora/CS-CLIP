import torch
from typing import Literal, Tuple

def get_alignment_svd(
    module: torch.nn.Module, 
    head: Literal['text', 'image'] = 'text'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the SVD (U, S, Vh) of the alignment weight matrix from the specified head.
    
    Args:
        module: The alignment module (CLIPAlignment or CLIPFTAlignment)
        head: Which alignment head to analyze ('text' or 'image')
        
    Returns:
        U, S, Vh: SVD of the alignment weight matrix (W = U @ diag(S) @ Vh)
    """
    # Extract the alignment linear layer depending on head
    mlp = getattr(module, f"{head}_mlp", None)
    if mlp is None:
        raise ValueError(f"Module does not have a {head}_mlp")
    if not isinstance(mlp, torch.nn.Sequential) or len(mlp) != 1:
        raise ValueError("MLP must be a single Linear layer (mlp_layers=1)")
    linear = mlp[0]
    if not isinstance(linear, torch.nn.Linear):
        raise ValueError("Layer is not Linear")
    W = linear.weight.data  # shape: (out_dim, in_dim)
    # SVD: W = U @ diag(S) @ Vh
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return U, S, Vh
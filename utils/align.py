import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import clip
import os
import logging
from collections import defaultdict
from utils.dist import ddp_gather_embeddings_dd
import torch.distributed as dist



def compute_image_embeddings_intermediate(dataloader, model, device, intermediate_layer_names, dtype=torch.float32):
    """
    Computes normalized CLIP image embeddings for all images at specified layers.

    Returns:
        embeddings_dict: {layer_name: embeddings tensor (N, D)}
    """
    embeddings_dict = defaultdict(list)
    N_total = 0

    image_layer_names = [name for name in intermediate_layer_names if name != 'final']
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Extract images
            if 'image_options' in batch:
                images = batch['image_options']
            else:
                if len(batch) == 4:
                    images, captions, neg_captions, neg_captions_all = batch
                else:
                    images, captions, neg_captions = batch
            images = images.to(device)

            # Register hooks for this batch
            activations = {}
            handles = []
            for name in image_layer_names:
                layer = dict(model.visual.named_modules())[name]
                def get_hook(n):
                    return lambda m, i, o: activations.setdefault(n, o.detach())
                handles.append(layer.register_forward_hook(get_hook(name)))

            # Forward pass
            final_emb = model.encode_image(images).float()

            # Process intermediate activations
            for name in image_layer_names:
                feat = activations[name]

                if feat.dim() == 3:
                    feat = feat.permute(1, 0, 2)  # (B, seq_len, dim)
                    emb = feat[:, 0, :]  # CLS token
                else:
                    emb = feat
                emb = emb.to(dtype)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
                embeddings_dict[name].append(emb.cpu())

            # Final embedding
            final_emb = final_emb / (final_emb.norm(dim=-1, keepdim=True) + 1e-8)
            final_emb = final_emb.to(dtype)
            embeddings_dict['final'].append(final_emb.cpu())

            # Remove hooks
            for h in handles:
                h.remove()

            N_total += images.size(0)

    # Concatenate tensors per layer
    for name in embeddings_dict:
        embeddings_dict[name] = torch.cat(embeddings_dict[name], dim=0)

    return embeddings_dict

def compute_image_embeddings_intermediate_ddp(dataloader, model, device, intermediate_layer_names, dtype=torch.float32, distributed=False, sampler=None):
    """
    Computes normalized CLIP image embeddings for all images at specified layers.
    For DDP, only main process returns the full dict; others return None.
    """
    from collections import defaultdict

    embeddings_dict = defaultdict(list)
    image_layer_names = [name for name in intermediate_layer_names if name != 'final']
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if 'image_options' in batch:
                images = batch['image_options']
            else:
                if len(batch) == 4:
                    images, _, _, _ = batch
                else:
                    images, _, _ = batch
            images = images.to(device)

            # Register hooks
            activations = {}
            handles = []
            for name in image_layer_names:
                layer = dict(model.visual.named_modules())[name]
                def get_hook(n):
                    return lambda m, i, o: activations.setdefault(n, o.detach())
                handles.append(layer.register_forward_hook(get_hook(name)))
            final_emb = model.encode_image(images).float()
            for name in image_layer_names:
                feat = activations[name]
                if feat.dim() == 3:
                    feat = feat.permute(1, 0, 2)
                    emb = feat[:, 0, :]
                else:
                    emb = feat
                emb = emb.to(dtype)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
                embeddings_dict[name].append(emb.cpu())
            final_emb = final_emb / (final_emb.norm(dim=-1, keepdim=True) + 1e-8)
            final_emb = final_emb.to(dtype)
            embeddings_dict['final'].append(final_emb.cpu())
            for h in handles:
                h.remove()
    # Now stack each part
    for name in embeddings_dict:
        embeddings_dict[name] = torch.cat(embeddings_dict[name], dim=0)

    # ---- DDP gather across all processes
    if distributed and dist.is_initialized():
        # Each process only has local data; gather on all
        for name in embeddings_dict:
            # Make everything contiguous
            embeddings_dict[name] = embeddings_dict[name].contiguous()
        # Gather on all processes, then only keep on main process
        embeddings_dict_full = ddp_gather_embeddings_dd(embeddings_dict, sampler)
        if dist.get_rank() == 0:
            return embeddings_dict_full
        else:
            return None
    else:
        return embeddings_dict
    
def compute_caption_embeddings_intermediate(captions, model, device, intermediate_layer_names, batch_size, dtype=torch.float32):
    """
    Computes normalized CLIP caption (text) embeddings for all captions at specified layers.

    Args:
        captions: List of caption strings
        model: CLIP model
        device: torch device
        intermediate_layer_names: list of string names for transformer layers (e.g. ["resblocks.3", ...])
        batch_size: int

    Returns:
        embeddings_dict: {layer_name: embeddings tensor (N, D)}
    """
    embeddings_dict = defaultdict(list)
    N_total = 0
    eos_token_id = model.tokenizer.eos_token_id if hasattr(model, "tokenizer") else 49407  # OpenAI CLIP default

    text_layer_names = [name for name in intermediate_layer_names if name != 'final']
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size)):
            text_batch = captions[i:i+batch_size]
            text_tokens = clip.tokenize(text_batch, truncate=True).to(device)  # (B, seq_len)

            # Register hooks for this batch
            activations = {}
            handles = []
            for name in text_layer_names:
                layer = dict(model.transformer.named_modules())[name]
                def get_hook(n):
                    return lambda m, inp, out: activations.setdefault(n, out.detach())
                handles.append(layer.register_forward_hook(get_hook(name)))

            # Forward pass
            _ = model.encode_text(text_tokens)

            # Pool EOS token for each layer
            eos_positions = (text_tokens == eos_token_id).float()
            eos_index = eos_positions.argmax(dim=1)  # (B,)

            for name in text_layer_names:
                feat = activations[name]
                if feat.dim() == 3 and feat.shape[0] == text_tokens.shape[1]:  # (seq_len, batch, dim)
                    feat = feat.permute(1, 0, 2)  # (batch, seq_len, dim)
                # Now proceed to index with eos_index as before:
                eos_emb = feat[torch.arange(feat.size(0)), eos_index, :]
                eos_emb = eos_emb / eos_emb.norm(dim=-1, keepdim=True)
                eos_emb = eos_emb.to(dtype)
                embeddings_dict[name].append(eos_emb.cpu())
                activations.pop(name)

            # Optionally save final embedding too
            final_emb = model.encode_text(text_tokens).to(dtype)
            final_emb = final_emb / final_emb.norm(dim=-1, keepdim=True)
            embeddings_dict['final'].append(final_emb.cpu())

            # Remove hooks!
            for h in handles:
                h.remove()

            N_total += len(text_batch)

    # Concatenate per layer
    for name in embeddings_dict:
        embeddings_dict[name] = torch.cat(embeddings_dict[name], dim=0)  # (N_total, D)

    return embeddings_dict

def compute_caption_embeddings_intermediate_ddp(captions, model, device, intermediate_layer_names, batch_size, dtype=torch.float32, distributed=False, sampler=None):
    from collections import defaultdict
    import torch
    import torch.distributed as dist

    embeddings_dict = defaultdict(list)
    eos_token_id = model.tokenizer.eos_token_id if hasattr(model, "tokenizer") else 49407

    text_layer_names = [name for name in intermediate_layer_names if name != 'final']
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size)):
            text_batch = captions[i:i+batch_size]
            text_tokens = clip.tokenize(text_batch, truncate=True).to(device)
            activations = {}
            handles = []
            for name in text_layer_names:
                layer = dict(model.transformer.named_modules())[name]
                def get_hook(n):
                    return lambda m, inp, out: activations.setdefault(n, out.detach())
                handles.append(layer.register_forward_hook(get_hook(name)))
            _ = model.encode_text(text_tokens)
            eos_positions = (text_tokens == eos_token_id).float()
            eos_index = eos_positions.argmax(dim=1)
            for name in text_layer_names:
                feat = activations[name]
                if feat.dim() == 3 and feat.shape[0] == text_tokens.shape[1]:
                    feat = feat.permute(1, 0, 2)
                eos_emb = feat[torch.arange(feat.size(0)), eos_index, :]
                eos_emb = eos_emb / eos_emb.norm(dim=-1, keepdim=True)
                eos_emb = eos_emb.to(dtype)
                embeddings_dict[name].append(eos_emb.cpu())
                activations.pop(name)
            final_emb = model.encode_text(text_tokens).to(dtype)
            final_emb = final_emb / final_emb.norm(dim=-1, keepdim=True)
            embeddings_dict['final'].append(final_emb.cpu())
            for h in handles:
                h.remove()
    for name in embeddings_dict:
        embeddings_dict[name] = torch.cat(embeddings_dict[name], dim=0)

    # ---- DDP gather
    if distributed and dist.is_initialized():
        for name in embeddings_dict:
            embeddings_dict[name] = embeddings_dict[name].contiguous()
        embeddings_dict_full = ddp_gather_embeddings_dd(embeddings_dict, sampler)
        if dist.get_rank() == 0:
            return embeddings_dict_full
        else:
            return None
    else:
        return embeddings_dict
    
def compute_image_embeddings_intermediate_batch(images, model, device, intermediate_layer_names, dtype=torch.float32, return_patches=False):
    """
    Computes normalized CLIP image embeddings for a batch of images at specified layers.

    Args:
        images: torch.Tensor of shape (B, C, H, W)
        model: CLIP model
        device: torch.device
        intermediate_layer_names: list of string names for visual transformer layers
        return_patches: If True, also return patch embeddings (for TQA models)

    Returns:
        embeddings_dict: {layer_name: embeddings tensor (B, D)}
                        If return_patches=True, also includes 'patches': (B, num_patches, D)
    """
    embeddings_dict = {}
    images = images.to(device)

    image_layer_names = [name for name in intermediate_layer_names if name != 'final']
    
    with torch.no_grad():
        # Register hooks for the intermediate layers
        activations = {}
        handles = []
        
        # For TQA, we need patch embeddings from the final transformer layer
        # Hook the last resblock to get full sequence
        patch_activations = {}
        patch_handles = []
        
        try:
            for name in image_layer_names:
                layer = dict(model.visual.named_modules())[name]  # FIX: Proper dict lookup
                def get_hook(n):
                    return lambda m, i, o: activations.setdefault(n, o.detach().cpu())  # FIX: Move to CPU immediately
                handles.append(layer.register_forward_hook(get_hook(name)))
            
            # Hook for patches - get output from last resblock before ln_post
            if return_patches:
                # Find the last resblock to hook
                resblocks = model.visual.transformer.resblocks
                last_resblock = resblocks[-1]
                def patch_hook(m, i, o):
                    # o is [seq_len, batch, dim] for ViT
                    patch_activations['last_resblock'] = o.detach()
                patch_handles.append(last_resblock.register_forward_hook(patch_hook))

            # Forward pass
            final_emb = model.encode_image(images).float()

            # Process intermediate activations
            for name in image_layer_names:
                feat = activations[name]  # Already on CPU
                if feat.dim() == 3:
                    feat = feat.permute(1, 0, 2)  # (B, seq_len, dim)
                    emb = feat[:, 0, :]  # CLS token for ViT
                else:
                    emb = feat
                emb = emb.to(dtype)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
                embeddings_dict[name] = emb  # Already on CPU

            # Final embedding
            final_emb = final_emb / (final_emb.norm(dim=-1, keepdim=True) + 1e-8)
            final_emb = final_emb.to(dtype)
            embeddings_dict['final'] = final_emb.cpu()  # Move to CPU
            
            # Extract patch embeddings for TQA
            if return_patches and 'last_resblock' in patch_activations:
                patch_feat = patch_activations['last_resblock']  # [seq_len, B, D]
                patch_feat = patch_feat.permute(1, 0, 2)  # [B, seq_len, D]
                # Apply layer norm if available
                if hasattr(model.visual, 'ln_post'):
                    patch_feat = model.visual.ln_post(patch_feat.to(device))
                # Separate CLS and patches
                cls_token = patch_feat[:, 0, :]  # [B, D]
                patches = patch_feat[:, 1:, :]  # [B, num_patches, D]
                # Normalize patches
                patches = patches / (patches.norm(dim=-1, keepdim=True) + 1e-8)
                embeddings_dict['patches'] = patches.cpu().to(dtype)
                embeddings_dict['global_unprojected'] = cls_token.cpu().to(dtype)

        finally:
            # Remove hooks even if an exception occurs
            for h in handles:
                h.remove()
            for h in patch_handles:
                h.remove()
            # Clear activations to free memory
            activations.clear()
            patch_activations.clear()
            # Force GPU memory cleanup
            torch.cuda.empty_cache()

    return embeddings_dict

def compute_caption_embeddings_intermediate_batch(texts, model, device, intermediate_layer_names, dtype=torch.float32):
    """
    Computes normalized CLIP caption (text) embeddings for a batch of captions at specified layers.

    Args:
        texts: List of caption strings (batch)
        model: CLIP model
        device: torch device
        intermediate_layer_names: list of string names for transformer layers (e.g. ["resblocks.3", ...])

    Returns:
        embeddings_dict: {layer_name: embeddings tensor (B, D)}
    """
    embeddings_dict = {}
    eos_token_id = model.tokenizer.eos_token_id if hasattr(model, "tokenizer") else 49407  # OpenAI CLIP default
    text_layer_names = [name for name in intermediate_layer_names if name != 'final']
    
    with torch.no_grad():
        # Tokenize batch
        import clip  # Remove if using open_clip or another implementation
        text_tokens = clip.tokenize(texts, truncate=True).to(device)  # (B, seq_len)

        # Register hooks for this batch
        activations = {}
        handles = []
        try:
            for name in text_layer_names:
                layer = dict(model.transformer.named_modules())[name]
                def get_hook(n):
                    return lambda m, inp, out: activations.setdefault(n, out.detach().cpu())  # FIX: Move to CPU immediately
                handles.append(layer.register_forward_hook(get_hook(name)))

            # Forward pass
            final_emb = model.encode_text(text_tokens).float()

            # Find EOS positions
            eos_positions = (text_tokens == eos_token_id).float()
            eos_index = eos_positions.argmax(dim=1)  # (B,)

            # Pool intermediate activations at EOS
            for name in text_layer_names:
                feat = activations[name]  # Already on CPU
                feat = feat.float()  
                if feat.dim() == 3 and feat.shape[0] == text_tokens.shape[1]:  # (seq_len, batch, dim)
                    feat = feat.permute(1, 0, 2)  # (batch, seq_len, dim)
                eos_emb = feat[torch.arange(feat.size(0)), eos_index.cpu(), :].to(dtype)  # (B, D) - use CPU eos_index
                eos_emb = eos_emb / (eos_emb.norm(dim=-1, keepdim=True) + 1e-8)
                embeddings_dict[name] = eos_emb  # Already on CPU

            final_emb = final_emb / (final_emb.norm(dim=-1, keepdim=True) + 1e-8)
            final_emb = final_emb.to(dtype)
            
            # CRITICAL: Check for NaN in final embeddings
            if torch.isnan(final_emb).any():
                logging.error("NaN detected in final caption embeddings after normalization!")
                logging.error(f"Input texts: {texts[:3]}...")  # Show first 3 texts
                logging.error(f"Final emb stats: min={final_emb.min().item():.6f}, max={final_emb.max().item():.6f}")
                # Check which samples have NaN
                nan_mask = torch.isnan(final_emb).any(dim=1)
                nan_indices = torch.where(nan_mask)[0].tolist()
                logging.error(f"NaN found in caption samples: {nan_indices}")
                if nan_indices:
                    for idx in nan_indices[:3]:  # Show details for first 3 NaN samples
                        logging.error(f"NaN sample {idx}: '{texts[idx]}'")
            
            embeddings_dict['final'] = final_emb.cpu()  # Move to CPU

        finally:
            # Remove hooks even if an exception occurs
            for h in handles:
                h.remove()
            # Clear activations to free memory
            activations.clear()
            # Force GPU memory cleanup
            torch.cuda.empty_cache()

    return embeddings_dict

def split_embeddings_dict(embeddings_dict, indices):
    # embeddings_dict: {layer_name: (N, D)}
    # indices: list or np.array of ints
    return {ln: emb[indices] for ln, emb in embeddings_dict.items()}

def dict_of_arrays_to_list_of_dicts(embeddings_dict):
    """
    Converts {layer_name: (N, D)} to [{layer_name: (D)}, ...] of length N
    """
    layer_names = list(embeddings_dict.keys())
    sizes = [v.shape[0] for v in embeddings_dict.values()]
    
    if len(set(sizes)) != 1:
        raise ValueError(f"Inconsistent batch sizes across layers: {sizes}")
    
    N = sizes[0]
    return [{ln: embeddings_dict[ln][i] for ln in layer_names} for i in range(N)]

def list_of_dicts_to_dict_of_arrays(list_of_dicts):
    """
    Converts a list of dicts [{layer_name: (D)}, ...] (length N)
    to a dict of arrays {layer_name: (N, D)}
    Works with torch.Tensor or numpy.ndarray values.
    """
    if not list_of_dicts:
        return {}

    layer_names = list(list_of_dicts[0].keys())
    from collections import defaultdict

    # Collect per layer
    out = defaultdict(list)
    for d in list_of_dicts:
        for ln in layer_names:
            out[ln].append(d[ln])

    # Stack per layer (assume torch or numpy)
    for ln in layer_names:
        vals = out[ln]
        # Torch or numpy detection
        if hasattr(vals[0], 'shape'):  # torch or numpy
            import torch
            import numpy as np
            if isinstance(vals[0], torch.Tensor):
                out[ln] = torch.stack(vals, dim=0)
            elif isinstance(vals[0], np.ndarray):
                out[ln] = np.stack(vals, axis=0)
            else:
                out[ln] = np.array(vals)
        else:
            out[ln] = np.array(vals)
    return dict(out)

def convert_list_of_dicts_to_dict_of_tensors(dict_list):
    if not dict_list:
        return {}
    keys = dict_list[0].keys()
    return {key: torch.stack([d[key] for d in dict_list], dim=0) for key in keys}

def extract_intermediate_features(input, model, device, layer_names, is_image=True, dtype=torch.float32):
    """
    Args:
        input: torch.Tensor (images or text tokens)
        model: CLIP model
        device: torch.device
        layer_names: list of strings
        is_image: True for image, False for text
    Returns:
        Dict[str, torch.Tensor]: {layer_name: features}
            - For text: embedding at EOS token for each layer (B, D)
            - For image: CLS/global pooled embedding for each layer (B, D)
    """

    activations = {}
    handles = []
    modules = model.visual if is_image else model.transformer
    modules_dict = dict(modules.named_modules())

    # Register hooks for all but 'final'
    for name in layer_names:
        if name == 'final':
            continue
        def get_hook(n):
            return lambda m, i, o: activations.setdefault(n, o)
        handles.append(modules_dict[name].register_forward_hook(get_hook(name)))

    # Forward pass (final embedding)
    if is_image:
        image = input.to(device)
        final_emb = model.encode_image(image)
    else:
        # If input is not already tokenized, tokenize it here
        text_tokens = input.to(device)
        final_emb = model.encode_text(text_tokens)

    # Remove hooks
    for h in handles:
        h.remove()

    # Compose output dict (main fix is for is_image=False)
    features = {}
    for name in layer_names:
        if name == 'final':
            final_emb = final_emb / final_emb.norm(dim=-1, keepdim=True)  # Normalize
            final_emb = final_emb.to(dtype)  # Convert to specified dtype
            features['final'] = final_emb
        else:
            feat = activations[name]
            if not is_image:
                # Pool EOS token
                eos_token_id = getattr(model, 'tokenizer', None)
                if eos_token_id is not None:
                    eos_token_id = model.tokenizer.eos_token_id
                else:
                    eos_token_id = 49407  # Default for OpenAI CLIP
                eos_positions = (text_tokens == eos_token_id).to(dtype)
                eos_index = eos_positions.argmax(dim=1)  # (B,)
                # Handle (seq_len, batch, dim) → (batch, seq_len, dim)
                if feat.dim() == 3 and feat.shape[0] == text_tokens.shape[1]:
                    feat = feat.permute(1, 0, 2)  # (B, seq_len, D)
                eos_emb = feat[torch.arange(feat.size(0)), eos_index, :].to(dtype)  # (B, D)
                features[name] = eos_emb / eos_emb.norm(dim=-1, keepdim=True)
            else:
                # For image: use [CLS] token or global token (assuming feat is (B, N, D))
                # Most CLIP ViTs: use first token
                if feat.dim() == 3:
                    features[name] = feat[:, 0, :] / feat[:, 0, :].norm(dim=-1, keepdim=True)
                else:
                    features[name] = feat / feat.norm(dim=-1, keepdim=True)

    features = {k: v.to(device).to(dtype) for k, v in features.items()}
    return features


def get_embeddings(
    dataset,
    indices,
    model_clip,
    model_align=None,
    device='cuda',
    batch_size=256,
    intermediate_image_layer_names=['final'],
    intermediate_text_layer_names=['final'],
    use_ft_model=False,
    return_layerwise_dict=False,
):
    """
    General function to get image, positive caption, and negative caption embeddings,
    with optional return of all intermediate layers.
    """
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    all_image_embeds = []
    all_caption_embeds = []
    all_neg_caption_embeds = []

    # For each layer, store lists for stacking later if requested
    if return_layerwise_dict:
        image_embeds_dict = {layer: [] for layer in intermediate_image_layer_names}
        caption_embeds_dict = {layer: [] for layer in intermediate_text_layer_names}
        neg_caption_embeds_dict = {layer: [] for layer in intermediate_text_layer_names}

    max_num_neg = 0
    # First pass: get max_num_neg
    for batch in dataloader:
        batch_dict = batch if isinstance(batch, dict) else {
            'image_options': batch[0],
            'caption_options': batch[1],
            'label': batch[2]
        }
        captions_list = batch_dict['caption_options']
        K = len(captions_list[0]) if isinstance(captions_list[0], list) else captions_list.shape[1]
        max_num_neg = max(max_num_neg, K - 1)
    # Reset dataloader
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(dataloader, desc="Embedding dataset"):
        batch_dict = batch if isinstance(batch, dict) else {
            'image_options': batch[0],
            'caption_options': batch[1],
            'label': batch[2]
        }
        imgs = batch_dict['image_options'].to(device)
        captions_list = batch_dict['caption_options']
        labels = batch_dict['label']

        B = len(captions_list)
        K = len(captions_list[0]) if isinstance(captions_list[0], list) else captions_list.shape[1]

        # ----- IMAGE EMBEDDINGS -----
        if use_ft_model:
            model_align.eval()
            with torch.no_grad():
                img_embs = model_align.encode_image(imgs)
                img_embs = img_embs.cpu().float()
                img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
            if return_layerwise_dict:
                for layer in intermediate_image_layer_names:
                    image_embeds_dict[layer].append(img_embs)  # Only returns final in this case
        else:
            model_clip.eval()
            if model_align:
                model_align.eval()
            with torch.no_grad():
                img_feat_dict = compute_image_embeddings_intermediate_batch(
                    imgs, model_clip, device, intermediate_image_layer_names
                )
                if model_align is not None:
                    aligned_dict = model_align.encode_image(img_feat_dict, return_dict=True)
                    img_embs_dict = aligned_dict
                else:
                    img_embs_dict = img_feat_dict

                # Always normalize each layer
                for layer in img_embs_dict:
                    emb = img_embs_dict[layer].cpu().float()
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    img_embs_dict[layer] = emb

            if return_layerwise_dict:
                for layer in intermediate_image_layer_names:
                    image_embeds_dict[layer].append(img_embs_dict[layer])
            else:
                img_embs = img_embs_dict['final']
                all_image_embeds.append(img_embs)

        # ----- CAPTION EMBEDDINGS -----
        # Flatten [B, K] into [B*K]
        if isinstance(captions_list[0], list):
            flat_captions = [c for sample in captions_list for c in sample]
        else:
            flat_captions = [c for sample in captions_list for c in sample]

        with torch.no_grad():
            if use_ft_model:
                import clip
                flat_captions_tokenized = clip.tokenize(flat_captions, truncate=True).to(device)
                cap_embs = model_align.encode_text(flat_captions_tokenized)
                if return_layerwise_dict:
                    for layer in intermediate_text_layer_names:
                        # Only one layer ("final") if using FT
                        caption_embeds_dict[layer].append(cap_embs.cpu().float() / cap_embs.norm(dim=-1, keepdim=True))
            else:
                cap_feat_dict = compute_caption_embeddings_intermediate_batch(
                    flat_captions, model_clip, device, intermediate_text_layer_names
                )
                if model_align is not None:
                    aligned_dict = model_align.encode_text(cap_feat_dict, return_dict=True)
                    cap_embs_dict = aligned_dict
                else:
                    cap_embs_dict = cap_feat_dict

                # Normalize
                for layer in cap_embs_dict:
                    emb = cap_embs_dict[layer].cpu().float()
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    cap_embs_dict[layer] = emb

            # For each layer, process pos/neg extraction
            for layer in intermediate_text_layer_names:
                if use_ft_model:
                    cap_embs = cap_embs.cpu().float()
                    cap_embs = cap_embs / cap_embs.norm(dim=-1, keepdim=True)
                    cap_embs_layer = cap_embs
                else:
                    cap_embs_layer = cap_embs_dict[layer]

                cap_embs_layer = cap_embs_layer.view(B, K, -1)
                if isinstance(labels, torch.Tensor):
                    labels_tensor = labels.to(cap_embs_layer.device)
                else:
                    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=cap_embs_layer.device)

                pos_cap_embs = cap_embs_layer[torch.arange(B), labels_tensor]  # [B, D]
                neg_cap_embs = []
                for i in range(B):
                    mask = torch.ones(K, dtype=torch.bool, device=cap_embs_layer.device)
                    mask[labels_tensor[i]] = False
                    neg_embs = cap_embs_layer[i][mask]  # [K-1, D]
                    # Pad to max_num_neg
                    if (K-1) < max_num_neg:
                        pad = torch.zeros((max_num_neg - (K-1), cap_embs_layer.size(-1)), dtype=neg_embs.dtype, device=neg_embs.device)
                        neg_embs = torch.cat([neg_embs, pad], dim=0)
                    neg_cap_embs.append(neg_embs)
                neg_cap_embs = torch.stack(neg_cap_embs, dim=0)  # [B, max_num_neg, D]

                if return_layerwise_dict:
                    caption_embeds_dict[layer].append(pos_cap_embs)
                    neg_caption_embeds_dict[layer].append(neg_cap_embs)
                else:
                    if layer == 'final':
                        all_caption_embeds.append(pos_cap_embs)
                        all_neg_caption_embeds.append(neg_cap_embs)

    # --- Collate and return ---
    if return_layerwise_dict:
        output = {}
        for layer in intermediate_image_layer_names:
            output.setdefault(layer, {})['image_embeds'] = torch.cat(image_embeds_dict[layer], dim=0)
        for layer in intermediate_text_layer_names:
            output.setdefault(layer, {})['caption_embeds'] = torch.cat(caption_embeds_dict[layer], dim=0)
            output[layer]['neg_caption_embeds'] = torch.cat(neg_caption_embeds_dict[layer], dim=0)
        return output
    else:
        all_image_embeds = torch.cat(all_image_embeds, dim=0)
        all_caption_embeds = torch.cat(all_caption_embeds, dim=0)
        all_neg_caption_embeds = torch.cat(all_neg_caption_embeds, dim=0)
        return all_image_embeds, all_caption_embeds, all_neg_caption_embeds


# =============================================================================
# TQA-Aware Similarity Computation
# =============================================================================

def is_tqa_model(model) -> bool:
    """
    Check if the model uses Text-Query Aggregation (TQA).
    
    TQA models require text to condition image embeddings, which means
    standard separate encode_image/encode_text doesn't fully utilize them.
    
    Args:
        model: The model to check (can be pipeline or head)
        
    Returns:
        True if the model uses TQA, False otherwise
    """
    # Check if model has a TQA head
    if hasattr(model, 'head'):
        head = model.head
        head_class_name = head.__class__.__name__
        if 'TextQueryAggregator' in head_class_name:
            return True
    
    # Check if model itself is a TQA head
    model_class_name = model.__class__.__name__
    if 'TextQueryAggregator' in model_class_name:
        return True
    
    # Check for TQA-specific attributes
    if hasattr(model, 'cross_attention') and hasattr(model, 'aggregate_with_text_query'):
        return True
    
    return False


def compute_pairwise_similarity_tqa(
    image_features: dict,
    text_features: torch.Tensor,
    model,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute pairwise similarity with TQA conditioning.
    
    For TQA models, image embeddings are conditioned on the corresponding text,
    so we need to pass text_features when encoding images.
    
    Args:
        image_features: Dict with 'global'/'global_unprojected' and optionally 'patches'
                       OR a tensor [N, D] for standard CLIP
        text_features: Text embeddings [N, D] (same batch size as images)
        model: The model (CLIPEndToEndPipeline or head)
        normalize: Whether to normalize outputs (default True)
        
    Returns:
        similarities: [N] pairwise similarity scores
    """
    # Get the head (for encoding)
    if hasattr(model, 'head'):
        head = model.head
    else:
        head = model
    
    # Check if TQA and we have patches
    use_tqa = is_tqa_model(model)
    has_patches = isinstance(image_features, dict) and 'patches' in image_features
    
    if use_tqa and has_patches:
        # TQA path: condition image on text
        image_embeds = head.encode_image(image_features, text_features=text_features)
    else:
        # Standard path
        image_embeds = head.encode_image(image_features)
    
    text_embeds = head.encode_text(text_features)
    
    # Normalize if needed
    if normalize:
        image_embeds = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        text_embeds = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Pairwise similarity (element-wise dot product, then sum)
    similarities = (image_embeds * text_embeds).sum(dim=-1)
    return similarities


def compute_binary_choice_similarity_tqa(
    image_features: dict,
    pos_text_features: torch.Tensor,
    neg_text_features: torch.Tensor,
    model,
    normalize: bool = True,
) -> tuple:
    """
    Compute similarities for binary choice tasks (e.g., SugarCrepe, Winoground).
    
    For TQA models, each (image, text) pair gets its own text-conditioned image embedding.
    
    Args:
        image_features: Dict with 'global' and 'patches', batch size N
        pos_text_features: Positive text embeddings [N, D]
        neg_text_features: Negative text embeddings [N, D]
        model: The model (CLIPEndToEndPipeline or head)
        normalize: Whether to normalize outputs
        
    Returns:
        Tuple of (pos_scores, neg_scores), each [N]
    """
    pos_scores = compute_pairwise_similarity_tqa(
        image_features, pos_text_features, model, normalize
    )
    neg_scores = compute_pairwise_similarity_tqa(
        image_features, neg_text_features, model, normalize
    )
    return pos_scores, neg_scores


def compute_retrieval_similarity_matrix_tqa(
    image_features: dict,
    text_features: torch.Tensor,
    model,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute full similarity matrix with TQA conditioning.
    
    For TQA models, this is O(N*M) because each (image, text) pair needs 
    its own conditioned image embedding.
    
    For standard CLIP, this is O(N+M) as usual.
    
    Args:
        image_features: Dict with 'global' and 'patches', batch size N
        text_features: Text embeddings [M, D]
        model: The model (CLIPEndToEndPipeline or head)
        normalize: Whether to normalize outputs
        
    Returns:
        similarity_matrix: [N, M]
    """
    # Get the head
    if hasattr(model, 'head'):
        head = model.head
    else:
        head = model
    
    # Check if TQA with patches
    use_tqa = is_tqa_model(model)
    has_patches = isinstance(image_features, dict) and 'patches' in image_features
    
    if not use_tqa or not has_patches:
        # Standard CLIP: compute separately and matmul
        image_embeds = head.encode_image(image_features)
        text_embeds = head.encode_text(text_features)
        
        if normalize:
            image_embeds = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-8)
            text_embeds = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        
        return image_embeds @ text_embeds.T
    
    # TQA path: O(N*M) computation
    # Extract image components
    vision_global = image_features.get('global_unprojected', 
                    image_features.get('global'))
    vision_patches = image_features.get('patches')
    
    N = vision_global.shape[0]
    M = text_features.shape[0]
    device = vision_global.device
    
    similarity_matrix = torch.zeros(N, M, device=device)
    
    # For each text, compute conditioned image embeddings
    for j in range(M):
        # Broadcast j-th text to all images
        text_j = text_features[j:j+1].expand(N, -1)  # [N, D]
        
        # Compute TQA-conditioned image embeddings
        image_input = {
            'global': vision_global,
            'global_unprojected': vision_global,
            'patches': vision_patches
        }
        image_embeds_j = head.encode_image(image_input, text_features=text_j)  # [N, D]
        
        # Get text embedding (encoded once)
        text_embed_j = head.encode_text(text_features[j:j+1])  # [1, D]
        
        if normalize:
            image_embeds_j = image_embeds_j / (image_embeds_j.norm(dim=-1, keepdim=True) + 1e-8)
            text_embed_j = text_embed_j / (text_embed_j.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute similarity column
        similarity_matrix[:, j] = (image_embeds_j @ text_embed_j.T).squeeze(-1)
    
    return similarity_matrix


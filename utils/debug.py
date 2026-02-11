import logging
import json
from typing import Any, Dict
from utils.cache import is_lazy_embedding_loader
import os
import torch
from utils.align import compute_image_embeddings_intermediate_batch, compute_caption_embeddings_intermediate_batch
import random, math
from utils.dist import is_main_process
import numpy as np

def debug_validate_embedding_order(
    image_embeddings,
    dataset,
    clip_model,
    device,
    args,
    sample_indices=None,
    layer_name="final",
    name="Image",
):
    """
    Validate that cached embeddings are aligned with dataset order by recomputing a few test samples.

    Args:
        image_embeddings: Lazy loader or dict of tensors (already computed/loaded embeddings)
        dataset: Dataset object
        clip_model: CLIP model used for encoding
        device: torch.device
        args: args namespace (must have embedding_path, force_float32, dataset, etc.)
        sample_indices: list of indices to test; default [0, 10, 100] (clamped to dataset length)
        layer_name: embedding layer to validate (default "final")
        name: String prefix for logging
    """
    if image_embeddings is None:
        logging.warning(f"{name} embeddings are None, skipping order validation")
        return

    dataset_size = len(dataset)
    if sample_indices is None:
        sample_indices = [0, min(10, dataset_size - 1), min(100, dataset_size - 1)]

    try:
        for test_idx in sample_indices:
            if test_idx >= dataset_size:
                continue

            # Extract sample image
            sample = dataset[test_idx]
            if isinstance(sample, tuple):
                test_image = sample[0]
            else:
                test_image = sample.get("image_options", sample)

            # Log tar mapping if available
            if hasattr(dataset, "index") and test_idx < len(dataset.index):
                tar_num, local_idx = dataset.index[test_idx]
                logging.info(f"Debug: dataset[{test_idx}] maps to tar {tar_num}, local_idx {local_idx}")
            else:
                tar_num, local_idx = None, None

            # Compute fresh embedding
            test_image = test_image.unsqueeze(0).to(device)
            with torch.no_grad():
                test_embed = compute_image_embeddings_intermediate_batch(
                    test_image, clip_model, device, [layer_name], dtype=torch.float32
                )[layer_name].cpu()

            # Cast to storage dtype
            target_dtype = torch.float32 if getattr(args, "force_float32", False) else torch.float16
            test_embed = test_embed.to(target_dtype)

            # Retrieve cached embedding
            try:
                if is_lazy_embedding_loader(image_embeddings):
                    cached_embed = image_embeddings.get_embeddings(layer_name, [test_idx])
                    test_embed = test_embed.float()  # lazy loaders always yield float32
                else:
                    cached_embed = image_embeddings[layer_name][test_idx:test_idx + 1].to(test_embed.dtype)

                # Compare
                similarity = torch.cosine_similarity(test_embed, cached_embed, dim=1).item()
                if similarity < 0.99:
                    logging.error(f"{name} embedding order mismatch at {test_idx}: similarity={similarity:.4f}")
                    logging.error(f"Test first 5: {test_embed[0, :5]}")
                    logging.error(f"Cached first 5: {cached_embed[0, :5]}")
                else:
                    logging.info(f"✓ {name} embedding order validated at {test_idx}: similarity={similarity:.4f}")

            except Exception as e:
                logging.warning(f"Could not validate {name} embedding at index {test_idx}: {e}")

            # Optional: save debug artifacts
            if test_idx <= 10:
                try:
                    import numpy as np
                    from PIL import Image
                    debug_dir = "debug_validation"
                    os.makedirs(debug_dir, exist_ok=True)

                    # Save image
                    image_np = test_image.squeeze(0).cpu()
                    if image_np.shape[0] == 3:  # CHW
                        image_np = image_np.permute(1, 2, 0).numpy()
                    else:
                        image_np = image_np.numpy()
                    if image_np.min() < 0:  # denormalize CLIP-preprocessed
                        mean = np.array([0.48145466, 0.4578275, 0.40821073])
                        std = np.array([0.26862954, 0.26130258, 0.27577711])
                        image_np = image_np * std + mean
                    image_np = np.clip(image_np, 0, 1)
                    Image.fromarray((image_np * 255).astype(np.uint8)).save(
                        f"{debug_dir}/validation_sample_{test_idx}.jpg"
                    )

                    # Save JSON metadata
                    import json
                    debug_info = {
                        "dataset_idx": test_idx,
                        "tar_mapping": {"tar_num": tar_num, "local_idx": local_idx} if tar_num is not None else None,
                        "similarity": float(similarity),
                        "test_embed_first_5": test_embed[0, :5].tolist(),
                        "cached_embed_first_5": cached_embed[0, :5].tolist(),
                    }
                    with open(f"{debug_dir}/validation_sample_{test_idx}.json", "w") as f:
                        json.dump(debug_info, f, indent=2)

                except Exception as e:
                    logging.warning(f"Failed to save debug artifacts for index {test_idx}: {e}")

    except Exception as outer_e:
        logging.warning(f"Could not run embedding order validation: {outer_e}")


def debug_validate_caption_embedding_order(
    caption_embeddings,
    dataset,
    clip_model,
    device,
    args,
    sample_indices=None,
    layer_name="final",
    name="Caption",
):
    """
    Validate that cached caption embeddings are aligned with dataset caption order by recomputing a few test samples.
    This verifies that CLIP is getting actual caption strings (not hashes) and that embeddings match.

    Args:
        caption_embeddings: Lazy loader or dict of tensors (already computed/loaded caption embeddings)
        dataset: Dataset object with .captions property
        clip_model: CLIP model used for encoding
        device: torch.device
        args: args namespace (must have embedding_path, force_float32, dataset, etc.)
        sample_indices: list of caption indices to test; default [0, 10, 100] (clamped to vocabulary size)
        layer_name: embedding layer to validate (default "final")
        name: String prefix for logging
    """
    if caption_embeddings is None:
        logging.warning(f"{name} embeddings are None, skipping order validation")
        return

    if not hasattr(dataset, 'get_captions_for_embedding'):
        logging.warning(f"Dataset has no 'get_captions_for_embedding' method, skipping {name} validation")
        return

    try:
        # Check if we should use shared memory approach (non-main processes)
        use_shared_memory = (hasattr(dataset, 'use_shared_caption_index') and 
                           dataset.use_shared_caption_index and
                           not hasattr(dataset, '_captions_for_embedding'))
        
        if use_shared_memory:
            # Use shared memory approach - get vocab size without loading all captions
            vocab_size = dataset.get_vocab_size()
            logging.info(f"Validating {name} embedding order with {vocab_size} captions (shared memory mode)...")
        else:
            # Use traditional approach - get captions list
            captions = dataset.get_captions_for_embedding()
            vocab_size = len(captions)
            logging.info(f"Validating {name} embedding order with {vocab_size} captions (local mode)...")
        
        if vocab_size == 0:
            logging.warning(f"Caption vocabulary is empty, skipping {name} validation")
            return

        if sample_indices is None:
            sample_indices = random.sample(range(vocab_size), k=min(10, vocab_size))

        for test_idx in sample_indices:
            if test_idx >= vocab_size:
                continue

            # Extract caption string using appropriate method
            if use_shared_memory:
                test_caption = dataset.get_caption_by_vocab_index(test_idx)
                if test_caption is None:
                    logging.warning(f"Could not retrieve caption for vocab index {test_idx} from shared memory")
                    continue
            else:
                test_caption = captions[test_idx]
            
            if not isinstance(test_caption, str):
                logging.error(f"Caption at index {test_idx} is not a string: {type(test_caption)} = {test_caption}")
                continue
                
            logging.info(f"Testing caption {test_idx}: '{test_caption[:50]}{'...' if len(test_caption) > 50 else ''}'")

            # Compute fresh embedding using actual caption string
            with torch.no_grad():
                test_embed = compute_caption_embeddings_intermediate_batch(
                    [test_caption], clip_model, device, [layer_name], dtype=torch.float32
                )[layer_name].cpu()

            # Cast to storage dtype
            target_dtype = torch.float32 if getattr(args, "force_float32", False) else torch.float16
            test_embed = test_embed.to(target_dtype)

            # Retrieve cached embedding
            try:
                if is_lazy_embedding_loader(caption_embeddings):
                    # For lazy loaders, the index should match caption vocab order
                    cached_embed = caption_embeddings.get_embeddings(layer_name, [test_idx])
                    test_embed = test_embed.float()  # lazy loaders always yield float32
                else:
                    # For direct arrays, we need to map caption vocab index to dataset index
                    # The embeddings are stored in dataset order, not caption vocab order
                    if hasattr(dataset, 'get_idx_to_ptr') and hasattr(dataset, '_vocab_split_indices'):
                        # Try to find the corresponding dataset index for this caption vocab index
                        try:
                            # Find which dataset index corresponds to this caption vocab index
                            dataset_idx = None
                            for d_idx, vocab_idx in enumerate(dataset._vocab_split_indices):
                                if vocab_idx == test_idx:
                                    dataset_idx = d_idx
                                    break
                            
                            if dataset_idx is not None:
                                if dataset_idx >= len(caption_embeddings[layer_name]):
                                    raise IndexError(f"Dataset index {dataset_idx} out of range for embedding array size {len(caption_embeddings[layer_name])}")
                                cached_embed = caption_embeddings[layer_name][dataset_idx:dataset_idx + 1].to(test_embed.dtype)
                            else:
                                # This is a serious problem - caption vocab index not found in dataset mapping
                                raise ValueError(f"Caption vocab index {test_idx} not found in dataset._vocab_split_indices. This indicates a serious index mapping bug.")
                        except Exception as mapping_error:
                            raise RuntimeError(f"Index mapping failed for caption {test_idx}: {mapping_error}. This indicates corrupted index mapping between vocabulary and embeddings.")
                    else:
                        # Fallback: assume direct indexing, but validate bounds
                        if test_idx >= len(caption_embeddings[layer_name]):
                            raise IndexError(f"Caption index {test_idx} out of range for embedding array size {len(caption_embeddings[layer_name])}. Expected vocabulary indices to match embedding array positions.")
                        cached_embed = caption_embeddings[layer_name][test_idx:test_idx + 1].to(test_embed.dtype)

                # Compare embeddings
                similarity = torch.cosine_similarity(test_embed, cached_embed, dim=1).item()
                if similarity < 0.99:
                    logging.error(f"{name} embedding order mismatch at caption {test_idx}: similarity={similarity:.4f}")
                    logging.error(f"Caption: '{test_caption[:100]}'")
                    logging.error(f"Fresh embedding first 5: {test_embed[0, :5]}")
                    logging.error(f"Cached embedding first 5: {cached_embed[0, :5]}")
                    
                    # Additional debugging for hash-based systems
                    if hasattr(dataset, 'caption_hash_to_idx'):
                        caption_hash = dataset._hash_caption(test_caption)
                        vocab_idx = dataset.caption_hash_to_idx.get(caption_hash, -1)
                        logging.error(f"Caption hash: {caption_hash:016x}, vocab_idx: {vocab_idx}")
                else:
                    logging.info(f"✓ {name} embedding order validated at {test_idx}: similarity={similarity:.4f}")

            except Exception as e:
                logging.warning(f"Could not validate {name} embedding at caption index {test_idx}: {e}")

            # Optional: save debug artifacts
            if test_idx <= 10:
                try:
                    debug_dir = "debug_validation"
                    os.makedirs(debug_dir, exist_ok=True)

                    # Save caption and embedding info
                    debug_info = {
                        "caption_idx": test_idx,
                        "caption_text": test_caption,
                        "caption_length": len(test_caption),
                        "similarity": float(similarity),
                        "fresh_embed_first_5": test_embed[0, :5].tolist(),
                        "cached_embed_first_5": cached_embed[0, :5].tolist(),
                        "embedding_layer": layer_name,
                        "vocab_size": vocab_size,
                    }
                    
                    # Add hash info if available
                    if hasattr(dataset, 'caption_hash_to_idx'):
                        caption_hash = dataset._hash_caption(test_caption)
                        debug_info.update({
                            "caption_hash": f"{caption_hash:016x}",
                            "hash_vocab_idx": dataset.caption_hash_to_idx.get(caption_hash, -1),
                            "uses_hash_system": True
                        })
                    else:
                        debug_info["uses_hash_system"] = False

                    with open(f"{debug_dir}/caption_validation_{test_idx}.json", "w") as f:
                        json.dump(debug_info, f, indent=2)

                except Exception as e:
                    logging.warning(f"Failed to save caption debug artifacts for index {test_idx}: {e}")

    except Exception as outer_e:
        logging.warning(f"Could not run caption embedding order validation: {outer_e}")


def debug_validate_embedding_size(embeddings, target_count, name="Embeddings"):
    """
    Validate that embeddings (lazy or regular) match the expected count.

    Args:
        embeddings: Lazy loader (TarBasedEmbeddingLoader, LazyEmbeddingLoader) or dict of tensors
        target_count: int, expected number of embeddings (e.g., len(dataset) or len(dataset.captions))
        name: str, label for logging ("Image", "Caption", etc.)
    """
    if embeddings is None:
        logging.warning(f"{name} embeddings are None, skipping validation")
        return

    # Lazy loader case
    if is_lazy_embedding_loader(embeddings):
        if hasattr(embeddings, "total_samples"):  
            # Tar-based loader
            embedding_size = embeddings.total_samples
            if embedding_size != target_count:
                logging.error(
                    f"{name} embeddings mismatch: {embedding_size} vs expected {target_count}"
                )
            else:
                for layer_name in embeddings.layer_names:
                    logging.info(f"✓ {name} embedding {layer_name} size: {embedding_size} matches expected {target_count}")
        elif "total_samples" in embeddings.metadata:  
            # Chunked loader
            for layer_name in embeddings.layer_names:
                embedding_size = embeddings.metadata["total_samples"][layer_name]
                if embedding_size != target_count:
                    logging.error(
                        f"{name} embedding {layer_name} mismatch: {embedding_size} vs expected {target_count}"
                    )
                else:
                    logging.info(f"✓ {name} embedding {layer_name} size: {embedding_size} matches expected {target_count}")
        else:
            logging.warning(f"{name} lazy embeddings do not expose total_samples; skipping validation")

    # Regular dict of tensors case
    else:
        for layer_name, tensor in embeddings.items():
            if tensor.shape[0] != target_count:
                logging.error(
                    f"{name} embedding {layer_name} mismatch: {tensor.shape[0]} vs expected {target_count}"
                )
            else:
                logging.info(f"✓ {name} embedding {layer_name} shape {tensor.shape} matches expected {target_count}")


def debug_check_nan_embeddings(emb_store, name: str, sample_size: int = 2048):
    """
    Checks for NaNs and all-zero rows:
      - regular dict of tensors: full scan per layer (small datasets)
      - lazy loader: random spot-check over `sample_size` rows per layer
    """
    try:
        if emb_store is None:
            logging.info(f"[{name}] no embeddings to check.")
            return

        if is_lazy_embedding_loader(emb_store):
            layer_names = getattr(emb_store, "layer_names", ["final"])
            total = getattr(emb_store, "total_samples", None)
            if total is None and hasattr(emb_store, "metadata"):
                ts = emb_store.metadata.get("total_samples", {})
                total = next(iter(ts.values())) if ts else None

            if total is None or total == 0:
                logging.info(f"[{name}] lazy: unknown/empty total samples; skipping NaN/zero check.")
                return

            take = min(sample_size, total)
            indices = random.sample(range(total), k=take)
            for ln in layer_names:
                x = emb_store.get_embeddings(ln, indices)
                nans = torch.isnan(x).sum().item()
                zero_rows = (x == 0).all(dim=1).sum().item()
                logging.info(
                    f"[{name}] lazy layer '{ln}': sample {take}/{total} rows -> "
                    f"NaN elems={nans}, all-zero rows={zero_rows}"
                )
        else:
            for ln, t in emb_store.items():
                nans = torch.isnan(t).sum().item()
                zero_rows = (t == 0).all(dim=1).sum().item()
                logging.info(
                    f"[{name}] layer '{ln}': NaN elems={nans}, all-zero rows={zero_rows}"
                )
    except Exception as e:
        logging.warning(f"[{name}] NaN/zero check failed: {e}")


def debug_validate_caption_vocab_count(dataset, caption_embeddings, text_embedding_path):
    """Ensure caption embeddings count == split-scoped vocab size."""
    if hasattr(dataset, 'get_caption_count'):
        total_captions = dataset.get_caption_count()
    elif hasattr(dataset, 'get_captions_for_embedding'):
        captions = dataset.get_captions_for_embedding()
        total_captions = len(captions)
    else:
        logging.error("[debug] Unable to determine total captions.")
        return

    expected = total_captions
    if is_lazy_embedding_loader(caption_embeddings):
        actual = getattr(caption_embeddings, 'total_samples', 0)
    else:
        actual = caption_embeddings.get('final', torch.empty(0)).shape[0] if caption_embeddings else 0

    if is_main_process():
        logging.info("🔍 Caption embedding validation:")
        logging.info(f"  - Split-scoped vocabulary size: {expected}")
        logging.info(f"  - Actual caption embeddings count: {actual}")

    if actual != expected:
        logging.error(f"❌ MISMATCH: Caption embeddings ({actual}) != split vocabulary ({expected})")
        logging.error("This will cause zero embeddings and NaN losses!")
        logging.error(f"Delete files matching: {text_embedding_path}*")
        raise RuntimeError(
            f"Caption embedding count mismatch: {actual} != {expected}. "
            "Embeddings were computed with a different vocabulary. "
            f"Please delete ({text_embedding_path}*) and rerun."
        )
    elif is_main_process():
        logging.info("✅ Caption embedding count matches split-scoped vocabulary")


def debug_dump_samples(dataset, split_dict, preprocess, out_dir="temp", n_train=5, n_val=5, seed=123):
    """Debug function to dump sample data for inspection."""
    import random
    try:
        os.makedirs(out_dir, exist_ok=True)
        rng = random.Random(seed)
        def _dump(split):
            indices = split_dict.get(split, {}).get('indices', [])
            pick = indices if len(indices) <= (n_train if split=='train' else n_val) else rng.sample(list(indices), (n_train if split=='train' else n_val))
            for i, idx in enumerate(pick):
                try:
                    sample = dataset[idx]
                    if isinstance(sample, dict) and 'caption_options' in sample:
                        with open(os.path.join(out_dir, f"{split}_{i}_idx-{idx}.txt"), 'w') as f:
                            for j, c in enumerate(sample['caption_options']):
                                f.write(f"{j}: {c}\n")

                    # Also save the image if available
                    if 'image_options' in sample:
                        img = sample['image_options']
                        if isinstance(img, torch.Tensor):
                            img = img.cpu().numpy()
                        elif isinstance(img, np.ndarray):
                            pass
                        else:
                            logging.warning(f"Unexpected image type: {type(img)} for index {idx}")
                
                    # Save the image
                    if isinstance(img, np.ndarray):
                        from PIL import Image
                        img_pil = Image.fromarray(img)
                        img_pil.save(os.path.join(out_dir, f"{split}_{i}_idx-{idx}.jpg"))
                    elif isinstance(img, torch.Tensor):
                        from torchvision.transforms.functional import to_pil_image
                        img_pil = to_pil_image(img)
                        img_pil.save(os.path.join(out_dir, f"{split}_{i}_idx-{idx}.jpg"))
                    else:
                        logging.warning(f"Cannot save image for index {idx}: unsupported type {type(img)}")
                except Exception:
                    pass
        _dump('train')
        _dump('val')
    except Exception:
        pass

def debug_caption_locality(dataset, sample_size=1000):
    """
    Debug function to analyze caption locality and index spans for I/O optimization.
    
    This function examines how caption indices are distributed across the dataset
    to identify large spans that cause I/O bottlenecks during text embedding loading.
    
    Args:
        dataset: Dataset object with caption access methods
        sample_size: Number of random samples to analyze for locality patterns
    """
    try:
        logging.info("=== DEBUGGING CAPTION LOCALITY ===")
        
        # Check if dataset has unified caption indices
        if not (hasattr(dataset, '_unified_indices') and dataset._unified_indices is not None):
            logging.info("Dataset has no unified caption indices - using legacy approach")
            # Legacy approach - analyze per-sample caption access
            try:
                caption_count = dataset.get_caption_count() if hasattr(dataset, 'get_caption_count') else 0
                if caption_count > 0 and hasattr(dataset, 'index'):
                    dataset_size = len(dataset)
                    sample_indices = random.sample(range(dataset_size), min(sample_size, dataset_size))
                    
                    caption_indices = []
                    for idx in sample_indices:
                        try:
                            sample = dataset[idx]
                            if isinstance(sample, dict) and 'caption_options' in sample:
                                # Get the selected caption index for this sample
                                if hasattr(dataset, '_get_caption_idx_for_sample'):
                                    cap_idx = dataset._get_caption_idx_for_sample(idx)
                                    if cap_idx is not None:
                                        caption_indices.append(cap_idx)
                        except Exception:
                            continue
                    
                    if caption_indices:
                        caption_indices.sort()
                        spans = []
                        for i in range(len(caption_indices) - 1):
                            span = caption_indices[i + 1] - caption_indices[i]
                            spans.append(span)
                        
                        logging.info(f"Legacy caption locality analysis ({len(caption_indices)} samples):")
                        logging.info(f"  - Caption index range: {min(caption_indices)} to {max(caption_indices)}")
                        logging.info(f"  - Average span: {np.mean(spans):.1f}")
                        logging.info(f"  - Max span: {max(spans) if spans else 0}")
                        logging.info(f"  - Spans > 1000: {sum(1 for s in spans if s > 1000)}")
                        logging.info(f"  - Spans > 10000: {sum(1 for s in spans if s > 10000)}")
                    else:
                        logging.info("No caption indices found for legacy analysis")
                else:
                    logging.info("No caption indices available for legacy analysis")
            except Exception as e:
                logging.warning(f"Legacy caption analysis failed: {e}")
                logging.info("Skipping legacy caption locality analysis")
            return
        
        # Unified caption indices analysis
        unified_indices = dataset._unified_indices
        total_samples = len(unified_indices)
        
        # Convert to numpy if it's a tensor, or to tensor if it's numpy for consistent processing
        if isinstance(unified_indices, torch.Tensor):
            unified_indices_np = unified_indices.cpu().numpy()
            unified_indices_torch = unified_indices
        else:
            unified_indices_np = unified_indices
            unified_indices_torch = torch.from_numpy(unified_indices)
        
        logging.info(f"Unified caption indices analysis ({total_samples} total samples):")
        logging.info(f"  - Caption index range: {unified_indices_np.min()} to {unified_indices_np.max()}")
        logging.info(f"  - Unique captions: {len(np.unique(unified_indices_np))}")
        
        # Sample for span analysis - work with torch tensors for consistent operations
        if total_samples > sample_size:
            sample_indices = torch.randperm(total_samples)[:sample_size].sort().values
            sampled_caption_indices = unified_indices_torch[sample_indices]
        else:
            sampled_caption_indices = unified_indices_torch
        
        # Calculate spans (gaps between consecutive caption indices)
        sorted_caption_indices = torch.sort(sampled_caption_indices).values
        spans = sorted_caption_indices[1:] - sorted_caption_indices[:-1]
        spans = spans[spans > 0]  # Remove duplicates (span = 0)
        
        if len(spans) > 0:
            spans_np = spans.cpu().numpy()
            
            logging.info(f"Caption locality statistics (sample size: {len(sampled_caption_indices)}):")
            logging.info(f"  - Average span: {np.mean(spans_np):.1f}")
            logging.info(f"  - Median span: {np.median(spans_np):.1f}")
            logging.info(f"  - Max span: {np.max(spans_np)}")
            logging.info(f"  - Min span: {np.min(spans_np)}")
            logging.info(f"  - Std deviation: {np.std(spans_np):.1f}")
            
            # Analyze problematic spans
            large_spans = spans_np[spans_np > 1000]
            very_large_spans = spans_np[spans_np > 10000]
            huge_spans = spans_np[spans_np > 100000]
            massive_spans = spans_np[spans_np > 1000000]
            
            logging.info(f"Span distribution analysis:")
            logging.info(f"  - Spans > 1,000: {len(large_spans)} ({len(large_spans)/len(spans_np)*100:.1f}%)")
            logging.info(f"  - Spans > 10,000: {len(very_large_spans)} ({len(very_large_spans)/len(spans_np)*100:.1f}%)")
            logging.info(f"  - Spans > 100,000: {len(huge_spans)} ({len(huge_spans)/len(spans_np)*100:.1f}%)")
            logging.info(f"  - Spans > 1,000,000: {len(massive_spans)} ({len(massive_spans)/len(spans_np)*100:.1f}%)")
            
            if len(massive_spans) > 0:
                logging.warning("⚠️  MASSIVE SPANS DETECTED (>1M) - This will cause severe I/O bottlenecks!")
                logging.warning(f"   Largest spans: {sorted(massive_spans, reverse=True)[:5]}")
            elif len(huge_spans) > 0:
                logging.warning("⚠️  HUGE SPANS DETECTED (>100k) - This may cause I/O performance issues!")
                logging.warning(f"   Largest spans: {sorted(huge_spans, reverse=True)[:5]}")
            elif len(very_large_spans) > 0:
                logging.info(f"ℹ️  Large spans detected (>10k): {len(very_large_spans)} instances")
                if len(very_large_spans) <= 10:
                    logging.info(f"   Large spans: {sorted(very_large_spans, reverse=True)}")
            else:
                logging.info("✅ Good locality - no problematic spans detected")
            
            # Percentile analysis
            percentiles = [50, 75, 90, 95, 99, 99.9]
            logging.info("Span percentiles:")
            for p in percentiles:
                val = np.percentile(spans_np, p)
                logging.info(f"  - {p:4.1f}%: {val:8.0f}")
        
        else:
            logging.info("No spans to analyze (all caption indices identical or insufficient data)")
        
        # Analyze vocabulary ordering if available
        if hasattr(dataset, '_vocab_split_indices'):
            vocab_indices = dataset._vocab_split_indices
            if len(vocab_indices) > 1:
                vocab_spans = []
                for i in range(len(vocab_indices) - 1):
                    span = vocab_indices[i + 1] - vocab_indices[i]
                    vocab_spans.append(span)
                
                if vocab_spans:
                    logging.info(f"Vocabulary ordering analysis:")
                    logging.info(f"  - Vocabulary size: {len(vocab_indices)}")
                    logging.info(f"  - Average vocab span: {np.mean(vocab_spans):.1f}")
                    logging.info(f"  - Max vocab span: {max(vocab_spans)}")
                    logging.info(f"  - Vocab spans > 1000: {sum(1 for s in vocab_spans if s > 1000)}")
        
        # Save detailed analysis if sample size is reasonable
        if len(spans) <= 50000:  # Don't save huge files
            try:
                debug_dir = "debug_validation"
                os.makedirs(debug_dir, exist_ok=True)
                
                analysis_data = {
                    "total_samples": int(total_samples),
                    "sampled_size": int(len(sampled_caption_indices)),
                    "caption_index_range": {
                        "min": int(unified_indices.min()),
                        "max": int(unified_indices.max())
                    },
                    "unique_captions": int(len(torch.unique(unified_indices))),
                    "span_statistics": {
                        "count": int(len(spans)),
                        "mean": float(np.mean(spans_np)) if len(spans) > 0 else 0,
                        "median": float(np.median(spans_np)) if len(spans) > 0 else 0,
                        "max": int(np.max(spans_np)) if len(spans) > 0 else 0,
                        "min": int(np.min(spans_np)) if len(spans) > 0 else 0,
                        "std": float(np.std(spans_np)) if len(spans) > 0 else 0,
                    },
                    "problematic_spans": {
                        "large_1k": int(len(large_spans)) if len(spans) > 0 else 0,
                        "very_large_10k": int(len(very_large_spans)) if len(spans) > 0 else 0,
                        "huge_100k": int(len(huge_spans)) if len(spans) > 0 else 0,
                        "massive_1m": int(len(massive_spans)) if len(spans) > 0 else 0,
                    }
                }
                
                if len(spans) > 0:
                    analysis_data["percentiles"] = {
                        f"p{p}": float(np.percentile(spans_np, p)) for p in [50, 75, 90, 95, 99, 99.9]
                    }
                    
                    # Include worst spans for debugging
                    if len(spans_np) > 0:
                        worst_spans = sorted(spans_np, reverse=True)[:20]
                        analysis_data["worst_spans"] = [int(s) for s in worst_spans]
                
                with open(f"{debug_dir}/caption_locality_analysis.json", "w") as f:
                    json.dump(analysis_data, f, indent=2)
                
                logging.info(f"Detailed locality analysis saved to: {debug_dir}/caption_locality_analysis.json")
                
            except Exception as e:
                logging.warning(f"Failed to save detailed locality analysis: {e}")
        
        logging.info("=== CAPTION LOCALITY DEBUG COMPLETE ===")
        
    except Exception as e:
        logging.error(f"Caption locality debug failed: {e}")
        import traceback
        traceback.print_exc()


def debug_dump_dataloader_samples(dataloader, dataset_name, out_dir="temp", n_batches=3, n_samples_per_batch=4):
    """
    Debug function to dump samples from a DataLoader for FT training verification.
    
    Args:
        dataloader: The DataLoader to sample from
        dataset_name: Name of the dataset (for file naming)
        out_dir: Output directory for debug files
        n_batches: Number of batches to sample
        n_samples_per_batch: Number of samples to save from each batch
    """
    try:
        import torch
        import torchvision.transforms.functional as TF
        
        os.makedirs(out_dir, exist_ok=True)
        logging.info(f"=== DEBUGGING DATALOADER SAMPLES ({dataset_name}) ===")
        
        dataloader_iter = iter(dataloader)
        
        for batch_idx in range(min(n_batches, len(dataloader))):
            try:
                batch = next(dataloader_iter)
                logging.info(f"Processing batch {batch_idx + 1}/{n_batches}")
                
                # Handle different batch formats
                if isinstance(batch, dict):
                    # Dictionary format (e.g., LAION400MNeg with tar-grouped batching)
                    images = batch.get("images")
                    pos_tokens = batch.get("pos_tokens")
                    neg_tokens = batch.get("neg_tokens", None)
                    pos_captions = batch.get("pos_captions", None)
                    neg_captions = batch.get("neg_captions", None)
                    
                    logging.info(f"Batch {batch_idx}: Dict format")
                    logging.info(f"  - Images shape: {images.shape if images is not None else 'None'}")
                    logging.info(f"  - Pos tokens shape: {pos_tokens.shape if pos_tokens is not None else 'None'}")
                    logging.info(f"  - Neg tokens shape: {neg_tokens.shape if neg_tokens is not None else 'None'}")
                    
                    batch_size = images.shape[0] if images is not None else 0
                    
                else:
                    # Tuple format (legacy)
                    if len(batch) >= 4:
                        images, pos_tokens, neg_tokens, extra = batch
                        pos_captions = None
                        neg_captions = None
                    elif len(batch) == 3:
                        images, pos_tokens, neg_tokens = batch
                        pos_captions = None
                        neg_captions = None
                    else:
                        logging.warning(f"Unexpected batch format with {len(batch)} elements")
                        continue
                    
                    logging.info(f"Batch {batch_idx}: Tuple format")
                    logging.info(f"  - Images shape: {images.shape}")
                    logging.info(f"  - Pos tokens shape: {pos_tokens.shape}")
                    logging.info(f"  - Neg tokens shape: {neg_tokens.shape if neg_tokens is not None else 'None'}")
                    
                    batch_size = images.shape[0]
                
                # Sample indices from this batch
                sample_indices = list(range(min(n_samples_per_batch, batch_size)))
                
                for sample_idx in sample_indices:
                    try:
                        # Save image
                        if images is not None and sample_idx < images.shape[0]:
                            img_tensor = images[sample_idx]  # [C, H, W]
                            
                            # Convert tensor to PIL Image
                            if img_tensor.dim() == 3:
                                # Denormalize if needed (assuming ImageNet normalization)
                                # CLIP typically uses mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
                                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                                
                                # Denormalize
                                img_tensor_denorm = img_tensor * std + mean
                                img_tensor_denorm = torch.clamp(img_tensor_denorm, 0, 1)
                                
                                # Convert to PIL
                                pil_image = TF.to_pil_image(img_tensor_denorm)
                                img_path = os.path.join(out_dir, f"batch{batch_idx}_sample{sample_idx}_image.jpg")
                                pil_image.save(img_path)
                                logging.info(f"Saved image: {img_path}")
                        
                        # Save text information
                        text_info = []
                        text_info.append(f"=== Batch {batch_idx}, Sample {sample_idx} ===\n")
                        
                        # Positive tokens/captions
                        if pos_tokens is not None and sample_idx < pos_tokens.shape[0]:
                            tokens = pos_tokens[sample_idx]
                            text_info.append(f"Positive tokens shape: {tokens.shape}")
                            text_info.append(f"Positive tokens (first 20): {tokens[:20].tolist()}")
                            text_info.append(f"Token range: {tokens.min().item()} to {tokens.max().item()}")
                        
                        # Positive captions (if available)
                        if pos_captions is not None and sample_idx < len(pos_captions):
                            text_info.append(f"Positive caption: {pos_captions[sample_idx]}")
                        
                        # Negative tokens/captions
                        if neg_tokens is not None and sample_idx < neg_tokens.shape[0]:
                            tokens = neg_tokens[sample_idx]
                            text_info.append(f"Negative tokens shape: {tokens.shape}")
                            text_info.append(f"Negative tokens (first 20): {tokens[:20].tolist()}")
                        
                        if neg_captions is not None and sample_idx < len(neg_captions):
                            text_info.append(f"Negative caption: {neg_captions[sample_idx]}")
                        
                        # Save text info
                        text_path = os.path.join(out_dir, f"batch{batch_idx}_sample{sample_idx}_text.txt")
                        with open(text_path, 'w') as f:
                            f.write('\n'.join(text_info))
                        
                        logging.info(f"Saved text info: {text_path}")
                        
                    except Exception as e:
                        logging.warning(f"Failed to save sample {sample_idx} from batch {batch_idx}: {e}")
                        
            except StopIteration:
                logging.info("Reached end of dataloader")
                break
            except Exception as e:
                logging.warning(f"Failed to process batch {batch_idx}: {e}")
                
        logging.info("=== DATALOADER DEBUG COMPLETE ===")
        logging.info(f"Debug files saved to: {out_dir}")
        
    except Exception as e:
        logging.error(f"Debug dump failed: {e}")
        import traceback
        traceback.print_exc()
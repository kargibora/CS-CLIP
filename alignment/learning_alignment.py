import logging
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from tqdm import tqdm

from utils.dist import is_main_process


# Training function for model with negative examples
class AllGatherVariableBatch(torch.autograd.Function):
    """
    Autograd-safe all_gather for variable local batch sizes using:
    pad -> all_gather (equal shapes) -> slice valid rows -> concat.
    Forward returns [sum_i bs_i, *trailing].
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        assert dist.is_initialized(), "torch.distributed must be initialized"
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # 1) Gather per-rank batch sizes (no grad needed)
        bs_local = x.shape[0]
        bs_t = torch.tensor([bs_local], device=x.device, dtype=torch.long)
        all_bs_list = [torch.zeros_like(bs_t) for _ in range(world_size)]
        dist.all_gather(all_bs_list, bs_t)
        batch_sizes = [int(t.item()) for t in all_bs_list]

        total_valid = sum(batch_sizes)
        trailing = x.shape[1:]

        if total_valid == 0:
            # everyone empty
            ctx.local_bs = 0
            ctx.offset = 0
            ctx.trailing = trailing
            ctx.batch_sizes = batch_sizes
            return x.new_zeros((0, *trailing))

        max_bs = max(batch_sizes)

        # 2) Pad locally to max_bs
        if bs_local < max_bs:
            pad = x.new_zeros((max_bs - bs_local, *trailing))
            x_pad = torch.cat([x, pad], dim=0)
        else:
            x_pad = x

        # 3) all_gather with equal-shaped tensors
        chunks = [x_pad.new_empty((max_bs, *trailing)) for _ in range(world_size)]
        dist.all_gather(chunks, x_pad)

        # 4) strip padding and concat valid rows (rank order)
        valid = [chunks[i][:batch_sizes[i]] for i in range(world_size) if batch_sizes[i] > 0]
        y = torch.cat(valid, dim=0)

        # Save slice info for backward
        offsets = [0]
        for bs in batch_sizes[:-1]:
            offsets.append(offsets[-1] + bs)

        ctx.local_bs = batch_sizes[rank]
        ctx.offset = offsets[rank]
        ctx.trailing = trailing
        ctx.batch_sizes = batch_sizes
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        if ctx.local_bs == 0:
            return grad_y.new_zeros((0, *ctx.trailing))
        start = ctx.offset
        end = start + ctx.local_bs
        return grad_y[start:end].contiguous()


# ---------- Optimized autograd Function (all_gather_into_tensor) ----------

class AllGatherVariableBatchOptimized(torch.autograd.Function):
    """
    Same semantics as above, but uses dist.all_gather_into_tensor for lower overhead.
    Requires PyTorch with dist.all_gather_into_tensor (>= 1.11; better >= 2.1).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        assert dist.is_initialized(), "torch.distributed must be initialized"
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # 1) Gather per-rank batch sizes
        bs_local = x.shape[0]
        bs_t = torch.tensor([bs_local], device=x.device, dtype=torch.long)
        bs_all = torch.empty(world_size, dtype=torch.long, device=x.device)
        dist.all_gather_into_tensor(bs_all, bs_t)  # -> [world_size]
        batch_sizes = bs_all.tolist()

        total_valid = int(bs_all.sum().item())
        trailing = x.shape[1:]

        if total_valid == 0:
            ctx.local_bs = 0
            ctx.offset = 0
            ctx.trailing = trailing
            ctx.batch_sizes = batch_sizes
            return x.new_zeros((0, *trailing))

        max_bs = int(bs_all.max().item())

        # 2) Pad locally to max_bs
        if bs_local < max_bs:
            pad = x.new_zeros((max_bs - bs_local, *trailing))
            x_pad = torch.cat([x, pad], dim=0)
        else:
            x_pad = x

        # 3) Gather into a single tensor of shape [world_size, max_bs, *trailing]
        gathered = torch.empty((world_size, max_bs, *trailing), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(gathered, x_pad)

        # 4) strip padding and concat valid rows (rank order)
        valid = []
        for i, bs in enumerate(batch_sizes):
            if bs > 0:
                valid.append(gathered[i, :bs])  # shape [bs, *trailing]
        y = torch.cat(valid, dim=0)

        # Save slice info for backward
        offsets = [0]
        for bs in batch_sizes[:-1]:
            offsets.append(offsets[-1] + bs)

        ctx.local_bs = batch_sizes[rank]
        ctx.offset = offsets[rank]
        ctx.trailing = trailing
        ctx.batch_sizes = batch_sizes
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        if ctx.local_bs == 0:
            return grad_y.new_zeros((0, *ctx.trailing))
        start = ctx.offset
        end = start + ctx.local_bs
        return grad_y[start:end].contiguous()


# ---------- Public helper ----------

def all_gather_variable_batch_with_grad(x: torch.Tensor, use_optimized: bool = True):
    """
    Returns:
        y: Tensor of shape [sum_i bs_i, *trailing], identical on all ranks.
        batch_sizes: Python list of per-rank sizes [bs_0, ..., bs_{p-1}].
    """
    if not dist.is_initialized():
        # Single process fallback
        return x, [int(x.shape[0])]

    # Use optimized path if available
    if use_optimized and hasattr(dist, "all_gather_into_tensor"):
        y = AllGatherVariableBatchOptimized.apply(x)
    else:
        y = AllGatherVariableBatch.apply(x)

    # Also return the batch sizes so caller can know N and per-rank offsets
    # (gather once more; cheap and avoids relying on ctx outside the Function)
    world_size = dist.get_world_size()
    bs_t = torch.tensor([x.shape[0]], device=x.device, dtype=torch.long)
    if hasattr(dist, "all_gather_into_tensor"):
        bs_all = torch.empty(world_size, dtype=torch.long, device=x.device)
        dist.all_gather_into_tensor(bs_all, bs_t)
        batch_sizes = bs_all.tolist()
    else:
        lst = [torch.zeros_like(bs_t) for _ in range(world_size)]
        dist.all_gather(lst, bs_t)
        batch_sizes = [int(t.item()) for t in lst]

    return y, batch_sizes

def synchronize_batch_processing(device, batch_idx, continue_training):
    """
    Synchronize batch processing across GPUs to handle uneven batch counts.
    Returns True if all GPUs should continue, False if any GPU is done.
    """
    world_size = dist.get_world_size()
    
    # Each GPU broadcasts whether it wants to continue
    continue_flags = [torch.zeros(1, device=device, dtype=torch.uint8) for _ in range(world_size)]
    local_flag = torch.tensor([1 if continue_training else 0], device=device, dtype=torch.uint8)
    dist.all_gather(continue_flags, local_flag)
    
    # All GPUs must have batches to continue
    all_continue = all(flag.item() > 0 for flag in continue_flags)
    
    return all_continue


def compute_contrastive_loss_with_merged_batches(
    image_embeddings, 
    text_embeddings, 
    neg_text_embeddings, 
    temperature, 
    device,
    loss_fn,
    skip_batch: bool = False,
    loss_kwargs: dict = None,
    entities_per_caption = None,
    num_entities_available = None,
    caption_valid_mask = None,
    paraphrase_embeddings = None,
    has_paraphrase = None,
):
    """
    Compute contrastive loss after merging batches from all distributed workers.
    Assumes `loss_fn` expects *global* (gathered) embeddings and uses global N for labels.
    
    Args:
        entities_per_caption: Optional[torch.Tensor] [B, N]
        num_entities_available: Optional[torch.Tensor] [B]
        caption_valid_mask: Optional[torch.Tensor] [B, 1+N] - validity mask for each caption
        paraphrase_embeddings: Optional[torch.Tensor] [B, D] - paraphrase text embeddings
        has_paraphrase: Optional[torch.Tensor] [B] - boolean mask for samples with paraphrases
    """
    if loss_kwargs is None:
        loss_kwargs = {}

    local_batch_size = 0 if skip_batch else int(image_embeddings.shape[0])

    # Early exit for empty/skip
    if skip_batch or local_batch_size == 0:
        dummy_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        return {
            'loss': dummy_loss,
            'accuracy': 0.0,
            'global_batch_size': 0,
            'local_batch_size': 0,
            'skip_batch': True
        }, 0

    if dist.is_initialized() and dist.get_world_size() > 1:
        # Detect multi-caption mode (3D tensors: [B, 1+N, D])
        is_multi_caption = text_embeddings.dim() == 3
        N_cap = text_embeddings.shape[1] if is_multi_caption else 1
        
        if is_multi_caption:
            # Flatten 3D [B, 1+N, D] -> 2D [B*(1+N), D] for all_gather
            B_local, _, D = text_embeddings.shape
            txt_flat = text_embeddings.view(B_local * N_cap, D).contiguous()
            neg_flat = neg_text_embeddings.view(B_local * N_cap, D).contiguous()
            
            # Gather all three streams
            img_all, img_sizes = all_gather_variable_batch_with_grad(image_embeddings)  # [B_total, D]
            txt_gathered, txt_sizes = all_gather_variable_batch_with_grad(txt_flat)  # [B_total*N_cap, D]
            neg_gathered, neg_sizes = all_gather_variable_batch_with_grad(neg_flat)  # [B_total*N_cap, D]
            
            # Unflatten back to 3D: [B_total*(1+N), D] -> [B_total, 1+N, D]
            total_B = sum(img_sizes)
            txt_all = txt_gathered.view(total_B, N_cap, D)
            neg_all = neg_gathered.view(total_B, N_cap, D)
        else:
            # Standard 2D mode [B, D]
            img_all, img_sizes = all_gather_variable_batch_with_grad(image_embeddings)
            txt_all, txt_sizes = all_gather_variable_batch_with_grad(text_embeddings)
            neg_all, neg_sizes = all_gather_variable_batch_with_grad(neg_text_embeddings)
        
        # Gather coverage information if present
        entities_all = None
        num_available_all = None
        valid_mask_all = None
        paraphrase_all = None
        has_paraphrase_all = None
        if entities_per_caption is not None:
            B, N = entities_per_caption.shape
            entities_flat = entities_per_caption.reshape(-1).float()
            entities_gathered, _ = all_gather_variable_batch_with_grad(entities_flat)
            total_B = sum(img_sizes)
            entities_all = entities_gathered.reshape(total_B, N).long()
        if num_entities_available is not None:
            num_available_all, _ = all_gather_variable_batch_with_grad(num_entities_available.float())
            num_available_all = num_available_all.long()  # Convert back to long
        if caption_valid_mask is not None:
            # caption_valid_mask is [B, 1+N], need to gather across GPUs
            B, M = caption_valid_mask.shape
            valid_flat = caption_valid_mask.reshape(-1).float()  # Convert bool to float for gather
            valid_gathered, _ = all_gather_variable_batch_with_grad(valid_flat)
            total_B = sum(img_sizes)
            valid_mask_all = valid_gathered.reshape(total_B, M).bool()  # Convert back to bool
        
        # Gather paraphrase embeddings if present
        if paraphrase_embeddings is not None:
            paraphrase_all, _ = all_gather_variable_batch_with_grad(paraphrase_embeddings)
        if has_paraphrase is not None:
            has_paraphrase_float = has_paraphrase.float()  # Convert bool to float for gather
            has_paraphrase_gathered, _ = all_gather_variable_batch_with_grad(has_paraphrase_float)
            has_paraphrase_all = has_paraphrase_gathered.bool()  # Convert back to bool

        N_img = sum(img_sizes)
        N_txt = sum(txt_sizes)
        N_neg = sum(neg_sizes)

        # Strict sanity check - differs between multi-caption and standard mode
        if is_multi_caption:
            # In multi-caption mode: txt/neg were flattened, so N_txt == N_img * N_cap
            # After unflattening: txt_all.size(0) == N_img and txt_all.shape[1] == N_cap
            expected_txt_size = N_img * N_cap
            if not (N_txt == N_neg == expected_txt_size):
                raise RuntimeError(
                    f"Global batch mismatch (multi-caption): N_img={N_img}, N_txt={N_txt}, N_neg={N_neg}, "
                    f"expected txt/neg size={expected_txt_size} (N_img * N_cap)"
                )
            if not (img_all.size(0) == txt_all.size(0) == neg_all.size(0) == N_img):
                raise RuntimeError(
                    f"Global batch mismatch after unflatten: img={img_all.size(0)}, txt={txt_all.size(0)}, neg={neg_all.size(0)}, expected={N_img}"
                )
        else:
            # Standard mode: all sizes should match
            if not (N_img == N_txt == N_neg == img_all.size(0) == txt_all.size(0) == neg_all.size(0)):
                raise RuntimeError(
                    f"Global batch mismatch: N_img={N_img}, N_txt={N_txt}, N_neg={N_neg}, "
                    f"gathered sizes: img={img_all.size(0)}, txt={txt_all.size(0)}, neg={neg_all.size(0)}"
                )

        if N_img > 0:
            # Add coverage to loss_kwargs if present
            if entities_all is not None:
                loss_kwargs = {**loss_kwargs, 'components_per_caption': entities_all}
            if num_available_all is not None:
                loss_kwargs = {**loss_kwargs, 'num_components_available': num_available_all}
            if valid_mask_all is not None:
                loss_kwargs = {**loss_kwargs, 'caption_valid_mask': valid_mask_all}
            # Add paraphrase embeddings if present
            if paraphrase_all is not None:
                loss_kwargs = {**loss_kwargs, 'paraphrase_embeddings': paraphrase_all}
            if has_paraphrase_all is not None:
                loss_kwargs = {**loss_kwargs, 'has_paraphrase': has_paraphrase_all}
            
            loss_dict = loss_fn(
                image_embeddings=img_all,
                pos_text_embeddings=txt_all,
                neg_text_embeddings=neg_all,
                temperature=temperature,
                device=device,
                **loss_kwargs
            )
        else:
            loss_dict = {
                'loss': torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True),
                'accuracy': 0.0
            }

        loss_dict['global_batch_size'] = N_img
        loss_dict['local_batch_size'] = local_batch_size
        return loss_dict, local_batch_size

    # Single-process
    # Add coverage to loss_kwargs if present
    if entities_per_caption is not None:
        loss_kwargs = {**loss_kwargs, 'components_per_caption': entities_per_caption}
    if num_entities_available is not None:
        loss_kwargs = {**loss_kwargs, 'num_components_available': num_entities_available}
    if caption_valid_mask is not None:
        loss_kwargs = {**loss_kwargs, 'caption_valid_mask': caption_valid_mask}
    # Add paraphrase embeddings if present
    if paraphrase_embeddings is not None:
        loss_kwargs = {**loss_kwargs, 'paraphrase_embeddings': paraphrase_embeddings}
    if has_paraphrase is not None:
        loss_kwargs = {**loss_kwargs, 'has_paraphrase': has_paraphrase}
    
    loss_dict = loss_fn(
        image_embeddings=image_embeddings,
        pos_text_embeddings=text_embeddings,
        neg_text_embeddings=neg_text_embeddings,
        temperature=temperature,
        device=device,
        **loss_kwargs
    )
    loss_dict['global_batch_size'] = local_batch_size
    loss_dict['local_batch_size'] = local_batch_size
    return loss_dict, local_batch_size


def train_model_multigpu_merged_batch(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    evaluate_fn=None,
    scheduler=None,
    metric_key='contrastive_accuracy',
    batch_unpack_fn=None,
    scheduler_on='epoch',
    cfg=None,
    gpu_wrapper=None,
    train_sampler=None,
    loss_kwargs=None,
    train_cfg=None,
    loss_cfg=None,
):
    """
    Minimal FT training loop:
    train each epoch, run validation on schedule, save best/last checkpoints.
    """

    if loss_kwargs is None:
        if loss_cfg is not None:
            from omegaconf import OmegaConf
            loss_kwargs = OmegaConf.to_container(loss_cfg, resolve=True)
            if not isinstance(loss_kwargs, dict):
                loss_kwargs = {}
        else:
            loss_kwargs = {}

    distributed = getattr(cfg.dist, 'distributed', False) if cfg else False
    should_log = is_main_process()
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0

    num_epochs = getattr(train_cfg, 'epochs', 10) if train_cfg else 10
    evaluate_every_n = getattr(train_cfg, 'eval_n', 1) if train_cfg else 1
    use_amp = getattr(train_cfg, 'use_amp', False) if train_cfg else False
    grad_clip_norm = getattr(train_cfg, 'grad_clip_norm', 0.0) if train_cfg else 0.0
    save_path = getattr(train_cfg, 'save_path', None) if train_cfg else None
    save_every_k_steps = getattr(train_cfg, 'save_every_k_steps', None) if train_cfg else None
    exp_name = getattr(cfg, 'exp_name', 'experiment') if cfg else 'experiment'

    if should_log:
        logging.info("Starting FT training for %d epochs", num_epochs)

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_metric_value = -float('inf')
    best_model_state = None
    best_epoch = -1
    global_step = 0

    epoch_iter = tqdm(
        range(num_epochs), 
        desc="Training",
        unit="epoch",
        ncols=100,
        colour='cyan',
    ) if should_log else range(num_epochs)

    for epoch in epoch_iter:
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        model.train()
        loss_sums = defaultdict(float)
        loss_counts = defaultdict(int)
        train_metrics = []
        
        batch_count = 0
        skipped_batches = 0
        processed_batches = 0

        train_iter = iter(train_loader)

        if should_log:
            total_batches = len(train_loader) if hasattr(train_loader, '__len__') else None
            pbar = tqdm(
                total=total_batches,
                desc=f"Epoch {epoch+1:3d}/{num_epochs}",
                leave=False,
                ncols=150,
                colour='yellow',
            )
        else:
            pbar = None

        while True:
            try:
                batch = next(train_iter)
                has_batch = True
            except StopIteration:
                has_batch = False
                batch = None

            if distributed:
                all_continue = synchronize_batch_processing(device, batch_count, has_batch)
                if not all_continue:
                    break
            elif not has_batch:
                break

            batch_count += 1
            skip_this_batch = True
            loss_dict = {'skip_batch': True, 'loss': torch.tensor(0.0, device=device)}

            if has_batch and batch is not None:
                optimizer.zero_grad()

                if batch_unpack_fn is None:
                    raise ValueError("batch_unpack_fn must be provided")

                try:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        unpack = batch_unpack_fn(batch, model, device)
                        pos_text_embeddings = unpack['text_embeddings']
                        neg_text_embeddings = unpack.get('neg_text_embeddings')
                        image_embeddings = unpack['image_embeddings']
                        temperature = unpack['temperature']
                        unpack_device = unpack['device']
                        entities_per_caption = unpack.get('entities_per_caption')
                        num_entities_available = unpack.get('num_entities_available')
                        caption_valid_mask = unpack.get('caption_valid_mask')  # [B, 1+N] NEW!
                        # Paraphrase embeddings for sentence alignment loss
                        paraphrase_embeddings = unpack.get('paraphrase_embeddings')  # [B, D] or None
                        has_paraphrase = unpack.get('has_paraphrase')  # [B] bool or None

                        if image_embeddings.shape[0] == 0:
                            image_embeddings = torch.zeros(1, image_embeddings.shape[1], device=device, dtype=image_embeddings.dtype)
                            pos_text_embeddings = torch.zeros(1, pos_text_embeddings.shape[1], device=device, dtype=pos_text_embeddings.dtype)
                            if neg_text_embeddings is not None:
                                neg_text_embeddings = torch.zeros(1, neg_text_embeddings.shape[1], device=device, dtype=neg_text_embeddings.dtype)
                            skip_this_batch = True
                        else:
                            skip_this_batch = False
                        
                        if not skip_this_batch:
                            if torch.isnan(image_embeddings).any() or torch.isinf(image_embeddings).any():
                                logging.error(f"Rank {rank}: Batch {batch_count} - NaN/Inf detected in image_embeddings!")
                                skip_this_batch = True
                            elif torch.isnan(pos_text_embeddings).any() or torch.isinf(pos_text_embeddings).any():
                                logging.error(f"Rank {rank}: Batch {batch_count} - NaN/Inf detected in text_embeddings!")
                                skip_this_batch = True
                            elif neg_text_embeddings is not None and neg_text_embeddings.shape[0] > 0:
                                if torch.isnan(neg_text_embeddings).any() or torch.isinf(neg_text_embeddings).any():
                                    logging.error(f"Rank {rank}: Batch {batch_count} - NaN/Inf detected in neg_text_embeddings!")
                                    skip_this_batch = True
                        
                        if neg_text_embeddings is None:
                            neg_text_embeddings = torch.zeros(0, pos_text_embeddings.shape[1],
                                                              dtype=pos_text_embeddings.dtype,
                                                              device=device)

                        loss_dict, local_batch_size = compute_contrastive_loss_with_merged_batches(
                            image_embeddings=image_embeddings,
                            text_embeddings=pos_text_embeddings,
                            neg_text_embeddings=neg_text_embeddings,
                            temperature=temperature,
                            device=unpack_device,
                            loss_fn=loss_fn,
                            skip_batch=skip_this_batch,
                            loss_kwargs=loss_kwargs,
                            entities_per_caption=entities_per_caption,
                            num_entities_available=num_entities_available,
                            caption_valid_mask=caption_valid_mask,
                            paraphrase_embeddings=paraphrase_embeddings,
                            has_paraphrase=has_paraphrase,
                        )

                        if not loss_dict.get('skip_batch', False):
                            if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                                loss_dict['learning_rate'] = optimizer.param_groups[0].get('lr', 0)

                except Exception as e:
                    logging.error(f"Rank {rank}: Error processing batch {batch_count}: {e}")
                    import traceback
                    logging.error(f"Rank {rank}: Traceback:\n{traceback.format_exc()}")
                    loss_dict = {
                        'loss': torch.tensor(0.0, device=device, requires_grad=True),
                        'accuracy': 0.0,
                        'skip_batch': True,
                        'skip_reason': f'Exception: {str(e)[:100]}'
                    }
                    skip_this_batch = True

            if not loss_dict.get('skip_batch', False):
                total_loss = loss_dict['loss']
                if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                    if use_amp and scaler is not None:
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        grad_norm = 0.0
                        param_count = 0
                        for param in model.parameters():
                            if param.grad is not None:
                                grad_norm += param.grad.data.norm(2).item() ** 2
                                param_count += param.numel()
                        grad_norm = grad_norm ** 0.5

                        if grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        grad_norm = 0.0
                        param_count = 0
                        for param in model.parameters():
                            if param.grad is not None:
                                grad_norm += param.grad.data.norm(2).item() ** 2
                                param_count += param.numel()
                        grad_norm = grad_norm ** 0.5

                        if grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                        optimizer.step()

                    loss_dict['grad_norm'] = grad_norm
                    loss_dict['param_count'] = param_count
                    processed_batches += 1

                    for k, v in loss_dict.items():
                        if k in ['global_batch_size', 'local_batch_size', 'skip_batch']:
                            continue
                        v = v.item() if torch.is_tensor(v) else float(v)
                        loss_sums[k] += v
                        loss_counts[k] += 1

                    acc = loss_dict.get("accuracy", None)
                    if acc is not None:
                        train_metrics.append(acc)
                else:
                    skipped_batches += 1
                    logging.warning(f"Rank {rank}: Batch {batch_count} - Skipped due to NaN/Inf loss (loss={total_loss.item() if torch.is_tensor(total_loss) else total_loss})")
            else:
                skipped_batches += 1

            if not loss_dict.get('skip_batch', False):
                global_step += 1

            if scheduler is not None and scheduler_on == 'step' and not skip_this_batch:
                scheduler.step()

            if (
                should_log
                and save_path
                and save_every_k_steps is not None
                and save_every_k_steps > 0
                and not loss_dict.get('skip_batch', False)
                and global_step % save_every_k_steps == 0
            ):
                ckpt_dir = os.path.join(save_path, f"{exp_name}")
                os.makedirs(ckpt_dir, exist_ok=True)
                base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
                step_ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{global_step}.pt")
                torch.save(base_model.state_dict(), step_ckpt_path)
                logging.info("Saved step checkpoint to %s", step_ckpt_path)

            if should_log and pbar is not None:
                postfix_dict = {}
                main_loss = 0.0

                if loss_dict and not loss_dict.get('skip_batch', False):
                    main_loss = loss_dict.get('total_loss', loss_dict.get('loss', 0))
                    if isinstance(main_loss, torch.Tensor):
                        main_loss = main_loss.item()
                    postfix_dict['Loss'] = f"{main_loss:.3f}"

                    if 'accuracy' in loss_dict:
                        acc = loss_dict['accuracy']
                        if isinstance(acc, torch.Tensor):
                            acc = acc.item()
                        postfix_dict['Acc'] = f"{acc:.3f}"

                    for k, v in loss_dict.items():
                        if k in ['accuracy', 'global_batch_size', 'learning_rate', 'temperature', 'grad_norm']:
                            continue
                        if k in ['loss', 'total_loss']:
                            continue
                        if isinstance(v, torch.Tensor):
                            v = v.item()
                        display_key = k.replace('_loss', '').replace('train_', '').title()[:8]
                        postfix_dict[display_key] = f"{v:.3f}"

                    if 'global_batch_size' in loss_dict:
                        postfix_dict['BS'] = int(loss_dict['global_batch_size'])

                    if 'learning_rate' in loss_dict:
                        lr = loss_dict['learning_rate']
                        postfix_dict['LR'] = f"{lr:.2e}"

                    if 'temperature' in loss_dict:
                        temp = loss_dict['temperature']
                        if isinstance(temp, torch.Tensor):
                            temp = temp.item()
                        postfix_dict['Temp'] = f"{temp:.3f}"

                    if 'grad_norm' in loss_dict:
                        grad_norm = loss_dict['grad_norm']
                        if isinstance(grad_norm, torch.Tensor):
                            grad_norm = grad_norm.item()
                        postfix_dict['GradNorm'] = f"{grad_norm:.2f}"
                else:
                    postfix_dict['Status'] = 'SKIPPED'

                batch_desc = f"Batch {batch_count}" + (f"/{total_batches}" if total_batches else "")
                pbar.set_description(f"Epoch {epoch+1:3d}/{num_epochs} | {batch_desc} | Loss: {main_loss:.3f}")
                pbar.set_postfix(postfix_dict)
                pbar.update(1)

        if should_log and pbar is not None:
            pbar.close()

        if distributed:
            all_batch_counts = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
            local_count = torch.tensor([processed_batches], device=device, dtype=torch.long)
            dist.all_gather(all_batch_counts, local_count)

            if should_log:
                counts = [c.item() for c in all_batch_counts]
                logging.info(f"Epoch {epoch+1}: Processed batches per GPU: {counts}, Total: {sum(counts)}")

        if distributed:
            dist.barrier()

            for k in loss_sums:
                try:
                    sum_t = torch.tensor(loss_sums[k], device=device, dtype=torch.float32)
                    count_t = torch.tensor(loss_counts[k], device=device, dtype=torch.float32)

                    dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count_t, op=dist.ReduceOp.SUM)

                    loss_sums[k] = sum_t.item()
                    loss_counts[k] = max(1, int(count_t.item()))
                except Exception as e:
                    logging.warning(f"Metric aggregation failed for {k}: {e}")

            if train_metrics:
                try:
                    acc_sum = torch.tensor(sum(train_metrics), device=device, dtype=torch.float32)
                    acc_count = torch.tensor(len(train_metrics), device=device, dtype=torch.float32)

                    dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

                    train_accuracy = acc_sum.item() / max(1, acc_count.item())
                except Exception as e:
                    logging.warning(f"Accuracy aggregation failed: {e}")
                    train_accuracy = 0.0
            else:
                train_accuracy = 0.0
        else:
            train_accuracy = float(sum(train_metrics)) / max(1, len(train_metrics)) if train_metrics else 0.0

        avg_losses = {f"train_{k}": loss_sums[k] / max(1, loss_counts[k]) for k in loss_sums}

        logs = {}
        if evaluate_fn is not None and (epoch + 1) % evaluate_every_n == 0:
            base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
            logs = evaluate_fn(base_model, val_loader, device, loss_fn, **loss_kwargs)

        if metric_key in logs and logs[metric_key] is not None:
            if logs[metric_key] > best_metric_value:
                best_metric_value = logs[metric_key]
                best_epoch = epoch
                base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
                best_model_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

        if should_log:
            base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
            parameters_dict = {}

            if hasattr(base_model, "get_alphas"):
                alphas = base_model.get_alphas()
                if alphas.get('image') is not None:
                    parameters_dict.update({f'parameters/alpha_image_{i}': a.item() for i, a in enumerate(alphas['image'])})
                if alphas.get('text') is not None:
                    parameters_dict.update({f'parameters/alpha_text_{i}': a.item() for i, a in enumerate(alphas['text'])})

            if hasattr(base_model, "temperature"):
                temperature = getattr(base_model, "temperature", None)
                if temperature is not None:
                    parameters_dict['parameters/temperature'] = temperature.item() if torch.is_tensor(temperature) else temperature

            parameters_dict['parameters/lr'] = optimizer.param_groups[0]['lr']
            parameters_dict['train/processed_batches'] = processed_batches
            parameters_dict['train/skipped_batches'] = skipped_batches

            if 'grad_norm' in avg_losses:
                parameters_dict['train/grad_norm'] = avg_losses['grad_norm']
            if 'param_count' in avg_losses:
                parameters_dict['train/param_count'] = avg_losses['param_count']

            logs.update(parameters_dict)
            logs.update(avg_losses)
            logs.update({'train_accuracy': train_accuracy})

            summary_parts = [
                f"epoch={epoch + 1}/{num_epochs}",
                f"train_loss={avg_losses.get('train_loss', avg_losses.get('train_total_loss', 0.0)):.4f}",
                f"train_acc={train_accuracy:.4f}",
                f"processed={processed_batches}",
                f"skipped={skipped_batches}",
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
            ]
            if logs.get(metric_key) is not None:
                summary_parts.append(f"{metric_key}={logs[metric_key]:.4f}")
            logging.info(" | ".join(summary_parts))

            if isinstance(epoch_iter, tqdm):
                epoch_metrics = {}
                main_loss = avg_losses.get('total_loss', avg_losses.get('loss', 0))
                epoch_metrics["Loss"] = f"{main_loss:.4f}"
                epoch_metrics["Acc"] = f"{train_accuracy:.4f}"
                epoch_metrics["Proc"] = f"{processed_batches}"
                if skipped_batches > 0:
                    epoch_metrics["Skip"] = f"{skipped_batches}"
                if hasattr(base_model, "temperature"):
                    temperature = getattr(base_model, "temperature", None)
                    if temperature is not None:
                        epoch_metrics["T"] = f"{temperature.item():.2f}" if torch.is_tensor(temperature) else f"{temperature:.2f}"
                epoch_metrics["LR"] = f"{optimizer.param_groups[0]['lr']:.1e}"
                if logs.get(metric_key) is not None:
                    epoch_metrics["Val"] = f"{logs[metric_key]:.4f}"
                epoch_iter.set_postfix(epoch_metrics)

        if scheduler is not None and scheduler_on == 'epoch':
            scheduler.step()

        if should_log and save_path:
            ckpt_dir = os.path.join(save_path, f"{exp_name}")
            os.makedirs(ckpt_dir, exist_ok=True)
            base_model = gpu_wrapper.get_base_model() if gpu_wrapper else model
            torch.save(base_model.state_dict(), os.path.join(ckpt_dir, "last_checkpoint.pt"))

    return {
        "best_model_state_dict": best_model_state,
        "best_epoch": best_epoch,
        "best_step": global_step,
        "best_metric_value": best_metric_value,
        "metric_key": metric_key,
        "final_global_step": global_step,
    }




def unpack_ft_multilayer(batch, model, device):
    """
    Unpacking function that handles both standard mode (single positive) and multi-caption mode (multiple positives).
    
    NEW FORMAT: Each positive caption has a paired negative.
    - pos_tokens: [B, 1+N, 77] where index 0 = original, 1:N+1 = entity/relation captions
    - neg_tokens: [B, 1+N, 77] where neg[i] is paired with pos[i]
    - paraphrase_tokens: [B, 77] - paraphrase of original caption (for sentence alignment loss)
    - has_paraphrase: [B] - boolean mask for valid paraphrases
    - caption_valid_mask: [B, 1+N] boolean mask for valid pos-neg pairs
    
    Returns:
        dict with:
            - image_embeddings: [B, D]
            - text_embeddings: [B, 1+N, D] - positive caption embeddings
            - neg_text_embeddings: [B, 1+N, D] - paired negative embeddings (NEW: now matches positives shape!)
            - paraphrase_embeddings: Optional[torch.Tensor] [B, D] - paraphrase embeddings (None if no paraphrases)
            - has_paraphrase: Optional[torch.Tensor] [B] - mask for valid paraphrases
            - temperature: scalar
            - device: device
            - num_positive_captions: int (1 for standard mode, 1+N for multi-caption mode)
            - entities_per_caption: Optional[torch.Tensor] [B, 1+N]
            - num_entities_available: Optional[torch.Tensor] [B]
            - caption_valid_mask: Optional[torch.Tensor] [B, 1+N] - boolean mask for valid pairs
    """
    entities_per_caption = None
    num_entities_available = None
    caption_valid_mask = None
    paraphrase_tokens = None
    has_paraphrase = None
    
    if isinstance(batch, dict):
        images = batch["images"]
        captions = batch["pos_tokens"]  # [B, 1+N, 77]
        neg_captions = batch.get("neg_tokens", batch.get("neg_token"))
        entities_per_caption = batch.get("entities_per_caption")  # [B, 1+N]
        num_entities_available = batch.get("num_entities_available")  # [B]
        caption_valid_mask = batch.get("caption_valid_mask")  # [B, 1+N]
        paraphrase_tokens = batch.get("paraphrase_tokens")  # [B, 77] or None
        has_paraphrase = batch.get("has_paraphrase")  # [B] or None
    else:
        # Tuple format (legacy)
        if len(batch) == 4:
            images, captions, neg_captions, _ = batch
        else:
            images, captions, neg_captions = batch
        
    images = images.to(device)
    captions = captions.to(device)  # [B, 1+N, 77]
    neg_captions = neg_captions.to(device)
    
    if entities_per_caption is not None:
        entities_per_caption = entities_per_caption.to(device)
    if num_entities_available is not None:
        num_entities_available = num_entities_available.to(device)
    if caption_valid_mask is not None:
        caption_valid_mask = caption_valid_mask.to(device)
    
    # Move paraphrase tensors to device if present
    if paraphrase_tokens is not None:
        paraphrase_tokens = paraphrase_tokens.to(device)
    if has_paraphrase is not None:
        has_paraphrase = has_paraphrase.to(device)
    
    # Detect mode: check if captions has multiple positives per sample
    batch_size = images.shape[0]
    num_positive_captions = captions.shape[1] if captions.dim() == 3 else 1
    
    # Handle DDP wrapped models
    base_model = model.module if hasattr(model, 'module') else model
    
    # Encode images (same for both modes)
    image_emb = base_model.encode_image(images)  # [B, D]
    
    # Encode positive text captions
    if num_positive_captions > 1:
        # Multi-caption mode: encode all positive captions
        # Reshape to [B*(1+N), 77] for batch encoding
        captions_flat = captions.view(-1, captions.shape[-1])  # [B*(1+N), 77]
        text_emb_flat = base_model.encode_text(captions_flat)  # [B*(1+N), D]
        # Reshape back to [B, 1+N, D]
        text_emb = text_emb_flat.view(batch_size, num_positive_captions, -1)  # [B, 1+N, D]
    else:
        # Standard mode: single positive caption per sample
        if captions.dim() == 3:
            captions = captions.squeeze(1)  # [B, 77]
        text_emb = base_model.encode_text(captions)  # [B, D]
        # Add dimension for consistency: [B, 1, D]
        text_emb = text_emb.unsqueeze(1)  # [B, 1, D]
    
    # Encode negative captions - NEW: handle paired negatives format [B, 1+N, 77]
    if neg_captions.dim() == 3 and neg_captions.shape[1] == num_positive_captions:
        # NEW FORMAT: Paired negatives [B, 1+N, 77] - one negative per positive
        neg_captions_flat = neg_captions.view(-1, neg_captions.shape[-1])  # [B*(1+N), 77]
        neg_text_emb_flat = base_model.encode_text(neg_captions_flat)  # [B*(1+N), D]
        # Reshape back to [B, 1+N, D]
        neg_text_emb = neg_text_emb_flat.view(batch_size, num_positive_captions, -1)  # [B, 1+N, D]
    elif neg_captions.dim() == 2:
        # LEGACY FORMAT: Single negative per sample [B, 77]
        neg_text_emb = base_model.encode_text(neg_captions)  # [B, D]
        # Broadcast to match positive shape: [B, 1, D] -> will be [B, 1+N, D] if needed
        neg_text_emb = neg_text_emb.unsqueeze(1)  # [B, 1, D]
        if num_positive_captions > 1:
            # Broadcast single negative to all positive captions
            # Note: .contiguous() is needed after expand for distributed all_gather
            neg_text_emb = neg_text_emb.expand(-1, num_positive_captions, -1).contiguous()  # [B, 1+N, D]
    else:
        # Unexpected format - log warning and try to handle
        logging.warning(f"Unexpected neg_captions shape: {neg_captions.shape}, attempting to handle...")
        if neg_captions.dim() > 2:
            # Take first negative if multiple
            neg_captions = neg_captions[:, 0, :]
        neg_text_emb = base_model.encode_text(neg_captions)  # [B, D]
        neg_text_emb = neg_text_emb.unsqueeze(1)  # [B, 1, D]
        if num_positive_captions > 1:
            # Note: .contiguous() is needed after expand for distributed all_gather
            neg_text_emb = neg_text_emb.expand(-1, num_positive_captions, -1).contiguous()

    # Encode paraphrase captions if available (for sentence alignment loss)
    paraphrase_emb = None
    if paraphrase_tokens is not None and has_paraphrase is not None:
        # Only encode if at least some samples have paraphrases
        if has_paraphrase.any():
            paraphrase_emb = base_model.encode_text(paraphrase_tokens)  # [B, D]
        # Note: samples without paraphrases will be masked out using has_paraphrase

    return {
        "image_embeddings": image_emb,  # [B, D]
        "text_embeddings": text_emb,  # [B, 1+N, D]
        "neg_text_embeddings": neg_text_emb,  # [B, 1+N, D]
        "paraphrase_embeddings": paraphrase_emb,  # [B, D] or None - paraphrase embeddings
        "has_paraphrase": has_paraphrase,  # [B] or None - mask for valid paraphrases
        "temperature": base_model.temperature.to(device),
        "device": device,
        "num_positive_captions": num_positive_captions,
        "entities_per_caption": entities_per_caption,  # [B, 1+N] or None
        "num_entities_available": num_entities_available,  # [B] or None
        "caption_valid_mask": caption_valid_mask,  # [B, 1+N] or None
    }

import logging
import os
import copy
import datetime
import traceback
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np

# Configure logging for this module
logger = logging.getLogger(__name__)

# Add these utility functions at the top of your main script

def is_distributed_launch_mode():
    """Check if we're running under torch.distributed.launch"""
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

def get_distributed_info():
    """Get distributed training information from environment or defaults"""
    if is_distributed_launch_mode():
        # torch.distributed.launch mode
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        world_size = int(os.environ['WORLD_SIZE'])
        return rank, local_rank, world_size
    else:
        # Single process or multiprocessing.spawn mode
        return 0, 0, 1

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def verify_process_group_health(rank, world_size):
    """Verify that the process group is healthy and can communicate"""
    try:
        # Test basic tensor communication
        test_tensor = torch.tensor([rank], dtype=torch.float32, device=torch.cuda.current_device())
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM, async_op=False)
        expected_sum = sum(range(world_size))
        
        if abs(test_tensor.item() - expected_sum) < 1e-6:
            logging.info(f"Rank {rank}: Process group health check passed")
            return True
        else:
            logging.error(f"Rank {rank}: Process group health check failed. Got {test_tensor.item()}, expected {expected_sum}")
            return False
            
    except Exception as e:
        logging.error(f"Rank {rank}: Process group health check failed with error: {e}")
        return False

def setup_distributed(rank=None, world_size=None):
    """Initialize distributed training with robust NCCL configuration
    
    Args:
        rank: Process rank (optional, will be read from environment if not provided)
        world_size: Total number of processes (optional, will be read from environment if not provided)
    """
    # When using torch.distributed.launch, rank and world_size are set via environment variables
    if rank is None:
        rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Use environment variables if available, otherwise set defaults
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    
    # For torch.distributed.launch, MASTER_PORT is usually set by the launcher
    # For multiprocessing.spawn, it should be set by the main process
    if 'MASTER_PORT' not in os.environ:
        # Only raise error if we're actually doing distributed training
        if world_size > 1:
            raise ValueError("MASTER_PORT environment variable must be set before calling setup_distributed")
    
    # Comprehensive NCCL environment configuration for stability
    nccl_settings = {
        'NCCL_TIMEOUT': '3600',  # 1 hour timeout for very large models
        'NCCL_BLOCKING_WAIT': '1',  # Enable blocking wait for better error reporting
        'NCCL_ASYNC_ERROR_HANDLING': '1',  # Better error handling
        'NCCL_DEBUG': 'WARN',  # Enable warnings but not verbose debug
        'NCCL_IB_DISABLE': '1',  # Disable InfiniBand if causing issues
        'NCCL_SOCKET_IFNAME': '^lo,docker',  # Avoid loopback and docker interfaces
        'NCCL_P2P_DISABLE': '1',  # Disable P2P for stability
        'NCCL_SHM_DISABLE': '1',  # Disable shared memory for stability
        'NCCL_TREE_THRESHOLD': '0',  # Force ring algorithm (more stable)
        'NCCL_MIN_NCHANNELS': '1',  # Use minimum channels for stability
        'NCCL_MAX_NCHANNELS': '1',  # Limit channels for stability
        'CUDA_LAUNCH_BLOCKING': '1',  # Synchronous CUDA kernel launches for debugging
    }
    
    # Apply NCCL settings only if not already set
    for key, value in nccl_settings.items():
        if key not in os.environ:
            os.environ[key] = value
    
    if rank == 0:
        logging.info(f"Using master port from environment: {os.environ['MASTER_PORT']}")
        logging.info(f"NCCL timeout set to: {os.environ.get('NCCL_TIMEOUT')} seconds")
        logging.info("Applied NCCL stability settings for robust communication")
    
    # Debug: print what each rank sees
    logging.info(f"Rank {rank}: MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")
    
    # Initialize the process group with extended timeout
    try:
        timeout = datetime.timedelta(seconds=3600)  # 1 hour for initialization
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
        torch.cuda.set_device(rank)
        
        # Verify process group health immediately
        if not verify_process_group_health(rank, world_size):
            raise RuntimeError(f"Process group health check failed on rank {rank}")
            
        logging.info(f"Successfully initialized distributed training for rank {rank}/{world_size}")
    except Exception as e:
        logging.error(f"Failed to initialize distributed training on rank {rank}: {e}")
        raise

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logging.info("Distributed process group destroyed")

def is_main_process():
    """Check if this is the main process (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_world_size():
    """Get the number of processes"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def get_rank():
    """Get the rank of current process"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

class MultiGPUWrapper:
    """Wrapper class to handle both DataParallel and DistributedDataParallel"""
    
    def __init__(self, model, args, device_ids=None):
        self.args = args
        self.original_model = model
        self.wrapped_model = None
        self.device_ids = device_ids
        
        if args.distributed:
            # DistributedDataParallel (recommended)
            self.setup_ddp()
        elif args.data_parallel and torch.cuda.device_count() > 1:
            # DataParallel (simpler but less efficient)
            self.setup_dp()
        else:
            # Single GPU or CPU
            self.wrapped_model = model
    
    def setup_dp(self):
        """Setup DataParallel"""
        logging.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        self.wrapped_model = DataParallel(self.original_model, device_ids=self.device_ids)
    
    def setup_ddp(self):
        """Setup DistributedDataParallel"""
        local_rank = get_rank()
        self.original_model = self.original_model.to(local_rank)
        self.wrapped_model = DistributedDataParallel(
            self.original_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # Important for models with conditional execution
        )
        logging.info(f"Using DistributedDataParallel on rank {local_rank}")
    
    def get_model(self):
        """Get the wrapped model"""
        return self.wrapped_model
    
    def get_base_model(self):
        """Get the underlying model (unwrapped)"""
        if isinstance(self.wrapped_model, (DataParallel, DistributedDataParallel)):
            return self.wrapped_model.module
        return self.wrapped_model

def create_distributed_dataloader(
    dataset,
    batch_size,
    *,
    shuffle: bool = True,
    num_workers: int = 2,
    distributed: bool = False,
    pin_memory: bool = True,
):
    """
    Build a DataLoader that works the same on 1-GPU and multi-GPU runs.

    Parameters
    ----------
    batch_size : int
        **Per-GPU** batch size.  The effective global batch is
        `batch_size × world_size` when `distributed=True`.
    distributed : bool
        If True, a `torch.utils.data.distributed.DistributedSampler`
        is attached and the DataLoader is returned on every rank.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
    sampler    : torch.utils.data.Sampler or None
    """
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=False,
        )
        # DataLoader must NOT shuffle when a sampler is supplied
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,          # ← **no more division by world_size**
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        # Disable prefetch for distributed training to reduce memory usage
        prefetch_factor=1 if distributed and num_workers > 0 else 2,
        collate_fn=getattr(dataset, "collate_fn", None),  # Use dataset's collate_fn if available
    )
    return dataloader, sampler


def distributed_train_wrapper(rank=None, world_size=None, args=None, train_fn=None):
    """
    Distributed training wrapper for use with both torch.multiprocessing.spawn and torch.distributed.launch.

    Args:
        rank: int, local rank of the process (for spawn mode)
        world_size: int, total number of GPUs (for spawn mode)  
        args: argparse.Namespace (shared across processes, be careful with modifications)
        train_fn: main training function (usually main(args))
        
    Note: When using torch.distributed.launch, rank and world_size are ignored and 
          read from environment variables instead.
    """
    try:
        # Detect if we're in torch.distributed.launch mode
        if is_distributed_launch_mode():
            # torch.distributed.launch mode - get info from environment
            rank, local_rank, world_size = get_distributed_info()
            logging.info(f"Running in torch.distributed.launch mode: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        else:
            # multiprocessing.spawn mode - use provided arguments
            if rank is None or world_size is None or args is None or train_fn is None:
                raise ValueError("In spawn mode, all arguments must be provided")
            local_rank = rank
            logging.info(f"Running in multiprocessing.spawn mode: rank={rank}, world_size={world_size}")
        
        # Note: MASTER_PORT should already be set by the main process before spawning
        # Don't set it here as it would cause race conditions
        
        setup_distributed(rank, world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        # Create a copy of args for this process to avoid modifying the shared args
        if args is not None:
            local_args = copy.copy(args)  # Shallow copy is sufficient
            local_args.rank = rank
            local_args.local_rank = local_rank
            local_args.device = device
            local_args.distributed = True
        else:
            # In torch.distributed.launch mode, args might be None if called from main
            # In this case, we assume main() will handle argument parsing
            local_args = None

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)
        
        seed = getattr(local_args, "seed", 42) if local_args else 42
        set_seed(seed + rank)

        if local_args is not None:
            # Broadcast args from rank 0 to all other ranks with retry logic
            # But preserve the device and rank settings which are rank-specific
            original_device = local_args.device
            original_rank = local_args.rank
            original_local_rank = local_args.local_rank
            local_args = broadcast_args_robust(local_args)
            # Restore rank-specific settings after broadcast
            local_args.device = original_device
            local_args.rank = original_rank
            local_args.local_rank = original_local_rank
            
            # Verify device setting is correct after broadcast
            logging.info(f"Rank {rank}: Device after broadcast: {local_args.device}")
        
        # Add multiple synchronization barriers with progressive timeouts
        for attempt in range(3):
            timeout = 60 * (attempt + 1)  # 60s, 120s, 180s
            if safe_barrier(timeout_seconds=timeout):
                logging.info(f"Rank {rank}: Synchronization barrier {attempt+1} passed")
                break
            else:
                logging.warning(f"Rank {rank}: Synchronization barrier {attempt+1} failed, retrying...")
                if attempt == 2:
                    raise RuntimeError(f"Rank {rank}: All synchronization attempts failed")

        # Final NCCL health check before training
        if not monitor_nccl_health():
            logging.error(f"Rank {rank}: Final NCCL health check failed")
            raise RuntimeError("NCCL communication is not working properly")

        if train_fn and local_args:
            train_fn(local_args)
        elif train_fn is None:
            # This means we were called from main() in launch mode, just return the args
            return local_args
        else:
            raise ValueError("train_fn is provided but local_args is None")
        
    except Exception as e:
        logging.error(f"Error in distributed training on rank {rank}: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise
    finally:
        try:
            # Ensure all processes reach cleanup
            if dist.is_initialized():
                safe_barrier(timeout_seconds=30)
            cleanup_distributed()
        except Exception as e:
            logging.warning(f"Rank {rank}: Error during cleanup: {e}")


def broadcast_args_robust(args):
    """Broadcast config from main process to others in DDP with error handling"""
    if not dist.is_initialized():
        return args
        
    import pickle
    rank = get_rank()
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if rank == 0:
                # Serialize args on rank 0
                serialized_args = pickle.dumps(vars(args))
                args_size = len(serialized_args)
                logging.info(f"Rank {rank}: Broadcasting {args_size} bytes of args (attempt {attempt+1})")
            else:
                serialized_args = None
                args_size = 0
            
            # First broadcast the size with timeout
            size_tensor = torch.tensor([args_size], dtype=torch.long, device=f'cuda:{rank}')
            
            # Use async operation with timeout
            handle = dist.broadcast(size_tensor, src=0, async_op=True)
            handle.wait()  # This will timeout if there are issues
            
            args_size = size_tensor.item()
            
            # Prepare buffer for data
            if rank == 0:
                data_tensor = torch.frombuffer(serialized_args, dtype=torch.uint8).cuda()
            else:
                data_tensor = torch.empty(args_size, dtype=torch.uint8, device=f'cuda:{rank}')
            
            # Broadcast the actual data with timeout
            handle = dist.broadcast(data_tensor, src=0, async_op=True)
            handle.wait()
            
            # Deserialize on non-rank-0 processes
            if rank != 0:
                serialized_args = data_tensor.cpu().numpy().tobytes()
                args_dict = pickle.loads(serialized_args)
                for k, v in args_dict.items():
                    setattr(args, k, v)
                logging.info(f"Rank {rank}: Successfully received broadcasted args (attempt {attempt+1})")
            
            return args  # Success
            
        except Exception as e:
            logging.warning(f"Rank {rank}: Args broadcast attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                logging.error(f"Rank {rank}: All broadcast attempts failed, using local args")
                # Don't raise error, use local args as fallback
                break
            
            # Wait a bit before retrying
            import time
            time.sleep(1)
    
    return args

def ddp_gather_embeddings_dd(embeddings_dict, local_indices):
    """
    Gather per-rank embedding chunks produced under DDP and
    restore the original (single-GPU) sample order.

    Parameters
    ----------
    embeddings_dict : Dict[str, Tensor]
        Each tensor has shape (N_local, D) on the current rank.
    local_indices : Sequence[int]
        The dataset indices **corresponding to the rows in the local
        tensors**.  For image loaders you can pass
        `sampler.indices` from the `DistributedSampler`.
        For manual caption sharding: `range(shard_start, shard_end)`.

    Returns
    -------
    gathered_dict : Dict[str, Tensor]
        Tensors of shape (N_total, D) on **all ranks**,
        reordered so that `gathered_dict[k][idx]` matches the
        single-GPU file produced with identical code.
    """
    if not dist.is_initialized():
        # Fallback – single-GPU
        return embeddings_dict

    world = dist.get_world_size()
    device = torch.device("cuda", dist.get_rank())

    # ---- 1. gather global indices -----------------------------------------
    local_idx = torch.as_tensor(local_indices, dtype=torch.long, device=device)
    idx_chunks = [torch.empty_like(local_idx) for _ in range(world)]
    dist.all_gather(idx_chunks, local_idx)
    full_idx = torch.cat(idx_chunks, dim=0)

    # ---- 2. gather each embedding layer -----------------------------------
    gathered = {}
    for name, local_tensor in embeddings_dict.items():
        # ensure contiguous memory before collective op
        local_tensor = local_tensor.to(device).contiguous()
        chunks = [torch.empty_like(local_tensor) for _ in range(world)]
        dist.all_gather(chunks, local_tensor)
        layer_full = torch.cat(chunks, dim=0)

        # ---- 3. restore ordering ------------------------------------------
        ordered = torch.empty_like(layer_full)
        ordered[full_idx] = layer_full
        gathered[name] = ordered.cpu()   # keep final result on CPU

    return gathered

def gather_embeddings_distributed(embeddings_dict, is_main_process, chunk_size=1000, timeout_seconds=1800):
    """
    Gather embeddings from all distributed processes with chunking to avoid timeouts.
    
    Args:
        embeddings_dict: Dict[str, torch.Tensor] - embeddings on each rank
        is_main_process: bool - whether this is the main process
        chunk_size: int - size of chunks to gather at once
        timeout_seconds: int - timeout for distributed operations
    
    Returns:
        Dict[str, torch.Tensor] - gathered embeddings (only valid on main process)
    """
    if not dist.is_initialized():
        return embeddings_dict
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}')
    
    # Set longer timeout for large embedding operations (if supported)
    old_timeout = None
    try:
        if hasattr(dist, 'get_default_timeout') and hasattr(dist, 'set_default_timeout'):
            old_timeout = dist.get_default_timeout()
            new_timeout = torch.timedelta(seconds=timeout_seconds)
            dist.set_default_timeout(new_timeout)
    except Exception as e:
        logging.warning(f"Could not set distributed timeout: {e}")
    
    try:
        gathered_embeddings = {}
        
        for layer_name, local_embedding in embeddings_dict.items():
            if is_main_process:
                logging.info(f"Gathering embeddings for layer: {layer_name}, local shape: {local_embedding.shape}")
            
            # Move to GPU for gathering
            local_tensor = local_embedding.to(device).contiguous()
            local_size = local_tensor.shape[0]
            embedding_dim = local_tensor.shape[1]
            
            # Gather sizes from all ranks
            size_tensor = torch.tensor(local_size, device=device, dtype=torch.long)
            all_sizes = [torch.zeros_like(size_tensor) for _ in range(world_size)]
            dist.all_gather(all_sizes, size_tensor)
            all_sizes = [s.item() for s in all_sizes]
            
            total_size = sum(all_sizes)
            max_local_size = max(all_sizes)
            
            if is_main_process:
                logging.info(f"Layer {layer_name}: total_size={total_size}, max_local_size={max_local_size}")
            
            # Handle empty tensors on some ranks
            if local_size == 0:
                local_tensor = torch.zeros(1, embedding_dim, device=device, dtype=local_tensor.dtype)
                local_size = 1
            
            # Pad to max size if needed
            if local_tensor.shape[0] < max_local_size:
                padding_size = max_local_size - local_tensor.shape[0]
                padding = torch.zeros(padding_size, embedding_dim, device=device, dtype=local_tensor.dtype)
                local_tensor = torch.cat([local_tensor, padding], dim=0)
            
            # Chunk-based gathering to avoid memory issues
            gathered_chunks = []
            num_chunks = (max_local_size + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, max_local_size)
                
                # Extract chunk from local tensor
                local_chunk = local_tensor[start_idx:end_idx]
                
                # Gather this chunk from all ranks
                chunks_from_all_ranks = [torch.zeros_like(local_chunk) for _ in range(world_size)]
                
                try:
                    dist.all_gather(chunks_from_all_ranks, local_chunk)
                except Exception as e:
                    if is_main_process:
                        logging.error(f"Failed to gather chunk {chunk_idx} for layer {layer_name}: {e}")
                    raise
                
                gathered_chunks.append(chunks_from_all_ranks)
            
            # Reconstruct full tensors on main process
            if is_main_process:
                final_embeddings = []
                for rank_idx in range(world_size):
                    rank_chunks = []
                    for chunk_group in gathered_chunks:
                        rank_chunks.append(chunk_group[rank_idx])
                    
                    # Concatenate chunks for this rank
                    rank_full = torch.cat(rank_chunks, dim=0)
                    
                    # Remove padding and handle empty tensors
                    actual_size = all_sizes[rank_idx]
                    if actual_size > 0:
                        rank_valid = rank_full[:actual_size]
                        final_embeddings.append(rank_valid)
                
                # Concatenate all ranks
                if final_embeddings:
                    gathered_embeddings[layer_name] = torch.cat(final_embeddings, dim=0).cpu()
                else:
                    # Handle case where all ranks have empty embeddings
                    gathered_embeddings[layer_name] = torch.empty(0, embedding_dim, dtype=local_embedding.dtype)
                
                logging.info(f"Layer {layer_name}: gathered shape = {gathered_embeddings[layer_name].shape}")
            else:
                # Non-main processes get empty dict
                pass
        
        # Synchronize all processes
        dist.barrier()
        
        if is_main_process:
            return gathered_embeddings
        else:
            return {}
            
    except Exception as e:
        if is_main_process:
            logging.error(f"Error in gather_embeddings_distributed: {e}")
        raise
    finally:
        # Restore original timeout (if we modified it)
        try:
            if old_timeout is not None and hasattr(dist, 'set_default_timeout'):
                dist.set_default_timeout(old_timeout)
        except Exception as e:
            logging.warning(f"Could not restore distributed timeout: {e}")

def monitor_nccl_health():
    """Monitor NCCL health and log relevant information"""
    if not dist.is_initialized():
        return
    
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Test basic communication
        test_tensor = torch.tensor([rank], dtype=torch.float32, device=torch.cuda.current_device())
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM, async_op=False)
        expected_sum = sum(range(world_size))
        
        if abs(test_tensor.item() - expected_sum) < 1e-6:
            logging.info(f"Rank {rank}: NCCL communication test passed")
            return True
        else:
            logging.error(f"Rank {rank}: NCCL communication test failed. Got {test_tensor.item()}, expected {expected_sum}")
            return False
            
    except Exception as e:
        logging.error(f"Rank {rank if 'rank' in locals() else '?'}: NCCL health check failed: {e}")
        return False

def safe_barrier(timeout_seconds=300):
    """Perform a distributed barrier with timeout and error handling"""
    if not dist.is_initialized():
        return True
        
    try:
        rank = dist.get_rank()
        logging.info(f"Rank {rank}: Entering barrier")
        
        # For older PyTorch versions that don't support timeout in barrier
        try:
            timeout = datetime.timedelta(seconds=timeout_seconds)
            dist.barrier(timeout=timeout)
        except TypeError:
            # Fallback for older PyTorch versions
            dist.barrier()
            
        logging.info(f"Rank {rank}: Barrier completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Rank {rank if 'rank' in locals() else '?'}: Barrier failed: {e}")
        return False

def setup_nccl_debug():
    """Setup NCCL debugging environment variables"""
    debug_settings = {
        'NCCL_DEBUG': 'INFO',
        'NCCL_DEBUG_SUBSYS': 'ALL',
        'NCCL_TREE_THRESHOLD': '0',  # Force ring algorithm
        'NCCL_MIN_NCHANNELS': '1',   # Use minimum channels for stability
    }
    
    for key, value in debug_settings.items():
        if key not in os.environ:
            os.environ[key] = value
            logging.info(f"Set {key}={value}")

# Add these arguments to your argparse
def add_multigpu_args(parser):
    """Add multi-GPU related arguments to parser"""
    parser.add_argument('--distributed', action='store_true', 
                       help='Use DistributedDataParallel for multi-GPU training (recommended)')
    parser.add_argument('--data_parallel', action='store_true',
                       help='Use DataParallel for multi-GPU training (simpler but less efficient)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training (set automatically by torch.distributed.launch)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers per GPU (default: 2, reduced for distributed efficiency)')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                       help='Pin memory for faster GPU transfer')
    parser.add_argument('--use_tar_batching', action='store_true', default=True,
                       help='Use tar-based batching for LAION dataset (recommended for better throughput)')
    parser.add_argument('--disk_gather', action='store_true', default=False,
                       help='Use disk-based gathering for large embeddings in distributed training')
    parser.add_argument('--split_by_tar', action='store_true', default=False,
                       help='Split train/val by tar files instead of samples (LAION only)')
    return parser

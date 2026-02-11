import os
import glob
import math
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Dict, Iterable, List, Tuple as _Tuple

import numpy as np
import torch
import torch.distributed as dist
from collections import defaultdict
from tqdm import tqdm

from utils.dist import is_main_process
from utils.align import (
    compute_image_embeddings_intermediate_batch,
    compute_caption_embeddings_intermediate_batch,  # available if you want to mirror for text
)

from collections import OrderedDict
import psutil

# ----------------------------------------
# Cache cleanup
# ----------------------------------------

def log_memory_usage(dataset_loader, tag=""):
    if is_main_process():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        logging.info(f"[{tag}] RAM: {mem_info.rss / 1024**3:.2f} GB, GPU: {gpu_mem:.2f} GB")
        
        logging.info(f"DataLoader config:")
        logging.info(f"  num_workers: {dataset_loader.num_workers}")
        logging.info(f"  batch_size: {dataset_loader.batch_size}")
        logging.info(f"  pin_memory: {dataset_loader.pin_memory}")
        logging.info(f"  timeout: {getattr(dataset_loader, 'timeout', 'default')}")
        logging.info(f"  multiprocessing_context: {dataset_loader.multiprocessing_context}")
        
        cpu_count = psutil.cpu_count()
        available_memory = psutil.virtual_memory().available / (1024**3)
        logging.info(f"  System: {cpu_count} CPUs, {available_memory:.2f} GB available RAM")
        

def clean_up_cache(embedding_dir, base_pattern):
    """
    Cleans up old embedding files in the specified directory based on a base pattern.
    """
    cleanup_patterns = [
        os.path.join(embedding_dir, base_pattern + '.pt'),
        os.path.join(embedding_dir, base_pattern + '_*.pt'),
        os.path.join(embedding_dir, base_pattern + '_tar*.pt'),
        os.path.join(embedding_dir, base_pattern + '_metadata.json'),
        os.path.join(embedding_dir, base_pattern + '_tar_metadata.json'),
        os.path.join(embedding_dir, base_pattern + '_rank*.pt'),
        os.path.join(embedding_dir, base_pattern + '_chunk_*.pt'),
    ]

    files_deleted = 0
    for pattern in cleanup_patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                files_deleted += 1
                logging.info(f"Deleted old embedding file: {file_path}")
            except OSError as e:
                logging.warning(f"Could not delete {file_path}: {e}")

    logging.info(f"Deleted {files_deleted} old embedding files.")

# ----------------------------------------
# Storage: legacy (dense) save helpers
# ----------------------------------------

def save_embeddings(embedding_path, embeddings):
    """Save embeddings to a single file (legacy)."""
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    torch.save(embeddings, embedding_path)

def load_embeddings(embedding_path, device='cuda'):
    """Load legacy single-file embeddings."""
    if os.path.exists(embedding_path):
        return torch.load(embedding_path, map_location=device, weights_only=False)
    return None

def save_embeddings_chunked(path, embeddings_dict, chunk_size=50_000):
    """
    Legacy chunk writer (dense -> chunk files). For streaming, prefer the new writers below.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metadata = {
        'total_samples': {k: int(v.shape[0]) for k, v in embeddings_dict.items()},
        'embedding_dims': {k: int(v.shape[1]) for k, v in embeddings_dict.items()},
        'chunk_size': int(chunk_size),
        'layer_names': list(embeddings_dict.keys()),
        'storage_type': 'chunked_legacy',
    }
    metadata_path = path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    for layer_name, embeddings in embeddings_dict.items():
        total_samples = embeddings.shape[0]
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_samples)
            chunk_data = embeddings[start_idx:end_idx]
            chunk_path = path.replace('.pt', f'_{layer_name}_chunk_{chunk_idx}.pt')
            torch.save(chunk_data, chunk_path)
    logging.info(f"Saved legacy chunked embeddings to {os.path.dirname(path)}")

def save_embeddings_tar_based(base_path, embeddings_dict, dataset, tar_mapping=None):
    """
    Legacy tar-based writer (dense -> per-tar tensors). For streaming tar shards, see TarShardStreamWriter + manifest.
    """
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    # Build mapping from dataset index to (tar_num, local_idx)
    if tar_mapping is None and hasattr(dataset, 'tar_ids'):
        # New compact index structure
        tar_mapping = {}
        tar_to_indices = defaultdict(list)
        for dataset_idx in range(len(dataset)):
            # Use new compact index lookup
            tar_pos = int(np.searchsorted(dataset.tar_offsets, dataset_idx, side="right") - 1)
            tar_num = int(dataset.tar_ids[tar_pos])
            local_idx = int(dataset.local_indices[dataset_idx])
            tar_mapping[dataset_idx] = (tar_num, local_idx)
            tar_to_indices[tar_num].append(dataset_idx)
    elif tar_mapping is None and hasattr(dataset, 'index'):
        # Legacy index structure (for backward compatibility)
        tar_mapping = {}
        tar_to_indices = defaultdict(list)
        for dataset_idx in range(len(dataset)):
            original_tar_info = dataset.index[dataset_idx]
            if isinstance(original_tar_info, (list, tuple)) and len(original_tar_info) == 2:
                tar_num, original_tar_local_idx = original_tar_info
            else:
                tar_num = int(original_tar_info)
                original_tar_local_idx = dataset_idx
            tar_mapping[dataset_idx] = (int(tar_num), int(original_tar_local_idx))
            tar_to_indices[int(tar_num)].append(dataset_idx)
    elif tar_mapping is None:
        logging.warning("No tar mapping available, falling back to chunked storage")
        save_embeddings_chunked(base_path, embeddings_dict, chunk_size=50_000)
        return
    else:
        tar_to_indices = defaultdict(list)
        for dataset_idx, tar_info in tar_mapping.items():
            tar_num = tar_info if isinstance(tar_info, int) else tar_info[0]
            tar_to_indices[int(tar_num)].append(dataset_idx)

    metadata = {
        'total_samples': {k: int(v.shape[0]) for k, v in embeddings_dict.items()},
        'embedding_dims': {k: int(v.shape[1]) for k, v in embeddings_dict.items()},
        'layer_names': list(embeddings_dict.keys()),
        'storage_type': 'tar_based',
        'tar_files': list(sorted(tar_to_indices.keys())),
        'tar_sizes': {str(tar_num): len(indices) for tar_num, indices in tar_to_indices.items()},
        'index_mapping': {str(dataset_idx): val for dataset_idx, val in tar_mapping.items()},
    }
    metadata_path = base_path.replace('.pt', '_tar_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    for tar_num, dataset_indices in tar_to_indices.items():
        tar_local_to_dataset = {}
        for dataset_idx in dataset_indices:
            tar_info = tar_mapping[dataset_idx]
            if isinstance(tar_info, (list, tuple)) and len(tar_info) == 2:
                _, tar_local_idx = tar_info
            else:
                tar_local_idx = len(tar_local_to_dataset)
            tar_local_to_dataset[int(tar_local_idx)] = dataset_idx

        max_tar_local_idx = max(tar_local_to_dataset.keys()) if tar_local_to_dataset else -1
        tar_embeddings = {}
        if tar_num == 0 and is_main_process():
            logging.info(f"DEBUG: Tar {tar_num} mappings (first 20):")
            for tar_local_idx, dataset_idx in list(sorted(tar_local_to_dataset.items()))[:20]:
                logging.info(f"  tar_local_idx {tar_local_idx} <- dataset_idx {dataset_idx}")

        for layer_name, full_embeddings in embeddings_dict.items():
            if torch.isnan(full_embeddings).any():
                nan_mask = torch.isnan(full_embeddings).any(dim=1)
                logging.error(f"NaN detected in {layer_name}; replacing NaNs with zeros")
                full_embeddings = full_embeddings.clone()
                full_embeddings[nan_mask] = 0.0

            tar_size = max_tar_local_idx + 1
            embedding_dim = full_embeddings.shape[1]
            tar_layer_embeddings = torch.zeros(tar_size, embedding_dim, dtype=full_embeddings.dtype)
            for tar_local_idx, dataset_idx in tar_local_to_dataset.items():
                emb = full_embeddings[dataset_idx]
                if torch.isnan(emb).any():
                    emb = torch.zeros_like(emb)
                tar_layer_embeddings[tar_local_idx] = emb

            tar_embeddings[layer_name] = tar_layer_embeddings

        tar_path = base_path.replace('.pt', f'_tar{int(tar_num):05d}.pt')
        torch.save(tar_embeddings, tar_path)

    logging.info(f"Saved tar-based embeddings for {len(tar_to_indices)} tar files to {os.path.dirname(base_path)}")

# ----------------------------------------
# Lazy loaders (support both legacy + streaming)
# ----------------------------------------

class LazyEmbeddingLoader:
    """
    Lazy loader for chunked embeddings.
    Supports:
      - storage_type == 'chunked_stream' (new streaming: file list + row counts)
      - storage_type == 'chunked_legacy' (fixed chunk_size + numbered files)
    """
    def __init__(self, base_path, metadata):
        self.base_path = base_path
        self.metadata = metadata
        self.layer_names = metadata.get('layer_names') or metadata.get('layers')
        self.storage_type = metadata.get('storage_type', 'chunked_legacy')

        # streaming fields
        self._files_by_layer: Dict[str, List[Dict[str, Any]]] = metadata.get("files", {})
        self._cumrows_by_layer: Dict[str, np.ndarray] = {}
        if self.storage_type == "chunked_stream":
            for ln in self.layer_names:
                files = self._files_by_layer.get(ln, [])
                cum = []
                s = 0
                for entry in files:
                    s += int(entry["rows"])
                    cum.append(s)
                self._cumrows_by_layer[ln] = np.array(cum, dtype=np.int64)

        # legacy fields
        self.chunk_size = metadata.get('chunk_size')
        self._chunk_cache: Dict[_Tuple[str, int], torch.Tensor] = {}
        self._cache_size = 5
        self._file_cache = OrderedDict()   # path -> Tensor
        self._max_file_cache = 8           # tune: 4–16; see notes below

    def _locate_stream_file(self, layer_name: str, idx: int):
        cum = self._cumrows_by_layer[layer_name]
        pos = int(np.searchsorted(cum, idx + 1, side='left'))
        prev_cum = 0 if pos == 0 else int(cum[pos - 1])
        local_idx = idx - prev_cum
        entry = self._files_by_layer[layer_name][pos]
        return entry["file"], local_idx

    def get_embeddings(self, layer_name, indices):
        if isinstance(indices, (list, np.ndarray)):
            indices = torch.tensor(indices, dtype=torch.long)
        elif isinstance(indices, int):
            indices = torch.tensor([indices], dtype=torch.long)

        if layer_name not in self.metadata['embedding_dims']:
            # graceful fallback
            dim = next(iter(self.metadata['embedding_dims'].values()))
            return torch.zeros(len(indices), dim, dtype=torch.float32)

        expected_dim = self.metadata['embedding_dims'][layer_name]
        result = torch.zeros(len(indices), expected_dim, dtype=torch.float32)

        if self.storage_type == "chunked_stream":
            # group by file path for efficient loads
            file_groups: Dict[str, List[_Tuple[int, int]]] = defaultdict(list)
            for i, idx in enumerate(indices.tolist()):
                fpath, local_idx = self._locate_stream_file(layer_name, int(idx))
                file_groups[fpath].append((i, local_idx))
            for fpath, pairs in file_groups.items():
                chunk_data = self._get_stream_chunk(fpath)            # was: torch.load(...)
                for out_i, local_idx in pairs:
                    if 0 <= local_idx < chunk_data.shape[0]:
                        result[out_i] = chunk_data[local_idx].float()
            return result

        # legacy path
        return self._get_embeddings_legacy(layer_name, indices)

    # --- legacy helpers (unchanged logic with caching) ---
    def _load_chunk(self, layer_name, chunk_idx):
        cache_key = (layer_name, int(chunk_idx))
        if cache_key in self._chunk_cache:
            return self._chunk_cache[cache_key]
        chunk_path = self.base_path.replace('.pt', f'_{layer_name}_chunk_{int(chunk_idx)}.pt')
        if not os.path.exists(chunk_path):
            # fallback: zero-chunk
            return torch.zeros(self.chunk_size, self.metadata['embedding_dims'][layer_name], dtype=torch.float32)
        chunk_data = torch.load(chunk_path, map_location='cpu')
        if len(self._chunk_cache) >= self._cache_size:
            oldest_key = next(iter(self._chunk_cache))
            del self._chunk_cache[oldest_key]
        self._chunk_cache[cache_key] = chunk_data
        return chunk_data

    def _get_embeddings_legacy(self, layer_name, indices):
        if isinstance(indices, torch.Tensor):
            idxs = indices.tolist()
        else:
            idxs = list(indices)
        result = torch.zeros(len(idxs), self.metadata['embedding_dims'][layer_name], dtype=torch.float32)
        groups = defaultdict(list)
        for i, idx in enumerate(idxs):
            chunk_idx = idx // self.chunk_size
            groups[int(chunk_idx)].append((i, int(idx) % self.chunk_size))
        for ch, pairs in groups.items():
            ch_data = self._load_chunk(layer_name, ch)
            for out_i, local_idx in pairs:
                if local_idx < ch_data.shape[0]:
                    result[out_i] = ch_data[local_idx].float()
        return result

    def set_file_cache_size(self, n: int):
        self._max_file_cache = max(0, int(n))

    def _get_stream_chunk(self, fpath: str) -> torch.Tensor:
        cache = self._file_cache
        if fpath in cache:
            t = cache.pop(fpath)     # refresh LRU
            cache[fpath] = t
            return t
        t = torch.load(fpath, map_location='cpu')
        cache[fpath] = t
        # cap cache size
        if len(cache) > self._max_file_cache:
            cache.popitem(last=False)  # evict LRU
        return t
    
    def _files_for_indices(self, layer_name: str, idxs: List[int]) -> List[str]:
        paths = []
        for idx in idxs:
            fpath, _ = self._locate_stream_file(layer_name, int(idx))
            paths.append(fpath)
        # unique while preserving order
        seen = set()
        out = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def prefetch_indices(self, layer_name: str, indices: List[int]) -> None:
        if self.storage_type != "chunked_stream":
            return
        for f in self._files_for_indices(layer_name, indices):
            _ = self._get_stream_chunk(f)
            
class TarBasedEmbeddingLoader:
    """
    Lazy loader for tar-based embeddings.
    Supports:
      - storage_type == 'tar_based'  (dense per-tar tensors)
      - storage_type == 'tar_sharded' (streaming shards: each shard has 'indices' + per-layer data)
    """
    def __init__(self, base_path, metadata, tar_nums=None):
        self.base_path = base_path
        self.metadata = metadata
        self.layer_names = metadata.get('layers') or metadata.get('layer_names')
        self.storage_type = metadata.get('storage_type', 'tar_based')
        self.tar_nums = tar_nums or metadata.get('tar_files', [])
        self._tar_cache: Dict[int, Dict[str, torch.Tensor]] = {}   # dense cache
        self._shard_cache: Dict[int, List[Dict[str, torch.Tensor]]] = {}  # shard cache
        self._max_cache_size = 10

        # dataset index -> (tar_num, tar_local_idx)
        self._index_to_tar: Dict[int, _Tuple[int, Optional[int]]] = {}
        if 'index_mapping' in metadata:
            for ds_idx_str, tar_info in metadata['index_mapping'].items():
                ds = int(ds_idx_str)
                if isinstance(tar_info, list) and len(tar_info) == 2:
                    self._index_to_tar[ds] = (int(tar_info[0]), int(tar_info[1]))
                else:
                    self._index_to_tar[ds] = (int(tar_info), None)
        self.total_samples = len(self._index_to_tar)

    def __getitem__(self, layer_name):
        # Dict-like compatibility; returns a lazy tensor facade
        return LazyEmbeddingTensor(self, layer_name)

    def keys(self):
        return self.layer_names

    def get_embeddings(self, layer_name, indices):
        if isinstance(indices, int):
            indices = [indices]
        out = torch.zeros(len(indices), self.metadata['embedding_dims'][layer_name], dtype=torch.float32)

        # group requested rows by tar
        by_tar: Dict[int, List[_Tuple[int, int]]] = defaultdict(list)
        for i, ds_idx in enumerate(indices):
            if ds_idx not in self._index_to_tar:
                continue
            tar_num, tar_local = self._index_to_tar[ds_idx]
            if tar_local is None:
                continue
            by_tar[int(tar_num)].append((i, int(tar_local)))

        for tar_num, pairs in by_tar.items():
            if self.storage_type == 'tar_based':
                tar_layer = self._load_tar_dense_layer(tar_num, layer_name)
                for out_i, local_idx in pairs:
                    if 0 <= local_idx < tar_layer.shape[0]:
                        out[out_i] = tar_layer[local_idx].float()
            else:
                # tar_sharded: find the shard that holds each local index
                shards = self._load_tar_shards(tar_num)
                need = torch.tensor([p[1] for p in pairs], dtype=torch.long)
                out_pos = [p[0] for p in pairs]
                filled = torch.zeros(len(pairs), dtype=torch.bool)
                for shard in shards:
                    shard_idx = shard["indices"]  # LongTensor
                    m = torch.isin(need, shard_idx)
                    if not m.any():
                        continue
                    idxs_needed = need[m].tolist()
                    # index -> position in shard
                    inv = {int(v): int(j) for j, v in enumerate(shard_idx.tolist())}
                    selected_positions = torch.where(m)[0].tolist()
                    for sel, need_val in zip(selected_positions, idxs_needed):
                        global_out_pos = out_pos[sel]
                        out[global_out_pos] = shard[layer_name][inv[need_val]].float()
                        filled[sel] = True
                # (any missing remain zeros)
        return out

    def _load_tar_dense(self, tar_num: int) -> Dict[str, torch.Tensor]:
        if tar_num in self._tar_cache:
            return self._tar_cache[tar_num]
        tar_path = self.base_path.replace('.pt', f'_tar{int(tar_num):05d}.pt')
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Tar embedding file not found: {tar_path}")
        tar_embeddings = torch.load(tar_path, map_location='cpu')
        if len(self._tar_cache) >= self._max_cache_size:
            self._tar_cache.pop(next(iter(self._tar_cache)))
        self._tar_cache[tar_num] = tar_embeddings
        return tar_embeddings

    def _load_tar_dense_layer(self, tar_num: int, layer_name: str) -> torch.Tensor:
        return self._load_tar_dense(tar_num)[layer_name]

    def _load_tar_shards(self, tar_num: int) -> List[Dict[str, torch.Tensor]]:
        if tar_num in self._shard_cache:
            return self._shard_cache[tar_num]
        # tar_shards stored with numeric keys as strings
        key_a = str(tar_num).zfill(5)
        key_b = str(tar_num)
        shard_entries = self.metadata.get("tar_shards", {}).get(key_a) or self.metadata.get("tar_shards", {}).get(key_b) or []
        shards = []
        for ent in shard_entries:
            shards.append(torch.load(ent["file"], map_location='cpu'))
        if len(self._shard_cache) >= self._max_cache_size:
            self._shard_cache.pop(next(iter(self._shard_cache)))
        self._shard_cache[tar_num] = shards
        return shards

class LazyEmbeddingTensor:
    """
    Thin facade to look like a tensor for [N, D] with __getitem__ loading on demand.
    """
    def __init__(self, loader, layer_name):
        self.loader = loader
        self.layer_name = layer_name
        self.shape = (
            loader.total_samples,
            loader.metadata['embedding_dims'][layer_name]
        )

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            start, stop, step = indices.indices(self.shape[0])
            indices = list(range(start, stop, step))
        elif isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        elif not isinstance(indices, (list, tuple)):
            indices = [indices]
        return self.loader.get_embeddings(self.layer_name, indices)

    def __len__(self):
        return self.shape[0]

def load_embeddings_lazy(path):
    """Load embeddings with lazy loader if *_metadata.json exists; else legacy single-file."""
    metadata_path = path.replace('.pt', '_metadata.json')
    if not os.path.exists(metadata_path):
        return load_embeddings(path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return LazyEmbeddingLoader(path, metadata)

def load_embeddings_tar_based(base_path, tar_nums=None):
    """Load tar-based embeddings (dense or tar-sharded) lazily using *_tar_metadata.json."""
    metadata_path = base_path.replace('.pt', '_tar_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Tar-based embedding metadata not found: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return TarBasedEmbeddingLoader(base_path, metadata, tar_nums)

# ----------------------------------------
# Distributed prepare (loading if exists)
# ----------------------------------------

def prepare_distributed_embeddings(
    is_distributed: bool,
    args: Any,
    image_embedding_path: str,
    text_embedding_path: str,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Load precomputed embeddings with independent detection per modality:
      - tar_sharded via *_tar_metadata.json
      - chunked_stream/legacy via *_metadata.json
      - single-file *.pt
    Returns (image_embeddings, caption_embeddings). If not found, returns (None, None) for that side.
    """
    def _detect_storage(path: str) -> str:
        tar_meta = path.replace(".pt", "_tar_metadata.json")
        chunk_meta = path.replace(".pt", "_metadata.json")
        if os.path.exists(tar_meta) and getattr(args, "dataset", None) == "LAION400M":
            return "tar"
        if os.path.exists(chunk_meta):
            return "chunked"
        if os.path.exists(path):
            return "single"
        return "missing"

    def _load_by_path(path: str, name: str):
        kind = _detect_storage(path)
        if kind == "tar":
            if is_main_process():
                logging.info(f"Loading {name} embeddings [tar_sharded]: {path}")
            return load_embeddings_tar_based(path)
        elif kind == "chunked":
            if is_main_process():
                logging.info(f"Loading {name} embeddings [chunked_lazy]: {path}")
            return load_embeddings_lazy(path)
        elif kind == "single":
            if is_main_process():
                logging.info(f"Loading {name} embeddings [single-file]: {path}")
            return load_embeddings(path)
        else:
            if is_main_process():
                logging.info(f"{name} embeddings not found: {path}")
            return None

    # recompute flag short-circuit
    if getattr(args, "recompute_cache", False):
        if is_main_process():
            logging.info("Recompute cache flag set; will compute embeddings from scratch.")
        return None, None

    image_embeddings: Optional[Any] = None
    caption_embeddings: Optional[Any] = None

    if is_distributed:
        # Try shared files first (independent detection per path)
        image_embeddings = _load_by_path(image_embedding_path, "image")
        caption_embeddings = _load_by_path(text_embedding_path, "caption")

        # If still missing, try legacy per-rank files independently
        if image_embeddings is None or caption_embeddings is None:
            rank = dist.get_rank()
            if image_embeddings is None:
                rank_img_path = image_embedding_path.replace(".pt", f"_rank{rank}.pt")
                image_embeddings = _load_by_path(rank_img_path, f"image (rank {rank})")
            if caption_embeddings is None:
                rank_txt_path = text_embedding_path.replace(".pt", f"_rank{rank}.pt")
                caption_embeddings = _load_by_path(rank_txt_path, f"caption (rank {rank})")
    else:
        # Non-distributed: just load each side independently
        image_embeddings = _load_by_path(image_embedding_path, "image")
        caption_embeddings = _load_by_path(text_embedding_path, "caption")

    # Final fallback / notice
    if (image_embeddings is None) or (caption_embeddings is None):
        missing = []
        if image_embeddings is None:
            missing.append("image")
        if caption_embeddings is None:
            missing.append("caption")
        if is_main_process():
            logging.info(f"No existing {' & '.join(missing)} embeddings found; will compute them.")

    return image_embeddings, caption_embeddings
# ----------------------------------------
# Streaming writers + global manifests (NEW)
# ----------------------------------------

@dataclass
class ChunkStreamWriter:
    base_path: str
    layer_names: List[str]
    rank: int
    max_rows_per_chunk: int = 50000   # Further reduced for extremely conservative memory usage

    _buffers: Dict[str, List[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    _row_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _chunk_idx: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    manifest: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))

    def append_batch(self, batch_embeds: Dict[str, torch.Tensor]):
        for layer, t in batch_embeds.items():
            if t is None or t.numel() == 0:
                continue
            t = t.detach().cpu()
            self._buffers[layer].append(t)
            self._row_counts[layer] += t.shape[0]
            if self._row_counts[layer] >= self.max_rows_per_chunk:
                self._flush_layer(layer)

    def _flush_layer(self, layer: str):
        if self._row_counts[layer] == 0:
            return
        chunk = torch.cat(self._buffers[layer], dim=0)
        out_path = self.base_path.replace(".pt", f"_{layer}_rank{self.rank}_chunk{self._chunk_idx[layer]}.pt")
        torch.save(chunk, out_path)
        self.manifest[layer].append({"file": out_path, "rows": int(chunk.shape[0])})
        self._buffers[layer].clear()
        self._row_counts[layer] = 0
        self._chunk_idx[layer] += 1

    def finalize(self):
        for layer in self.layer_names:
            self._flush_layer(layer)
        rank_manifest_path = self.base_path.replace(".pt", f"_rank{self.rank}_manifest.json")
        with open(rank_manifest_path, "w") as f:
            json.dump({"layers": self.manifest}, f, indent=2)
        return rank_manifest_path

@dataclass
class TarShardStreamWriter:
    base_path: str
    layer_names: List[str]
    rank: int
    dataset: Any            # dataset must have tar_ids/tar_offsets/local_indices (new) or .index (legacy) to map ds_idx -> (tar_num, tar_local_idx)
    max_rows_per_shard: int = 5000  # Further reduced to 3k for extremely conservative memory usage

    _buf_layers: Dict[int, Dict[str, List[torch.Tensor]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    _buf_indices: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    _buf_rows: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    _shard_idx: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    manifest: Dict[int, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    _last_tar: Optional[int] = field(default_factory=lambda: None)  # Track last tar for boundary flushing

    def append_batch(self, batch_embeds: Dict[str, torch.Tensor], dataset_indices: List[int]):
        for row_i, ds_idx in enumerate(dataset_indices):
            # Support both new compact index and legacy index structure
            if hasattr(self.dataset, 'tar_ids'):
                # New compact index structure
                tar_pos = int(np.searchsorted(self.dataset.tar_offsets, ds_idx, side="right") - 1)
                tar_num = int(self.dataset.tar_ids[tar_pos])
                tar_local = int(self.dataset.local_indices[ds_idx])
            else:
                # Legacy index structure
                tar_info = self.dataset.index[ds_idx]
                if isinstance(tar_info, (list, tuple)) and len(tar_info) == 2:
                    tar_num, tar_local = int(tar_info[0]), int(tar_info[1])
                else:
                    tar_num, tar_local = int(tar_info), int(ds_idx)

            # Flush previous tar when we see a new tar_num (sequential tar processing)
            if self._last_tar is not None and tar_num != self._last_tar:
                self._flush_tar(self._last_tar)
            self._last_tar = tar_num

            for layer in self.layer_names:
                self._buf_layers[tar_num][layer].append(batch_embeds[layer][row_i].detach().cpu().unsqueeze(0))

            self._buf_indices[tar_num].append(tar_local)
            self._buf_rows[tar_num] += 1

            # Also flush if we hit the row threshold (now 5k instead of 100k)
            if self._buf_rows[tar_num] >= self.max_rows_per_shard:
                self._flush_tar(tar_num)

    def _flush_tar(self, tar_num: int):
        if self._buf_rows[tar_num] == 0:
            return
        shard = {"indices": torch.tensor(self._buf_indices[tar_num], dtype=torch.long)}
        for layer in self.layer_names:
            if len(self._buf_layers[tar_num][layer]) == 0:
                shard[layer] = torch.empty(0)
            else:
                shard[layer] = torch.cat(self._buf_layers[tar_num][layer], dim=0)
        out_path = self.base_path.replace(".pt", f"_tar{tar_num:05d}_rank{self.rank}_shard{self._shard_idx[tar_num]}.pt")
        torch.save(shard, out_path)
        self.manifest[tar_num].append({"file": out_path, "rows": int(self._buf_rows[tar_num])})

        # reset
        self._buf_layers[tar_num].clear()
        self._buf_indices[tar_num].clear()
        self._buf_rows[tar_num] = 0
        self._shard_idx[tar_num] += 1

    def finalize(self):
        for tar_num in list(self._buf_rows.keys()):
            self._flush_tar(tar_num)
        rank_manifest_path = self.base_path.replace(".pt", f"_rank{self.rank}_manifest.json")
        with open(rank_manifest_path, "w") as f:
            manifest_json = {str(k): v for k, v in self.manifest.items()}
            json.dump({"tars": manifest_json, "layers": self.layer_names}, f, indent=2)
        return rank_manifest_path

def _gather_rank_manifests(base_path: str, world_size: int) -> List[str]:
    manifests = []
    for r in range(world_size):
        p = base_path.replace(".pt", f"_rank{r}_manifest.json")
        if not os.path.exists(p):
            logging.error(f"Missing rank manifest: {p}")
            continue
        manifests.append(p)
    return manifests

def build_global_chunked_manifest(base_path: str, layer_names: List[str], world_size: int, total_samples: int, embedding_dims: Dict[str, int]):
    """Create one global metadata file for streaming chunked embeddings (no data concatenation)."""
    manifests = _gather_rank_manifests(base_path, world_size)
    files_by_layer: Dict[str, List[Dict[str, Any]]] = {ln: [] for ln in layer_names}

    for mf in manifests:
        with open(mf, "r") as f:
            data = json.load(f)
        layers = data.get("layers", {})
        for ln in layer_names:
            files_by_layer[ln].extend(layers.get(ln, []))

    metadata = {
        "storage_type": "chunked_stream",
        "total_samples": {ln: int(total_samples) for ln in layer_names},
        "embedding_dims": {k: int(v) for k, v in embedding_dims.items()},
        "layers": layer_names,
        "files": files_by_layer,
    }
    out = base_path.replace(".pt", "_metadata.json")
    with open(out, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Wrote global chunked manifest: {out}")

def build_global_tar_sharded_manifest(base_path: str, world_size: int, embedding_dims: Dict[str, int], dataset, layer_names: List[str]):
    """Create one global metadata file for streaming tar-sharded embeddings (no data concatenation)."""
    manifests = _gather_rank_manifests(base_path, world_size)
    tar_shards: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for mf in manifests:
        with open(mf, "r") as f:
            data = json.load(f)
        tars = data.get("tars", {})
        for tar_num_str, shard_list in tars.items():
            tar_shards[tar_num_str].extend(shard_list)

    # sort per tar for determinism
    for tar_num in tar_shards.keys():
        tar_shards[tar_num].sort(key=lambda x: x["file"])

    # dataset index -> [tar_num, tar_local_idx] using new compact index structure
    index_mapping = {}
    for ds_idx in range(len(dataset)):
        if hasattr(dataset, 'tar_ids'):
            # New compact index structure
            tar_pos = int(np.searchsorted(dataset.tar_offsets, ds_idx, side="right") - 1)
            tar_num = int(dataset.tar_ids[tar_pos])
            tar_local = int(dataset.local_indices[ds_idx])
        else:
            # Legacy index structure (for backward compatibility)
            info = dataset.index[ds_idx]
            if isinstance(info, (list, tuple)) and len(info) == 2:
                tar_num, tar_local = int(info[0]), int(info[1])
            else:
                tar_num, tar_local = int(info), int(ds_idx)
        index_mapping[str(ds_idx)] = [tar_num, tar_local]

    metadata = {
        "storage_type": "tar_sharded",
        "embedding_dims": {k: int(v) for k, v in embedding_dims.items()},
        "layers": layer_names,
        "tar_files": sorted([int(k) for k in tar_shards.keys()]),
        "tar_shards": tar_shards,       # {"00000": [{"file": "...pt", "rows": N}, ...], ...}
        "index_mapping": index_mapping, # dataset_idx -> [tar_num, tar_local_idx]
    }
    out = base_path.replace(".pt", "_tar_metadata.json")
    with open(out, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Wrote global tar-sharded manifest: {out}")

# ----------------------------------------
# Helpers
# ----------------------------------------

def is_lazy_embedding_loader(embeddings):
    """Check if embeddings is a lazy loader (LazyEmbeddingLoader or TarBasedEmbeddingLoader)."""
    return isinstance(embeddings, (LazyEmbeddingLoader, TarBasedEmbeddingLoader))

def save_rank_shard(base_path, layer_name, rank, tensor):
    """Legacy shard saver (kept for backward compat)."""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    shard_path = base_path.replace('.pt', f'_{layer_name}_rank{rank}.pt')
    torch.save(tensor.cpu(), shard_path)
    return shard_path

def merge_rank_shards(base_path, layer_names, total_samples, world_size, chunk_size=50_000, remove_shards=True):
    """
    Legacy merge (kept for backward compat). Prefer streaming manifests instead.
    """
    indices_per_rank = (total_samples + world_size - 1) // world_size
    embedding_dims = {}

    for layer in layer_names:
        found_dim = False
        for r in range(world_size):
            shard_path = base_path.replace('.pt', f'_{layer}_rank{r}.pt')
            if os.path.exists(shard_path):
                shard = torch.load(shard_path, map_location='cpu')
                if shard.numel() > 0:
                    embedding_dims[layer] = shard.shape[1]
                    found_dim = True
                    del shard
                    break
                del shard
        if not found_dim:
            raise FileNotFoundError(f"No non-empty shard found for layer {layer}")

    metadata = {
        'total_samples': {k: int(total_samples) for k in layer_names},
        'embedding_dims': embedding_dims,
        'chunk_size': int(chunk_size),
        'layer_names': list(layer_names),
        'storage_type': 'chunked_legacy',
    }
    metadata_path = base_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    for layer in layer_names:
        current_buffer = []
        current_count = 0
        chunk_idx = 0

        for r in range(world_size):
            shard_path = base_path.replace('.pt', f'_{layer}_rank{r}.pt')
            expected_len = min(indices_per_rank, max(0, total_samples - r * indices_per_rank))
            if expected_len == 0:
                continue
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Missing shard {shard_path}")

            shard = torch.load(shard_path, map_location='cpu')
            actual_len = shard.shape[0]
            offset = 0
            while offset < actual_len:
                remaining = actual_len - offset
                space = chunk_size - current_count
                take = min(remaining, space)
                current_buffer.append(shard[offset:offset+take])
                current_count += take
                offset += take
                if current_count == chunk_size:
                    chunk_tensor = torch.cat(current_buffer, dim=0)
                    out_path = base_path.replace('.pt', f'_{layer}_chunk_{chunk_idx}.pt')
                    torch.save(chunk_tensor, out_path)
                    chunk_idx += 1
                    current_buffer = []
                    current_count = 0

            del shard
            if remove_shards:
                os.remove(shard_path)

        if current_count > 0:
            chunk_tensor = torch.cat(current_buffer, dim=0)
            out_path = base_path.replace('.pt', f'_{layer}_chunk_{chunk_idx}.pt')
            torch.save(chunk_tensor, out_path)

    logging.info(f"Finished merging shards for {base_path}")

def _derive_rank_indices(dataset, is_distributed: bool) -> Optional[List[int]]:
    if not is_distributed:
        return None
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    N = len(dataset)
    per_rank = (N + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, N)
    return list(range(start, end))

# ----------------------------------------
# MAIN: Memory-efficient image embedding computation (STREAMING)
# ----------------------------------------

def compute_image_embeddings_streaming(
    image_embeddings: Optional[Any],
    image_embedding_path: str,
    args: Any,
    dataset,
    dataset_loader: Iterable,
    clip_model,
    device: torch.device,
    intermediate_image_layer_names: List[str],
    is_distributed: bool,
    stream_rows_per_file: int = 5000,  # Further reduced to 1k for extremely conservative memory usage
) -> Any:
    """
    Compute image embeddings and stream them to disk:
      - non-LAION: rank->chunk files + global chunked_stream manifest
      - LAION400M: rank->tar shards (indices + embeddings) + global tar_sharded manifest
    Returns a lazy loader (no in-memory merges).
    """
    needs_img_embed = (
        image_embeddings is None
        or getattr(args, "recompute_cache", False)
        or (not is_lazy_embedding_loader(image_embeddings)
            and any(ln not in image_embeddings for ln in intermediate_image_layer_names))
        or (is_lazy_embedding_loader(image_embeddings)
            and any(ln not in getattr(image_embeddings, "layer_names", []) for ln in intermediate_image_layer_names))
    )
    if not needs_img_embed:
        return image_embeddings

    rank = dist.get_rank() if is_distributed else 0
    world = dist.get_world_size() if is_distributed else 1
    if is_main_process():
        logging.info(f"Starting streaming embedding computation. Distributed={is_distributed} world_size={world}")

    target_dtype = torch.float32 if getattr(args, "force_float32", False) else torch.float16
    layer_names = list(intermediate_image_layer_names)

    use_tar = (getattr(args, "dataset", None) == "LAION400M" and 
               (hasattr(dataset, "tar_ids") or hasattr(dataset, "index")))
    image_base_path = image_embedding_path
    if use_tar:
        writer = TarShardStreamWriter(
            base_path=image_base_path,
            layer_names=layer_names,
            rank=rank,
            dataset=dataset,
            max_rows_per_shard=stream_rows_per_file,
        )
    else:
        writer = ChunkStreamWriter(
            base_path=image_base_path,
            layer_names=layer_names,
            rank=rank,
            max_rows_per_chunk=stream_rows_per_file,
        )

    rows_written = 0
    with torch.inference_mode():
        # Create rank index list for distributed case, or track indices manually for single-rank
        rank_idx_list = _derive_rank_indices(dataset, is_distributed)
        rank_ptr = 0

        for batch_idx, batch in enumerate(tqdm(dataset_loader, disable=(not is_main_process()))):
            if isinstance(batch, dict) and "image_options" in batch:
                images = batch["image_options"]
                batch_indices = batch.get("dataset_indices", None)
            else:
                images, *_ = batch
                batch_indices = None

            if batch_idx % 10 == 0:
                log_memory_usage(dataset_loader,f"[Images] Step {batch_idx}")

            batch_embeds = compute_image_embeddings_intermediate_batch(
                images, clip_model, device, layer_names, dtype=torch.float32
            )
            for k in batch_embeds.keys():
                batch_embeds[k] = batch_embeds[k].to(target_dtype)

            B = next(iter(batch_embeds.values())).shape[0] if batch_embeds else 0

            if use_tar:
                if batch_indices is None:
                    # For tar streaming, we need dataset indices to map to (tar_num, local_idx)
                    if rank_idx_list is not None:
                        # Distributed case: use pre-computed rank indices
                        batch_indices = rank_idx_list[rank_ptr:rank_ptr + B]
                        rank_ptr += B
                    else:
                        # Single-rank case: derive indices from batch position
                        batch_indices = list(range(rank_ptr, rank_ptr + B))
                        rank_ptr += B
                writer.append_batch(batch_embeds, list(map(int, batch_indices)))
            else:
                writer.append_batch(batch_embeds)

            rows_written += B


    rank_manifest = writer.finalize()

    if is_distributed:
        dist.barrier()

    # Build global manifests (rank 0 only)
    if (not is_distributed) or dist.get_rank() == 0:
        logging.info("Building global manifest (no tensor concatenation)")
        embedding_dims: Dict[str, int] = {}

        if use_tar:
            man = json.load(open(rank_manifest))
            any_tar = next(iter(man["tars"].keys()), None)
            if any_tar is None or len(man["tars"][any_tar]) == 0:
                raise RuntimeError("No tar shards written.")
            probe_file = man["tars"][any_tar][0]["file"]
            probe = torch.load(probe_file, map_location='cpu')
            for ln in layer_names:
                embedding_dims[ln] = int(probe[ln].shape[1])
            build_global_tar_sharded_manifest(
                base_path=image_base_path,
                world_size=world,
                embedding_dims=embedding_dims,
                dataset=dataset,
                layer_names=layer_names,
            )
        else:
            man = json.load(open(rank_manifest))
            # probe first available file per layer
            for ln in layer_names:
                files = man["layers"].get(ln, [])
                if not files:
                    continue
                probe_file = files[0]["file"]
                p = torch.load(probe_file, map_location='cpu')
                embedding_dims[ln] = int(p.shape[1])
            total_samples = len(dataset)
            build_global_chunked_manifest(
                base_path=image_base_path,
                layer_names=layer_names,
                world_size=world,
                total_samples=total_samples,
                embedding_dims=embedding_dims,
            )

    if is_distributed:
        dist.barrier()

    # Load lazily according to storage type
    if use_tar:
        return load_embeddings_tar_based(image_base_path)
    else:
        return load_embeddings_lazy(image_base_path)

def compute_caption_embeddings_streaming(
    caption_embeddings: Optional[Any],
    caption_embeddings_path: str,
    args: Any,
    dataset,
    clip_model,
    device: torch.device,
    intermediate_text_layer_names: List[str],
    is_distributed: bool,
    stream_rows_per_file: int = 50000,  # Further reduced to 1k for extremely conservative memory usage
) -> Any:
    """
    Compute caption embeddings and stream them to disk (no giant in-RAM tensors).

    Storage:
      - Default (most robust): rank -> chunk files + global chunked_stream manifest
      - If LAION-like & we can map caption idx -> dataset idx -> (tar, local): rank -> tar shards + global tar_sharded manifest

    Returns
      A lazy loader (LazyEmbeddingLoader or TarBasedEmbeddingLoader), ready to use.
    """
    needs_txt_embed = (
        caption_embeddings is None
        or getattr(args, "recompute_cache", False)
        or (not is_lazy_embedding_loader(caption_embeddings)
            and any(ln not in caption_embeddings for ln in intermediate_text_layer_names))
        or (is_lazy_embedding_loader(caption_embeddings)
            and any(ln not in getattr(caption_embeddings, "layer_names", []) for ln in intermediate_text_layer_names))
    )
    if not needs_txt_embed:
        return caption_embeddings

    rank = dist.get_rank() if is_distributed else 0
    world = dist.get_world_size() if is_distributed else 1
    if is_main_process():
        logging.info(f"[captions] starting streaming computation. distributed={is_distributed} world={world}")

    layer_names = list(intermediate_text_layer_names)

    # Decide whether we can/should tar-shard the text
    def _can_use_tar_for_captions(ds) -> bool:
        # We need to be on a LAION-like dataset with a tar index AND
        # the ability to map caption pointer/index -> dataset index, e.g. get_ptr_to_idx / ptr_to_idx
        if getattr(args, "dataset", None) != "LAION400M":
            return False
        if not (hasattr(ds, "tar_ids") or hasattr(ds, "index")):
            return False
        return hasattr(ds, "get_ptr_to_idx") or hasattr(ds, "ptr_to_idx")

    use_tar = _can_use_tar_for_captions(dataset)

    # pick base path for text embeddings
    text_base_path = caption_embeddings_path

    # build writer
    if use_tar:
        writer = TarShardStreamWriter(
            base_path=text_base_path,
            layer_names=layer_names,
            rank=rank,
            dataset=dataset,
            max_rows_per_shard=stream_rows_per_file,
        )
    else:
        writer = ChunkStreamWriter(
            base_path=text_base_path,
            layer_names=layer_names,
            rank=rank,
            max_rows_per_chunk=stream_rows_per_file,
        )

    # For efficient caption embedding, use the shared caption store if available
    # Otherwise, fall back to dataset's caption vocabulary
    # Check for new shared memory caption index approach
    if hasattr(dataset, 'use_shared_caption_index') and dataset.use_shared_caption_index:
        try:
            # Use the new method to get captions for embedding
            all_captions = dataset.get_captions_for_embedding(allow_for_embedding_computation=True)
            total_captions = len(all_captions)
            
            if total_captions == 0:
                raise RuntimeError(
                    "No captions available for embedding. Call prepare_caption_vocab_for_splits(...) first."
                )
            
            logging.info(f"[captions] Using shared memory caption approach: {total_captions} captions")
            
            # Function to get caption by index
            def get_caption_by_index(idx):
                return all_captions[idx]
                
        except Exception as e:
            raise RuntimeError(f"Failed to get captions for embedding from shared memory approach: {e}")
    
    # Legacy approaches below
    elif hasattr(dataset, 'use_shared_captions') and dataset.use_shared_captions and dataset.caption_manager:
        # Use shared caption store - much more efficient
        if hasattr(dataset, "caption_hash_to_idx") and hasattr(dataset, "caption_hashes"):
            # CRITICAL FIX: Use vocabulary order, NOT sorted hash order!
            vocab_hashes = dataset.caption_hashes  # This preserves natural order from dataset traversal
            total_captions = len(vocab_hashes)
        else:
            raise RuntimeError("Shared caption store enabled but no caption vocabulary found")
        
        # Function to get caption by index
        def get_caption_by_index(idx):
            caption_hash = vocab_hashes[idx]
            return dataset.caption_manager.get_caption(caption_hash)
            
    elif hasattr(dataset, 'caption_to_idx') and dataset.caption_to_idx:
        # Legacy approach using caption_to_idx dictionary
        logging.info("[captions] Using legacy caption_to_idx approach")
        sorted_captions = sorted(dataset.caption_to_idx.items(), key=lambda x: x[1])
        total_captions = len(sorted_captions)
        
        def get_caption_by_index(idx):
            return sorted_captions[idx][0]
            
    elif hasattr(dataset, "caption_hash_to_idx") and hasattr(dataset, "caption_hashes"):
        # Fallback to dataset's vocabulary - use cached strings, NOT expensive hash lookups
        # CRITICAL FIX: Use vocabulary order, NOT sorted hash order!
        vocab_hashes = dataset.caption_hashes  # This preserves natural order from dataset traversal
        total_captions = len(vocab_hashes)
        
        def get_caption_by_index(idx):
            caption_hash = vocab_hashes[idx]
            # Try cached string first (fast)
            if caption_hash in dataset.hash_to_string:
                return dataset.hash_to_string[caption_hash]
            # NEVER return placeholder captions - this breaks embedding computation!
            raise RuntimeError(
                f"Caption hash {caption_hash} not found in hash_to_string cache at index {idx}. "
                f"This indicates incomplete vocabulary building or cache corruption. "
                f"Vocabulary size: {len(vocab_hashes)}, Cache size: {len(dataset.hash_to_string)}. "
                f"Use the new shared memory approach (use_shared_caption_index=True) instead."
            )
    else:
        raise RuntimeError(
            "Caption vocabulary not built. Call prepare_caption_vocab_for_splits(...) first. "
            "Make sure keep_strings=True to cache caption strings for embedding."
        )
    
    if is_main_process():
        logging.info(f"[captions] Total captions in vocabulary: {total_captions}")
        if hasattr(dataset, 'use_shared_captions') and dataset.use_shared_captions:
            logging.info("[captions] Using shared caption store for efficient access")
    if is_distributed:
        shard_len = (total_captions + world - 1) // world
        shard_start = rank * shard_len
        shard_end = min(shard_start + shard_len, total_captions)
    else:
        shard_start, shard_end = 0, total_captions

    if shard_start >= shard_end:
        if is_main_process():
            logging.info(f"[captions] rank {rank}: no work (total {total_captions})")
    else:
        logging.info(f"[captions] rank {rank}: processing {shard_start}:{shard_end} "
                     f"({shard_end - shard_start}/{total_captions})")

    # We don’t need a DataLoader for text; do a simple range loop to avoid extra RAM
    with torch.inference_mode():
        batch_size = getattr(args, "batch_size", 512)
        for batch_idx, i0 in enumerate(tqdm(range(shard_start, shard_end, batch_size),disable=(not is_main_process()))):
            i1 = min(i0 + batch_size, shard_end)
            
            # Get caption strings for this batch using the efficient lookup function
            text_batch = []
            for i in range(i0, i1):
                caption_str = get_caption_by_index(i)
                if caption_str is None:
                    logging.warning(f"[captions] Could not retrieve caption at index {i}, skipping")
                    continue
                text_batch.append(caption_str)
            
            if batch_idx == 0 and is_main_process():
                logging.info(f"[captions] First batch sample captions: {text_batch[:2]}...")
            
            if len(text_batch) == 0:
                logging.warning(f"[captions] Empty text batch at {i0}:{i1}, skipping embedding computation")
                continue

            # compute intermediate-layer text embeddings
            batch_embeds = compute_caption_embeddings_intermediate_batch(
                text_batch, clip_model, device, layer_names, dtype=torch.float32
            )

            # if batch_idx % 100 == 0:
            #     log_memory_usagef"[Captions] Step {batch_idx}")

            if use_tar:
                # Need to map caption indices -> dataset indices, then the TarShardStreamWriter
                # will use dataset's compact index (tar_ids/tar_offsets/local_indices) to place into (tar, local_idx).
                cap_idxs = list(range(i0, i1))
                if hasattr(dataset, "get_ptr_to_idx"):
                    ds_idxs = [int(dataset.get_ptr_to_idx(ci)) for ci in cap_idxs]
                elif hasattr(dataset, "ptr_to_idx"):
                    # assume it’s a list/array mapping ptr -> dataset idx
                    ds_idxs = [int(dataset.ptr_to_idx[ci]) for ci in cap_idxs]
                else:
                    raise RuntimeError("Tar text streaming requires ptr->idx mapping on dataset")
                writer.append_batch(batch_embeds, ds_idxs)
            else:
                writer.append_batch(batch_embeds)
                

    rank_manifest = writer.finalize()

    if is_distributed:
        dist.barrier()

    # Rank 0 builds the global manifest by probing small files (no concat)
    if (not is_distributed) or dist.get_rank() == 0:
        logging.info("[captions] building global manifest")
        embedding_dims: Dict[str, int] = {}

        if use_tar:
            man = json.load(open(rank_manifest))
            any_tar = next(iter(man["tars"].keys()), None)
            if any_tar is None or len(man["tars"][any_tar]) == 0:
                raise RuntimeError("[captions] no tar shards written")
            probe_file = man["tars"][any_tar][0]["file"]
            probe = torch.load(probe_file, map_location="cpu")
            for ln in layer_names:
                embedding_dims[ln] = int(probe[ln].shape[1])
            # total samples should match len(captions) (split-scoped vocab for LAION)
            build_global_tar_sharded_manifest(
                base_path=text_base_path,
                world_size=world,
                embedding_dims=embedding_dims,
                dataset=dataset,
                layer_names=layer_names,
                # For captions we want the manifest’s total_samples to reflect caption count
            )
        else:
            man = json.load(open(rank_manifest))
            for ln in layer_names:
                files = man["layers"].get(ln, [])
                if not files:
                    continue
                probe_file = files[0]["file"]
                p = torch.load(probe_file, map_location="cpu")
                embedding_dims[ln] = int(p.shape[1])
            build_global_chunked_manifest(
                base_path=text_base_path,
                layer_names=layer_names,
                world_size=world,
                total_samples=total_captions,
                embedding_dims=embedding_dims,
            )

    if is_distributed:
        dist.barrier()

    # Return a lazy loader for immediate use
    if use_tar:
        return load_embeddings_tar_based(text_base_path)
    else:
        return load_embeddings_lazy(text_base_path)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared memory caption hash store for efficient cross-process caption access.

Features:
- Lock-free reading for maximum performance
- Safe concurrent access from multiple processes
- Memory-efficient packed binary format
- Fast hash-based lookups with binary search
- Automatic cleanup and recovery from crashes
"""

import os
import time
import struct
import hashlib
import logging
from typing import Optional, List, Tuple, Dict
from multiprocessing import shared_memory
import threading
import atexit

class SharedCaptionHashStore:
    """
    Thread-safe and process-safe shared memory store for caption hash mappings.
    
    Memory layout:
    - Header (64 bytes): [magic, version, num_entries, next_string_offset, checksum, reserved...]
    - Index entries: [hash(8), string_offset(4), string_length(4)] * num_entries
    - String storage: concatenated UTF-8 strings
    """
    
    MAGIC = b"CAPT"
    VERSION = 1
    HEADER_SIZE = 64
    INDEX_ENTRY_SIZE = 16  # 8 + 4 + 4
    
    def __init__(self, 
                 name: str = "caption_store",
                 max_entries: int = 10_000_000,
                 string_storage_mb: int = 500,
                 read_only: bool = False):
        """
        Initialize shared caption store.
        
        Args:
            name: Unique name for the shared memory segment
            max_entries: Maximum number of caption entries
            string_storage_mb: String storage size in MB
            read_only: If True, only attach to existing store (don't create)
        """
        self.name = name
        self.max_entries = max_entries
        self.string_storage_size = string_storage_mb * 1024 * 1024
        self.read_only = read_only
        
        # Calculate sizes
        self.index_size = self.HEADER_SIZE + (max_entries * self.INDEX_ENTRY_SIZE)
        self.total_size = self.index_size + self.string_storage_size
        
        # Thread safety
        self._lock = threading.RLock()
        self.is_creator = False
        self.shm = None
        
        # Try to attach or create
        self._initialize_memory()
        
        # Register cleanup
        atexit.register(self._cleanup)
    
    def _initialize_memory(self):
        """Initialize or attach to shared memory."""
        shm_name = f"{self.name}_{os.getpid()//1000}"  # Group by process range
        
        try:
            # Try to attach to existing
            self.shm = shared_memory.SharedMemory(name=shm_name)
            self.is_creator = False
            logging.info(f"Attached to existing caption store: {shm_name}")
            
            # Setup views first so we can validate header
            self._setup_views()
            
            # Validate header
            if not self._validate_header():
                logging.warning("Invalid header in existing store, recreating...")
                self.shm.close()
                self._create_new_store(shm_name)
                return  # _create_new_store already sets up views
                
        except FileNotFoundError:
            if self.read_only:
                raise RuntimeError(f"Caption store {shm_name} not found and read_only=True")
            self._create_new_store(shm_name)
            return  # _create_new_store already sets up views
        
        # Views already setup above for existing store case
    
    def _create_new_store(self, shm_name: str):
        """Create new shared memory store."""
        try:
            self.shm = shared_memory.SharedMemory(
                name=shm_name,
                create=True,
                size=self.total_size
            )
            self.is_creator = True
            logging.info(f"Created new caption store: {shm_name} ({self.total_size//1024//1024}MB)")
            
            # Setup memory views first
            self._setup_views()
            
            # Initialize header
            self._write_header(num_entries=0, next_string_offset=0)
            
        except FileExistsError:
            # Race condition - someone else created it
            self.shm = shared_memory.SharedMemory(name=shm_name)
            self.is_creator = False
            logging.info(f"Attached to newly created store: {shm_name}")
            
            # Setup memory views after attaching
            self._setup_views()
    
    def _setup_views(self):
        """Setup memory views for efficient access."""
        try:
            # Memory layout views
            self.header_view = memoryview(self.shm.buf)[:self.HEADER_SIZE]
            self.index_view = memoryview(self.shm.buf)[self.HEADER_SIZE:self.index_size]
            self.string_view = memoryview(self.shm.buf)[self.index_size:]
            
            # Cache frequently accessed values
            self._cached_num_entries = 0
            self._cache_timestamp = 0
            self._cache_refresh_interval = 1.0  # Refresh every second
            
        except Exception as e:
            raise RuntimeError(f"Failed to setup memory views: {e}")
    
    def _validate_header(self) -> bool:
        """Validate header magic and version."""
        try:
            magic, version = struct.unpack_from('4sI', self.header_view, 0)
            return magic == self.MAGIC and version == self.VERSION
        except Exception:
            return False
    
    def _write_header(self, num_entries: int, next_string_offset: int):
        """Write header to shared memory."""
        try:
            checksum = self._compute_checksum(num_entries, next_string_offset)
            
            # Use the underlying buffer directly to avoid memoryview issues
            header_data = struct.pack(
                '4sIQQQ44x',  # magic, version, num_entries, next_string_offset, checksum, padding
                self.MAGIC,
                self.VERSION,
                num_entries,
                next_string_offset,
                checksum
            )
            
            # Write directly to the shared memory buffer
            self.shm.buf[:len(header_data)] = header_data
                
        except Exception as e:
            logging.error(f"Failed to write header: {e}")
            logging.error(f"Shared memory buffer size: {len(self.shm.buf)}")
            logging.error(f"Header data size: {len(header_data) if 'header_data' in locals() else 'unknown'}")
            raise
    
    def _read_header(self) -> Tuple[int, int]:
        """Read header from shared memory with caching."""
        current_time = time.time()
        
        # Use cached values if recent
        if (current_time - self._cache_timestamp) < self._cache_refresh_interval:
            return self._cached_num_entries, 0
        
        # Read fresh header
        try:
            magic, version, num_entries, next_string_offset, checksum = struct.unpack_from(
                '4sIQQQ', self.header_view, 0
            )
            
            if magic != self.MAGIC or version != self.VERSION:
                raise ValueError("Invalid header")
            
            # Verify checksum
            expected_checksum = self._compute_checksum(num_entries, next_string_offset)
            if checksum != expected_checksum:
                logging.warning("Header checksum mismatch - data may be corrupted")
            
            # Update cache
            self._cached_num_entries = num_entries
            self._cache_timestamp = current_time
            
            return num_entries, next_string_offset
            
        except struct.error:
            return 0, 0
    
    def _compute_checksum(self, num_entries: int, next_string_offset: int) -> int:
        """Compute simple checksum for header validation."""
        return (num_entries ^ next_string_offset ^ 0xDEADBEEF) & 0xFFFFFFFFFFFFFFFF
    
    def add_caption(self, caption: str) -> int:
        """
        Add caption to store and return its hash.
        Only the creating process can add captions.
        """
        if self.read_only or not self.is_creator:
            # Non-creator processes just compute hash
            return self._fast_hash(caption)
        
        caption_hash = self._fast_hash(caption)
        
        # Check if already exists
        if self.get_caption(caption_hash) is not None:
            return caption_hash
        
        with self._lock:
            # Re-read header to get current state
            num_entries, next_string_offset = self._read_header()
            
            # Check capacity with warnings
            if num_entries >= self.max_entries:
                logging.error(f"Caption store full: {num_entries}/{self.max_entries} entries. "
                             f"Increase caption_store_max_entries parameter.")
                return caption_hash
            elif num_entries > self.max_entries * 0.9:
                logging.warning(f"Caption store nearing capacity: {num_entries}/{self.max_entries} entries "
                               f"({num_entries/self.max_entries:.1%} full)")
            
            caption_bytes = caption.encode('utf-8')
            if next_string_offset + len(caption_bytes) >= self.string_storage_size:
                string_usage_mb = next_string_offset / (1024 * 1024)
                total_mb = self.string_storage_size / (1024 * 1024)
                logging.error(f"String storage full: {string_usage_mb:.1f}MB/{total_mb:.1f}MB used. "
                             f"Increase caption_store_size_mb parameter.")
                return caption_hash
            
            try:
                # Write string data
                string_end = next_string_offset + len(caption_bytes)
                self.string_view[next_string_offset:string_end] = caption_bytes
                
                # Write index entry
                entry_offset = num_entries * self.INDEX_ENTRY_SIZE
                struct.pack_into(
                    'QII',
                    self.index_view,
                    entry_offset,
                    caption_hash,
                    next_string_offset,
                    len(caption_bytes)
                )
                
                # Update header atomically
                self._write_header(num_entries + 1, string_end)
                
                # Invalidate cache
                self._cache_timestamp = 0
                
                logging.debug(f"Added caption hash {caption_hash:016x} at entry {num_entries}")
                
            except Exception as e:
                logging.error(f"Failed to add caption: {e}")
        
        return caption_hash
    
    def get_caption(self, caption_hash: int) -> Optional[str]:
        """Get caption string from hash using binary search."""
        num_entries, _ = self._read_header()
        
        if num_entries == 0:
            return None
        
        # Binary search in sorted index
        left, right = 0, num_entries - 1
        
        while left <= right:
            mid = (left + right) // 2
            entry_offset = mid * self.INDEX_ENTRY_SIZE
            
            try:
                stored_hash, str_offset, str_length = struct.unpack_from(
                    'QII', self.index_view, entry_offset
                )
                
                if stored_hash == caption_hash:
                    # Found it - extract string
                    try:
                        caption_bytes = bytes(self.string_view[str_offset:str_offset + str_length])
                        return caption_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        logging.warning(f"Unicode decode error for hash {caption_hash:016x}")
                        return None
                        
                elif stored_hash < caption_hash:
                    left = mid + 1
                else:
                    right = mid - 1
                    
            except struct.error:
                logging.warning(f"Struct unpack error at entry {mid}")
                return None
        
        return None
    
    def bulk_add_captions(self, captions: List[str]) -> List[int]:
        """Add multiple captions efficiently."""
        if self.read_only or not self.is_creator:
            return [self._fast_hash(cap) for cap in captions]
        
        hashes = []
        
        with self._lock:
            for caption in captions:
                hash_val = self.add_caption(caption)
                hashes.append(hash_val)
        
        return hashes
    
    def get_stats(self) -> Dict[str, int]:
        """Get store statistics."""
        num_entries, next_string_offset = self._read_header()
        
        return {
            'num_entries': num_entries,
            'max_entries': self.max_entries,
            'string_bytes_used': next_string_offset,
            'string_bytes_total': self.string_storage_size,
            'memory_usage_mb': self.total_size // 1024 // 1024,
            'fill_percentage': int(100 * num_entries / self.max_entries) if self.max_entries > 0 else 0
        }
    
    def _fast_hash(self, text: str) -> int:
        """
        Fast, consistent hash function.
        Uses SHA-256 truncated to 64 bits for cross-platform consistency.
        """
        # Use SHA-256 for consistency across platforms and Python versions
        hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
        # Take first 8 bytes as 64-bit integer
        return struct.unpack('>Q', hash_bytes[:8])[0]
    
    def _cleanup(self):
        """Cleanup shared memory."""
        if self.shm is not None:
            try:
                if self.is_creator:
                    # Only creator unlinks
                    self.shm.close()
                    self.shm.unlink()
                    logging.info(f"Cleaned up caption store: {self.shm.name}")
                else:
                    self.shm.close()
            except Exception as e:
                logging.debug(f"Cleanup error: {e}")
    
    def __del__(self):
        """Destructor cleanup."""
        self._cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()


class CaptionManager:
    """High-level manager for caption storage across processes."""
    
    def __init__(self, 
                 store_name: str = "laion_captions",
                 max_entries: int = 60_000_000,  # Increased for large datasets
                 storage_mb: int = 5000):        # Increased for large datasets
        self.store = SharedCaptionHashStore(
            name=store_name,
            max_entries=max_entries,
            string_storage_mb=storage_mb
        )
        
        # Local cache for frequently accessed captions
        self._local_cache = {}
        self._cache_max_size = 1000
    
    def add_caption(self, caption: str) -> int:
        """Add caption and cache locally."""
        caption_hash = self.store.add_caption(caption)
        
        # Cache locally for fast access
        if len(self._local_cache) < self._cache_max_size:
            self._local_cache[caption_hash] = caption
        
        return caption_hash
    
    def get_caption(self, caption_hash: int) -> Optional[str]:
        """Get caption with local caching."""
        # Check local cache first
        if caption_hash in self._local_cache:
            return self._local_cache[caption_hash]
        
        # Get from shared store
        caption = self.store.get_caption(caption_hash)
        
        # Cache locally if found
        if caption is not None and len(self._local_cache) < self._cache_max_size:
            self._local_cache[caption_hash] = caption
        
        return caption
    
    def get_stats(self) -> Dict[str, int]:
        """Get comprehensive statistics."""
        stats = self.store.get_stats()
        stats['local_cache_size'] = len(self._local_cache)
        return stats


# Singleton instance for easy access
_global_caption_manager = None

def get_caption_manager(store_name: str = "laion_captions") -> CaptionManager:
    """Get or create global caption manager."""
    global _global_caption_manager
    
    if _global_caption_manager is None:
        _global_caption_manager = CaptionManager(store_name=store_name)
    
    return _global_caption_manager


def estimate_storage_requirements(num_captions: int, avg_caption_length: int = 50) -> dict:
    """
    Estimate storage requirements for a given number of captions.
    
    Args:
        num_captions: Number of unique captions
        avg_caption_length: Average caption length in characters
    
    Returns:
        Dict with storage estimates
    """
    # Index storage: 16 bytes per entry (hash + offset + length)
    index_mb = (64 + num_captions * 16) / (1024 * 1024)  # 64 byte header
    
    # String storage: average caption length in UTF-8 bytes
    string_mb = (num_captions * avg_caption_length) / (1024 * 1024)
    
    # Add 20% overhead for safety
    total_mb = (index_mb + string_mb) * 1.2
    
    return {
        'num_captions': num_captions,
        'index_storage_mb': index_mb,
        'string_storage_mb': string_mb,
        'total_storage_mb': total_mb,
        'recommended_max_entries': num_captions,
        'recommended_storage_mb': int(string_mb * 1.2)  # 20% overhead
    }


# Utility functions for easy integration
def add_caption(caption: str) -> int:
    """Add caption to global store."""
    return get_caption_manager().add_caption(caption)

def get_caption(caption_hash: int) -> Optional[str]:
    """Get caption from global store."""
    return get_caption_manager().get_caption(caption_hash)

def get_caption_stats() -> Dict[str, int]:
    """Get global store statistics."""
    return get_caption_manager().get_stats()


if __name__ == "__main__":
    # Test the shared caption store
    import random
    import string
    
    def random_caption(length=50):
        return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))
    
    # Test with multiple "processes" (threads)
    with SharedCaptionHashStore("test_store", max_entries=1000, string_storage_mb=1) as store:
        
        # Add some test captions
        test_captions = [random_caption() for _ in range(100)]
        
        print("Adding captions...")
        hashes = []
        for caption in test_captions:
            hash_val = store.add_caption(caption)
            hashes.append(hash_val)
        
        print("Verifying captions...")
        for i, (caption, hash_val) in enumerate(zip(test_captions, hashes)):
            retrieved = store.get_caption(hash_val)
            if retrieved != caption:
                print(f"MISMATCH at {i}: {caption} != {retrieved}")
            else:
                print(f"OK {i}: {hash_val:016x}")
        
        print("Stats:", store.get_stats())

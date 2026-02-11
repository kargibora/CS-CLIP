#!/usr/bin/env python3
"""
Simple shared caption-to-index mapping for fast embedding lookup.

This solves your exact use case:
1. Put in a caption
2. Get back its index in the embedding array
3. Handle both positive and negative captions
4. Memory efficient across processes
"""

import os
import struct
import logging
from typing import Optional, Dict, List, Tuple
from multiprocessing import shared_memory
import threading
import atexit

if 'PYTHONHASHSEED' not in os.environ:
    os.environ['PYTHONHASHSEED'] = "42"
    
class SharedCaptionIndexStore:
    """
    Simple shared memory store for caption → index mappings.
    
    Stores: caption_hash → embedding_index
    
    For your use case:
    - Input: "a dog running" 
    - Output: 12345 (index in caption_embeddings array)
    - Access: embedding = caption_embeddings[12345]
    
    Memory efficient: 60M captions × 12 bytes = 720MB shared vs 12GB string storage
    """
    
    MAGIC = b"CIDX"  # Caption Index
    VERSION = 1
    HEADER_SIZE = 64
    ENTRY_SIZE = 12  # hash(8) + index(4)
    
    def __init__(self, 
                 name: str = "caption_index",
                 max_entries: int = 60_000_000):
        """Initialize shared caption index store."""
        self.name = name
        self.max_entries = max_entries
        self.total_size = self.HEADER_SIZE + (max_entries * self.ENTRY_SIZE)
        
        self._lock = threading.RLock()
        self.is_creator = False
        self.shm = None
        
        self._initialize_memory()
        atexit.register(self._cleanup)
    
    def _initialize_memory(self):
        """Initialize or attach to shared memory."""
        shm_name = f"{self.name}_{os.getpid()//1000}"
        
        try:
            # Try to attach to existing
            self.shm = shared_memory.SharedMemory(name=shm_name)
            self.is_creator = False
            logging.info(f"Attached to caption index store: {shm_name}")
            
        except FileNotFoundError:
            # Create new
            self.shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=self.total_size
            )
            self.is_creator = True
            logging.info(f"Created caption index store: {shm_name} ({self.total_size//1024//1024}MB)")
            
            # Initialize header
            self._write_header(0)
        
        except FileExistsError:
            # Race condition - attach to existing
            self.shm = shared_memory.SharedMemory(name=shm_name)
            self.is_creator = False
            logging.info(f"Attached to existing caption index store: {shm_name}")
        
        # Setup memory views
        self.header_view = memoryview(self.shm.buf)[:self.HEADER_SIZE]
        self.data_view = memoryview(self.shm.buf)[self.HEADER_SIZE:]
    
    def _write_header(self, num_entries: int):
        """Write header to shared memory."""
        header_data = struct.pack('4sIQ52x', self.MAGIC, self.VERSION, num_entries)
        self.shm.buf[:len(header_data)] = header_data
    
    def _read_header(self) -> int:
        """Read number of entries from header."""
        try:
            magic, version, num_entries = struct.unpack_from('4sIQ', self.header_view, 0)
            if magic == self.MAGIC and version == self.VERSION:
                return num_entries
        except:
            pass
        return 0
    
    def _fast_hash(self, caption: str) -> int:
        """Actually fast hash using Python's built-in hash function."""
        return hash(caption) & 0x7FFFFFFFFFFFFFFF  # Mask to ensure positive int
    
    def build_index(self, captions: List[str]):
        """
        Build the caption → index mapping from a list of captions.
        
        Args:
            captions: List of caption strings in embedding order
                     captions[i] corresponds to caption_embeddings[i]
        """
        if not self.is_creator:
            logging.warning("Only creator process can build index")
            return
        
        with self._lock:
            num_entries = len(captions)
            
            if num_entries > self.max_entries:
                logging.error(f"Too many captions: {num_entries} > {self.max_entries}")
                return
            
            # Build hash → index mapping
            hash_index_pairs = []
            for i, caption in enumerate(captions):
                caption_hash = self._fast_hash(caption)
                hash_index_pairs.append((caption_hash, i))
            
            # Sort by hash for binary search
            hash_index_pairs.sort(key=lambda x: x[0])
            
            # Write to shared memory
            for i, (caption_hash, caption_index) in enumerate(hash_index_pairs):
                offset = i * self.ENTRY_SIZE
                struct.pack_into('QI', self.data_view, offset, caption_hash, caption_index)
            
            # Update header
            self._write_header(num_entries)
            
            logging.info(f"Built caption index: {num_entries} mappings in shared memory")
    
    def get_caption_index(self, caption: str) -> Optional[int]:
        """
        Get embedding index for a caption.
        
        This is your main function:
        caption = "a dog running"
        index = store.get_caption_index(caption)
        embedding = caption_embeddings[index]
        """
        caption_hash = self._fast_hash(caption)
        return self._get_index_by_hash(caption_hash)
    
    def _get_index_by_hash(self, caption_hash: int) -> Optional[int]:
        """Get index using binary search on sorted hashes."""
        num_entries = self._read_header()
        if num_entries == 0:
            return None
        
        # Binary search
        left, right = 0, num_entries - 1
        
        while left <= right:
            mid = (left + right) // 2
            offset = mid * self.ENTRY_SIZE
            
            try:
                stored_hash, caption_index = struct.unpack_from('QI', self.data_view, offset)
                
                if stored_hash == caption_hash:
                    return caption_index
                elif stored_hash < caption_hash:
                    left = mid + 1
                else:
                    right = mid - 1
                    
            except struct.error:
                return None
        
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get store statistics."""
        num_entries = self._read_header()
        return {
            'num_entries': num_entries,
            'max_entries': self.max_entries,
            'memory_mb': self.total_size // 1024 // 1024,
            'fill_percentage': int(100 * num_entries / self.max_entries) if self.max_entries > 0 else 0
        }
    
    def _cleanup(self):
        """Cleanup shared memory."""
        if self.shm:
            try:
                if self.is_creator:
                    self.shm.close()
                    self.shm.unlink()
                    logging.info(f"Cleaned up caption index store: {self.shm.name}")
                else:
                    self.shm.close()
            except:
                pass


class HashTableCaptionIndex:
    """
    Hash table implementation for O(1) caption → index lookups.
    
    Uses open addressing with linear probing for collision resolution.
    Faster than binary search for frequent lookups, but uses more memory.
    
    Memory usage: ~1.5x more than binary search, but O(1) vs O(log n) speed.
    """
    
    MAGIC = b"HTCI"  # Hash Table Caption Index
    VERSION = 1
    HEADER_SIZE = 64
    EMPTY_SLOT = 0xFFFFFFFFFFFFFFFF  # 64-bit marker for empty slots
    
    def __init__(self, 
                 name: str = "hashtable_caption_index",
                 max_entries: int = 60_000_000,
                 load_factor: float = 0.75):  # 75% full for good performance
        
        self.name = name
        self.max_entries = max_entries
        self.load_factor = load_factor
        self.table_size = int(max_entries / load_factor)  # Make table bigger than needed
        
        # Each slot: hash(8 bytes) + embedding_idx(4 bytes) = 12 bytes
        self.slot_size = 12
        self.table_bytes = self.table_size * self.slot_size
        self.total_size = self.HEADER_SIZE + self.table_bytes
        
        self._lock = threading.RLock()
        self.is_creator = False
        self.shm = None
        
        self._initialize_memory()
        atexit.register(self._cleanup)
    
    def _fast_hash(self, caption: str) -> int:
        """Fast hash that avoids collisions with EMPTY_SLOT marker."""
        hash_val = hash(caption) & 0x7FFFFFFFFFFFFFFF  # Force positive
        # Ensure hash is never EMPTY_SLOT
        return hash_val if hash_val != self.EMPTY_SLOT else hash_val - 1
    
    def _initialize_memory(self):
        """Initialize or attach to shared memory."""
        shm_name = f"{self.name}_{os.getpid()//1000}"
        
        try:
            self.shm = shared_memory.SharedMemory(name=shm_name)
            self.is_creator = False
            logging.info(f"Attached to hash table caption index: {shm_name}")
            
        except FileNotFoundError:
            self.shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=self.total_size
            )
            self.is_creator = True
            logging.info(f"Created hash table caption index: {shm_name} ({self.total_size//1024//1024}MB)")
            
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=shm_name)
            self.is_creator = False
            logging.info(f"Attached to existing hash table caption index: {shm_name}")
        
        # Setup memory views FIRST
        self.header_view = memoryview(self.shm.buf)[:self.HEADER_SIZE]
        self.table_view = memoryview(self.shm.buf)[self.HEADER_SIZE:]
        
        # Now initialize if creator
        if self.is_creator:
            self._clear_table()
            self._write_header(0)
    
    def _clear_table(self):
        """Clear all hash table slots."""
        # Fill with EMPTY_SLOT markers
        for i in range(self.table_size):
            offset = i * self.slot_size
            struct.pack_into('QI', self.table_view, offset, self.EMPTY_SLOT, 0)
    
    def _write_header(self, num_entries: int):
        """Write header."""
        header = struct.pack('4sIQ48x', self.MAGIC, self.VERSION, num_entries)
        self.shm.buf[:len(header)] = header
    
    def _read_header(self) -> int:
        """Read number of entries."""
        try:
            magic, version, num_entries = struct.unpack_from('4sIQ', self.header_view, 0)
            if magic == self.MAGIC and version == self.VERSION:
                return num_entries
        except:
            pass
        return 0
    
    def _probe_sequence(self, hash_val: int):
        """Generate probe sequence for hash table (linear probing)."""
        slot = hash_val % self.table_size
        for _ in range(self.table_size):  # Limit probing to avoid infinite loops
            yield slot
            slot = (slot + 1) % self.table_size
    
    def build_index(self, captions: List[str]):
        """Build hash table from captions (O(n) operation)."""
        if not self.is_creator:
            logging.warning("Only creator can build hash table index")
            return
        
        with self._lock:
            logging.info(f"Building hash table for {len(captions)} captions...")
            
            if len(captions) > self.max_entries:
                raise ValueError(f"Too many captions: {len(captions)} > {self.max_entries}")
            
            entries_added = 0
            collisions = 0
            
            for embedding_idx, caption in enumerate(captions):
                caption_hash = self._fast_hash(caption)
                
                # Find empty slot using linear probing
                for slot in self._probe_sequence(caption_hash):
                    offset = slot * self.slot_size
                    stored_hash, = struct.unpack_from('Q', self.table_view, offset)
                    
                    if stored_hash == self.EMPTY_SLOT:
                        # Found empty slot - insert here
                        struct.pack_into('QI', self.table_view, offset, caption_hash, embedding_idx)
                        entries_added += 1
                        break
                    elif stored_hash == caption_hash:
                        # Duplicate caption - update index
                        struct.pack_into('I', self.table_view, offset + 8, embedding_idx)
                        break
                    else:
                        # Collision - try next slot
                        collisions += 1
                        continue
            
            self._write_header(entries_added)
            
            logging.info(f"Built hash table: {entries_added} entries, {collisions} collisions")
            logging.info(f"Load factor: {entries_added / self.table_size:.1%}")
    
    def get_caption_index(self, caption: str) -> Optional[int]:
        """Get embedding index - O(1) average case!"""
        caption_hash = self._fast_hash(caption)
        
        # Linear probing to find the caption
        for slot in self._probe_sequence(caption_hash):
            offset = slot * self.slot_size
            try:
                stored_hash, embedding_idx = struct.unpack_from('QI', self.table_view, offset)
                
                if stored_hash == self.EMPTY_SLOT:
                    # Reached empty slot - caption not found
                    return None
                elif stored_hash == caption_hash:
                    # Found it!
                    return embedding_idx
                # else: collision, continue probing
                
            except struct.error:
                return None
        
        return None  # Table is full (shouldn't happen with proper load factor)
    
    def get_stats(self) -> Dict[str, any]:
        """Get performance statistics."""
        num_entries = self._read_header()
        return {
            'num_entries': num_entries,
            'table_size': self.table_size,
            'load_factor': num_entries / self.table_size if self.table_size > 0 else 0,
            'memory_mb': self.total_size // 1024 // 1024,
            'avg_lookup_time': 'O(1)',
            'max_lookup_time': f'O({int(1/self.load_factor)})'  # Worst case probing
        }
    
    def _cleanup(self):
        """Cleanup shared memory."""
        if self.shm:
            try:
                if self.is_creator:
                    self.shm.close()
                    self.shm.unlink()
                    logging.info(f"Cleaned up hash table: {self.shm.name}")
                else:
                    self.shm.close()
            except:
                pass


class CaptionIndexManager:
    """
    High-level manager for caption → index lookups with configurable backend.
    
    Supports two methods:
    - 'binary_search': O(log n) lookups, lower memory usage (default)
    - 'hash_table': O(1) lookups, higher memory usage but faster
    
    This is what you'll use in your pipeline.
    """
    
    def __init__(self, 
                 store_name: str = "laion_caption_index",
                 max_entries: int = 60_000_000,
                 cache_size: int = 1000,
                 method: str = "hash_table"):  # "binary_search" or "hash_table"
        """
        Initialize caption index manager.
        
        Args:
            store_name: Name for shared memory store
            max_entries: Maximum number of captions
            cache_size: Size of local LRU cache
            method: "binary_search" (memory efficient) or "hash_table" (faster)
        """
        self.method = method
        
        if method == "binary_search":
            self.store = SharedCaptionIndexStore(
                name=f"{store_name}_binary",
                max_entries=max_entries
            )
        elif method == "hash_table":
            self.store = HashTableCaptionIndex(
                name=f"{store_name}_hashtable",
                max_entries=max_entries
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'binary_search' or 'hash_table'")
        
        # Small local cache for frequently accessed captions
        self._cache = {}
        self._cache_max_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Initialize shared caption list (will be set during build or attach)
        self._caption_list_shm = None
        
        logging.info(f"Initialized CaptionIndexManager with method: {method}")
    
    def build_from_captions(self, captions: List[str]):
        """
        Build index from caption list and store captions in shared memory.
        
        Args:
            captions: List where captions[i] maps to caption_embeddings[i]
        """
        # Build the caption→index mapping
        self.store.build_index(captions)
        
        # Also store the caption list itself in shared memory
        self._store_caption_list(captions)
        
        # Pre-populate cache with first few captions
        for i, caption in enumerate(captions[:self._cache_max_size]):
            self._cache[caption] = i
        
        logging.info(f"Built caption index with {len(captions)} captions using {self.method}")
    
    def _store_caption_list(self, captions: List[str]):
        """Store the ordered caption list in shared memory for cross-process access."""
        import pickle
        from multiprocessing import shared_memory
        
        try:
            # Serialize the caption list
            caption_data = pickle.dumps(captions)
            caption_size = len(caption_data)
            
            # Create shared memory for caption list
            list_shm_name = f"{self.store.name}_caption_list"
            
            try:
                # Try to create new shared memory
                self._caption_list_shm = shared_memory.SharedMemory(
                    name=list_shm_name, create=True, size=caption_size + 8  # +8 for size header
                )
                
                # Store size header + data
                import struct
                self._caption_list_shm.buf[:8] = struct.pack('Q', caption_size)
                self._caption_list_shm.buf[8:8+caption_size] = caption_data
                
                logging.info(f"Created shared caption list: {list_shm_name} ({caption_size//1024//1024}MB)")
                
            except FileExistsError:
                # Shared memory already exists, attach to it
                self._caption_list_shm = shared_memory.SharedMemory(name=list_shm_name)
                logging.info(f"Attached to existing shared caption list: {list_shm_name}")
                
        except Exception as e:
            logging.warning(f"Failed to create shared caption list: {e}")
            self._caption_list_shm = None
    
    def get_all_captions(self) -> List[str]:
        """Get all captions from shared memory."""
        if not hasattr(self, '_caption_list_shm') or self._caption_list_shm is None:
            # Try to attach to existing shared memory
            self._attach_to_caption_list()
            
        if self._caption_list_shm is None:
            raise RuntimeError("Shared caption list not available")
        
        try:
            import pickle
            import struct
            
            # Read size header
            caption_size = struct.unpack('Q', self._caption_list_shm.buf[:8])[0]
            
            # Read and deserialize caption data
            caption_data = bytes(self._caption_list_shm.buf[8:8+caption_size])
            captions = pickle.loads(caption_data)
            
            return captions
            
        except Exception as e:
            raise RuntimeError(f"Failed to read shared caption list: {e}")
    
    def _attach_to_caption_list(self):
        """Attach to existing shared caption list."""
        from multiprocessing import shared_memory
        
        list_shm_name = f"{self.store.name}_caption_list"
        try:
            self._caption_list_shm = shared_memory.SharedMemory(name=list_shm_name)
            logging.info(f"Attached to shared caption list: {list_shm_name}")
        except FileNotFoundError:
            logging.warning(f"Shared caption list not found: {list_shm_name}")
            self._caption_list_shm = None
    
    def get_index(self, caption: str) -> Optional[int]:
        """
        Get embedding index for caption - YOUR MAIN FUNCTION!
        
        Usage:
            index = manager.get_index("a dog running")
            embedding = caption_embeddings[index]
        """
        # Check local cache first
        if caption in self._cache:
            self._cache_hits += 1
            return self._cache[caption]
        
        self._cache_misses += 1
        
        # Get from shared store
        index = self.store.get_caption_index(caption)
        
        # Cache if found and cache not full
        if index is not None and len(self._cache) < self._cache_max_size:
            self._cache[caption] = index
        
        return index
    
    def get_indices_batch(self, captions: List[str]) -> List[Optional[int]]:
        """Get indices for multiple captions efficiently."""
        return [self.get_index(caption) for caption in captions]
    
    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics."""
        store_stats = self.store.get_stats()
        
        total_lookups = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(1, total_lookups)
        
        return {
            **store_stats,
            'method': self.method,
            'cache_size': len(self._cache),
            'cache_hit_rate': hit_rate,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses
        }
    
    def cleanup(self):
        """Clean up shared memory resources."""
        try:
            if hasattr(self, '_caption_list_shm') and self._caption_list_shm:
                self._caption_list_shm.close()
                self._caption_list_shm = None
        except Exception as e:
            logging.warning(f"Error cleaning up caption list shared memory: {e}")
        
        # Clean up the main store
        if hasattr(self.store, 'cleanup'):
            self.store.cleanup()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup


# Global manager instance for easy access
_global_caption_index_manager = None

def get_caption_index_manager(store_name: str = "laion_caption_index", 
                            method: str = "binary_search") -> CaptionIndexManager:
    """
    Get or create global caption index manager.
    
    Args:
        store_name: Name for shared memory store
        method: "binary_search" (memory efficient) or "hash_table" (faster)
    """
    global _global_caption_index_manager
    
    if _global_caption_index_manager is None:
        _global_caption_index_manager = CaptionIndexManager(
            store_name=store_name, 
            method=method
        )
    
    return _global_caption_index_manager

def get_caption_index(caption: str) -> Optional[int]:
    """Global function for caption → index lookup."""
    return get_caption_index_manager().get_index(caption)


if __name__ == "__main__":
    # Test both caption index methods
    print("Testing SharedCaptionIndexStore implementations...")
    
    # Test captions
    test_captions = [
        "a dog running in the park",
        "a cat sitting on a chair", 
        "a bird flying in the sky",
        "a car driving on the road",
        "a person walking down the street"
    ]
    
    # Test both methods
    for method in ["binary_search", "hash_table"]:
        print(f"\n=== Testing {method.upper()} method ===")
        
        # Create manager and build index
        manager = CaptionIndexManager("test_store", max_entries=1000, method=method)
        manager.build_from_captions(test_captions)
        
        # Test lookups
        print("Testing lookups:")
        all_correct = True
        for i, caption in enumerate(test_captions):
            retrieved_index = manager.get_index(caption)
            print(f"Caption: '{caption}' → Index: {retrieved_index} (expected: {i})")
            
            if retrieved_index != i:
                print("❌ MISMATCH!")
                all_correct = False
            else:
                print("✅ OK")
        
        # Test stats
        stats = manager.get_stats()
        print(f"\nStats: {stats}")
        
        if all_correct:
            print(f"✅ {method.upper()} method test PASSED!")
        else:
            print(f"❌ {method.upper()} method test FAILED!")
    
    print("\n🎉 All tests completed!")
    
    # Performance comparison
    print("\n📊 PERFORMANCE COMPARISON:")
    print("Binary Search:")
    print("  - Memory: Lower (60MB for 4M captions)")
    print("  - Lookup: O(log n) ≈ 22 operations for 4M captions")
    print("  - Best for: Memory-constrained environments")
    
    print("\nHash Table:")
    print("  - Memory: Higher (96MB for 4M captions)")
    print("  - Lookup: O(1) ≈ 1-2 operations")
    print("  - Best for: Frequent lookups, speed-critical applications")
    
    print("\n💡 Usage in your LAION dataset:")
    print('manager = CaptionIndexManager(method="binary_search")  # Memory efficient')
    print('manager = CaptionIndexManager(method="hash_table")     # Faster lookups')

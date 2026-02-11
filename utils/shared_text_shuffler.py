"""
Shared TextShuffler singleton for efficient multi-GPU, multi-worker usage.

Problem:
--------
With 8 GPUs and N workers per GPU, creating a TextShuffler per dataset instance
would spawn 8*N spaCy models, consuming excessive memory and initialization time.

Solution:
---------
Use a module-level singleton pattern where:
1. The spaCy model is loaded exactly ONCE per process (not per dataset instance)
2. All dataset instances in the same process share the same TextShuffler
3. Fork-safe: each worker process gets its own copy after fork
4. Lazy initialization: only loaded when actually used

Usage:
------
# In your dataset __getitem__:
from utils.shared_text_shuffler import get_text_shuffler

shuffler = get_text_shuffler()  # Returns shared instance
negatives = shuffler.shuffle_nouns_and_adj(caption)

Memory savings:
--------------
Before: 8 GPUs × 8 workers × ~500MB = ~32GB spaCy models
After:  8 GPUs × 1 model × ~500MB = ~4GB spaCy models

"""

import logging
import os
from typing import Optional

# Module-level singleton
_TEXT_SHUFFLER_INSTANCE: Optional[object] = None
_SHUFFLER_PID: Optional[int] = None


def get_text_shuffler():
    """
    Get the shared TextShuffler instance for this process.
    
    This function ensures that:
    1. Only one spaCy model is loaded per process (not per dataset instance)
    2. Fork-safe: each worker gets its own instance after fork
    3. Lazy: model only loaded when first accessed
    
    Returns:
        TextShuffler instance shared across all datasets in this process
    """
    global _TEXT_SHUFFLER_INSTANCE, _SHUFFLER_PID
    
    current_pid = os.getpid()
    
    # Check if we need to create/recreate the shuffler
    # (first access or after fork into a new process)
    if _TEXT_SHUFFLER_INSTANCE is None or _SHUFFLER_PID != current_pid:
        logging.info(f"[SharedTextShuffler] Initializing TextShuffler for PID {current_pid}")
        
        # Import here to avoid loading spaCy at module import time
        from utils.perturbations import TextShuffler
        
        _TEXT_SHUFFLER_INSTANCE = TextShuffler()
        _SHUFFLER_PID = current_pid
        
        logging.info(f"[SharedTextShuffler] TextShuffler ready for PID {current_pid}")
    
    return _TEXT_SHUFFLER_INSTANCE


def reset_text_shuffler():
    """
    Reset the singleton (useful for testing or explicit cleanup).
    
    Note: In normal usage, you don't need to call this.
    The singleton will automatically handle process forks.
    """
    global _TEXT_SHUFFLER_INSTANCE, _SHUFFLER_PID
    
    _TEXT_SHUFFLER_INSTANCE = None
    _SHUFFLER_PID = None
    
    logging.info("[SharedTextShuffler] Reset singleton")


def get_shuffler_stats():
    """
    Get statistics about the current shuffler instance.
    
    Returns:
        dict with:
            - 'initialized': bool, whether shuffler exists
            - 'pid': int or None, process ID that owns the shuffler
            - 'current_pid': int, current process ID
            - 'needs_reinit': bool, whether a new instance is needed
    """
    current_pid = os.getpid()
    
    return {
        'initialized': _TEXT_SHUFFLER_INSTANCE is not None,
        'pid': _SHUFFLER_PID,
        'current_pid': current_pid,
        'needs_reinit': _TEXT_SHUFFLER_INSTANCE is None or _SHUFFLER_PID != current_pid
    }

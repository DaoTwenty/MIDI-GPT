"""
Shared utilities for MIDI-GPT: cache directory resolution and cached index files.

Cache root is controlled by the MIDIGPT_CACHE env var (default: ~/.midigpt).
Usable by both training (dataset filter indices) and inference (model files).
"""

import hashlib
import os
from pathlib import Path


def cache_dir() -> Path:
    """Return (and create) the MIDI-GPT cache root directory."""
    root = Path(os.environ.get("MIDIGPT_CACHE", "~/.midigpt")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def cached_indices(key: str) -> Path:
    """Path to a .npy file for storing a list of valid dataset indices."""
    return cache_dir() / f"valid_{key}.npy"


def file_cache_key(path: str, **extra) -> str:
    """Stable MD5 key derived from a file's path, size, and any extra kwargs.

    Deliberately does NOT include mtime: this dataset is rsync/Globus-synced
    across 5 clusters, and every sync rewrites mtime, which would silently
    invalidate the whole filter cache and force a multi-hour recompute per
    cluster on every transfer. path+size is stable across clusters (same
    canonical path, size preserved by rsync) and still changes if a shard's
    content is replaced. (If a shard were ever replaced with different content
    of the exact same byte size, delete the cache dir to force a rebuild.)
    """
    stat = Path(path).stat()
    raw = f"{path}:{stat.st_size}"
    for k, v in sorted(extra.items()):
        raw += f":{k}={v}"
    return hashlib.md5(raw.encode()).hexdigest()

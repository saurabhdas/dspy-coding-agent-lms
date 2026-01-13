"""Response caching for Claude Code LM."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached response entry.

    Attributes:
        value: The cached value.
        timestamp: When the entry was created (Unix timestamp).
        ttl: Time-to-live in seconds (None for no expiration).
    """

    value: Any
    timestamp: float
    ttl: float | None = None

    def is_expired(self) -> bool:
        """Check if the cache entry has expired.

        Returns:
            True if expired, False otherwise.
        """
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class ResponseCache:
    """Two-level response cache (memory + disk).

    Provides fast caching for Claude Code responses with both in-memory
    caching for performance and disk caching for persistence.

    Attributes:
        cache_dir: Directory for disk cache storage.
        default_ttl: Default time-to-live for cache entries.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        default_ttl: float | None = None,
        max_memory_entries: int = 100,
    ) -> None:
        """Initialize the response cache.

        Args:
            cache_dir: Directory for disk cache. Defaults to
                ~/.cache/dspy-coding-agent-lms
            default_ttl: Default TTL in seconds. None for no expiration.
            max_memory_entries: Maximum entries in memory cache.
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/dspy-coding-agent-lms")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.max_memory_entries = max_memory_entries

        # In-memory cache for performance
        self._memory_cache: dict[str, CacheEntry] = {}

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for a cache key.

        Args:
            key: The cache key (hash).

        Returns:
            Path to the cache file.
        """
        return self.cache_dir / f"{key}.json"

    def _evict_memory_if_needed(self) -> None:
        """Evict oldest entries from memory cache if over limit."""
        while len(self._memory_cache) >= self.max_memory_entries:
            # Remove oldest entry
            oldest_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k].timestamp,
            )
            del self._memory_cache[oldest_key]

    def get(self, key: str) -> Any | None:
        """Get a value from cache.

        Checks memory cache first, then disk cache.

        Args:
            key: The cache key.

        Returns:
            The cached value, or None if not found or expired.
        """
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired():
                logger.debug("Cache hit (memory): %s", key[:16])
                return entry.value
            else:
                del self._memory_cache[key]
                logger.debug("Cache expired (memory): %s", key[:16])

        # Check file cache
        path = self._get_cache_path(key)
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                entry = CacheEntry(
                    value=data["value"],
                    timestamp=data["timestamp"],
                    ttl=data.get("ttl"),
                )
                if not entry.is_expired():
                    # Populate memory cache
                    self._evict_memory_if_needed()
                    self._memory_cache[key] = entry
                    logger.debug("Cache hit (disk): %s", key[:16])
                    return entry.value
                else:
                    # Clean up expired disk entry
                    path.unlink(missing_ok=True)
                    logger.debug("Cache expired (disk): %s", key[:16])
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning("Cache read error for %s: %s", key[:16], e)
                path.unlink(missing_ok=True)

        logger.debug("Cache miss: %s", key[:16])
        return None

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set a value in cache.

        Stores in both memory and disk cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds (overrides default).
        """
        ttl = ttl if ttl is not None else self.default_ttl
        entry = CacheEntry(
            value=value,
            timestamp=time.time(),
            ttl=ttl,
        )

        # Memory cache
        self._evict_memory_if_needed()
        self._memory_cache[key] = entry

        # File cache
        path = self._get_cache_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "value": value,
                        "timestamp": entry.timestamp,
                        "ttl": ttl,
                    },
                    f,
                )
            logger.debug("Cache set: %s", key[:16])
        except (OSError, TypeError) as e:
            logger.warning("Cache write error for %s: %s", key[:16], e)

    def delete(self, key: str) -> None:
        """Delete a cache entry.

        Removes from both memory and disk cache.

        Args:
            key: The cache key.
        """
        self._memory_cache.pop(key, None)
        path = self._get_cache_path(key)
        path.unlink(missing_ok=True)
        logger.debug("Cache deleted: %s", key[:16])

    def clear(self) -> None:
        """Clear all cache entries from both memory and disk."""
        self._memory_cache.clear()
        for path in self.cache_dir.glob("*.json"):
            path.unlink(missing_ok=True)
        logger.debug("Cache cleared")

    def clear_expired(self) -> int:
        """Remove all expired entries from cache.

        Returns:
            Number of entries removed.
        """
        removed = 0

        # Clear expired memory entries
        expired_keys = [
            k for k, v in self._memory_cache.items() if v.is_expired()
        ]
        for key in expired_keys:
            del self._memory_cache[key]
            removed += 1

        # Clear expired disk entries
        for path in self.cache_dir.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                entry = CacheEntry(
                    value=data["value"],
                    timestamp=data["timestamp"],
                    ttl=data.get("ttl"),
                )
                if entry.is_expired():
                    path.unlink()
                    removed += 1
            except (json.JSONDecodeError, KeyError, OSError):
                path.unlink(missing_ok=True)
                removed += 1

        logger.debug("Cleared %d expired entries", removed)
        return removed

    def stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        disk_count = len(list(self.cache_dir.glob("*.json")))
        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": disk_count,
            "max_memory_entries": self.max_memory_entries,
        }


def compute_cache_key(
    prompt: str,
    json_schema: dict[str, Any] | None = None,
    config_dict: dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    """Compute a cache key from request parameters.

    Creates a deterministic hash from the request parameters that can
    be used as a cache key.

    Args:
        prompt: The prompt string.
        json_schema: Optional JSON schema for structured output.
        config_dict: Configuration dictionary.
        **kwargs: Additional parameters to include in the key.

    Returns:
        SHA-256 hash string as cache key.
    """
    key_data = {
        "prompt": prompt,
        "json_schema": json_schema,
        "config": config_dict,
        "kwargs": kwargs,
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()

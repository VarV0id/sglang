# SPDX-License-Identifier: Apache-2.0
"""LRU-bounded file HiCache backend.

Subclasses HiCacheFile to add an on-disk size cap with LRU (mtime) eviction.
On every set(), if the directory is approaching the cap, the oldest .bin
files are unlinked until usage drops to the low watermark. Persistence,
file layout, and read path are unchanged from the parent backend.

Config (HiCacheStorageConfig.extra_config dict):
  max_size_gb: int, default 80
      Maximum on-disk footprint in GiB. Eviction starts at 95% of this
      value; the evictor drains down to 85% to amortize.

Env override:
  SGLANG_HICACHE_LRU_MAX_SIZE_GB
"""

import logging
import os
import threading
from typing import Any, List, Optional, Tuple

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig

logger = logging.getLogger(__name__)


class LRUHiCacheFile(HiCacheFile):
    DEFAULT_MAX_SIZE_GB = 80
    HIGH_WATERMARK = 0.95
    LOW_WATERMARK = 0.85

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        file_path: str = "/tmp/hicache",
    ) -> None:
        super().__init__(storage_config, file_path)

        max_size_gb = self.DEFAULT_MAX_SIZE_GB
        if storage_config.extra_config:
            cfg_value = storage_config.extra_config.get("max_size_gb")
            if cfg_value is not None:
                try:
                    max_size_gb = int(cfg_value)
                except (TypeError, ValueError):
                    logger.warning(
                        f"LRUHiCacheFile: ignoring non-integer extra_config.max_size_gb={cfg_value!r}"
                    )
        env_value = os.environ.get("SGLANG_HICACHE_LRU_MAX_SIZE_GB")
        if env_value:
            try:
                max_size_gb = int(env_value)
            except ValueError:
                logger.warning(
                    f"LRUHiCacheFile: ignoring non-integer SGLANG_HICACHE_LRU_MAX_SIZE_GB={env_value!r}"
                )

        self.max_size_bytes = int(max_size_gb) * (1024 ** 3)
        self.high_watermark_bytes = int(self.max_size_bytes * self.HIGH_WATERMARK)
        self.low_watermark_bytes = int(self.max_size_bytes * self.LOW_WATERMARK)

        self._lock = threading.Lock()
        self._current_bytes = self._scan_existing()

        # Optional access-counter for observability (cheap; flipped via env).
        self._verbose = os.environ.get("SGLANG_HICACHE_LRU_VERBOSE") == "1"
        self._stats = {"set": 0, "set_bytes": 0, "get_hit": 0, "get_miss": 0, "evicted": 0}

        logger.info(
            "LRUHiCacheFile initialized: path=%s max_size_gb=%d current=%.2f GiB",
            self.file_path,
            max_size_gb,
            self._current_bytes / (1024 ** 3),
        )

    def _scan_existing(self) -> int:
        total = 0
        if not os.path.isdir(self.file_path):
            return total
        try:
            with os.scandir(self.file_path) as it:
                for entry in it:
                    if not entry.is_file() or not entry.name.endswith(".bin"):
                        continue
                    try:
                        total += entry.stat().st_size
                    except OSError:
                        continue
        except OSError as e:
            logger.warning("LRUHiCacheFile: scan %s failed: %s", self.file_path, e)
        return total

    def _evict_locked(self, incoming_bytes: int) -> None:
        """Drain to low watermark. Caller holds self._lock."""
        target = self.low_watermark_bytes - incoming_bytes
        if target < 0:
            target = 0
        if self._current_bytes <= target:
            return

        entries: List[Tuple[float, int, str]] = []
        try:
            with os.scandir(self.file_path) as it:
                for entry in it:
                    if not entry.is_file() or not entry.name.endswith(".bin"):
                        continue
                    try:
                        st = entry.stat()
                    except OSError:
                        continue
                    entries.append((st.st_mtime, st.st_size, entry.path))
        except OSError as e:
            logger.warning("LRUHiCacheFile: scandir during evict failed: %s", e)
            return

        entries.sort(key=lambda t: t[0])  # oldest first

        freed = 0
        evicted = 0
        for _mtime, size, path in entries:
            if self._current_bytes <= target:
                break
            try:
                os.unlink(path)
            except FileNotFoundError:
                continue
            except OSError as e:
                logger.warning("LRUHiCacheFile: unlink %s failed: %s", path, e)
                continue
            self._current_bytes -= size
            freed += size
            evicted += 1

        if evicted:
            logger.info(
                "LRUHiCacheFile: evicted %d pages (%.2f GiB), current=%.2f GiB",
                evicted,
                freed / (1024 ** 3),
                self._current_bytes / (1024 ** 3),
            )

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        # Short-circuit on existing key — match parent behavior, avoid double-counting.
        if self.exists(key):
            return True

        incoming = 0
        if value is not None:
            try:
                incoming = int(value.numel()) * int(value.element_size())
            except AttributeError:
                incoming = 0  # unknown size; skip eviction guard

        if incoming > 0:
            with self._lock:
                if self._current_bytes + incoming > self.high_watermark_bytes:
                    self._evict_locked(incoming)

        ok = super().set(key, value, target_location, target_sizes)

        if ok and incoming > 0:
            with self._lock:
                self._current_bytes += incoming
                self._stats["set"] += 1
                self._stats["set_bytes"] += incoming
                if self._verbose and self._stats["set"] % 200 == 0:
                    logger.info(
                        "LRUHiCacheFile stats: set=%d (%.2f GiB written), get_hit=%d, get_miss=%d, evicted=%d, current=%.2f GiB",
                        self._stats["set"],
                        self._stats["set_bytes"] / (1024 ** 3),
                        self._stats["get_hit"],
                        self._stats["get_miss"],
                        self._stats["evicted"],
                        self._current_bytes / (1024 ** 3),
                    )
        return ok

    def get(
        self,
        key: str,
        target_location,
        target_sizes=None,
    ):
        result = super().get(key, target_location, target_sizes)
        # Bookkeeping only. Parent already logs on miss at WARNING; we count both
        # so a periodic stats line gives a clear hit/miss ratio.
        with self._lock:
            if result is None:
                self._stats["get_miss"] += 1
            else:
                self._stats["get_hit"] += 1
            total = self._stats["get_hit"] + self._stats["get_miss"]
            if self._verbose and total > 0 and total % 200 == 0:
                logger.info(
                    "LRUHiCacheFile reads: hit=%d miss=%d hit_rate=%.1f%% (current=%.2f GiB)",
                    self._stats["get_hit"],
                    self._stats["get_miss"],
                    100.0 * self._stats["get_hit"] / total,
                    self._current_bytes / (1024 ** 3),
                )
        return result

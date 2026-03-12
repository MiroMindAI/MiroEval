import json
import os
import threading
import logging

logger = logging.getLogger(__name__)


class FileCache:
    """Thread-safe file-backed JSON cache."""

    def __init__(self, cache_dir: str, cache_name: str):
        self._cache_dir = cache_dir
        self._cache_name = cache_name
        self._file_path = os.path.join(cache_dir, f"{cache_name}_cache.json")
        self._lock = threading.Lock()
        self._data = {}
        os.makedirs(cache_dir, exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info(f"Cache '{self._cache_name}' loaded: {len(self._data)} entries")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache '{self._cache_name}': {e}")
                self._data = {}

    def _save(self):
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get(self, key: str):
        with self._lock:
            return self._data.get(key)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def set(self, key: str, value):
        with self._lock:
            self._data[key] = value
            self._save()

    def batch_set(self, items: dict):
        with self._lock:
            self._data.update(items)
            self._save()

    def clear(self):
        with self._lock:
            self._data = {}
            self._save()

    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def keys(self) -> list:
        with self._lock:
            return list(self._data.keys())

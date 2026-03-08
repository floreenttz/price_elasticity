"""Local filesystem storage implementation."""

import os
import pickle
from glob import glob as fs_glob
from typing import Any

import pandas as pd


class LocalStorage:
    """Storage backend for local filesystem."""

    def __init__(self, base_path: str = "."):
        """
        Initialize local storage.

        Args:
            base_path: Base directory for relative paths.
        """
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to base_path."""
        # Strip s3:// prefix if present (allows reusing S3 paths for local testing)
        if path.startswith("s3://"):
            # Extract path after bucket name
            parts = path[5:].split("/", 1)
            path = parts[1] if len(parts) > 1 else ""

        if os.path.isabs(path):
            return path
        return os.path.join(self.base_path, path)

    def read_parquet(self, path: str) -> pd.DataFrame:
        """Read a parquet file from local filesystem."""
        full_path = self._resolve_path(path)
        return pd.read_parquet(full_path)

    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        """Write a DataFrame to parquet on local filesystem."""
        full_path = self._resolve_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        df.to_parquet(full_path, index=False)

    def read_pickle(self, path: str) -> Any:
        """Read a pickle file from local filesystem."""
        full_path = self._resolve_path(path)
        with open(full_path, "rb") as f:
            return pickle.load(f)

    def write_pickle(self, obj: Any, path: str) -> None:
        """Write an object to pickle on local filesystem."""
        full_path = self._resolve_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            pickle.dump(obj, f)

    def glob(self, pattern: str) -> list[str]:
        """Find files matching a glob pattern on local filesystem."""
        full_pattern = self._resolve_path(pattern)
        return fs_glob(full_pattern, recursive=True)

    def exists(self, path: str) -> bool:
        """Check if a path exists on local filesystem."""
        full_path = self._resolve_path(path)
        return os.path.exists(full_path)

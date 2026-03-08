"""Storage protocol for abstracting file I/O."""

from typing import Protocol, Any
import pandas as pd


class Storage(Protocol):
    """Protocol for storage backends (S3, local filesystem, etc.)."""

    def read_parquet(self, path: str) -> pd.DataFrame:
        """Read a parquet file from storage."""
        ...

    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        """Write a DataFrame to parquet in storage."""
        ...

    def read_pickle(self, path: str) -> Any:
        """Read a pickle file from storage."""
        ...

    def write_pickle(self, obj: Any, path: str) -> None:
        """Write an object to pickle in storage."""
        ...

    def glob(self, pattern: str) -> list[str]:
        """Find files matching a glob pattern."""
        ...

    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        ...

"""Client adapter protocol for abstracting client-specific logic."""

from typing import Protocol, Any
import pandas as pd

from ..storage.base import Storage


class ClientAdapter(Protocol):
    """
    Protocol defining the interface for client-specific adapters.

    Each client (Hoogvliet, Spar, etc.) implements this interface to handle
    their specific data loading, path construction, and processing needs.
    """

    @property
    def name(self) -> str:
        """Client identifier (e.g., 'hoogvliet', 'spar')."""
        ...

    @property
    def competitors(self) -> list[str]:
        """List of competitor identifiers for price distance features."""
        ...

    @property
    def country(self) -> str:
        """Country code (e.g., 'NL', 'BE')."""
        ...

    @property
    def category_column(self) -> str:
        """Column name for product category grouping."""
        ...

    @property
    def price_column(self) -> str:
        """Column name for product price."""
        ...

    def load_raw_data(
        self,
        config: dict,
        storage: Storage,
        subset: bool = False,
        chunk_size: int = 10,
    ) -> pd.DataFrame:
        """
        Load raw sales data for this client.

        Args:
            config: Configuration dictionary with paths and settings.
            storage: Storage backend for reading files.
            subset: If True, load a random 50% sample (for testing)
            chunk_size: Number of files to process per chunk (where applicable).

        Returns:
            Raw sales DataFrame.
        """
        ...

    def load_external_data(self, config: dict, storage: Storage) -> dict[str, Any]:
        """
        Load any additional data files specific to this client.

        Args:
            config: Configuration dictionary.
            storage: Storage backend.

        Returns:
            Dictionary of additional data (e.g., {'names': df, 'cpi_categories': df}).
            Empty dict if no external data needed.
        """
        ...

    def get_artifact_path(self, artifact: str, config: dict, frequency: str = "daily") -> str:
        """
        Get the storage path for a pipeline artifact.

        Args:
            artifact: Artifact identifier (e.g., 'preprocessed_data', 'feature_data',
                     'estimations', 'elasticities', 'price_grid', 'model').
            config: Configuration dictionary with base paths.
            frequency: 'daily' or 'weekly'

        Returns:
            Full storage path for the artifact.
        """
        ...

    def get_cpi_data(self, config: dict, storage: Storage) -> pd.DataFrame | None:
        """
        Load CPI data appropriate for this client.

        Args:
            config: Configuration dictionary.
            storage: Storage backend.

        Returns:
            CPI DataFrame (may be global or category-level depending on client),
            or None if CPI is not used.
        """
        ...

    def get_store_columns_to_ffill(self) -> list[str]:
        """
        Get columns that need forward-fill imputation.

        Returns:
            List of column names (empty if none).
        """
        ...

    def get_columns_to_drop(self) -> list[str]:
        """
        Get columns to drop during preprocessing.

        Returns:
            List of column names to drop (empty if none).
        """
        ...

    def add_output_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add client-specific metadata columns to output DataFrames.

        Args:
            df: Output DataFrame (elasticities, etc.)

        Returns:
            DataFrame with any additional metadata columns.
        """
        ...

    def get_category_cpi_mapper(self) -> dict[str, str] | None:
        """
        Get mapping from product categories to CPI categories.

        Returns:
            Dictionary mapping category names to CPI category names,
            or None if not applicable.
        """
        ...

"""Hoogvliet client adapter."""

import gc
import os
from typing import Any

import pandas as pd

from ..storage.base import Storage


class HoogvlietAdapter:
    """
    Adapter for Hoogvliet client.

    Hoogvliet data characteristics:
    - Partitioned parquet files (loaded in chunks for memory management)
    - Global CPI only (no category-level CPI)
    - No priceline concept
    - No store metrics columns
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "hoogvliet"

    @property
    def country(self) -> str:
        return "NL"

    @property
    def competitors(self) -> list[str]:
        return ["ah", "lidl", "dirck_iii", "vomar", "gall_gall", "jumbo", "dirk"]

    @property
    def category_column(self) -> str:
        return "category_name_level3"

    @property
    def price_column(self) -> str:
        return "product_selling_price"

    def load_raw_data(
        self,
        config: dict,
        storage: Storage,
        subset: bool = False,
        chunk_size: int = 10,
    ) -> pd.DataFrame:
        """
        Load raw sales data from partitioned parquet files.

        Hoogvliet stores data in partitioned folders (e.g., year_week=2023_01/*.parquet).
        Files are loaded in chunks to manage memory.

        Args:
            config: Configuration dictionary.
            storage: Storage backend.
            subset: If True, load only 10% of files (for testing).
            chunk_size: Number of files to process per chunk.

        Returns:
            Combined DataFrame from all parquet files.
        """
        s3_config = config["s3_dir"]
        bucket = s3_config["bucket"].rstrip("/")
        data_prefix = s3_config["data_prefix"]

        # Build pattern for year_week partition files only
        # (exclude processed files like feature_data.parquet, preprocessed_data.parquet)
        pattern = f"{bucket}/{data_prefix}year_week=*/*.parquet"

        # Get all parquet files
        all_files = storage.glob(f"s3://{pattern}")

        if not all_files:
            raise ValueError(f"No parquet files found matching {pattern}")

        # Optionally subset for testing
        if subset:
            import numpy as np
            rng = np.random.default_rng(42)
            k = max(1, int(len(all_files) * 0.1))
            all_files = rng.choice(all_files, size=k, replace=False).tolist()

        # Load in chunks to manage memory
        chunks = []
        for i in range(0, len(all_files), chunk_size):
            chunk_files = all_files[i : i + chunk_size]
            chunk_dfs = [storage.read_parquet(f) for f in chunk_files]
            chunk_df = pd.concat(chunk_dfs, ignore_index=True)
            chunks.append(chunk_df)

            # Clear memory
            del chunk_dfs
            gc.collect()

        # Combine all chunks
        data = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

        return data

    def load_external_data(self, config: dict, storage: Storage) -> dict[str, Any]:
        """Hoogvliet doesn't need external data files."""
        return {}

    def get_artifact_path(self, artifact: str, config: dict, frequency: str = "daily") -> str:
        """
        Get storage path for an artifact.

        Args:
            artifact: One of 'preprocessed_data', 'feature_data', 'estimations',
                     'elasticities', 'price_grid', 'model'.
            config: Configuration dictionary.
            frequency: 'daily' or 'weekly'.

        Returns:
            Full S3 path for the artifact.
        """
        s3_config = config["s3_dir"]
        bucket = s3_config["bucket"].rstrip("/")
        data_prefix = s3_config["data_prefix"]
        output_prefix = s3_config.get("output_prefix", data_prefix)

        artifact_map = {
            "preprocessed_data": s3_config.get(
                f"preprocessed_data{'_weekly' if frequency == 'weekly' else ''}",
                f"preprocessed_data{'_weekly' if frequency == 'weekly' else ''}.parquet",
            ),
            "feature_data": s3_config.get(
                f"feature_data{'_weekly' if frequency == 'weekly' else ''}",
                f"feature_data{'_weekly' if frequency == 'weekly' else ''}.parquet",
            ),
            "estimations": s3_config.get(
                f"estimations{'_weekly' if frequency == 'weekly' else ''}",
                f"estimations{'_weekly' if frequency == 'weekly' else ''}.parquet",
            ),
            "elasticities": s3_config.get(
                f"elasticities{'_weekly' if frequency == 'weekly' else ''}",
                f"elasticities{'_weekly' if frequency == 'weekly' else ''}.parquet",
            ),
            "price_grid": s3_config.get("price_grid", "price_grid.pkl"),
            "model": s3_config.get("model", "trained_model.joblib"),
        }

        filename = artifact_map.get(artifact)
        if filename is None:
            raise ValueError(f"Unknown artifact: {artifact}")

        # Use output_prefix for results, data_prefix for intermediate files
        if artifact in ("elasticities",):
            return f"s3://{bucket}/{output_prefix}{filename}"
        elif artifact in ("model",):
            model_prefix = s3_config.get("model_prefix", data_prefix)
            return f"s3://{bucket}/{model_prefix}{filename}"
        else:
            return f"s3://{bucket}/{data_prefix}{filename}"

    def get_cpi_data(self, config: dict, storage: Storage) -> pd.DataFrame | None:
        """
        Load global CPI data.

        Hoogvliet uses only global CPI (not category-specific).
        """
        if not config.get("filters", {}).get("cpi", False):
            return None

        cpi_path = config.get("data_folder", "") + "cpi_alle_bestedingen.parquet"
        return storage.read_parquet(cpi_path)

    def get_store_columns_to_ffill(self) -> list[str]:
        """Hoogvliet doesn't have store metric columns."""
        return []

    def get_columns_to_drop(self) -> list[str]:
        """Hoogvliet doesn't have extra price columns to drop."""
        return []

    def add_output_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hoogvliet doesn't add extra metadata."""
        return df

    def get_category_cpi_mapper(self) -> dict[str, str] | None:
        """Hoogvliet doesn't use category-specific CPI."""
        return None

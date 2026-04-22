"""IJsvogel client adapter."""

import gc
from typing import Any

import numpy as np
import pandas as pd

from ..storage.base import Storage

class IJsvogel:
    """
    Adapter for Ijsvogel client.

    IJsvogel data characteristics:
    - Single parquet file (product x channel x date)
    - Channel concept: store_priceline_code (FRM, WEB, PPB, BOL, BEL)
    - Competitor prices from Google Shopping (aggregated: min, max, avg, median)
    - Global CPI only (no category-level CPI)
    - category_level_3 for category grouping
    """

    def __init__(self, channel: str | None = None):
        """
        Initialize IJsvogel adapter.

        Args:
            channel: Channel name ('FRM', 'WEB', 'PPB', 'BOL', 'BEL').
                     If None, all channels are loaded
        """
        self.channel = channel

    @property
    def name(self) -> str:
        return "ijsvogel"

    @property
    def country(self) -> str:
        return "NL"

    @property
    def competitors(self) -> list[str]:
        # Google Shopping prices aggregated into 4 features.
        # features.py will look for {competitor}_prices columns, so these map to:
        # google_shopping_min_price, google_shopping_max_price, etc.
        return ['zooplus', 'intratuin', 'welkoop', 'ranzijn', 'plein', 'brekz', 'hornbach', 'medpets', 'maxizoo', 'google_shopping_median']

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
        Load raw sales data from a single parquet file, filtered to this channel.

        Args:
        config: Configuration dictionary.
        storage: Storage backend.
        subset: If True, load a random 50% sample (for testing).
        chunk_size: Ignored for IJsvogel (single file).

        Returns:
        Sales DataFrame filtered to the configured channel.
        """
        s3_config = config["s3_dir"]
        bucket = s3_config["bucket"].rstrip("/")
        data_prefix = s3_config["data_prefix"]

        file_path = f"s3://{bucket}/{data_prefix}"
        data = storage.read_parquet(file_path)

        # Filter to the configured channel (skip if channel is None -> all channels)
        if self.channel is not None:
            data = data[data["store_priceline_code"] == self.channel].copy()

        if subset:
            rng = np.random.default_rng(42)
            k = max(1, int(len(data) * 0.50))
            data = data.iloc[rng.choice(len(data), size=k, replace=False)].reset_index(drop=True)

        return data

    def load_external_data(self, config: dict, storage: Storage) -> dict[str, Any]:
        """IJsvogel doesn't need external data files."""
        return {}

    def get_artifact_path(self, artifact: str, config: dict, frequency: str = "daily") -> str:
        """
        Get storage path for an artifact, including channel prefix.

        Args:
            artifact: One of 'preprocessed_data', 'feature_data', 'estimations', 'elasticities', 'price_grid', 'model'.
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

        # Add channel prefix to filename 
        filename = f"{self.channel}_{filename}"

        if artifact in ("elasticities",):
            return f"s3://{bucket}/{output_prefix}{filename}"
        elif artifact in ("model",):
            model_prefix = s3_config.get("model_prefix", output_prefix)
            return f"s3://{bucket}/{model_prefix}{filename}"
        else:
            artifacts_prefix = s3_config.get("artifacts_prefix", output_prefix)
            return f"s3://{bucket}/{artifacts_prefix}{filename}"

    def get_cpi_data(self, config: dict, storage: Storage) -> pd.DataFrame | None:
        """Load global CPI data. IJsvogel uses only global CPI."""
        if not config.get("filters", {}).get("cpi", False):
            return None

        cpi_path = config.get("data_folder", "") + "cpi_alle_bestedingen.parquet"
        return storage.read_parquet(cpi_path)

    def get_store_columns_to_ffill(self) -> list[str]:
        """Return columns that need forward-fill imputation."""
        return ["number_of_pricelines_per_week"]

    def get_columns_to_drop(self) -> list[str]:
        """IJsvogel doesn't have extra price columns to drop."""
        return []

    def add_output_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add channel column to output DataFrames."""
        df = df.copy()
        if self.channel is not None:
            df["channel"] = self.channel
        return df

    def get_category_cpi_mapper(self) -> dict[str, str] | None:
        """IJsvogel doesn't use category-specific CPI."""
        return None




        
"""Spar client adapter."""

import os
from typing import Any

import pandas as pd

from ..storage.base import Storage


# CPI category file names for Spar
SPAR_CPI_CATEGORIES = [
    "voedingsmiddelen",
    "brood_en_granen",
    "vlees",
    "vis_en_schaal_en_schelpdieren",
    "melk_kaas_en_eieren",
    "olien_en_vetten",
    "fruit",
    "groenten",
    "suiker_zoetwaren_en_ijs",
    "voedingsmiddelen_neg",
    "alcoholvrije_dranken",
    "koffie_thee_en_cacao",
    "mineraalwater_frisdr_en_sappen",
    "alcoholhoudende_dranken",
    "gedistilleerde_dranken",
    "wijn",
    "bier",
    "tabak",
    "medische_producten_apparaten",
    "persoonlijke_verzorging",
    "producten_voor_huisdieren",
    "kranten_boeken_en_schrijfwaren",
    "dagelijks_onderhoud_van_de_woning",
]


class SparAdapter:
    """
    Adapter for Spar client.

    Spar data characteristics:
    - Single aggregated parquet file per priceline
    - Category-level CPI (23 category files)
    - Priceline concept (Enjoy, City, Highway, Buurt, etc.)
    - Store metrics columns (need forward-fill)
    - Multiple price column variants
    """

    def __init__(self, priceline: str):
        """
        Initialize Spar adapter.

        Args:
            priceline: Priceline name (e.g., 'Enjoy', 'City', 'Highway', 'Buurt').
        """
        self.priceline = priceline

    @property
    def name(self) -> str:
        return "spar"

    @property
    def country(self) -> str:
        return "NL"

    @property
    def competitors(self) -> list[str]:
        return ["ah", "plus"]

    @property
    def category_column(self) -> str:
        return "category_name_level2"

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
        Load raw sales data from a single aggregated parquet file.

        Spar stores data in a single file per priceline: {priceline}_aggregated.parquet

        Args:
            config: Configuration dictionary.
            storage: Storage backend.
            subset: Ignored for Spar (single file).
            chunk_size: Ignored for Spar (single file).

        Returns:
            Sales DataFrame.
        """
        s3_config = config["s3_dir"]
        bucket = s3_config["bucket"].rstrip("/")
        data_prefix = s3_config["data_prefix"]

        # Build the path: {bucket}/{data_prefix}/{priceline}_aggregated.parquet
        file_path = f"s3://{bucket}/{data_prefix}{self.priceline}_aggregated.parquet"

        return storage.read_parquet(file_path)

    def load_external_data(self, config: dict, storage: Storage) -> dict[str, Any]:
        """
        Load additional data files for Spar.

        Spar needs:
        - Product names data (pe_names.parquet)
        - Category CPI files (23 files)

        Args:
            config: Configuration dictionary.
            storage: Storage backend.

        Returns:
            Dictionary with:
            - 'names': Product names DataFrame
            - 'cpi_categories': Combined category CPI DataFrame
        """
        s3_config = config["s3_dir"]
        bucket = s3_config["bucket"].rstrip("/")
        data_prefix = s3_config["data_prefix"]
        data_folder = config.get("data_folder", "")

        result = {}

        # Load product names
        names_path = f"s3://{bucket}/{data_prefix}pe_names.parquet"
        if storage.exists(names_path):
            result["names"] = storage.read_parquet(names_path)

        # Load category CPI files
        if config.get("filters", {}).get("cpi", False):
            cpi_dfs = []
            for category_name in SPAR_CPI_CATEGORIES:
                cpi_path = f"{data_folder}{category_name}.parquet"
                try:
                    cpi_df = storage.read_parquet(cpi_path)
                    cpi_dfs.append(cpi_df)
                except Exception:
                    # Skip missing CPI files
                    pass

            if cpi_dfs:
                result["cpi_categories"] = pd.concat(cpi_dfs, ignore_index=True)

        return result

    def get_artifact_path(self, artifact: str, config: dict, frequency: str = "daily") -> str:
        """
        Get storage path for an artifact, including priceline prefix.

        Args:
            artifact: Artifact identifier.
            config: Configuration dictionary.
            frequency: 'daily' or 'weekly'.

        Returns:
            Full S3 path with priceline prefix.
        """
        s3_config = config["s3_dir"]
        bucket = s3_config["bucket"].rstrip("/")
        data_prefix = s3_config["data_prefix"]
        output_prefix = s3_config.get("output_prefix", data_prefix)

        # Get base filename from config
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

        base_filename = artifact_map.get(artifact)
        if base_filename is None:
            raise ValueError(f"Unknown artifact: {artifact}")

        # Add priceline prefix to filename
        filename = f"{self.priceline}_{base_filename}"

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

        Note: Category-level CPI is loaded via load_external_data().
        This returns global CPI for baseline calculations.
        """
        if not config.get("filters", {}).get("cpi", False):
            return None

        cpi_path = config.get("data_folder", "") + "cpi_alle_bestedingen.parquet"
        return storage.read_parquet(cpi_path)

    def get_store_columns_to_ffill(self) -> list[str]:
        """Return store metric columns that need forward-fill."""
        return ["unique_stores_last_7_days", "unique_stores_last_30_days"]

    def get_columns_to_drop(self) -> list[str]:
        """Return price columns to drop when using product_selling_price."""
        return ["calculated_price", "calculated_price_after_discount"]

    def add_output_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add priceline column to output DataFrames."""
        df = df.copy()
        df["priceline"] = self.priceline
        return df

    def get_category_cpi_mapper(self) -> dict[str, str] | None:
        """
        Return mapping from product categories to CPI categories.

        This should be loaded from cat_mapper.json but for now
        returns None (to be implemented with actual mapping).
        """
        # TODO: Load from cat_mapper.json
        return None

    def merge_product_names(self, data: pd.DataFrame, names_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge product category names into sales data.

        Spar sales data needs category names joined from pe_names.parquet.

        Args:
            data: Sales DataFrame.
            names_df: Product names DataFrame.

        Returns:
            Merged DataFrame with category columns.
        """
        return pd.merge(
            data,
            names_df[
                [
                    "product_code",
                    "product_category_level1",
                    "product_category_level2",
                    "product_category_level3",
                ]
            ],
            on="product_code",
            how="left",
        )

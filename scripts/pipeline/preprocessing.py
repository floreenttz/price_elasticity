"""
Refactored DataPreprocessor using client adapters.

This module provides a client-agnostic data preprocessing pipeline.
Client-specific logic is handled by ClientAdapter implementations.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from ..clients.base import ClientAdapter
from ..storage.base import Storage
from ..functions import rename_columns, remove_outliers, interpolate_cpi


def get_logger(name: str = "preprocessing") -> logging.Logger:
    """Create a logger for preprocessing."""
    os.makedirs("logs", exist_ok=True)
    current_time = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
    log_file = os.path.join("logs", f"{name}_{current_time}.log")

    logger = logging.getLogger(f"{name}_{current_time}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


class DataPreprocessor:
    """
    Client-agnostic data preprocessor.

    Uses a ClientAdapter to handle all client-specific logic (data loading,
    path construction, CPI handling, etc.). The preprocessing pipeline itself
    is the same for all clients.
    """

    def __init__(
        self,
        client: ClientAdapter,
        config: dict,
        storage: Storage,
        logger: logging.Logger | None = None,
        frequency: str = "daily",
    ):
        """
        Initialize the preprocessor.

        Note: Does NOT run the pipeline. Call run() to execute.

        Args:
            client: Client adapter for client-specific logic.
            config: Configuration dictionary.
            storage: Storage backend for I/O.
            logger: Optional logger instance.
            frequency: 'daily' or 'weekly'.
        """
        self.client = client
        self.config = config
        self.storage = storage
        self.logger = logger or get_logger()
        self.frequency = frequency

        # Extract config values
        self.filters = config.get("filters", {})
        self.price_column = config.get("price_column", "product_selling_price")
        self.target = config.get("target", "quantity_sold")
        self.category = config.get("category", client.category_column)
        self.price_grid_freq = config.get("price_grid_freq", 7)
        self.min_grid_perc = config.get("min_grid_perc", -10)
        self.max_grid_perc = config.get("max_grid_perc", 10)

        # Data containers
        self.data: pd.DataFrame | None = None
        self.external_data: dict[str, Any] = {}
        self.price_grid: dict[str, list] = {}

    def run(self, subset: bool = False, overview: bool = True) -> pd.DataFrame:
        """
        Execute the full preprocessing pipeline.

        Args:
            subset: If True, load only a subset of data (for testing).
            overview: If True, generate overview report.

        Returns:
            Preprocessed DataFrame.
        """
        self.logger.info(f"Starting preprocessing for client: {self.client.name}")

        # Load data
        self._load_data(subset)
        self._load_external_data()

        # Preprocessing steps
        self._filter_columns()
        self._convert_columns()
        self._deduplicate_columns()

        if self.filters.get("filter_product_status", False):
            self._filter_product_status()
        if self.filters.get("filter_live_products", False):
            self._filter_live_products()
        if self.filters.get("filter_sufficient_price_levels", False):
            self._filter_sufficient_price_levels()

        self._remove_missing_values()
        self._remove_outliers()

        if self.filters.get("filter_low_selling_products", False):
            self._filter_low_selling_products()

        self._further_preprocessing()

        if self.filters.get("cpi", False):
            self._calculate_cpi()

        self.data = self.data.reset_index(drop=True)
        self._drop_columns()

        if self.frequency == "weekly":
            self._resample_data()

        if overview:
            self._generate_overview_report()

        self._create_price_grid()

        # Round all float columns to 2 decimal places
        float_cols = self.data.select_dtypes(include=["float"]).columns
        self.data[float_cols] = self.data[float_cols].round(2)

        self.logger.info("Preprocessing complete!")
        return self.data

    def save(self) -> None:
        """Save preprocessed data and price grid to storage."""
        # Save preprocessed data
        data_path = self.client.get_artifact_path(
            "preprocessed_data", self.config, self.frequency
        )
        self.storage.write_parquet(self.data, data_path)
        self.logger.info(f"Saved preprocessed data to {data_path}")

        # Save price grid
        grid_path = self.client.get_artifact_path("price_grid", self.config)
        self.storage.write_pickle(self.price_grid, grid_path)
        self.logger.info(f"Saved price grid to {grid_path}")

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def _load_data(self, subset: bool = False) -> None:
        """Load raw sales data using client adapter."""
        self.logger.info("Loading raw data...")
        self.data = self.client.load_raw_data(self.config, self.storage, subset=subset)
        self.logger.info(f"Loaded {len(self.data):,} rows")

    def _load_external_data(self) -> None:
        """Load external data using client adapter."""
        self.logger.info("Loading external data...")
        self.external_data = self.client.load_external_data(self.config, self.storage)

        # If client provides product names, merge them
        if "names" in self.external_data and hasattr(self.client, "merge_product_names"):
            self.data = self.client.merge_product_names(
                self.data, self.external_data["names"]
            )
            self.logger.info("Merged product names")

    # -------------------------------------------------------------------------
    # Filtering Steps
    # -------------------------------------------------------------------------

    def _filter_columns(self) -> None:
        """Filter columns (matches original data_preprocessing.py)."""
        self.logger.info("Filtering columns...")

        # Determine revenue column
        if 'revenue_after' in self.data.columns:
            self.revenue = 'revenue_after'
        else:
            self.revenue = 'revenue_before'

        # Rename columns to standard names
        self.data = rename_columns(self.data)

        # Calculate price from revenue if price column doesn't exist
        if self.price_column not in self.data.columns:
            self.data[self.price_column] = (
                self.data[self.revenue] / self.data["quantity_sold"]
            )

        self.logger.info(f"Unique products: {self.data['product_code'].nunique()}")
        self.logger.info("Columns filtered successfully.")

    def _convert_columns(self) -> None:
        """Convert column types (matches original data_preprocessing.py)."""
        self.logger.info("Columns conversion started.")

        # Convert date with specific formats
        try:
            self.data['date'] = pd.to_datetime(self.data['date'], format="%Y%m%d")
        except ValueError:
            try:
                self.data['date'] = pd.to_datetime(self.data['date'], format="%Y-%m-%d")
            except ValueError:
                raise ValueError("Date format not recognized. Please specify the correct format.")

        # Convert product_code to string
        self.data['product_code'] = self.data['product_code'].astype(str)
        self.logger.info("Columns conversion completed successfully.")

    def _deduplicate_columns(self) -> None:
        """Ensure all column names are unique by adding suffixes to duplicates."""
        cols = self.data.columns.tolist()
        if len(cols) != len(set(cols)):
            self.logger.warning(f"Found {len(cols) - len(set(cols))} duplicate columns")

            # Make columns unique manually (matches original)
            new_cols = []
            seen = {}
            for col in cols:
                if col not in seen:
                    seen[col] = 1
                    new_cols.append(col)
                else:
                    seen[col] += 1
                    new_cols.append(f"{col}_{seen[col]}")

            self.data.columns = new_cols
            self.logger.info(f"Fixed duplicate columns. New columns: {new_cols[:10]}...")

    def _filter_product_status(self) -> None:
        """Filter by product status if column exists."""
        self.logger.info("Filtering by product status...")
        if "product_status" in self.data.columns:
            before = len(self.data)
            self.data = self.data[self.data["product_status"] == "Active"]
            self.logger.info(f"Filtered {before - len(self.data)} inactive products")

    def _filter_live_products(self) -> None:
        """Keep only products with recent sales."""
        self.logger.info("Filtering live products...")
        before = self.data["product_code"].nunique()

        max_date = self.data["date"].max()
        cutoff = max_date - pd.Timedelta(days=self.price_grid_freq)
        recent_products = self.data[self.data["date"] >= cutoff]["product_code"].unique()
        self.data = self.data[self.data["product_code"].isin(recent_products)]

        after = self.data["product_code"].nunique()
        self.logger.info(f"Filtered {before - after} inactive products")

    def _filter_sufficient_price_levels(self) -> None:
        """Keep products with enough price variation."""
        self.logger.info("Filtering by price levels...")
        min_levels = self.filters.get("sufficient_price_levels", 5)
        before = self.data["product_code"].nunique()

        price_levels = self.data.groupby("product_code")[self.price_column].nunique()
        valid_products = price_levels[price_levels >= min_levels].index
        self.data = self.data[self.data["product_code"].isin(valid_products)]

        after = self.data["product_code"].nunique()
        self.logger.info(f"Filtered {before - after} products with insufficient price levels")

    def _filter_low_selling_products(self) -> None:
        """Filter products with low median sales."""
        self.logger.info("Filtering low selling products...")
        min_sales = self.filters.get("min_sales", 1)
        before = self.data["product_code"].nunique()

        median_sales = self.data.groupby("product_code")[self.target].median()
        valid_products = median_sales[median_sales >= min_sales].index
        self.data = self.data[self.data["product_code"].isin(valid_products)]

        after = self.data["product_code"].nunique()
        self.logger.info(f"Filtered {before - after} low selling products")

    # -------------------------------------------------------------------------
    # Cleaning Steps
    # -------------------------------------------------------------------------

    def _remove_missing_values(self) -> None:
        """Remove rows with missing category values (matches original)."""
        self.logger.info("Removing missing values in category columns...")
        before = len(self.data)

        # Remove rows where category columns have nulls (matches original)
        category_columns = [col for col in self.data.columns if 'category' in col]
        self.data = self.data.dropna(subset=category_columns)
        self.data.sort_values(by=["product_code", "date"], inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        after = len(self.data)
        self.logger.info(f"Removed {before - after} rows with missing values")
        self.logger.info(f"Unique products: {self.data['product_code'].nunique()}")

    def _remove_outliers(self) -> None:
        """Remove outliers from price column."""
        self.logger.info(f"Removing outliers in {self.price_column}...")
        before = len(self.data)

        # Use IQR method per product on price column (matches original)
        self.data = self.data.groupby("product_code").apply(
            lambda g: remove_outliers(g, self.price_column)
        ).reset_index(drop=True)
        self.data.sort_values(by=["product_code", "date"], inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        after = len(self.data)
        self.logger.info(f"Removed {before - after} outlier rows")

    def _further_preprocessing(self) -> None:
        """
        Apply additional preprocessing based on competitor info.
        Matches original data_preprocessing.py behavior.
        """
        if self.config.get("competitor_info", False):
            self.logger.info("Updating competitor prices...")
            competitors = self.config.get("competitors", [])
            for competitor in competitors:
                try:
                    content_factor_col = f"{competitor}_content_factor"
                    price_col = f"{competitor}_price"

                    if price_col in self.data.columns and content_factor_col in self.data.columns:
                        self.data[price_col] = (
                            self.data[price_col] * self.data[content_factor_col]
                        )

                        # Calculate null percentage
                        null_percentage = self.data[price_col].isnull().sum() / len(self.data) * 100

                        if null_percentage > 59:
                            print(f"{price_col} column has {int(round(null_percentage))}% nulls and it will be removed")
                            self.data.drop(columns=[price_col, content_factor_col], inplace=True)

                except Exception as e:
                    self.logger.error(f"Error updating {competitor} prices: {e}")

            self.logger.info("Competitor prices updated.")
        else:
            self.logger.info("Data doesn't include competitor prices.")

    # -------------------------------------------------------------------------
    # CPI Calculation
    # -------------------------------------------------------------------------

    def _calculate_cpi(self) -> None:
        """Merge daily CPI into data and filter to dates where CPI is available."""
        self.logger.info("Calculating CPI adjustments...")

        cpi_data = self.client.get_cpi_data(self.config, self.storage)
        if cpi_data is None:
            self.logger.info("CPI data not available, skipping")
            return
            
        # Interpolate monthly CPI to daily granularity
        cpi_daily = interpolate_cpi(cpi_data)

        # Filter data to only dates covered by CPI
        max_cpi_date = cpi_daily['date'].max()
        before = len(self.data)
        self.data = self.data[self.data['date'] <= max_cpi_date]
        self.logger.info(
            f"Filtered to CPI coverage (up to {max_cpi_date.date()}): "     
            f"removed {before - len(self.data)} rows"
        )

        # Merge CPI into data
        self.data = self.data.merge(cpi_daily, on='date', how='left')
        self.logger.info("CPI merged into data")

        # Create CPI-adjusted price using base year CPI (average CPI of base year)
        base_year = self.config.get("cpi_base_year", 2023)
        base_year_mask = cpi_daily['date'].dt.year == base_year
        base_cpi = cpi_daily.loc[base_year_mask, 'cpi'].mean()
        self.data['adjusted_price'] = self.data[self.price_column] * (base_cpi / self.data['cpi'])
        self.logger.info(f"Created adjusted_price using base year {base_year} (base CPI: {base_cpi:.2f})")

    # -------------------------------------------------------------------------
    # Column Management
    # -------------------------------------------------------------------------

    def _drop_columns(self) -> None:
        """Drop unnecessary columns."""
        self.logger.info("Dropping columns...")

        # Standard columns to drop if present
        cols_to_drop = [
            "category_code_level1",
            "category_code_level2",
            "category_code_level3",
            "category_name",
        ]

        # Add client-specific columns to drop
        cols_to_drop.extend(self.client.get_columns_to_drop())

        # Drop columns that exist
        existing = [c for c in cols_to_drop if c in self.data.columns]
        if existing:
            self.data = self.data.drop(columns=existing)
            self.logger.info(f"Dropped columns: {existing}")

    # -------------------------------------------------------------------------
    # Resampling
    # -------------------------------------------------------------------------

    def _resample_data(self) -> None:
        """Resample data to weekly frequency."""
        self.logger.info("Resampling to weekly...")

        # Remove duplicates
        self.data = self.data.drop_duplicates(
            subset=["product_code", "date"], keep="last"
        )

        # Add week column
        self.data["year_week"] = (
            self.data["date"].dt.isocalendar().year.astype(str)
            + "_"
            + self.data["date"].dt.isocalendar().week.astype(str).str.zfill(2)
        )

        # Aggregate by product and week
        agg_dict = {
            self.target: "sum",
            self.price_column: "mean",
            "date": "first",
        }

        # Add competitor prices
        for comp in self.client.competitors:
            comp_col = f"{comp}_price"
            if comp_col in self.data.columns:
                agg_dict[comp_col] = "mean"

        self.data = self.data.groupby(["product_code", "year_week"]).agg(agg_dict).reset_index()
        self.logger.info(f"Resampled to {len(self.data)} weekly records")

    # -------------------------------------------------------------------------
    # Price Grid
    # -------------------------------------------------------------------------

    def _create_price_grid(self) -> None:
        """Create price grid for elasticity simulation."""
        self.logger.info("Creating price grid...")

        # Get reference rows for price grid
        # If updated=False and test_end_date is set, use the test week rows
        # If updated=True, use the last N rows (latest available data)
        test_end_date = self.config.get("test_end_date")
        updated = self.config.get("updated", False)

        if test_end_date and not updated:
            test_end = pd.Timestamp(test_end_date)
            test_start = test_end - pd.Timedelta(days=self.price_grid_freq - 1)
            last_n = self.data[
                (self.data["date"] >= test_start) & (self.data["date"] <= test_end)
            ]
            self.logger.info(f"Building price grid from test week: {test_start.date()} to {test_end.date()}")
        else:
            last_n = (
                self.data.sort_values(["product_code", "date"])
                .groupby("product_code")
                .tail(self.price_grid_freq)
            )


        # Calculate number of price points
        num_prices = abs(self.min_grid_perc) + abs(self.max_grid_perc) + 1

        # Create price grid per product
        self.price_grid = {}
        for product_code, group in last_n.groupby("product_code"):
            product_prices = []
            for _, row in group.iterrows():
                base_price = row[self.price_column]
                if base_price > 0:
                    prices = np.linspace(
                        base_price * (1 + self.min_grid_perc / 100),
                        base_price * (1 + self.max_grid_perc / 100),
                        num_prices,
                    ).tolist()
                    product_prices.append(prices)
            if product_prices:
                self.price_grid[str(product_code)] = product_prices

        self.logger.info(f"Created price grid for {len(self.price_grid)} products")

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------

    def _generate_overview_report(self) -> None:
        """Generate overview statistics."""
        self.logger.info("Generating overview report...")

        stats = {
            "num_products": self.data["product_code"].nunique(),
            "num_rows": len(self.data),
            "date_range": f"{self.data['date'].min()} to {self.data['date'].max()}",
            "num_categories": self.data[self.category].nunique() if self.category in self.data.columns else 0,
            "price_range": f"{self.data[self.price_column].min():.2f} - {self.data[self.price_column].max():.2f}",
        }

        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")

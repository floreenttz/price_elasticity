"""
Refactored ElasticityCalculator using client adapters.

This module provides client-agnostic price elasticity calculation.
"""

import logging
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from ..clients.base import ClientAdapter
from ..storage.base import Storage


def get_logger(name: str = "elasticity") -> logging.Logger:
    """Create a logger for elasticity calculation."""
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


class ElasticityCalculator:
    """
    Client-agnostic price elasticity calculator.

    Calculates price elasticity from demand model predictions
    using log-log regression on price-quantity pairs.
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
        Initialize the elasticity calculator.

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
        self.price_column = config.get("price_column", "product_selling_price")
        self.category_column = config.get("category", client.category_column)

        # Data containers
        self.estimations: pd.DataFrame | None = None
        self.preprocessed_data: pd.DataFrame | None = None
        self.curve_results: pd.DataFrame | None = None
        self.elasticity_df: pd.DataFrame | None = None
        self.elasticities_by_category: pd.DataFrame | None = None

    def run(
        self,
        estimations: pd.DataFrame | None = None,
        preprocessed_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Execute the elasticity calculation pipeline.

        Args:
            estimations: Predictions DataFrame. If None, loads from storage.
            preprocessed_data: Preprocessed data for category info. If None, loads.

        Returns:
            DataFrame with elasticity values.
        """
        self.logger.info(f"Starting elasticity calculation for client: {self.client.name}")

        # Load data
        if estimations is not None:
            self.estimations = estimations.copy()
        else:
            self._load_estimations()

        if preprocessed_data is not None:
            self.preprocessed_data = preprocessed_data.copy()
        else:
            self._load_preprocessed_data()

        # Rename prediction column for consistency
        if "predicted_quantity" in self.estimations.columns:
            self.estimations = self.estimations.rename(
                columns={"predicted_quantity": "predict_sales"}
            )

        # Calculate demand curves and elasticities
        self._calculate_demand_curves()
        self._calculate_elasticities()
        self._calculate_elasticities_per_category()

        # Add client metadata
        self.elasticity_df = self.client.add_output_metadata(self.elasticity_df)

        self.logger.info("Elasticity calculation complete!")
        return self.elasticity_df

    def save(self) -> None:
        """Save elasticity results to storage."""
        path = self.client.get_artifact_path("elasticities", self.config, self.frequency)
        self.storage.write_parquet(self.elasticity_df, path)
        self.logger.info(f"Saved elasticities to {path}")

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def _load_estimations(self) -> None:
        """Load model predictions from storage."""
        self.logger.info("Loading estimations...")
        path = self.client.get_artifact_path("estimations", self.config, self.frequency)
        self.estimations = self.storage.read_parquet(path)
        self.logger.info(f"Loaded {len(self.estimations):,} predictions")

    def _load_preprocessed_data(self) -> None:
        """Load preprocessed data for category info."""
        self.logger.info("Loading preprocessed data...")
        path = self.client.get_artifact_path("preprocessed_data", self.config, self.frequency)
        self.preprocessed_data = self.storage.read_parquet(path)
        self.logger.info(f"Loaded {len(self.preprocessed_data):,} rows")

    # -------------------------------------------------------------------------
    # Demand Curve Fitting
    # -------------------------------------------------------------------------

    def _calculate_demand_curves(self) -> None:
        """Fit demand curves for each product using log-log regression."""
        self.logger.info("Calculating demand curves...")

        # Price column in estimations
        price_col = f"{self.price_column}_calc"

        results = []
        grouped = self.estimations.groupby("product_code")

        for product_code, group in grouped:
            curve_result = self._fit_demand_curve(group, price_col)
            if curve_result is not None:
                curve_result["product_code_elasticity"] = product_code
                results.append(curve_result)

        if results:
            self.curve_results = pd.concat(results, ignore_index=True)
            self.curve_results = self.curve_results.sort_values(
                ["product_code_elasticity", "price"]
            ).reset_index(drop=True)
        else:
            self.curve_results = pd.DataFrame()

        self.logger.info(f"Fitted demand curves for {len(results)} products")

    @staticmethod
    def _fit_demand_curve(
        group: pd.DataFrame, price_col: str
    ) -> pd.DataFrame | None:
        """
        Fit a log-log demand curve: log(Q) = a + b*log(P)

        Args:
            group: DataFrame with price and quantity data.
            price_col: Name of the price column.

        Returns:
            DataFrame with fitted curve, or None if fitting failed.
        """
        if group.empty:
            return None

        # Filter valid rows (positive price and quantity)
        g = group.copy()
        g = g[(g["predict_sales"] > 0) & (g[price_col] > 0)]

        if len(g) < 2:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                x = np.log(g[price_col].astype(float).values)
                y = np.log(g["predict_sales"].astype(float).values)

                # Linear regression in log space
                b, a = np.polyfit(x, y, 1)

                # Predict for original prices
                prices = group[price_col].astype(float).values
                fitted = np.full(len(prices), np.nan)

                mask = prices > 0
                fitted[mask] = np.exp(a + b * np.log(prices[mask]))

                result = pd.DataFrame({
                    "price": group[price_col].values,
                    "sales": fitted,
                })

                return result.dropna(subset=["sales"])

            except Exception:
                return None

    # -------------------------------------------------------------------------
    # Elasticity Calculation
    # -------------------------------------------------------------------------

    def _calculate_elasticities(self) -> None:
        """Calculate price elasticity for each product."""
        self.logger.info("Calculating elasticities...")

        if self.curve_results.empty:
            self.elasticity_df = pd.DataFrame(columns=["product_code", "elasticity"])
            return

        elasticities = {}

        grouped = self.curve_results.groupby("product_code_elasticity")

        for product_code, group in grouped:
            if len(group) < 3:
                continue

            # Get middle point
            middle_idx = len(group) // 2
            middle_row = group.iloc[middle_idx]
            current_price = middle_row["price"]

            # Get adjacent points
            prev_next = group[
                (group["price"].shift(1) == current_price)
                | (group["price"].shift(-1) == current_price)
            ]

            if prev_next.empty:
                continue

            # Calculate elasticity: % change in Q / % change in P
            pct_change_q = prev_next["sales"].pct_change().dropna()
            pct_change_p = prev_next["price"].pct_change().dropna()

            if len(pct_change_q) > 0 and len(pct_change_p) > 0:
                elasticity = (pct_change_q.iloc[0] / pct_change_p.iloc[0])
                elasticities[product_code] = elasticity

        # Build elasticity DataFrame
        if elasticities:
            self.elasticity_df = pd.DataFrame(
                [
                    {"product_code": pc, "elasticity": e}
                    for pc, e in elasticities.items()
                ]
            )

            # Filter zero elasticities and average across sequences
            self.elasticity_df = self.elasticity_df[
                self.elasticity_df["elasticity"] != 0
            ]
            self.elasticity_df = (
                self.elasticity_df.groupby("product_code")["elasticity"]
                .mean()
                .reset_index()
            )
        else:
            self.elasticity_df = pd.DataFrame(columns=["product_code", "elasticity"])

        self.logger.info(
            f"Calculated elasticities for {len(self.elasticity_df)} products"
        )

    # -------------------------------------------------------------------------
    # Category Aggregation
    # -------------------------------------------------------------------------

    def _calculate_elasticities_per_category(self) -> None:
        """Aggregate elasticities by category."""
        self.logger.info("Calculating elasticities per category...")

        if self.elasticity_df.empty:
            self.elasticities_by_category = pd.DataFrame()
            return

        # Keep only non-positive elasticities (economically sensible)
        valid_elasticities = self.elasticity_df[self.elasticity_df["elasticity"] <= 0]

        # Merge with category info
        merged = pd.merge(
            valid_elasticities,
            self.preprocessed_data[["product_code", self.category_column]].drop_duplicates(),
            on="product_code",
            how="left",
        )

        # Calculate aggregates per category
        self.elasticities_by_category = (
            merged.groupby(self.category_column)
            .agg(
                elasticity_median=("elasticity", "median"),
                elasticity_mean=("elasticity", "mean"),
                num_products=("product_code", "nunique"),
            )
            .reset_index()
        )

        # Round values
        self.elasticities_by_category["elasticity_median"] = (
            self.elasticities_by_category["elasticity_median"].round(2)
        )
        self.elasticities_by_category["elasticity_mean"] = (
            self.elasticities_by_category["elasticity_mean"].round(2)
        )

        # Filter categories with enough products
        min_products = 5
        self.elasticities_by_category = self.elasticities_by_category[
            self.elasticities_by_category["num_products"] >= min_products
        ]

        # Add client metadata
        self.elasticities_by_category = self.client.add_output_metadata(
            self.elasticities_by_category
        )

        self.logger.info(
            f"Calculated elasticities for {len(self.elasticities_by_category)} categories"
        )

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Get summary statistics of elasticity results."""
        if self.elasticity_df.empty:
            return {"total_products": 0}

        elasticities = self.elasticity_df["elasticity"]

        return {
            "total_products": len(self.elasticity_df),
            "elastic_count": len(elasticities[(elasticities < -1) & (elasticities > -5)]),
            "inelastic_count": len(elasticities[(elasticities > -1) & (elasticities < 0)]),
            "positive_count": len(elasticities[elasticities > 0]),
            "outlier_count": len(elasticities[elasticities < -5]),
            "median_elasticity": elasticities.median(),
            "mean_elasticity": elasticities.mean(),
        }

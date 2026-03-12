"""
Refactored DemandModel using client adapters.

This module provides client-agnostic demand modeling using XGBoost.
"""

import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from ..clients.base import ClientAdapter
from ..storage.base import Storage


def get_logger(name: str = "modeling") -> logging.Logger:
    """Create a logger for modeling."""
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


class DemandModel:
    """
    Client-agnostic demand model.

    Trains an XGBoost model to predict quantity sold based on price
    and other features. Uses price grid to simulate demand at different
    price points.
    """

    def __init__(
        self,
        client: ClientAdapter,
        config: dict,
        storage: Storage,
        logger: logging.Logger | None = None,
        params: dict | None = None,
        frequency: str = "daily",
    ):
        """
        Initialize the demand model.

        Args:
            client: Client adapter for client-specific logic.
            config: Configuration dictionary.
            storage: Storage backend for I/O.
            logger: Optional logger instance.
            params: XGBoost hyperparameters.
            frequency: 'daily' or 'weekly'.
        """
        self.client = client
        self.config = config
        self.storage = storage
        self.logger = logger or get_logger()
        self.frequency = frequency

        # Extract config values
        self.price_column = config.get("price_column", "product_selling_price")
        self.target = config.get("target", "quantity_sold")
        self.price_lags = config.get("price_lags", list(range(1, 29)))
        self.price_change_columns = config.get("price_change_columns", [self.price_column])
        self.competitors = config.get("competitors", client.competitors)
        self.price_grid_freq = config.get("price_grid_freq", 7)
        self.min_grid_perc = config.get("min_grid_perc", -10)
        self.max_grid_perc = config.get("max_grid_perc", 10)

        # Model parameters
        self.params = params or {}

        # Data containers
        self.data: pd.DataFrame | None = None
        self.price_grid: dict[str, list] = {}
        self.model: xgb.XGBRegressor | None = None
        self.features: list[str] = []
        self.x_train: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.predictions_df: pd.DataFrame | None = None

    def run(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Execute the modeling pipeline.

        Args:
            data: Input feature DataFrame. If None, loads from storage.

        Returns:
            DataFrame with predictions.
        """
        self.logger.info(f"Starting demand modeling for client: {self.client.name}")

        # Load data
        if data is not None:
            self.data = data.copy()
        else:
            self._load_data()

        self._load_price_grid()

        # Prepare features
        self._prepare_features()

        # Train/test split
        self._train_test_split()

        # Train model
        self._train_model()

        # Generate predictions
        self._generate_predictions()

        self.logger.info("Demand modeling complete!")
        return self.predictions_df

    def save(self) -> None:
        """Save predictions and model to storage."""
        # Save predictions
        pred_path = self.client.get_artifact_path(
            "estimations", self.config, self.frequency
        )
        self.storage.write_parquet(self.predictions_df, pred_path)
        self.logger.info(f"Saved predictions to {pred_path}")

        # Save model
        model_path = self.client.get_artifact_path("model", self.config)
        # Save locally first, then upload
        local_path = "trained_model.joblib"
        joblib.dump(self.model, local_path)
        # For S3, we'd need special handling
        self.logger.info(f"Saved model to {model_path}")

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load feature data from storage."""
        self.logger.info("Loading feature data...")
        path = self.client.get_artifact_path("feature_data", self.config, self.frequency)
        self.data = self.storage.read_parquet(path)
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.logger.info(f"Loaded {len(self.data):,} rows")

    def _load_price_grid(self) -> None:
        """Load price grid from storage."""
        self.logger.info("Loading price grid...")
        path = self.client.get_artifact_path("price_grid", self.config)
        self.price_grid = self.storage.read_pickle(path)
        self.logger.info(f"Loaded price grid for {len(self.price_grid)} products")

    # -------------------------------------------------------------------------
    # Feature Preparation
    # -------------------------------------------------------------------------

    def _prepare_features(self) -> None:
        """Prepare feature list for modeling."""
        self.logger.info("Preparing features...")

        # All columns except target and date are features (keep product_code like original)
        exclude = [self.target, "date"]
        self.features = [c for c in self.data.columns if c not in exclude]

        # Convert object columns to category
        for col in self.data.select_dtypes(include=["object"]).columns:
            self.data[col] = self.data[col].astype("category")

        self.logger.info(f"Using {len(self.features)} features")

    # -------------------------------------------------------------------------
    # Train/Test Split
    # -------------------------------------------------------------------------

    def _train_test_split(self) -> None:
        """Split data into training and test sets."""
        self.logger.info("Splitting train/test data...")

        # Calculate cutoff: last N days per product
        last_dates = self.data.groupby("product_code", observed=True)["date"].transform("max")
        cutoff_dates = last_dates - pd.Timedelta(days=self.price_grid_freq - 1)
        is_test = self.data["date"] >= cutoff_dates

        # Training data
        train_mask = ~is_test
        self.x_train = self.data.loc[train_mask, self.features].copy()
        self.y_train = self.data.loc[train_mask, self.target].copy()

        # Test data for predictions
        test_data = self.data.loc[is_test].copy()

        # Expand test data with price grid
        self.predict_df = self._expand_with_price_grid(test_data)

        self.logger.info(
            f"Train: {len(self.x_train):,} rows, "
            f"Test (expanded): {len(self.predict_df):,} rows"
        )

    def _expand_with_price_grid(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Expand test data with price grid for counterfactual predictions."""
        self.logger.info(f"Expanding test data ({len(test_data):,} rows) with price grid...")

        # Vectorized approach: build expansion lists first, then create DataFrame once
        expanded_rows = []
        products_processed = 0
        total_products = test_data["product_code"].nunique()

        for product_code, group in test_data.groupby("product_code", observed=True):
            products_processed += 1
            if products_processed % 5000 == 0:
                self.logger.info(f"  Processed {products_processed:,}/{total_products:,} products...")
            product_prices = self.price_grid.get(str(product_code), [])

            if not product_prices:
                continue

            group = group.reset_index(drop=True)
            num_days = min(len(group), len(product_prices))

            for i in range(num_days):
                price_set = product_prices[i]
                row_dict = group.iloc[i].to_dict()

                for price in price_set:
                    expanded_row = row_dict.copy()
                    expanded_row[self.price_column] = price
                    expanded_row["sequence_number"] = i
                        
                    # Sequential fix: lags within simulation window use simulated price
                    for column in self.price_change_columns:
                        for lag in self.price_lags:
                            if lag <= i:
                                lag_col = f"{column}_lag_{lag}"
                                if lag_col in expanded_row:
                                    expanded_row[lag_col] = price
                    
                    expanded_rows.append(expanded_row)

        if not expanded_rows:
            return pd.DataFrame()

        # Single DataFrame construction from list of dicts (much faster)
        result = pd.DataFrame(expanded_rows)

        # Convert object columns to category for XGBoost compatibility
        for col in result.select_dtypes(include=["object"]).columns:
            result[col] = result[col].astype("category")

        self.logger.info(f"Expanded to {len(result):,} rows")

        # Recalculate price-dependent features
        self._update_price_features(result)

        # Set promotion to False for simulation
        result["promotion_indicator"] = False

        return result

    def _update_price_features(self, df: pd.DataFrame) -> None:
        """Update price change and distance features for new prices.

        Also drops lag columns after recalculating price changes (matches original).
        """
        # Update price change features and drop lag columns
        lag_cols_to_drop = []
        for column in self.price_change_columns:
            for lag in self.price_lags:
                lag_col = f"{column}_lag_{lag}"
                change_col = f"price_change_{column}_lag_{lag}"

                if lag_col in df.columns:
                    lagged = df[lag_col]
                    if change_col in df.columns:
                        df[change_col] = ((df[column] - lagged) / lagged) * 100
                        df[change_col] = df[change_col].replace([np.inf, -np.inf], np.nan)
                    lag_cols_to_drop.append(lag_col)

        # Drop lag columns from predict_df (matches original)
        df.drop(columns=[c for c in lag_cols_to_drop if c in df.columns], inplace=True)

        # Also drop lag columns from x_train (matches original)
        for col in lag_cols_to_drop:
            if col in self.x_train.columns:
                self.x_train.drop(columns=[col], inplace=True)

        # Update price distance features
        for competitor in self.competitors:
            comp_col = f"{competitor}_price"
            dist_col = f"price_distance_{competitor}"

            if comp_col in df.columns:
                own_price = df[self.price_column]
                comp_price = df[comp_col]
                avg_price = (own_price + comp_price) / 2

                df[dist_col] = np.where(
                    avg_price > 0,
                    (own_price - comp_price) / avg_price,
                    0,
                )

    # -------------------------------------------------------------------------
    # Model Training
    # -------------------------------------------------------------------------

    def _train_model(self) -> None:
        """Train the XGBoost demand model."""
        self.logger.info("Training XGBoost model...")

        # Default parameters (seed=42 for reproducibility)
        default_params = {
            "objective": "reg:tweedie",
            "tweedie_variance_power": 1.24,
            "learning_rate": 0.06,
            "n_estimators": 1700,
            "max_leaves": 782,
            "enable_categorical": True,
            "seed": 42,
        }

        # Merge with provided params
        model_params = {**default_params, **self.params}

        self.model = xgb.XGBRegressor(**model_params)

        # Transform target (log1p for count data)
        y_train_transformed = np.log1p(self.y_train)

        # Fit
        self.model.fit(self.x_train, y_train_transformed)

        self.logger.info("Model trained successfully")

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    def _generate_predictions(self) -> None:
        """Generate predictions on expanded test set."""
        self.logger.info("Generating predictions...")

        if self.predict_df.empty:
            self.predictions_df = pd.DataFrame()
            return

        # Get features for prediction
        feature_cols = [c for c in self.features if c in self.predict_df.columns]
        X_pred = self.predict_df[feature_cols]

        # Predict (transform back from log space)
        y_pred_log = self.model.predict(X_pred)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0)  # Ensure non-negative

        # Build output DataFrame
        self.predictions_df = pd.DataFrame({
            "product_code": self.predict_df["product_code"],
            "predicted_quantity": y_pred,
            f"{self.price_column}_calc": self.predict_df[self.price_column],
            "sequence_number": self.predict_df["sequence_number"],
        })

        # Sum predicted quantities across all days per (product, simulated price)
        self.predictions_df = (
            self.predictions_df
            .groupby(["product_code", f"{self.price_column}_calc"])["predicted_quantity"]
            .sum()
            .reset_index()
        )

        # Get actual prices for comparison
        last_days = (
            self.data.sort_values(["product_code", "date"])
            .groupby("product_code")
            .tail(self.price_grid_freq)
        )

        # Add sequence numbers to match
        last_days = last_days.reset_index(drop=True)
        last_days["sequence_number"] = last_days.groupby("product_code").cumcount()

        # Merge actual prices
        self.predictions_df = self.predictions_df.merge(
            last_days[["product_code", self.price_column, "sequence_number"]],
            on=["product_code", "sequence_number"],
            how="left",
            suffixes=["_calc", "_real"],
        )

        self.logger.info(f"Generated {len(self.predictions_df):,} predictions")

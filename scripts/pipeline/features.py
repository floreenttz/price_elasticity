"""
Refactored FeatureEngineer using client adapters.

This module provides client-agnostic feature engineering for demand modeling.
"""

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import holidays
from holidays.countries import Netherlands

import calendar
import datetime as dt

from ..clients.base import ClientAdapter
from ..storage.base import Storage


def get_logger(name: str = "features") -> logging.Logger:
    """Create a logger for feature engineering."""
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

class ExtendedNLHolidays(Netherlands):
    """Dutch holidays extended with commercially relevant dates missing from the default module."""

    def _populate(self, year):
        Netherlands._populate(self, year)
        # Default module only counts Bevrijdingsdag every 5 years; include it annually
        try:
            self.pop_named("Bevrijdingsdag")
        except KeyError:
            pass
        self[dt.date(year, 5, 5)] = "Bevrijdingsdag"
        self[dt.date(year, 12, 5)] = "Sinterklaas"
        self[dt.date(year, 12, 24)] = "Kerstavond"

        c = calendar.Calendar(firstweekday=calendar.MONDAY)
        self[c.monthdatescalendar(year, 6)[2][-1]] = "Vaderdag"   # 3rd Sunday of June
        self[c.monthdatescalendar(year, 5)[1][-1]] = "Moederdag"  # 2nd Sunday of May


class FeatureEngineer:
    """
    Client-agnostic feature engineer.

    Creates features for demand modeling:
    - Calendar features (day of week, month, etc.)
    - Weather features
    - Lag features for price
    - Price change features
    - Price distance features (vs competitors)
    """

    def __init__(
        self,
        client: ClientAdapter,
        config: dict,
        storage: Storage,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the feature engineer.

        Args:
            client: Client adapter for client-specific logic.
            config: Configuration dictionary.
            storage: Storage backend for I/O.
            logger: Optional logger instance.
        """
        self.client = client
        self.config = config
        self.storage = storage
        self.logger = logger or get_logger()

        # Extract config values
        self.price_column = config.get("price_column", "product_selling_price")
        self.price_lags = config.get("price_lags", list(range(1, 29)))
        self.price_change_columns = config.get("price_change_columns", [self.price_column])
        self.competitors = config.get("competitors", client.competitors)

        # Data container
        self.data: pd.DataFrame | None = None

    def run(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Execute the feature engineering pipeline.

        Args:
            data: Input DataFrame. If None, loads from storage.

        Returns:
            DataFrame with engineered features.
        """
        self.logger.info(f"Starting feature engineering for client: {self.client.name}")

        # Load data if not provided
        if data is not None:
            self.data = data.copy()
        else:
            self._load_data()

        # Feature engineering steps
        self._calendar_features()
        self._weather_features()
        self._holiday_features()
        self._lag_features()
        self._price_change_features()
        self._price_distance_features()
        self._convert_dtypes()

        # Round all float columns to 2 decimal places                                       
        float_cols = self.data.select_dtypes(include=["float"]).columns                     
        self.data[float_cols] = self.data[float_cols].round(2) 

        self.logger.info("Feature engineering complete!")
        return self.data

    def save(self, frequency: str = "daily") -> None:
        """Save feature data to storage."""
        path = self.client.get_artifact_path("feature_data", self.config, frequency)
        self.storage.write_parquet(self.data, path)
        self.logger.info(f"Saved feature data to {path}")

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def _load_data(self, frequency: str = "daily") -> None:
        """Load preprocessed data from storage."""
        self.logger.info("Loading preprocessed data...")
        path = self.client.get_artifact_path("preprocessed_data", self.config, frequency)
        self.data = self.storage.read_parquet(path)
        self.logger.info(f"Loaded {len(self.data):,} rows")

    # -------------------------------------------------------------------------
    # Calendar Features
    # -------------------------------------------------------------------------

    def _calendar_features(self) -> None:
        """Create calendar-based features."""
        self.logger.info("Creating calendar features...")

        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data["day_of_week"] = self.data["date"].dt.day_of_week
        self.data["day_of_month"] = self.data["date"].dt.day
        self.data["day_of_year"] = self.data["date"].dt.day_of_year
        self.data["month"] = self.data["date"].dt.month
        self.data["year"] = self.data["date"].dt.year
        self.data["is_month_start"] = self.data["date"].dt.is_month_start
        self.data["is_month_end"] = self.data["date"].dt.is_month_end
        self.data["is_quarter_start"] = self.data["date"].dt.is_quarter_start
        self.data["is_quarter_end"] = self.data["date"].dt.is_quarter_end

        self.logger.info("Calendar features created")

    # -------------------------------------------------------------------------
    # Holiday Features
    # -------------------------------------------------------------------------

    def _holiday_features(self) -> None:
        """Create holiday proximity features."""
        self.logger.info("Adding holiday features...")

        start_year = self.data["date"].min().year
        # If 'country' is not checked
        # nl_holidays = ExtendedNLHolidays(years=range(start_year, datetime.now().year + 1))
        # holiday_df = pd.DataFrame(list(nl_holidays.items()), columns=["date", "holiday"])
        years = range(start_year, datetime.now().year + 1)

        if self.client.country == "NL":
            client_holidays = ExtendedNLHolidays(years=years)
        else:
            client_holidays = holidays.country_holidays(self.client.country, years=years)

        holiday_df = pd.DataFrame(list(client_holidays.items()), columns=["date", "holiday"])
        holiday_df["date"] = pd.to_datetime(holiday_df["date"])

        # Build a full date range and merge holidays in
        start_date = datetime(start_year, 1, 1)
        end_date = datetime.now() + timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        df = pd.DataFrame({"date": date_range})

        holiday_df = pd.merge(df, holiday_df, on="date", how="outer")
        holiday_df["holiday"] = holiday_df["holiday"].fillna("No holiday")

        # Days until next holiday
        group = holiday_df["holiday"].iloc[::-1].ne("No holiday").cumsum()
        holiday_df["time_until_next_holiday"] = group.groupby(group).cumcount()

        # Days since previous holiday
        group = holiday_df["holiday"].ne("No holiday").cumsum()
        holiday_df["time_after_previous_holiday"] = group.groupby(group).cumcount()

        self.data = self.data.merge(holiday_df, on="date", how="left")
        self.logger.info("Holiday features added")


    # -------------------------------------------------------------------------
    # Weather Features
    # -------------------------------------------------------------------------

    def _weather_features(self) -> None:
        """Merge weather data."""
        self.logger.info("Adding weather features...")

        s3_config = self.config.get("s3_dir", {})
        weather_path = s3_config.get("weather")

        if weather_path and self.storage.exists(weather_path):
            try:
                weather_df = self.storage.read_parquet(weather_path)

                # Drop columns that might cause issues
                cols_to_drop = ["mean_temp_over_years", "len_rain"]
                weather_df = weather_df.drop(
                    columns=[c for c in cols_to_drop if c in weather_df.columns]
                )

                # Ensure date types match
                weather_df["date"] = pd.to_datetime(weather_df["date"])
                self.data["date"] = pd.to_datetime(self.data["date"])

                # Merge
                self.data = pd.merge(self.data, weather_df, on="date", how="left")
                self.logger.info("Weather features added")
            except Exception as e:
                self.logger.warning(f"Could not load weather data: {e}")
        else:
            self.logger.info("Weather data not available")
            

    # -------------------------------------------------------------------------
    # Lag Features
    # -------------------------------------------------------------------------

    def _lag_features(self) -> None:
        """Create lag features for price columns."""
        self.logger.info(f"Creating lag features for lags: {self.price_lags}")

        for column in self.price_change_columns:
            if column not in self.data.columns:
                continue

            for lag in self.price_lags:
                lag_col = f"{column}_lag_{lag}"
                self.data[lag_col] = self.data.groupby("product_code")[column].shift(lag)

        self.logger.info("Lag features created")

    # -------------------------------------------------------------------------
    # Price Change Features
    # -------------------------------------------------------------------------

    def _price_change_features(self) -> None:
        """Calculate price change percentages."""
        self.logger.info("Creating price change features...")

        for column in self.price_change_columns:
            if column not in self.data.columns:
                continue

            for lag in self.price_lags:
                lag_col = f"{column}_lag_{lag}"
                change_col = f"price_change_{column}_lag_{lag}"

                if lag_col in self.data.columns:
                    lagged = self.data[lag_col]
                    self.data[change_col] = ((self.data[column] - lagged) / lagged) * 100
                    self.data[change_col] = self.data[change_col].replace(
                        [np.inf, -np.inf], np.nan
                    )

        self.logger.info("Price change features created")

    # -------------------------------------------------------------------------
    # Price Distance Features
    # -------------------------------------------------------------------------

    def _price_distance_features(self) -> None:
        """Calculate price distance vs competitors."""
        self.logger.info("Creating price distance features...")

        for competitor in self.competitors:
            comp_col = f"{competitor}_price"
            if comp_col in self.data.columns:
                dist_col = f"price_distance_{competitor}"
                own_price = self.data[self.price_column]
                comp_price = self.data[comp_col]
                avg_price = (own_price + comp_price) / 2

                # Avoid division by zero
                self.data[dist_col] = np.where(
                    avg_price > 0,
                    (own_price - comp_price) / avg_price,
                    0,
                )

        self.logger.info("Price distance features created")

    # -------------------------------------------------------------------------
    # Type Conversion
    # -------------------------------------------------------------------------

    def _convert_dtypes(self) -> None:
        """Convert column types for model compatibility."""
        self.logger.info("Converting data types...")

        # Convert boolean columns
        if "promotion_indicator" in self.data.columns:
            self.data["promotion_indicator"] = self.data["promotion_indicator"].astype(bool)

        # Convert string columns to category for XGBoost
        for col in self.data.select_dtypes(include=["object"]).columns:
            self.data[col] = self.data[col].astype("category")

        self.logger.info("Data types converted")

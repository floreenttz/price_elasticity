import json
import time
import numpy as np
import pandas as pd
import holidays
from pandas.tseries.offsets import MonthBegin
import os
import logging
import yaml
import calendar
import datetime
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


def get_logger():
    os.makedirs('logs', exist_ok=True)
    current_time = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
    log_file = os.path.join('logs', f"feature_engineering_{current_time}.log")
  
    # Create a new logger
    logger = logging.getLogger(current_time)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger

def log_decorator(method):
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = method(self, *args, **kwargs)  # Our original method is executed here
        end = time.time()
        elapsed_time = end - start
        self.logger.info(f"Execution time: {elapsed_time:.2f} seconds")
        self.logger.info("-" * 40)
        return result
    return wrapper


class GradientBoostingFeaturizer:
    """
    A class responsible for feature engineering tailored for gradient boosting models.
    """

    def __init__(self, config_file, client_name=None, priceline_name=None, preprocessed_data=None, logger=None, debug=False, frequency='daily'):
        """
        Initializes the GradientBoostingFeaturizer class.

        Args:
            config_file (str): Path to the configuration file.
            client_name (str, optional): Name of the client. Defaults to None.
            preprocessed_data (pd.DataFrame, optional): Preprocessed data from DataPreprocessor. Defaults to None.
            frequency (str): Frequency of the data ('daily' or 'weekly'). Defaults to 'daily'.

        Raises:
            AssertionError: If client_name is not provided or if preprocessed_data is empty.
        """

        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger

        self.debug = debug
        self.config_file = config_file
        self.priceline_name = priceline_name
        self.frequency = frequency  # Frequency parameter added
        self.current_script_path = os.path.dirname(os.path.abspath("__file__"))
        self._load_config()
        self.data = pd.read_parquet(self.data_from_s3)
        self.competitors = self.config['competitors']
        self.implement_features()
        self.save_to_s3()


    def _load_config(self):
        """
        This function attempts to load configuration settings from the specified YAML configuration file.
        It logs success or failure messages and updates progress accordingly.

        Returns:
            dict: The loaded configuration settings.

        Raises:
            Exception: If there is an error while loading the configuration.
        """
        try:
            logging.info("-" * 40)
            self.logger.info("Loading configuration...")

            path = os.path.join(self.current_script_path, "config_files/", self.config_file)
            
            with open(path, "r") as stream:
                self.config = yaml.safe_load(stream)

            self.config_s3 = self.config['s3_dir']
            self.client_name = self.config['client_name']
            self.price_column = self.config['price_column']
            self.price_change_columns = self.config['price_change_columns']
            self.weather_data_s3 = self.config_s3['weather']

            if self.client_name == 'spar':
                self.priceline_name = self.config['priceline_name']
                    
            self.bucket = os.path.join(self.config_s3['s3'], self.config_s3['bucket'])
            if self.client_name == 'spar':
                if self.frequency== 'weekly':
                    self.data_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'], self.priceline_name + '_' + self.config_s3['preprocessed_data_weekly'])
                    self.data_s3_dir = os.path.join(self.bucket, self.config_s3['data_prefix'], self.priceline_name + '_' + self.config_s3['feature_data_weekly'])
                else:
                    self.data_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'], self.priceline_name + '_' + self.config_s3['preprocessed_data'])
                    self.data_s3_dir = os.path.join(self.bucket, self.config_s3['data_prefix'], self.priceline_name + '_' + self.config_s3['feature_data'])
            else:
                if self.frequency=='weekly':
                    self.data_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'], self.config_s3['preprocessed_data_weekly'])
                    self.data_s3_dir = os.path.join(self.bucket, self.config_s3['data_prefix'], self.config_s3['feature_data_weekly'])

                else:
                    self.data_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'], self.config_s3['preprocessed_data'])
                    self.data_s3_dir = os.path.join(self.bucket, self.config_s3['data_prefix'], self.config_s3['feature_data'])

            self.target = self.config["target"]
            self.lags =  self.config["price_lags"]

            self.logger.info("Configuration loaded successfully.")
            self.logger.info("-" * 40)
        except Exception as e:
            self.logger.error("Error loading configuration: %s", str(e))
            raise e

    def implement_features(self):
        """
        Method for implementing features during initialization of the main class.
        """
        self.calendar_features()
        self.weather_features()
        #self.holiday_features()
        self.lag_features()
        self.calculate_price_changes()
        self.price_distance_features()
        self._change_dtypes()

    @log_decorator
    def calendar_features(self):
        """
        Constructs calendar-related features based on date attributes.
        Adapts the features depending on the frequency of the data.

        Raises:
            Exception: If an error occurs while creating date-related features.
        """
        try:
            self.logger.info("Creating date-related features...")
            self.data['date'] = pd.to_datetime(self.data['date'])
            
            if self.frequency == 'daily':
                # Daily frequency features
                self.data["day_of_week"] = self.data.date.dt.day_of_week
                self.data["day_of_month"] = self.data.date.dt.day
                self.data["day_of_year"] = self.data.date.dt.day_of_year
                self.data["month"] = self.data.date.dt.month
                self.data["year"] = self.data.date.dt.year


            elif self.frequency == 'weekly':
                # Weekly frequency features
                self.data["week_of_year"] = self.data.date.dt.isocalendar().week
                self.data["year"] = self.data.date.dt.year
                # Custom feature to indicate the start of the week
                self.data["is_week_start"] = self.data.date.dt.weekday == 0  # Monday as week start

            self.logger.info("Date-related features created successfully.")
        except Exception as e:
            self.logger.error("Error creating date-related features: %s", str(e))
            raise e

    @log_decorator
    def holiday_features(self):
        """
        Constructs holiday-related features based on Dutch holidays.
        Adapts the features depending on the frequency of the data.
        
        Raises:
            Exception: If an error occurs while creating holiday-related features.
        """
        try:
            self.logger.info("Creating holiday-related features...")
            start_year = self.data.date.min().year

            holiday_df = pd.DataFrame([ExtendedNLHolidays(years=range(start_year, datetime.now().year))])
            holiday_df = pd.melt(holiday_df)
            holiday_df.columns = ['date', 'holiday']
            holiday_df.date = pd.to_datetime(holiday_df.date)

            # Create date range with all dates
            start_date = datetime(start_year, 1, 1)
            end_date = datetime.now() + timedelta(days=365)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            df = pd.DataFrame({'date': date_range})

            # Merge holidays with all dates
            holiday_df = pd.merge(df, holiday_df, on='date', how='outer')

            # Fill in non-holidays
            holiday_df.holiday = holiday_df.holiday.fillna('No holiday')

            # Create holiday features dynamically
            if self.frequency == 'daily':
                # Time until and after the holiday for daily data
                group = holiday_df.holiday.iloc[::-1].ne('No holiday').cumsum()
                holiday_df['time_until_next_holiday'] = group.groupby(group).cumcount()
                group = holiday_df.holiday.iloc[:].ne('No holiday').cumsum()
                holiday_df['time_after_previous_holiday'] = group.groupby(group).cumcount()
                
                # Merge holiday features with the main dataset
                self.data = pd.merge(self.data, holiday_df, on='date', how='left')
                
            elif self.frequency == 'weekly':
                # Indicate if there was a holiday in the week
                holiday_df['week_start'] = holiday_df['date'] - pd.to_timedelta(holiday_df['date'].dt.weekday, unit='D')
                weekly_holidays = holiday_df.groupby('week_start')['holiday'].apply(lambda x: (x != 'No holiday').any()).reset_index()
                weekly_holidays.rename(columns={'holiday': 'has_holiday'}, inplace=True)
                
                # Merge based on week start
                self.data['week_start'] = self.data['date'] - pd.to_timedelta(self.data['date'].dt.weekday, unit='D')
                self.data = pd.merge(self.data, weekly_holidays, on='week_start', how='left')
                self.data.drop(columns=['week_start'], inplace=True)

            self.logger.info("Holiday-related features created successfully.")
        except Exception as e:
            self.logger.error("Error creating holiday-related features: %s", str(e))
            raise e

    @log_decorator
    def weather_features(self):
        """
        Constructs weather-related features by merging weather data with the main dataset.
        Adapts to the frequency of the data (daily or weekly).

        Raises:
            Exception: If an error occurs while incorporating weather features.
        """
        try:
            self.logger.info("Loading weather data...")
            self.weather_df = pd.read_parquet(self.weather_data_s3)
            self.weather_df.drop(columns=['mean_temp_over_years', 'len_rain'], inplace=True)
            self.weather_df.date = self.weather_df.date.astype("datetime64[ns]")
            self.data.date = self.data.date.astype("datetime64[ns]")

            if self.frequency == 'weekly':
                # Resample weather data to weekly if the main data is weekly
                self.weather_df.set_index('date', inplace=True)
                self.weather_df = self.weather_df.resample('W-MON', label='left', closed='left').mean().reset_index()
                # Rounding multiple columns
                self.weather_df = self.weather_df.round({
                    'mean_temp': 1,
                    'min_temp': 1,
                    'max_temp': 1,
                    'sum_rain': 0,
                    'perc_sun': 0
                })

            # Merge weather data with the main dataset
            self.data = pd.merge(self.data, self.weather_df, on="date", how="left")
            self.logger.info("Loading weather data completed successfully.")
        except Exception as e:
            self.logger.error("Error loading weather data: %s", str(e))
            raise e
            
    @log_decorator
    def lag_features(self):
        """
        Constructs lag-based features for specified columns.

        Raises:
            Exception: If an error occurs while creating lag features.
        """
        
        try:
            self.logger.info("Creating lag features...")
            for column in self.price_change_columns:
                for lag in self.lags:
                    self.calc_lag(column=column, lag=lag)
            self.logger.info("Creating lag features completed successfully.")
        except Exception as e:
            self.logger.error("Error creating lag features: %s", str(e))
            raise e
            
    
    def calc_lag(self, column, lag):
        """
        Generates lag features for a specific column and lag value.

        Args:
            column (str): Name of the column to apply lag.
            lag (int): Lag value.
        """
        self.data[f"{column}_lag_{lag}"] = self.data.groupby(["product_code"])[column].shift(lag)

    def calculate_price_changes(self):
        """
        Calculates percentage changes in prices for lagged features.
        Adapts to the frequency of the data (daily or weekly).
        """
        try:
            self.logger.info("Calculating price change features...")
            for column in self.price_change_columns:
                for lag in self.lags:
                    column_name = f"price_change_{column}_lag_{lag}"
                    lagged_prices = self.data.groupby(["product_code"])[column].shift(lag)
                    self.data[column_name] = ((self.data[column] - lagged_prices) / lagged_prices) * 100
                    self.data[column_name].replace([np.inf, -np.inf], np.nan, inplace=True)
            self.logger.info("Price change features calculated successfully.")
        except Exception as e:
            self.logger.error("Error calculating price change features: %s", str(e))
            raise e

    def price_distance_features(self):
        """
        Calculates distance between the price and competitor prices.
        This remains unchanged regardless of frequency as it calculates relative differences.
        """
        try:
            self.logger.info("Calculating price distance features...")
            for competitor in self.competitors:
                if competitor + '_price' in self.data.columns:
                    self.data[f"price_distance_{competitor}"] = (self.data[self.price_column] - self.data[f"{competitor}_price"]) / ((self.data[self.price_column] + self.data[f"{competitor}_price"]) / 2)
            self.logger.info("Price distance features calculated successfully.")
        except Exception as e:
            self.logger.error("Error calculating price distance features: %s", str(e))
            raise e

    def _change_dtypes(self):
        """
        Change data types of certain columns to optimize memory usage.
        """
        try:
            self.logger.info("Changing data types to optimize memory usage...")
            self.data['promotion_indicator'] = self.data['promotion_indicator'].astype(bool)
            self.logger.info("Data types changed successfully.")
        except Exception as e:
            self.logger.error("Error changing data types: %s", str(e))
            raise e

    @log_decorator
    def save_to_s3(self):
        """
        Saves the processed data back to S3.
        """
        try:
            self.data.reset_index(drop=True, inplace=True)
            self.data.to_parquet(self.data_s3_dir)
            self.logger.info("Data saved to S3 successfully.")
        except Exception as e:
            self.logger.error("Error saving data to S3: %s", str(e))
            raise e


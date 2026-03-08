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
from datetime import datetime
from datetime import timedelta
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

# class ExtendedNLHolidays(holidays.NL):
#     """
#     An enhanced version of the python Holidays module tailored for the Netherlands.

#     The default Holidays module misses out on certain important Dutch holidays. This extension
#     aims to cover those holidays to provide a more comprehensive list. Specifically, it includes
#     holidays like "Bevrijdingsdag", "Vaderdag", "Sinterklaas", "Kerstavond", and "Moederdag".

#     Notes:
#         - Bevrijdingsdag: While the default module acknowledges this holiday once every 5 years (as per
#           legal mandate), this extension includes it annually.
#         - Vaderdag: Represented as the third Sunday of June.
#         - Moederdag: Represented as the second Sunday of May.

#     Usage:
#         holidays = ExtendedNLHolidays(years=2023)
#         print(holidays.get('2023-05-05'))  # Outputs: "Bevrijdingsdag"
#     """

#     def _populate(self, year):
#         holidays.NL._populate(self, year)
#         # It only counts bevrijdingsdag every 5 years, as it i s only legally mandated as such
#         # It's however important for us, so to prevent it from happening twice a year remove the default
#         self.pop_named("Bevrijdingsdag")
#         self[datetime.date(year, 5, 5)] = "Bevrijdingsdag"
#         self[datetime.date(year, 12, 5)] = "Sinterklaas"
#         self[datetime.date(year, 12, 24)] = "Kerstavond"

#         # Used to select xth day of yth month
#         c = calendar.Calendar(firstweekday=calendar.MONDAY)
#         # 3rd sunday of june
#         self[c.monthdatescalendar(year, 6)[2][-1]] = "Vaderdag"
#         # 2nd sunday of may
#         self[c.monthdatescalendar(year, 5)[1][-1]] = "Moederdag"


class GradientBoostingFeaturizer:
    """
    A class responsible for feature engineering tailored for gradient boosting models.
    """

    def __init__(self, config_file, client_name=None, priceline_name=None, preprocessed_data=None, logger=None, debug=False):
        """
        Initializes the GradientBoostingFeaturizer class.

        Args:
            config_file (str): Path to the configuration file.
            client_name (str, optional): Name of the client. Defaults to None.
            preprocessed_data (pd.DataFrame, optional): Preprocessed data from DataPreprocessor. Defaults to None.

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
        self.current_script_path = os.path.dirname(os.path.abspath("__file__"))
        self._load_config()
        self.data = pd.read_parquet(self.data_from_s3)
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

            path = os.path.join(self.current_script_path, self.config_file)
            
            with open(path, "r") as stream:
                self.config = yaml.safe_load(stream)

            self.config_s3 = self.config['s3_dir']
            self.client_name = self.config['client_name']
            self.price_column = self.config['price_column']
            self.competitors = self.config['competitors']
            self.price_change_columns = self.config['price_change_columns']
            self.weather_data_s3 = self.config_s3['weather']

            if self.client_name == 'spar':
                self.priceline_name = self.config['priceline_name']
                    
            self.bucket = os.path.join(self.config_s3['s3'], self.config_s3['bucket'])
            if self.client_name == 'spar':
                self.data_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'], self.priceline_name + '_' + self.config_s3['preprocessed_data'])
                self.data_s3_dir = os.path.join(self.bucket, self.config_s3['data_prefix'], self.priceline_name + '_' + self.config_s3['feature_data'])
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
        Method for implementing features during initialization
        of the main class.
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

        Raises:
            Exception: If an error occurs while creating date-related features.
        """
        try:
            self.logger.info("Creating date-related features...")
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data["day_of_week"] = self.data.date.dt.day_of_week
            self.data["day_of_month"] = self.data.date.dt.day
            self.data["day_of_year"] = self.data.date.dt.day_of_year
            self.data["month"] = self.data.date.dt.month
            self.data["year"] = self.data.date.dt.year
            self.data["is_month_start"] = self.data.date.dt.is_month_start
            self.data["is_month_end"] = self.data.date.dt.is_month_end
            self.data["is_quarter_start"] = self.data.date.dt.is_quarter_start
            self.data["is_quarter_end"] = self.data.date.dt.is_quarter_end
            
            self.logger.info("Date-related features created successfully.")
        except Exception as e:
            self.logger.error("Error creating date-related features: %s", str(e))
            raise e
            
    @log_decorator
    def holiday_features(self):
        
        start_year = self.data.date.min().year
        
        holiday_df = pd.DataFrame([ExtendedNLHolidays(years=range(start_year, datetime.datetime.today().year))])
        holiday_df = pd.melt(holiday_df)
        holiday_df.columns = ['date', 'holiday']
        holiday_df.date = pd.to_datetime(holiday_df.date)

        # create dateframe with all dates
        start_date = datetime.datetime(start_year, 1, 1)
        end_date = datetime.datetime.now() + timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        df = pd.DataFrame({'date': date_range})

        # merge holidays with all dates
        holiday_df = pd.merge(df, holiday_df, on='date', how='outer')

        # fill in non-holidays
        holiday_df.holiday =  holiday_df.holiday.fillna('No holiday')

        # create holiday features
        group = holiday_df.holiday.iloc[::-1].ne('No holiday').cumsum()
        holiday_df['time_untilnextholiday'] = group.groupby(group).cumcount()
        group = holiday_df.holiday.iloc[:].ne('No holiday').cumsum()
        holiday_df['time_afterpreviousholiday'] = group.groupby(group).cumcount()

        self.data = self.data.merge(holiday_df, on='date', how='left')

    @log_decorator
    def weather_features(self):
        """
        Constructs weather-related features by merging weather data with the main dataset.

        Raises:
            Exception: If an error occurs while incorporating weather features.
        """

        try:
            self.logger.info("Loading weather data...")
            self.weather_df = pd.read_parquet(self.weather_data_s3)
            self.weather_df.drop(columns=['mean_temp_over_years', 'len_rain'], inplace=True)
            self.weather_df.date = self.weather_df.date.astype("datetime64[ns]")
            self.data.date = self.data.date.astype("datetime64[ns]")
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
        
        for column in self.price_change_columns:
            for lag in self.lags:
                column_name = f"price_change_{column}_lag_{lag}"
                lagged_prices = self.data.groupby(["product_code"])[column].shift(lag)
                self.data[column_name] = ((self.data[column] - lagged_prices) / lagged_prices) * 100
                self.data[column_name].replace([np.inf, -np.inf], np.nan, inplace=True)

    def price_distance_features(self):
        
        for competitor in self.competitors:
            self.data[f"price_distance_{competitor}"] = (self.data[self.price_column] - self.data[f"{competitor}_price"]) / ((self.data[self.price_column] + self.data[f"{competitor}_price"]) / 2)

                
    def _change_dtypes(self):
        
        self.data['promotion_indicator'] = self.data['promotion_indicator'].astype(bool)

        
    @log_decorator
    def save_to_s3(self):
        self.data.reset_index(drop=True, inplace=True)
        self.data.to_parquet(self.data_s3_dir)
        

import os
import sys
import io
import time
import logging
import json
import pickle
import datetime
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
from pandarallel import pandarallel
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin
import swifter
from tqdm import tqdm
from tabulate import tabulate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .functions import *
import yaml
import s3fs
import boto3

from .functions import *

warnings.filterwarnings("ignore")

# Add the parent directory to the system path to handle relative imports correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..', '..')))

def get_logger():
    
    current_time = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
    log_file = os.path.join("logs", f"data_preprocessing_{current_time}.log")
  
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
        self._update_progress(self.pbar)
        end = time.time()
        elapsed_time = end - start
        self.logger.info(f"Execution time: {elapsed_time:.2f} seconds")
        self.logger.info("-" * 40)
        return result
    return wrapper

class DataPreprocessor:
    """
    This class is responsible for preprocessing data based on a specified client configuration.

    Attributes:
        config_file (str): Path to the configuration file.
        client_name (str): Name of the client for which data will be processed.
    """

    def __init__(self, config_file, logger=None, overview=True, subset=False, debug=False, frequency = 'daily'):
        """
        Initializes the DataPreprocessor class.

        Args:
            config_file (str): Path to the configuration file.
            client_name (str, optional): Name of the client. Defaults to None.

        Raises:
            AssertionError: If client_name and priceline name are not provided or are invalid.
        """

        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger
            
        # Initialize progress bar variables
        self.total_tasks = 18
        self.frequency = frequency
        self.completed_tasks = 0
        self.pbar = tqdm(total=self.total_tasks, desc="Overall Progress")
        self.config_file = config_file
        self.debug = debug
        self._load_config()
        self._load_sales_data(subset)
        self.preprocess_data(overview=overview)
        self._create_price_grid()
        self.save_to_s3()

    def _update_progress(self, pbar):
        """Update overall progress and display progress bar."""
        self.completed_tasks += 1
        self.pbar.update(1)

    @log_decorator
    def _load_config(self):
        """
        This function attempts to load configuration settings from the specified YAML configuration file.
        It logs success or failure messages and updates progress accordingly.

        Raises:
            Exception: If there is an error while loading the configuration.
        """

        try:
            self.logger.info("Loading configuration...")
            
            with open(f"{self.config_file}", "r") as stream:
                self.config = yaml.safe_load(stream)

            self.client_name = self.config['client_name']
            self.config_s3 = self.config['s3_dir']
            self.category = self.config["category"]
            self.filters = self.config["filters"]
            self.price_grid_freq = self.config["price_grid_freq"]
            self.category = self.config['category']
            self.min_grid_perc = self.config['min_grid_perc']
            self.max_grid_perc = self.config['max_grid_perc']
            # Spar has different pricelines
            if self.client_name == 'spar':
                self.priceline_name = self.config['priceline_name']

            self.price_column = self.config['price_column']
            self.bucket = os.path.join(self.config_s3['s3'], self.config_s3['bucket'])
            self.pe_names_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'], 'pe_names.parquet')
            self.cpi_s3 = os.path.join(self.config['data_folder'], 'cpi_alle_bestedingen.parquet')
            self.raw_data_s3_prefix = os.path.join(self.bucket, self.config_s3['data_prefix'])
            
            if self.client_name == 'spar':
                if self.frequency=='weekly':
                    self.data_s3_dir = os.path.join(self.bucket, self.config_s3['data_prefix'], self.priceline_name + '_' + self.config_s3['preprocessed_data_weekly'])
                
                else:
                    self.data_s3_dir = os.path.join(self.bucket, self.config_s3['data_prefix'], self.priceline_name + '_' + self.config_s3['preprocessed_data'])

            else:
                if self.frequency=='weekly':
                    self.data_s3_dir = os.path.join(self.bucket, self.config_s3['data_prefix'], self.config_s3['preprocessed_data_weekly'])
                else:
                    self.data_s3_dir = os.path.join(self.bucket, self.config_s3['data_prefix'], self.config_s3['preprocessed_data'])
                
            self.logger.info("Configuration loaded successfully.")
        except Exception as e:
            self.logger.error("Error loading configuration: %s", str(e))
            raise e

    # @log_decorator
    # def _load_sales_data(self, subset):
    #     """
    #     This function attempts to load sales data from the specified CSV file, automatically detecting
    #     the appropriate separator. It gogs success or failure messages and update progress accordingly.

    #     Returns:
    #         pandas.DataFrame: The loaded sales data as a DataFrame.

    #     Raises:
    #         Exception: If there is an error while loading the sales data.
    #     """

    #     try:
    #         self.logger.info("Loading sales data...")
    #         if self.client_name == 'spar':
    #             self.data_dir = os.path.join(self.raw_data_s3_prefix, self.priceline_name + '_aggregated.parquet')
    #         else:
    #             self.data_dir = os.path.join(self.raw_data_s3_prefix, 'raw_data.parquet')

    #         if 'parquet' in self.data_dir:
    #             self.data = pd.read_parquet(self.data_dir)
    #             self.test = self.data.copy()
    #         else:
    #             raise f"Unsupported file format: {self.data_dir}"
        
    #         # Take a subset of data by selecting the 3 biggest categories
    #         if subset:
    #             cats = self.data['product_category_code_level2'].value_counts().nlargest(3).index
    #             self.data = self.data[self.data['product_category_code_level2'].isin(cats)]
                        
    #         self.logger.info("Sales data loaded successfully.")
    #     except Exception as e:
    #         self.logger.error("Error loading sales data: %s", str(e))
    #         raise e

    @log_decorator
    def _load_other_data(self):
        """

        This function loads various datasets including product names, CPI categories, category mappers,
        and total CPI data. It performs necessary transformations and merge data into the
        main dataset based on specified conditions.

        Raises:
            Exception: If there is an error while loading or processing the additional data.
        """
        try:
            self.logger.info("Loading external and internal data...")
            # get data path and load the data
            if self.client_name == 'spar':
                # names data
                self.names_data = pd.read_parquet(self.pe_names_s3)
                
                self.data = pd.merge(
                    self.data,
                    self.names_data[
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
                
                file_names = [
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
                    "dagelijks_onderhoud_van_de_woning"
                ]

                cat_dfs = []
                for name in file_names:
                    cat_df = pd.read_parquet(f"{self.config['data_folder']}{name}.parquet")
                    cat_dfs.append(cat_df)
                                
                self.cpi_cat = pd.concat(cat_dfs)
                
            with open(f"cat_mapper.json") as f:
                self.cat_mapper = json.load(f)

            # load total cpi data
            self.cpi = pd.read_parquet(self.cpi_s3)
            self.cpi['month_year'] = pd.to_datetime(self.cpi['month_year']).dt.strftime('%Y-%m')

            self.logger.info("External and internal data loaded successfully.")

        except Exception as e:
            self.logger.error("Error loading external and internal data: %s", str(e))
            raise e

    @log_decorator
    def _load_sales_data(self, subset):
        """
        This function loads sales data from the specified location.
        For Hoogvliet, it handles multiple parquet files in chunks to avoid memory issues.
        """
        try:
            self.logger.info("Loading sales data...")
            self.logger.info(f"Client name: {self.client_name}")
            
            if self.client_name == 'spar':
                # SPAR: single file approach
                self.data_dir = os.path.join(self.raw_data_s3_prefix, self.priceline_name + '_aggregated.parquet')
                self.logger.info(f"Reading SPAR file: {self.data_dir}")
                if 'parquet' in self.data_dir:
                    self.data = pd.read_parquet(self.data_dir)
                    self.test = self.data.copy()
                else:
                    raise Exception(f"Unsupported file format: {self.data_dir}")
            
            elif self.client_name == 'hoogvliet':
                # HOOGVLIET: multiple parquet files in partitioned folders - load in chunks
                import s3fs
                
                self.logger.info(f"Reading all parquet files from: {self.raw_data_s3_prefix}")
                
                # Initialize S3 filesystem
                fs = s3fs.S3FileSystem()
                
                # Get all parquet files recursively
                all_parquet_files = fs.glob(f"{self.raw_data_s3_prefix}**/*.parquet")
                self.logger.info(f"Found {len(all_parquet_files)} parquet files")

                if subset:
                    rng = np.random.default_rng(42)
                    factor = 0.1
                    k = max(1, int(len(all_parquet_files) * factor))
                    all_parquet_files = rng.choice(all_parquet_files, size=k, replace=False).tolist()
                
                if not all_parquet_files:
                    raise Exception(f"No parquet files found in {self.raw_data_s3_prefix}")
                
                # Process files in chunks of 10 to avoid memory issues
                chunk_size = 10
                chunks = []
                
                for i in range(0, len(all_parquet_files), chunk_size):
                    chunk_files = all_parquet_files[i:i+chunk_size]
                    self.logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(all_parquet_files)-1)//chunk_size + 1} ({len(chunk_files)} files)")
                    
                    chunk_dfs = []
                    for file in chunk_files:
                        self.logger.info(f"  Reading {file}")
                        with fs.open(file, 'rb') as f:
                            df = pd.read_parquet(f)
                        chunk_dfs.append(df)
                    
                    # Combine chunk files
                    chunk_df = pd.concat(chunk_dfs, ignore_index=True)
                    chunks.append(chunk_df)
                    
                    self.logger.info(f"  Chunk rows: {len(chunk_df):,}")
                    
                    # Optional: clear memory
                    del chunk_dfs
                    import gc
                    gc.collect()
                
                # Combine all chunks into final dataframe
                self.data = pd.concat(chunks, ignore_index=True)
                self.test = self.data.copy()
                self.logger.info(f"Loaded {len(self.data):,} total rows from {len(all_parquet_files)} files")
                
                # Clear chunks from memory
                del chunks
                gc.collect()
            
            else:
                # Fallback for any other client
                self.data_dir = os.path.join(self.raw_data_s3_prefix, 'raw_data.parquet')
                self.logger.info(f"Reading default file: {self.data_dir}")
                if 'parquet' in self.data_dir:
                    self.data = pd.read_parquet(self.data_dir)
                    self.test = self.data.copy()
                else:
                    raise Exception(f"Unsupported file format: {self.data_dir}")
        
            # Take a subset of data by selecting the 3 biggest categories (if subset=True)
            if subset:
                # Find a category column to use for subset
                category_cols = [col for col in self.data.columns if 'category' in col and 'level' in col]
                if category_cols:
                    cat_col = category_cols[0]
                    cats = self.data[cat_col].value_counts().nlargest(3).index
                    self.data = self.data[self.data[cat_col].isin(cats)]
                    self.logger.info(f"Subset to top 3 categories in {cat_col}")
                            
            self.logger.info("Sales data loaded successfully.")
            
        except Exception as e:
            self.logger.error("Error loading sales data: %s", str(e))
            raise e
            
    @log_decorator
    def _filter_columns(self):
        """
        This function filters the dataset to include only specified columns from the configuration.

        It renames columns if necessary and filter the dataset to retain only the specified
        common columns along with any extra columns defined in the client's configuration.

        Raises:
            Exception: If there is an error while filtering or processing columns.
        """
        try:
            self.logger.info("Filtering columns...")
            if 'revenue_after' in self.data.columns:
                self.revenue = 'revenue_after'
            else:
                self.revenue = 'revenue_before'
            self.data = rename_columns(self.data)

            if self.price_column not in self.data.columns:
                
                self.data[self.price_column] = (
                    self.data[self.revenue] / self.data["quantity_sold"]
                )
            
            self.logger.info(self.data.product_code.nunique())
            self.logger.info("Columns filtered successfully.")
        except Exception as e:
            self.logger.error("Error filtering columns: %s", str(e))
            raise e

    @log_decorator
    def _deduplicate_columns(self):
        """
        Ensure all column names are unique by adding suffixes to duplicates.
        """
        # Check for duplicate columns
        cols = self.data.columns.tolist()
        if len(cols) != len(set(cols)):
            self.logger.warning(f"Found {len(cols) - len(set(cols))} duplicate columns")
            
            # Make columns unique manually
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

    # @log_decorator
    # def _clean_quantity_sold(self):
    #     """
    #     This function cleans the 'quantity_sold' column in the dataset.

    #     It performs specific cleaning operations on the 'quantity_sold' column,
    #     including handling consecutive NaN values and removing rows with
    #     excessive NaN values.

    #     Raises:
    #         Exception: If there is an error during the quantity sold cleaning process.
    #     """
        
    #     self.logger.info("Cleaning quantity sold...")
    #     self.out_of_stock_days = 7
    #     self.ignore_auto_of_stock_days = 2

    #     try:            
    #         self.data = self.data.sort_values(["product_code", "date"]).reset_index(
    #             drop=True
    #         )
    #         # Function to count number of zero-sales days in a row
    #         def count_nan_sequences(group):
    #             is_nan = group['quantity_sold'].isna().astype(int)
    #             shifts = is_nan.diff().fillna(0).ne(0).cumsum()
    #             nan_sequence_count = is_nan.groupby(shifts).transform('sum')
    #             group['nan_sequence_count'] = nan_sequence_count
    #             return group

    #         df_copy = self.data.copy()
    #         df = self.data[['product_code', 'date', 'quantity_sold']].copy()
    #         df['quantity_sold'] = df['quantity_sold'].replace(0, np.nan)
    #         # workaround for the upcoming deprecation, find a better way
    #         df = df.groupby('product_code', group_keys=False).apply(count_nan_sequences, include_groups=False)
    #         df['product_code'] = df_copy.product_code
    #         df = df[df.nan_sequence_count < self.out_of_stock_days]

    #         self.data = df[['date', 'product_code']].merge(df_copy,
    #                                                 how='left',
    #                                                 on=['product_code', 'date'])
    #         self.data['quantity_sold'] = self.data['quantity_sold'].fillna(0)
    #         self.logger.info("Cleaning quantity sold completed.")

    #     except Exception as e:
    #         self.logger.error("Error in cleaning of quantity sold: %s", str(e))
    #         raise e

    @log_decorator
    def _clean_quantity_sold(self):
        """
        This function cleans the 'quantity_sold' column by removing long sequences of zeros/NaNs.
        Simplified version that avoids complex groupby operations.
        """
        
        self.logger.info("Cleaning quantity sold...")
        self.out_of_stock_days = 7
    
        try:
            # Sort data
            self.data = self.data.sort_values(["product_code", "date"]).reset_index(drop=True)
            
            # Create a mask for rows to keep
            keep_mask = pd.Series(True, index=self.data.index)
            
            # Process each product group
            for product_code in self.data['product_code'].unique():
                product_mask = self.data['product_code'] == product_code
                product_indices = self.data[product_mask].index
                
                # Get quantity sold for this product
                quantities = self.data.loc[product_indices, 'quantity_sold'].values
                
                # Replace 0 with NaN for counting
                quantities = np.where(quantities == 0, np.nan, quantities)
                
                # Find sequences of NaNs
                is_nan = np.isnan(quantities)
                
                if len(is_nan) == 0:
                    continue
                    
                # Find start and end of NaN sequences
                diff = np.diff(np.concatenate(([0], is_nan.astype(int), [0])))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                # Mark long sequences for removal
                for start, end in zip(starts, ends):
                    sequence_length = end - start
                    if sequence_length >= self.out_of_stock_days:
                        # Mark these indices to be removed
                        seq_indices = product_indices[start:end]
                        keep_mask.loc[seq_indices] = False
            
            # Apply the mask
            original_count = len(self.data)
            self.data = self.data[keep_mask].reset_index(drop=True)
            
            # Fill remaining NaN with 0
            self.data['quantity_sold'] = self.data['quantity_sold'].fillna(0)
            
            self.logger.info(f"Cleaning completed. Removed {original_count - len(self.data)} rows")
            self.logger.info(f"Products after cleaning: {self.data.product_code.nunique()}")
            
        except Exception as e:
            self.logger.error(f"Error in cleaning quantity sold: {str(e)}")
            raise e

    # @log_decorator
    # def _impute_missing_values(self):
    #     """
    #     This function performs missing values imputation based on specified conditions.

    #     It fills missing values in the 'calculated_price' column using the 'product_selling_price'
    #     when 'quantity_sold' is zero and 'calculated_price' is null. Additionally, it fills other
    #     missing values using forward fill method. Ensures that the dataset does not contain infinite
    #     values after imputation.

    #     Raises:
    #         Exception: If an error occurs during missing values imputation.
    #     """

    #     try:
    #         self.logger.info("Starting missing values imputation...")
    #         # Implement missing values imputation based on config
    #         # Condition to fill NaN values in 'best_selling_price'
    #         condition = (
    #             ((self.data["quantity_sold"] == 0) | (self.data["quantity_sold"].isna()))
    #             & (self.data["product_selling_price"].notnull())
    #             & (self.data[self.price_column].isnull())
    #         )

    #         # Fill NaN values based on the condition
    #         self.data.loc[condition, self.price_column] = self.data.loc[
    #             condition, "product_selling_price"
    #         ]

    #         self.data = self.data[self.data[self.price_column] != np.inf]
    #         if self.client_name == 'spar':
                
    #             # Fill missing values for each product with the last known non-null value
    #             self.data["unique_stores_last_7_days"] = self.data.groupby("product_code")[
    #                 "unique_stores_last_7_days"
    #             ].ffill()

    #             self.data["unique_stores_last_30_days"] = self.data.groupby("product_code")[
    #                 "unique_stores_last_30_days"
    #             ].ffill() 
            
    #         self.data = self.data[~(self.data[self.price_column].isnull())]
    #         self.data = self.data[self.data[self.price_column]>=0]
    #         self.data_export = self.data.copy()
    #         self.data.promotion_indicator = self.data.promotion_indicator.fillna(0)
            
    #         self.logger.info("Missing data imputation completed successfully.")
    #         self.logger.info(self.data.product_code.nunique())

    #     except Exception as e:
    #         self.logger.error("Error in missing data imputation: %s", str(e))
    #         raise e

    @log_decorator
    def _impute_missing_values(self):
        """
        This function performs missing values imputation based on specified conditions.
        Simplified version to avoid multidimensional indexing issues.
        """
        
        try:
            self.logger.info("Starting missing values imputation...")
            
            # Make sure we're working with a clean DataFrame
            self.data = self.data.reset_index(drop=True)
            
            # Condition to fill NaN values in price column
            condition = (
                ((self.data["quantity_sold"] == 0) | (self.data["quantity_sold"].isna()))
                & (self.data["product_selling_price"].notnull())
                & (self.data[self.price_column].isnull())
            )
            
            # Convert condition to 1D numpy array to avoid multidimensional issues
            condition_array = condition.values.flatten() if hasattr(condition, 'values') else condition
            
            # Fill NaN values based on the condition
            self.data.loc[condition_array, self.price_column] = self.data.loc[
                condition_array, "product_selling_price"
            ].values
    
            # Remove infinite values
            self.data = self.data[self.data[self.price_column] != np.inf]
            
            # SPAR-specific store count imputation
            if self.client_name == 'spar':
                for col in ['unique_stores_last_7_days', 'unique_stores_last_30_days']:
                    if col in self.data.columns:
                        self.data[col] = self.data.groupby("product_code")[col].ffill()
            
            # Remove rows where price column is still null
            self.data = self.data[~(self.data[self.price_column].isnull())]
            
            # Ensure non-negative prices
            self.data = self.data[self.data[self.price_column] >= 0]
            
            # Store a copy and fill promotion indicator
            self.data_export = self.data.copy()
            self.data['promotion_indicator'] = self.data['promotion_indicator'].fillna(0)
            
            self.logger.info(f"Products after imputation: {self.data.product_code.nunique()}")
            self.logger.info("Missing data imputation completed successfully.")
            
        except Exception as e:
            self.logger.error(f"Error in missing data imputation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    @log_decorator
    def _convert_columns(self):
        """
        This function converts columns in the dataset to the appropriate format.

        It converts the 'date' column to datetime format based on the specified format string.
        Changes 'product_code' to categorical.
        Logs successful column conversions.

        Raises:
            Exception: If there is an error during column conversion.
        """
        try:
            self.logger.info('Columns conversion started.')
            # Attempt to parse dates as "YYYYMMDD"
            self.data['date'] = pd.to_datetime(self.data['date'], format="%Y%m%d")
        except ValueError:
            try:
                # If the above format fails, attempt to parse dates as "YYYY-MM-DD"
                self.data['date'] = pd.to_datetime(self.data['date'], format="%Y-%m-%d")
            except ValueError:
                raise ValueError("Date format not recognized. Please specify the correct format.")

        self.data['product_code'] = self.data['product_code'].astype(str)
        # self.data = self.data[self.data['date'] <= '2024-07-01']
        self.logger.info('Columns conversion completed succesfully.')
        
    @log_decorator
    def _filter_non_negative(self):
        """
        This function filters the dataset to retain only rows with non-negative values.

        It filters rows based on conditions to keep only those with non-negative 'quantity_sold'
        and 'revenue'. Logs successful completion of filtering.

        Raises:
            Exception: If an error occurs during filtering of non-negative values.
        """

        try:
            self.logger.info("Starting filtering of positive values...")
            self.logger.info(f"Start: {self.data.product_code.nunique()}")
            # Keep only rows with non-negative quantity and revenue
            self.data = self.data[(((self.data["quantity_sold"] >= 0) | (self.data["quantity_sold"].isna())) & 
                                   ((self.data[self.revenue] >= 0) | (self.data[self.revenue].isna())))
                                  ].reset_index(drop=True)
            
            self.data.drop(columns=[self.revenue], inplace=True)
            self.logger.info(f"End: {self.data.product_code.nunique()}")
            self.logger.info("Filtering of positive values completed successfully.")
        except Exception as e:
            self.logger.error("Error in filtering of positive values: %s", str(e))
            raise e
            
            
    @log_decorator
    def _filter_product_status(self):
        """
        This function filters the dataset to retain only rows with products that are in 'Actief', 'Afbouw' or 'Goedgekeurd' status.
        The rest of the products are about to be stopped.

        Raises:
            Exception: If an error occurs during filtering active products.
        """

        try:
            self.logger.info("Starting filtering of active products...")
            self.logger.info(f"Start: {self.data.product_code.nunique()}")
            # Keep only rows with active products
            self.product_status_prods = pd.DataFrame(self.data.groupby('product_status')['product_code'].nunique()).reset_index()
            self.data = self.data[self.data["product_status"].isin(['Actief', 'Afbouw', 'Goedgekeurd'])].reset_index(drop=True)
            self.data.drop(columns=["product_status"], inplace=True)
            self.logger.info(f"End: {self.data.product_code.nunique()}")
            self.logger.info("Filtering of active products completed successfully.")
        except Exception as e:
            self.logger.error("Error in filtering of active products: %s", str(e))
            raise e

    @log_decorator
    def _further_preprocessing(self):
        """
        This function performs additional preprocessing based on special features in the configuration.

        If the client's configuration specifies the presence of competitor information,
        update competitor prices in the data accordingly. This involves adjusting prices
        based on content factors for different competitors (e.g., ah_price, jumbo_price).
        Log success or failure messages and update progress accordingly.

        Raises:
            Exception: If there is an error while updating competitor prices.
        """

        
        if self.config["competitor_info"]:
            self.logger.info("Updating competitor prices...")
            for competitor in self.config["competitors"]:
                try:

                    self.data[f"{competitor}_price"] = (
                        self.data[f"{competitor}_price"] * self.data[f"{competitor}_content_factor"]
                    )
                    
                    # Calculate the percentage of null values for each column
                    null_percentage = self.data[f"{competitor}_price"].isnull().sum() / self.data.shape[0] * 100

                    if null_percentage > 59:
                        print(f"{competitor}_price column has {np.round(null_percentage)}% nulls and it will be removed")
                        self.data.drop(columns=[f"{competitor}_price", f"{competitor}_content_factor"], inplace=True)
                
                except Exception as e:
                    self.logger.error(f"Error updating {competitor} prices: {e}")
                
                #self.data.drop(columns=[f"{competitor}_content_factor"], inplace=True)
            self.logger.info("Competitor prices updated.")
        else:
            self.logger.info("Data doesn't include competitor prices.")
            
    @log_decorator
    def _filter_sufficient_price_levels(self):
        """
        This function filters products based on the number of unique price levels.

        It filters products to retain only those with a specified number of unique 'calculated_price'
        values. Logs successful completion of the filtering process.

        Raises:
            Exception: If there is an error during the filtering of price levels.
        """

        try:
            self.logger.info("Starting filter of sufficient price levels...")
            self.logger.info(f"Start: {self.data.product_code.nunique()}")
            # Filter products with sufficient number of price levels based on config
            num_prices = pd.DataFrame(
                self.data.groupby("product_code")[self.price_column].nunique()
            ).reset_index()
            prods_prices = num_prices[num_prices[self.price_column] >= self.filters['sufficient_price_levels']]
            self.data = self.data[
                self.data.product_code.isin(prods_prices.product_code)
            ].reset_index(drop=True)
            self.logger.info(f"End: {self.data.product_code.nunique()}")
            self.logger.info("Filter of sufficient price levels completed successfully.")

        except Exception as e:
            self.logger.error("Error in filter of sufficient price levels: %s", str(e))
            raise e

    @log_decorator
    def _filter_live_products(self):
        """
        This function filters the dataset to include only rows corresponding to live products.

        Determines live products based on the maximum 'date' value per 'product_code'.
        Retains only rows corresponding to live products in the dataset.

        Raises:
            Exception: If there is an error during the filtering of live products.
        """

        try:
            self.logger.info("Starting the retrieval of live products...")
            self.logger.info(f"Start: {self.data.product_code.nunique()}")
            
            self.cutoff_dates = self.data.date.max() - pd.to_timedelta(self.price_grid_freq - 1, unit='d')
            self.date_range = pd.date_range(start=self.cutoff_dates, end=self.data.date.max())
            self.prods_max_dates = self.data[self.data.date.isin(self.date_range)].product_code.unique()
            self.data = self.data[self.data.product_code.isin(self.prods_max_dates)].reset_index(drop=True)

            self.logger.info("Retrieval of live products completed successfully...")
            self.logger.info(self.data.product_code.nunique())
            self.logger.info(f"End: {self.data.product_code.nunique()}")
        except Exception as e:
            self.logger.error("Error in retrieval of live products: %s", str(e))
            raise e
    
    @log_decorator
    def _filter_low_selling_products(self):
        """
        This function filters out products that are low-selling.

        Raises:
            Exception: If there is an error during the filtering of live products.
        """

        try:
            self.logger.info("Starting the removal of low-selling products...")
            self.logger.info(f"Start: {self.data.product_code.nunique()}")
            
            # Calculate the median quantity sold per product code
            self.median_sales = self.data.groupby('product_code')['quantity_sold'].median()

            # Identify low-selling products based on the median
            self.low_selling_products = self.median_sales[self.median_sales < self.filters['min_sales']]
            # Remove low-selling products from the original data
            self.data = self.data[~self.data['product_code'].isin(self.low_selling_products.index)]

            self.logger.info("Removal of low-selling products completed successfully...")
            self.logger.info(self.data.product_code.nunique())
            self.logger.info(f"End: {self.data.product_code.nunique()}")
        except Exception as e:
            self.logger.error("Error in removal of low-selling products: %s", str(e))
            raise e
   
    # @log_decorator
    # def _calculate_cpi(self):
    #     """
    #     This function calculates Consumer Price Index (CPI) and adjust prices accordingly.

    #     Calculates adjusted CPI values and price adjustments based on specified conditions
    #     and configurations. Logs successful completion of CPI calculations.

    #     Raises:
    #         Exception: If there is an error during CPI calculation or price adjustment.
    #     """


    #     try:
    #         self.logger.info("Starting the calculation of CPI...")

    #         self.logger.info(f"CPI data exists: {hasattr(self, 'cpi')}")
    #         if hasattr(self, 'cpi'):
    #             self.logger.info(f"CPI columns: {self.cpi.columns.tolist()}")
    #             self.logger.info(f"CPI shape: {self.cpi.shape}")
                    
    #         df = self.data.sort_values(["product_code", "date"]).reset_index(drop=True)
            
    #         if self.client_name == 'spar':
    #             df["category_name"] = df[self.category].map(self.cat_mapper)
    #             daily_cpi_data = self.cpi_cat.groupby("category_name").apply(interpolate_cpi).reset_index(drop=True)
    #             df = pd.merge(df, daily_cpi_data, on=["date", "category_name"])

    #         daily_cpi_data = interpolate_cpi(self.cpi)
    #         daily_cpi_data['month_year'] = daily_cpi_data['date'].dt.to_period('M')
    #         df = pd.merge(df, daily_cpi_data, on="date")

    #         # Get the base_year_cpi for each product_code for both cpi and cpi_per_category
    #         base_cpi = df.sort_values('date').groupby('product_code').first().reset_index()
    #         if self.client_name == 'spar':
    #             base_cpi = base_cpi[['product_code', 'cpi', 'cpi_per_category']].rename(
    #                 columns={'cpi': 'base_year_cpi',
    #                          'cpi_per_category': 'base_year_cpi_per_category'})
                
    #         else:
    #             base_cpi = base_cpi[['product_code', 'cpi']].rename(
    #                 columns={'cpi': 'base_year_cpi'})

    #         df = df.merge(base_cpi, on='product_code')

    #         # Calculate the adjusted prices
    #         df['adjusted_price'] = df[self.price_column] * (df['base_year_cpi'] / df['cpi'])
    #         #df['adjusted_price_category'] = df[self.price_column] * (df['base_year_cpi_per_category'] / df['cpi_per_category'])
    #         if self.client_name == 'spar':
    #             df = df.drop(columns=['base_year_cpi', 'base_year_cpi_per_category', 'month_year'])
    #         else:
    #             df = df.drop(columns=['base_year_cpi', 'month_year'])
                

    #         self.data = df.sort_values(by=["date"]).reset_index(drop=True)
    #         self.logger.info("Calculation of CPI completed successfully...")

    #     except Exception as e:
    #         self.logger.error("Error in calculation of CPI: %s", str(e))
    #         raise e

    @log_decorator
    def _calculate_cpi(self):
        """
        This function calculates Consumer Price Index (CPI) and adjust prices accordingly.
        Simplified version for Hoogvliet.
        """
        try:
            self.logger.info("Starting the calculation of CPI...")
            
            # Check if CPI data exists
            if not hasattr(self, 'cpi') or self.cpi is None or len(self.cpi) == 0:
                self.logger.error("CPI data not available. Skipping CPI adjustment.")
                # Create dummy adjusted_price if needed
                self.data['adjusted_price'] = self.data[self.price_column]
                return
            
            self.logger.info(f"CPI data shape: {self.cpi.shape}")
            
            df = self.data.sort_values(["product_code", "date"]).reset_index(drop=True)
            
            # Get daily CPI data
            daily_cpi_data = interpolate_cpi(self.cpi)
            daily_cpi_data['month_year'] = daily_cpi_data['date'].dt.to_period('M')
            df = pd.merge(df, daily_cpi_data, on="date", how="left")
    
            # Get the base_year_cpi for each product_code
            base_cpi = df.sort_values('date').groupby('product_code').first().reset_index()
            
            # Check if 'cpi' column exists after merge
            if 'cpi' not in df.columns:
                self.logger.error("CPI column not found after merge. Skipping CPI adjustment.")
                self.data['adjusted_price'] = self.data[self.price_column]
                return
                
            base_cpi = base_cpi[['product_code', 'cpi']].rename(columns={'cpi': 'base_year_cpi'})
            df = df.merge(base_cpi, on='product_code')
    
            # Calculate the adjusted prices
            df['adjusted_price'] = df[self.price_column] * (df['base_year_cpi'] / df['cpi'])
            
            # Drop temporary columns
            df = df.drop(columns=['base_year_cpi', 'month_year'], errors='ignore')
            
            self.data = df.sort_values(by=["date"]).reset_index(drop=True)
            self.logger.info("Calculation of CPI completed successfully!")
    
        except Exception as e:
            self.logger.error(f"Error in calculation of CPI: {str(e)}")
            self.logger.info("Creating adjusted_price as copy of price column as fallback")
            self.data['adjusted_price'] = self.data[self.price_column]
        
    @log_decorator
    def _remove_outliers(self):
        
        try:
            self.logger.info(f"Removing outliers in {self.price_column}...")
            
            # Apply the remove_outliers function to each group
            self.data = self.data.groupby('product_code').apply(lambda group: remove_outliers(group, self.price_column)).reset_index(drop=True)
            self.data.sort_values(by=["product_code", "date"], inplace=True)
            self.data.reset_index(drop=True, inplace=True)

            self.logger.info(f"Removed outliers successfully in {self.price_column}...")
        except Exception as e:
            self.logger.error("Error in removing outliers: %s", str(e))
            raise e
    
    @log_decorator
    def _remove_missing_values(self):
        
        try:
            self.logger.info(f"Removing missing values in category columns...")
            category_columns = [col for col in self.data.columns if 'category' in col]
            self.data = self.data.dropna(subset=category_columns) 
            self.data.sort_values(by=["product_code", "date"], inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            self.logger.info(self.data.product_code.nunique())
            self.logger.info(f"Removed missing values...")
        except Exception as e:
            self.logger.error("Error in removing missing values: %s", str(e))
            raise e
       
    @log_decorator
    def _create_price_grid(self):
        
        if self.data.index.name == 'date':
            self.data.reset_index(inplace = True)
        
            
        # Extract the last x prices for each product_code
        self.last_n_prices = (self.data.sort_values(by=["product_code", "date"])
                             .groupby("product_code")
                             .tail(self.price_grid_freq)).reset_index(drop=True)

        price_dict = {}

        for _, row in self.last_n_prices.iterrows():
            
            self.num_prices = abs(self.min_grid_perc) + abs(self.max_grid_perc) + 1  # The number of prices, this will be 26 for this range
            product_code = row['product_code']
            base_price = row[self.price_column]

            # Generate list of prices using configurable range and number of points
            prices = np.linspace((1 + self.min_grid_perc / 100) * base_price, 
                                 (1 + self.max_grid_perc / 100) * base_price, 
                                 self.num_prices).tolist()
            
            
#             product_code = row['product_code']
#             base_price = row[self.price_column]                

#             # Generate list of 21 prices from 90% to 110% of the base price
#             prices = np.linspace(0.925 * base_price, 1.075 * base_price, 21).tolist()
# #             # prices = [round(x, 3) for x in prices]
            price_dict.setdefault(product_code, []).append(prices)
            
        self.price_grid = price_dict

    @log_decorator
    def _drop_columns(self):
        """
        This function drops specified columns from the dataset.

        It removes 'category_code_level1', 'category_code_level2', 
        'category_code_level3' columns.
        Logs successful column drops.

        Raises:
            KeyError: If any of the columns to drop are not found in the dataset.
        """
        if 'category_code_level1' in self.data.columns:
            columns_to_drop = ['category_code_level1', 'category_code_level2', 'category_code_level3', 'category_name']
            self.data.drop(columns=columns_to_drop, inplace=True, errors='raise')
        if self.client_name=='spar':
            if self.price_column == 'product_selling_price':
                self.data.drop(columns=['calculated_price', 'calculated_price_after_discount'], inplace=True)

    @log_decorator
    def _resample_data(self):
        
        """
        This function resamples the data from daily to weekly level.

        Raises:
            KeyError: If there is a mistake within the process.
        """
        
        pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count())
        if self.frequency == 'weekly':
            
            # Remove duplicate entries based on 'product_code' and 'date'
            self.data = self.data.drop_duplicates(subset=['product_code', 'date'], keep='last')

#             # Identify price changes
#             self.data['price_change'] = (self.data.groupby(['product_code'])['product_selling_price'].diff().fillna(0) != 0).astype(int)

#             # Identify groups of consecutive days without price change
#             self.data['last_price_change_group'] = self.data.groupby('product_code')['price_change'].cumsum()

            # Set 'date' as index to use in the calculation
            self.data.set_index('date', inplace=True)
#             days_since_last_change = pd.Series(self.data.groupby(['product_code', 'last_price_change_group']).parallel_apply(
#                 lambda x: (x.index - x.index.min()).days  # Convert TimedeltaIndex to days
#             ).explode().astype(int).reset_index(level=[0], drop=True).reset_index(drop=True))

#             self.data['days_since_last_change'] = days_since_last_change.values
            # Step 1: Create a column to flag where the price changes (1 if it changed, 0 otherwise)
            self.data['price_change'] = self.data.groupby('product_code')['product_selling_price'].diff().fillna(0) != 0

            # Step 2: Create a group for each period between price changes (cumsum of price change flags)
            self.data['price_change_group'] = self.data.groupby('product_code')['price_change'].cumsum()

            # Step 3: For each product and each price change group, calculate the days since the last change
            self.data['days_since_last_change'] = self.data.groupby(['product_code', 'price_change_group']).cumcount()
            self.data['weeks_since_last_change'] = (self.data['days_since_last_change'] // 7).astype(int)
            
            # Resample data by weeks starting from Monday and aggregate
            self.weekly_data = self.data.groupby(['product_code']).resample('W-MON', label='left', closed='left').agg({
                'product_selling_price': 'last',  # Take the last value of the week, not the mean
                'quantity_sold': 'sum',           # Sum quantity sold over the week
                'brand_name': 'last',
                'category_name_level1': 'last',
                'category_name_level2': 'last',
                'category_name_level3': 'last',
                'days_since_last_change': 'last',
                'unique_stores_last_7_days': 'last',
                'unique_stores_last_30_days': 'last',
                'product_status': 'last',
                'promotion_indicator': 'sum',
                'ah_price': 'last',
                'plus_price': 'last',
                'cpi': 'last', 
                'cpi_per_category': 'last', 
                'adjusted_price': 'last', 
                'adjusted_price_category': 'last'

            }).reset_index()
            
            self.weekly_data = self.weekly_data.groupby('product_code').apply(lambda group: group.ffill()).reset_index(drop=True)

#             days_since_last_change = self.data.groupby(['product_code']).resample('W-MON', label='left', closed='left').apply(custom_aggregation).reset_index(drop=True)

#             #Add the custom aggregated column to the weekly data
#             self.weekly_data['days_since_last_change'] = days_since_last_change
        else:
            pass

        
    def preprocess_data(self, overview=True):
        """
        This method initiates the data preprocessing pipeline by calling various private methods sequentially.
        Loads additional data, filters and converts columns, performs data cleaning and imputation,
        and filters out specific data subsets.
        Logs progress at each step and generates an overview report at the end.
        """

        self.logger.info("Starting data preprocessing...")
        self._load_other_data()
        self._filter_columns()
        self._convert_columns()
        self._deduplicate_columns()
        if self.filters['filter_product_status']:
            self._filter_product_status()
        if self.filters['filter_live_products']:
            self._filter_live_products()
        if self.filters['filter_sufficient_price_levels']:
            self._filter_sufficient_price_levels()
        self._remove_missing_values()
        # self._clean_quantity_sold()
        # self._impute_missing_values()
        # self._filter_non_negative()
        self._remove_outliers()
        if self.filters['filter_low_selling_products']:
            self._filter_low_selling_products()
        self._further_preprocessing()
        if self.filters['cpi']:
            self._calculate_cpi()
        self.data = self.data.reset_index(drop=True)
        self._drop_columns()
        if self.frequency == 'weekly':
            self._resample_data()
        self.pbar.close()
        if overview:
            self.generate_overview_report()
        self.logger.info("Data preprocessing is finished!")
  
    def compute_statistics(self):
        """
        This method computes various statistical summaries of the processed data.
        Calculates the number of unique categories, unique dates, min/max/median prices,
        and the percentage of null values in a specific column.
        Also computes the count of unique products per category.
        Returns these statistics as a DataFrame.
        """

        # Calculate statistics
        num_categories = self.data[self.category].nunique()
        #unique_dates = self.data["date"].nunique()
        num_prods = self.data.product_code.nunique()
        #num_low_selling = self.low_selling_products.index.nunique()
        min_price = np.round(self.data[self.price_column].min(), 2)
        max_price = np.round(self.data[self.price_column].max(), 2)
        median_price = np.round(self.data[self.price_column].median(), 2)
        percentage_nulls_price = np.round(
            sum(pd.isnull(self.data["product_selling_price"]) / self.data.shape[0])
            * 100,
            2,
        )

        category_prod_counts = (
            self.data.groupby(self.category)["product_code"]
            .nunique()
            .reset_index()
            .rename(columns={"product_code": "num_products"})
            .sort_values("num_products", ascending=False)
        )

        # Store statistics in a DataFrame
        statistics_table = pd.DataFrame(
            {
                "Number of Categories": [num_categories],
                "Number of products": [num_prods],
               # "Number of low selling products": [num_low_selling],
                "Min Price": [min_price],
                "Max Price": [max_price],
                "Median Price": [median_price],
                "Percentage nulls price": [percentage_nulls_price],
            }
        )

        return (statistics_table, category_prod_counts)
    
    @log_decorator
    def generate_overview_report(self):
        """
        This function generates an interactive overview report using Plotly.
        Calls `compute_statistics` to gather data for creating tables.
        Creates two interactive tables (`fig1` and `fig2`) containing statistical summaries.
        Displays the tables using Plotly.
        """
       
        tables = self.compute_statistics()
        values_overview = [tables[0][column] for column in tables[0].columns.tolist()]

        fig = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0.01,
            specs=[[{"type": "table"}], [{"type": "table"}]]
        )

        fig.add_trace(
            go.Table(
                header=dict(
                    values=tables[0].columns,
                    line_color="darkslategray",
                    fill_color="lightskyblue",
                    align="left",
                ),
                cells=dict(
                    values=values_overview,
                    line_color="darkslategray",
                    fill_color="lightcyan",
                    align="left",
                ),
            ),
            row=1, col=1
        )

        if self.client_name == 'spar':
            title = "Column insights and number of products per category for priceline " + self.priceline_name
        else:
            title = "Column insights and number of products per category"

        fig.add_trace(
            go.Table(
                header=dict(
                    values=tables[1].columns,
                    line_color="darkslategray",
                    fill_color="lightskyblue",
                    align="left",
                ),
                cells=dict(
                    values=[
                        tables[1].iloc[:, 0],
                        tables[1].iloc[:, 1],
                    ],
                    line_color="darkslategray",
                    fill_color="lightcyan",
                    align="left",
                ),
            ),
            row=2, col=1
        )

        fig.update_layout(
            autosize=True,
            height=500,
            title=dict(
                text=title,
                x=0.5,
                y=0.95,
                font=dict(size=18)
            ),
            margin=dict(l=10, r=10, t=50, b=50)
        )

        fig.show()

#         if self.client_name == 'spar':
#         # Create pie chart for number of products per product status
#             fig3 = px.pie(self.product_status_prods, values='product_code', names='product_status')
#             fig3.update_layout(
#                 margin=dict(t=50, l=50, r=50, b=0.5),
#                 title=dict(
#                     text='Number of products per product status',
#                     x=0.5,
#                     y=0.99,
#                     font=dict(size=18)
#                 )
#             )
        
#             fig3.show()
        
    def save_to_s3(self):
        
        if self.frequency == 'daily':
            self.data.to_parquet(self.data_s3_dir)
        else:
            self.weekly_data.to_parquet(self.data_s3_dir)
        file = os.path.join(self.bucket, self.config_s3['data_prefix'], 'price_grid.pkl')
        pickle_data = pickle.dumps(self.price_grid)
        with s3fs.S3FileSystem().open(file, 'wb') as f:
            f.write(pickle_data)
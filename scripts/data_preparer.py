import os
import sys
import time
import boto3
import s3fs
import warnings
import logging
import pandas as pd
import polars as pl
from tqdm import tqdm
from datetime import datetime, timedelta
from scripts.group_by_dynamic_polars import group_by_dynamic_right_aligned

warnings.filterwarnings("ignore")

def get_logger():
    
    current_time = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
    log_file = os.path.join("logs", f"data_loader_{current_time}.log")
  
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

class DataPreparer:
    
    """
        This class is responsible for aggregating the initial data from a store/receipt level to a product/date level, based on a specified client configuration.

        Attributes:
            config_file (str): Path to the configuration file.
            client_name (str): Name of the client for which data will be processed.
    """
        
    def __init__(self, logger=None):
        
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger
            
        self.cutoff_date = pl.datetime(2024, 8, 1)
        
        for priceline in ['Enjoy', 'Buurt',  'City', 'Highway', 'Neutraal']:
            self.priceline = priceline
            self._load_data()
            self._fix_weighted_products()
            self._select_data()
            self._aggregate_data()
            self._save_to_s3()

    @log_decorator
    def _load_data(self):
        self.data = pl.read_parquet(f"downloads/{self.priceline}.parquet")
 
    @log_decorator
    def _fix_weighted_products(self):

        self.data = self.data.with_columns(
            pl.when(pl.col('weight_sold') > 0)
            .then(pl.col('calculated_price') * pl.col('quantity_sold') / pl.col('weight_sold'))
            .otherwise(pl.col('product_selling_price'))
            .round(2)
            .alias('calculated_price')
        )
    
    @log_decorator
    def _aggregate_data(self):

        df = self.data.with_columns(pl.col('date_code').str.strptime(pl.Date, format='%Y%m%d').alias('date').cast(pl.Datetime("us")))
        df = df.unique(subset=['date', 'product_code', 'store_code'])
        df = df.sort(by=["product_code", "date"])

        self.aggregate_days = [7, 30]
        
        for days in self.aggregate_days:
            # Group_by_dynamic to create a rolling window of 7 days within each product_code group
            result = (
                group_by_dynamic_right_aligned(
                    df,
                    index_column="date",
                    every=timedelta(days=1), 
                    period=timedelta(days=days),
                    by="product_code", 
                )
                .agg([
                    pl.col("store_code").n_unique().alias(f"unique_stores_last_{days}_days")
                ])
            )

            result = result.with_columns(pl.col("product_code"))
            result = result.sort(by=["product_code", "date"])
            result = result.with_columns(pl.col("date").cast(pl.Datetime("us")))
            df = df.join(result, on=["product_code", "date"], how="left")

        self.data = df.group_by(['product_code', 'date']).agg([
            pl.col('quantity_sold').sum(),
            pl.col('unique_stores_last_7_days').first(),
            pl.col('unique_stores_last_30_days').first(),
            pl.col('product_selling_price').first(),
            pl.col('product_category_code_level1').first(),
            pl.col('product_category_code_level2').first(),
            pl.col('product_category_code_level3').first(),
            pl.col('revenue_before_discount_incl_vat').sum(),
            pl.col('product_status').first(),
            pl.col('promo_ind').first(),
            pl.col('calculated_price').mean(),
            pl.col('calculated_price_after_discount').mean(),
            pl.col('ah_price').first(),
            pl.col('plus_price').first(),
            pl.col('plus_content_factor').first(),
            pl.col('ah_content_factor').first(),
            pl.col('brand_name').first()
        ])
       
    @log_decorator
    def _select_data(self):
                

        df = self.data.drop_nulls(subset=['product_selling_price'])

        df = df.with_columns(
            pl.col('date_code').str.strptime(pl.Date, format='%Y%m%d').alias('date')
        )

        df = df.filter(pl.col("date") <= self.cutoff_date)
        df = df.sort("date")
        df = df.filter(pl.col("quantity_sold") > 0)

        df = df.group_by(['product_code', 'date', 'store_code']).agg([
            pl.col('product_selling_price').mean().alias('product_selling_price_mean'),
            pl.col('calculated_price').mean().alias('calculated_price_mean'),
            pl.col('quantity_sold').sum().alias('quantity_sold_sum')
        ])

        df = df.with_columns([
            pl.col('product_selling_price_mean').shift(1).over(['product_code', 'store_code']).alias('product_selling_price_shifted'),
            pl.col('calculated_price_mean').shift(1).over(['product_code', 'store_code']).alias('calculated_price_shifted')
        ])

        df = df.with_columns([
            ((pl.col('product_selling_price_mean') - pl.col('product_selling_price_shifted')) * 100).round(2).alias('product_selling_price_change'),
            ((pl.col('calculated_price_mean') - pl.col('calculated_price_shifted')) * 100).round(2).alias('calculated_price_price_change')
        ])

        df = df.filter(pl.col('product_selling_price_change') != 0)

        selection = df.select([
                    pl.col('product_code'),
                    pl.col('store_code'),
                    pl.col('product_selling_price_change'),
                    pl.col('calculated_price_price_change')
                ])
    
        # Calculate correlation between product selling price and calculated price when product selling price has changed
        # Select high correlations > these stores follow the set product prices by Spar
        product_store = []
        grouped = selection.group_by(['product_code', 'store_code'], maintain_order=True)
        for (product_code, store_code), group in tqdm(grouped):
            if group.height > 1:
                corr = group.select([
                    pl.col('product_selling_price_change'),
                    pl.col('calculated_price_price_change')
                ]).corr()

                if corr[0, 1] > 0.9:
                    product_store.append(f"{product_code}_{int(store_code)}")
            
        self.data = self.data.with_columns((pl.col('product_code') + "_" + pl.col('store_code').cast(pl.Int32).cast(pl.Utf8)).alias('product_store_combined'))
        self.data = self.data.filter(pl.col('product_store_combined').is_in(product_store)).drop('product_store_combined')
        
    @log_decorator
    def _save_to_s3(self):
        
        self.data_s3_dir = "s3://prime-rel-ml/data-analytics/price-elasticity/spar/data/"
        
        fs = s3fs.S3FileSystem()
        with fs.open(f"{self.data_s3_dir}{self.priceline}_aggregated.parquet", 'wb') as f:
            self.data.write_parquet(f)
    
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import curve_fit
import glob
import logging
from pandarallel import pandarallel
import datetime
from datetime import datetime
from datetime import timedelta
import warnings
from tqdm import tqdm
import os
import boto3
import yaml
import time
import plotly.graph_objects as go
import argparse
from .functions import *

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


def get_logger():
    
    os.makedirs('logs', exist_ok=True)
    current_time = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
    log_file = os.path.join('logs', f"price_elasticity_{current_time}.log")
  
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

class Price_Elasticity:
    """
    This class calculates and analyzes price elasticities for products. It manages the data processing, 
    feature engineering, model estimations and calculation of elasticities.
    """

    def __init__(self, config_file, client_name=None, priceline_name=None, logger=None, frequency = 'daily'):
        """
        The class is being initialized with the provided configuration file and client name. 
        It loads the configuration, sets up the model, and begins the process of calculating elasticities.
        """
        
        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger
            
        # Initialize progress bar variables
        self.frequency = frequency
        self.total_tasks = 5  # Total number of tasks/functions
        self.completed_tasks = 0  # Counter for completed tasks
        self.pbar = tqdm(total=self.total_tasks, desc="Overall Progress")
        self._load_config(config_file)
        self.client_name = self.config['client_name']

        self.price_column = self.config['price_column']
        self.config_s3 = self.config['s3_dir']
        self.bucket = os.path.join(self.config_s3['s3'], self.config_s3['bucket'])
        if self.client_name == 'spar':
            self.priceline_name = self.config['priceline_name']
    
        self.category_level = self.config["category"]
        
        if self.category_level == 'brand_name':
            self.name = 'brand'
        else:
            self.name = 'category'
            
        if self.client_name == 'spar':
            
            if self.frequency == 'daily':
                self.estimations_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'] + self.priceline_name + '_' + self.config_s3['estimations'])
                self.output_to_s3 = os.path.join(self.bucket, self.config_s3['output_prefix'], self.priceline_name + '_' + self.name + '_' + self.config_s3['elasticities'])
                self.data_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'] + self.priceline_name + '_' + 'preprocessed_data.parquet')
            else:
                self.estimations_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'] + self.priceline_name + '_' + self.config_s3['estimations_weekly'])
                self.output_to_s3 = os.path.join(self.bucket, self.config_s3['output_prefix'], self.priceline_name + '_' + self.name + '_' + self.config_s3['elasticities_weekly'])
                self.data_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'] + self.priceline_name + '_' + 'preprocessed_data_weekly.parquet')
                
        else:
            
            if self.frequency == 'daily':
                self.data_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'], 'preprocessed_data.parquet')
                self.estimations_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'], self.config_s3['estimations'])
                self.output_to_s3 = os.path.join(self.bucket, self.config_s3['output_prefix'] + '_'  + self.name + '_' +  self.config_s3['elasticities'])
            else:
                self.data_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'], 'preprocessed_data_weekly.parquet')
                self.estimations_from_s3 = os.path.join(self.bucket, self.config_s3['data_prefix'], self.config_s3['estimations_weekly'])
                self.output_to_s3 = os.path.join(self.bucket, self.config_s3['output_prefix'] + '_'  + self.name + '_' +  self.config_s3['elasticities_weekly'])
            

        
        self.num_products = self.config["prods_per_category"]
        self.get_data_from_s3()
        self.run_process()
        # self.save_to_s3()
        # self.save_to_s3_for_dashboards()

    def _update_progress(self, pbar):
        """Update overall progress and display progress bar."""
        self.completed_tasks += 1
        self.pbar.update(1)

    @log_decorator
    def _load_config(self, config_file):
        """
        This function attempts to load configuration settings from the specified YAML configuration file.
        It logs success or failure messages and updates progress accordingly.
        """

        try:
            self.logger.info("Loading configuration...")
            path = os.path.join(config_file)

            with open(path, "r") as stream:
                self.config = yaml.safe_load(stream)
            self.logger.info("Configuration loaded successfully.")
        except Exception as e:
            self.logger.error("Error loading configuration: %s", str(e))
            raise e
           
    @log_decorator
    def get_data_from_s3(self):

        try:
            self.logger.info("Data extraction from s3 has started...")
            self.preprocessed_data = pd.read_parquet(self.data_from_s3)
            self.estimations = pd.read_parquet(self.estimations_from_s3)
            self.estimations = self.estimations.rename(
                columns={"predicted_quantity": "predict_sales"}
            )
            
            self.logger.info("Data extraction from s3 has been completed...")
        except Exception as e:
            self.logger.error("Error splitting the data: %s", str(e))
            raise e
        
    # @staticmethod
    # def richards_growth(x, A, K, B, k):
    #     """
    #     Implements the Richards growth model equation.

    #     Parameters:
    #     x : The input data, typically representing time or a similar independent variable.
    #     A : The lower asymptote of the curve.
    #     K : The upper asymptote of the curve.
    #     B : A shape parameter that affects the curve's symmetry.
    #     k : float
    #         The growth rate or steepness of the curve.

    #     Returns:
    #         The values of the Richards growth model evaluated at each point in x.
    #     """

    #     equation = A + (K - A) / (1 + np.exp(-k * x)) ** (1 / B)
    #     return equation

    @staticmethod
    def log_log_demand(price, a, b):
        """
        Log-log demand curve:
            log(Q) = a + b*log(P)  =>  Q = exp(a) * P^b
        """
        price = np.asarray(price, dtype=float)
        return np.exp(a) * np.power(price, b)

#     @staticmethod
#     def demand_curve(group):
#         """
#         Fits a Richards growth curve to the given group's data and returns the fitted curve.

#         Parameters:
#         group : DataFrame
#             A DataFrame containing e.g 'calculated_price_calc' and 'predict_sales' columns, 
#             representing the adjusted prices and corresponding predicted sales.

#         Returns:
#         DataFrame
#             A DataFrame containing the original product code, the adjusted prices, and the 
#             sales predicted by the fitted Richards growth curve. Returns an empty DataFrame 
#             in case of an error during curve fitting.
#         """
#         if group.empty:
#             print("Empty group passed to demand_curve.")
#             return pd.DataFrame()

#         initial_guess = [
#             np.max(group["predict_sales"]),
#             1,
#             np.min(group[f"product_selling_price_calc"]),
#             0.5,
#         ]
        
#         # Suppress warnings
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')

#             try:
#                 # Curve fitting
#                 params, _ = curve_fit(
#                     Price_Elasticity.richards_growth,
#                     group[f"product_selling_price_calc"],
#                     group["predict_sales"],
#                     p0=initial_guess,
#                     maxfev=10000,
#                 )

#                 # Calculate the curve fit using the estimated parameters
#                 curve_fit_data = Price_Elasticity.richards_growth(
#                     group[f"product_selling_price_calc"], *params
#                 )
#                 # **Change 1: Added a Named Function Instead of Lambda**
# #                 def fitted_curve(x):
# #                     return PriceElasticity.richards_growth(x, *params)

# #                 # **Change 2: Use the named function to calculate curve_fit_data**
# #                 # Calculate the curve fit using the estimated parameters
# #                 curve_fit_data = fitted_curve(group["calculated_price_after_discount_calc"])

#                 if (curve_fit_data.var() >= 0) & (curve_fit_data.var() < 1):

#                     # Curve fitting
#                     params, _ = curve_fit(
#                         Price_Elasticity.richards_growth,
#                         group[f"product_selling_price_calc"],
#                         group["predict_sales"],
#                         maxfev=100000,
#                     )

#                     # Calculate the curve fit using the estimated parameters
#                     curve_fit_data = Price_Elasticity.richards_growth(
#                         group[f"product_selling_price_calc"], *params
#                     )
#                 # Add the data to the result DataFrame
#                 result_data = pd.DataFrame(
#                     {
#                         "product_code_elasticity": group["product_code"].values[0],
#                         "price": group[f"product_selling_price_calc"],
#                         "sales": curve_fit_data,
#                     }
#                 )

#                 return result_data
#             except Exception as e:
#                 # self.logger.info(f"Error processing product {group['product_code'].values[0]}: {e}")
#                 return pd.DataFrame()

    @staticmethod
    def demand_curve(group):
        """
        Fits a log-log regression (power-law) demand curve:
            log(Q) = a + b*log(P)
        Returns a DataFrame with fitted sales for the group's prices.
        """
        if group.empty:
            print("Empty group passed to demand_curve.")
            return pd.DataFrame()
    
        # Keep only valid rows for log transforms
        g = group.copy()
        g = g[(g["predict_sales"] > 0) & (g["product_selling_price_calc"] > 0)]
        if g.empty:
            return pd.DataFrame()
    
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
    
            try:
                x = np.log(g["product_selling_price_calc"].astype(float).values)
                y = np.log(g["predict_sales"].astype(float).values)
    
                # Linear regression in log space: y = a + b*x
                # np.polyfit returns [b, a]
                b, a = np.polyfit(x, y, 1)
    
                # Predict for original group's prices (only where price > 0)
                prices_all = group["product_selling_price_calc"].astype(float).values
                curve_fit_data = np.full(shape=len(prices_all), fill_value=np.nan, dtype=float)
    
                mask_price_pos = prices_all > 0
                curve_fit_data[mask_price_pos] = np.exp(a + b * np.log(prices_all[mask_price_pos]))
    
                result_data = pd.DataFrame(
                    {
                        "product_code_elasticity": group["product_code"].values[0],
                        "price": group["product_selling_price_calc"],
                        "sales": curve_fit_data,
                    }
                )
    
                # If you want: drop rows where we couldn't compute (price<=0)
                result_data = result_data.dropna(subset=["sales"])
    
                return result_data
    
            except Exception:
                return pd.DataFrame()

    @log_decorator
    def calculate_demand_curve(self):
        """
        This method calculates the demand curve for each product by fitting a Richards growth function to the sales data.
        It uses parallel processing to speed up the computation across multiple product groups.
        """
        try:
            self.logger.info("Calculating the demand curve...")
            pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count())
            start_time = time.time()

            # Group by 'product_code' and apply the curve fitting function to each group
            self.curve_results = (
                self.estimations.groupby(["product_code", "sequence_number"], observed=False)
                .parallel_apply(lambda group: Price_Elasticity.demand_curve(group))
                .reset_index().drop(['product_code', 'level_2'], axis=1)
            )

            elapsed_time = (time.time() - start_time) / 60
            self.logger.info(f"\nParallelization completed in {elapsed_time:.2f} minutes")

            self.curve_results = self.curve_results.sort_values(
                by=["product_code_elasticity", "sequence_number", "price"]
            ).reset_index(drop=True)
        
            self.curve_merged = pd.merge(self.curve_results,
            self.estimations[["product_code", "sequence_number"]],
            left_on=["product_code_elasticity", "sequence_number"],
            right_on=["product_code", "sequence_number"],
            how="inner")
            
            # TO DO: FIND OUT WHY THERE ARE DUPLICATES IN SELF.curve_results
            self.curve_merged = self.curve_merged.drop_duplicates().drop(
                "product_code", axis=1
            )
            self.curve_merged = self.curve_merged.sort_values(
                by=["product_code_elasticity", "sequence_number", "price"]
            ).reset_index(drop=True)
            self.logger.info("Demand curve is calculated successfully.")
        except Exception as e:
            self.logger.info("Error calculating the demand curve: %s", str(e))
            raise e

    # @log_decorator
    # def calculate_demand_curve_product(self, product):
    #     """
    #     This method calculates the demand curve for a single product and plots it using Plotly Express.

    #     Parameters:
    #     - product (str): The product code for which the demand curve is to be calculated.

    #     Steps:
    #     1. Extract data for the specified product.
    #     2. Perform curve fitting using the Richards growth function.
    #     3. Plot the original sales data and the fitted demand curve using Plotly Express.
    #     """
    #     try:
    #         self.logger.info("Calculating demand curve for one product...")
            
    #         product = self.estimations[self.estimations["product_code"] == product].reset_index(drop=True)
            
    #         price = product[f"product_selling_price_calc"]
    #         quantity_sold = product.predict_sales
    #         initial_guess = [np.max(quantity_sold), 1, np.min(price), 0.5]

    #         # Curve fitting
    #         params, _ = curve_fit(
    #             Price_Elasticity.richards_growth,
    #             price,
    #             quantity_sold,
    #             p0=initial_guess,
    #             maxfev=100000,
    #         )

    #         # Calculate the curve fit using the estimated parameters
    #         curve_fit_data = Price_Elasticity.richards_growth(price, *params)

    #         if (curve_fit_data.var() >= 0) & (curve_fit_data.var() < 1):
    #             # Curve fitting
    #             params, _ = curve_fit(
    #                 Price_Elasticity.richards_growth, price, quantity_sold, maxfev=100000
    #             )

    #             # Calculate the curve fit using the estimated parameters
    #             curve_fit_data = Price_Elasticity.richards_growth(price, *params)
                
    #         curve_fit_df = {'Price': price, 'Quantity Sold': curve_fit_data}

    #         # Plot the original data
    #         fig = px.scatter(
    #             product,
    #             x=self.price_column + "_calc",
    #             y="predict_sales",
    #             labels={"Quantity Sold": "Quantity Sold"},
    #             title="Original Data vs Curve Fit",
    #         )
            
    #         # Add the curve fit data
    #         fig.add_scatter(
    #             x=curve_fit_df["Price"],
    #             y=curve_fit_df["Quantity Sold"],
    #             mode="lines",
    #             name="Richards Curve Fit",
    #             line=dict(color="red"),
    #         )

    #         fig.show()
    #         self.logger.info("The demand curve for one product is calculated successfully.")
    #     except Exception as e:
    #         self.logger.info("Error calculating the demand curve for one product: %s", str(e))
    #         raise e

    @log_decorator
    def calculate_demand_curve_product(self, product):
        """
        Calculates the demand curve for a single product using log-log regression and plots it.
        Model: log(Q) = a + b*log(P)  =>  Q_hat = exp(a) * P^b
        """
        try:
            self.logger.info("Calculating demand curve for one product (log-log regression)...")
    
            product_df = self.estimations[self.estimations["product_code"] == product].reset_index(drop=True)
    
            price = product_df[f"product_selling_price_calc"].astype(float)
            quantity_sold = product_df["predict_sales"].astype(float)
    
            # Keep only valid rows for log transforms
            mask = (price > 0) & (quantity_sold > 0)
            price_fit = price[mask]
            qty_fit = quantity_sold[mask]
    
            if len(price_fit) < 2:
                self.logger.info(f"Not enough valid points to fit log-log curve for product {product}.")
                return
    
            # Fit: log(Q) = a + b*log(P)
            x = np.log(price_fit.values)
            y = np.log(qty_fit.values)
            b, a = np.polyfit(x, y, 1)  # slope=b, intercept=a
    
            # Predict curve for all prices (only where price > 0)
            curve_fit_data = np.full(shape=len(price), fill_value=np.nan, dtype=float)
            mask_price_pos = price.values > 0
            curve_fit_data[mask_price_pos] = np.exp(a + b * np.log(price.values[mask_price_pos]))
    
            curve_fit_df = {"Price": price, "Quantity Sold": curve_fit_data}
    
            # Plot the original data
            fig = px.scatter(
                product_df,
                x=self.price_column + "_calc",
                y="predict_sales",
                labels={"predict_sales": "Quantity Sold"},
                title=f"Original Data vs Log-Log Curve Fit ({product})",
            )
    
            # Add the fitted curve
            fig.add_scatter(
                x=curve_fit_df["Price"],
                y=curve_fit_df["Quantity Sold"],
                mode="lines",
                name="Log-Log Curve Fit",
                line=dict(color="red"),
            )
    
            fig.show()
            self.logger.info("The demand curve for one product is calculated successfully (log-log).")
    
        except Exception as e:
            self.logger.info("Error calculating the demand curve for one product: %s", str(e))
            raise e
 
    @log_decorator
    def calculate_elasticities(self):
        """
        This function calculates price elasticities for each product. Groups data by product, computes elasticities based on sales and 
        price changes. Results are stored in a DataFrame for further analysis.
        """
        try:
            self.logger.info("Calculating elasticities...")
                        
            elasticities = {}
                        
            grouped = self.curve_merged.groupby(["product_code_elasticity", "sequence_number"], observed=False)
            for (product_code, sequence_number), group in grouped:
                middle_index = len(group) // 2
                middle_row = group.iloc[middle_index]
                current_price = middle_row["price"]
                current_price_rows = group[group["price"] == current_price]
                prev_next_data = group[
                    (group["price"].shift(1) == current_price) | (group["price"].shift(-1) == current_price)
                ]
                if prev_next_data.empty:
                    continue
                calculated_elasticity = list(
                    prev_next_data["sales"].pct_change().dropna()
                    / prev_next_data["price"].pct_change().dropna()
                )[0]
                elasticities[(product_code, sequence_number)] = calculated_elasticity
            
            self.elasticity_df = pd.DataFrame(list(elasticities.items()), columns=['product_info', 'elasticity'])

            # Split the product_info column into separate product_code and sequence_number columns
            self.elasticity_df[['product_code', 'sequence_number']] = pd.DataFrame(self.elasticity_df['product_info'].tolist(), index=self.elasticity_df.index)
            self.temp = self.elasticity_df.copy()
            self.elasticity_df = self.elasticity_df[self.elasticity_df.elasticity != 0]
            self.elasticity_df = self.elasticity_df.drop(columns=['product_info'])
            self.elasticity_df = self.elasticity_df.groupby(['product_code']).elasticity.mean().reset_index()


            self.logger.info("Elasticities are calculated successfully.")
        except Exception as e:
            self.logger.error("Error in calculating elasticities: %s", str(e))
            raise e
            
    def count_elasticities(self):
        """Counts the types of elasticities"""

        conditions = [
            (-5 < self.elasticity_df['elasticity']) & (self.elasticity_df['elasticity'] < -1),
            (self.elasticity_df['elasticity'] < -5),
            (-1 < self.elasticity_df['elasticity']) & (self.elasticity_df['elasticity'] < 0),
            (self.elasticity_df['elasticity'] > 0)
        ]

        # Apply conditions to get product lists
        self.elastic_products = self.elasticity_df.loc[conditions[0], 'product_code'].tolist()
        self.outlier_products = self.elasticity_df.loc[conditions[1], 'product_code'].tolist()
        self.inelastic_products = self.elasticity_df.loc[conditions[2], 'product_code'].tolist()
        self.positive_elastic_products = self.elasticity_df.loc[conditions[3], 'product_code'].tolist()

        # Count the products based on conditions
        self.elastic_count = len(self.elastic_products)
        self.outlier_count = len(self.outlier_products)
        self.inelastic_count = len(self.inelastic_products)
        self.positive_count = len(self.positive_elastic_products)

    @log_decorator
    def calculate_elasticities_per_category(self):
        """
        This function calculates median price elasticities per category. Cleans the elasticity data, merges with product categories, 
        computes product counts, and calculates median elasticities. Filters out categories with fewer than a certain number of products 
        and logs insufficient categories.
        """
        
        try:
            self.logger.info("Calculating elasticities per category...")
            
            self.positive_elasticities = self.elasticity_df[self.elasticity_df.elasticity > 0]
            # Drops elasticities of 0 (unable to fit the curve)
            #self.elasticity_df = self.elasticity_df[self.elasticity_df.elasticity != 0]
            # Computes the mean elasticity per product
            self.elasticity_df = self.elasticity_df.groupby(['product_code']).elasticity.mean().reset_index()
            self.percent_positive_elasticities = self.elasticity_df[self.elasticity_df.elasticity > 0].shape[0]/self.elasticity_df.shape[0]*100
            # Counts the types of products
            self.count_elasticities()

            # Merge elasticities with categories
            self.merged_elasticities = pd.merge(
                self.elasticity_df[self.elasticity_df.elasticity <= 0],
                self.preprocessed_data[["product_code", self.category_level]],
                on="product_code",
                how="left",
            ).drop_duplicates()
            
            if self.client_name == 'spar':
                self.merged_elasticities['priceline'] = self.priceline_name 
            
            # if self.client_name == 'hoogvliet':
            #     self.merged_elasticities = elasticity_mapping(self.merged_elasticities, self.category_level)
                
            # Create product counts per category and merge with previous dataset
            prods_per_category = self.merged_elasticities.groupby(self.category_level, observed=False)[
                "product_code"
            ].count()
            
            self.merged_elasticities = pd.merge(
                self.merged_elasticities,
                pd.DataFrame(prods_per_category)
                .reset_index()
                .rename(columns={"product_code": "num_prods"}),
                on=self.category_level,
            )

            # Create a df with the mean and median per category
            self.elasticities = (
                self.merged_elasticities.groupby('product_code', observed=False)
                .agg(
                    elasticity_median=('elasticity', 'median'),
                    elasticity_mean=('elasticity', 'mean'),
                    **{self.category_level: (self.category_level, 'first')}
                )
                .reset_index()
                .groupby(self.category_level, observed=False)
                .agg(
                    elasticity_median=("elasticity_median", lambda x: np.round(x.median(), 2)),
                    elasticity_mean=("elasticity_mean", lambda x: np.round(x.mean(), 2)),
                    num_products=("product_code", "count"),
                )
                .reset_index()
            )
            
            #calculate weighted elasticities
            self.result = self.preprocessed_data.groupby(self.category_level).apply(lambda x: calculate_sales_fraction(x, self.category_level) ).reset_index(drop=True)
            self.merged_elasticities = pd.merge(self.merged_elasticities, self.result, on = [self.category_level, 'product_code'], how = 'inner')
            self.merged_elasticities['weighted_elasticities'] = self.merged_elasticities['elasticity'] * self.merged_elasticities['sales_fraction']
            self.grouped_weighted = pd.DataFrame(self.merged_elasticities.groupby(self.category_level)['weighted_elasticities'].sum()).reset_index()
            self.elasticities = pd.merge(self.elasticities, self.grouped_weighted, on = self.category_level, how = 'inner')
            self.elasticities['weighted_elasticities'] = np.round(self.elasticities['weighted_elasticities'], 2)
            self.elasticities = self.elasticities[[self.category_level, 'elasticity_mean', 'elasticity_median', 'weighted_elasticities', 'num_products']]
            insufficient_categories = self.elasticities[self.elasticities.num_products < 5]
            if not insufficient_categories.empty:
                self.num_insufficient_cat = insufficient_categories[self.category_level].nunique()
                self.logger.info(
                    f"There are {self.num_insufficient_cat} categories with less than {self.num_products} products."
                )
            else:
                self.num_insufficient_cat = 0
            self.elasticities = self.elasticities[self.elasticities.num_products >= 5]
            self.elasticities.reset_index(drop=True, inplace=True)
            if self.client_name == 'spar':
                self.elasticities['priceline'] = self.priceline_name 
            self.elasticities = self.elasticities.sort_values('num_products', ascending = False).reset_index(drop=True)
            self.logger.info("Median elasticities are calculated successfully.")
        except Exception as e:
            self.logger.error("Error loading configuration: %s", str(e))
            raise e
        
    def show_insights(self):
        
        # Create an interactive table using Plotly Express
        fig1 = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=['elastic_pe_count', 'inelastic_pe_count', 'positive_pe_count', 'outlier_pe_count', 'number_sufficient_categories', 'number_insufficient_categories'],
                        line_color="darkslategray",
                        fill_color="lightskyblue",
                        align="left",
                    ),
                    cells=dict(
                        values=[self.elastic_count, self.inelastic_count, self.positive_count, self.outlier_count, self.elasticities.shape[0] , self.num_insufficient_cat], # 2nd column
                        line_color="darkslategray",
                        fill_color="lightcyan",
                        align="left",
                    ),
                )
            ]
        )

        fig1.update_layout(
            autosize=True,
            title=dict(
                x=0.5,
                y=0.95,
                font=dict(size=18),
            ),
        )
        fig1.show()
        
    def beautify_table(self):
        """
        This function creates a plotly table. This table displays the median elasticities data with 
        customized header and cell styles. The layout is adjusted for height, font size, and margins to enhance 
        visual appeal.
        """
            
        try:
            self.logger.info("Creating table with median elasticities...")
            
            # Create an interactive table using Plotly Express
            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=self.elasticities.columns[:-1],
                            line_color="darkslategray",
                            fill_color="lightskyblue",
                            align="left",
                        ),
                        cells=dict(
                            values=[
                                self.elasticities.iloc[:, 0],
                                self.elasticities.iloc[:, 1],
                                self.elasticities.iloc[:, 2],
                                self.elasticities.iloc[:, 3],
                                self.elasticities.iloc[:, 4],
                            ],
                            line_color="darkslategray",
                            fill_color="lightcyan",
                            align="left",
                        ),
                    )
                ]
            )
            if self.client_name == 'spar':
                title="Column insights and number of products per category for priceline " + self.priceline_name
            else:
                title="Column insights and number of products per category"
            fig.update_layout(
                autosize=True,
                title=dict(text = title,
                x=0.5,
                y=0.95,
                
                font=dict(size=18),
            ),
                font=dict(size=12),
                margin=dict(t=75, l=50, r=50, b=50),
            )

            fig.show()
            self.logger.info("Table with median elasticities is created successfully.")
        except Exception as e:
            self.logger.error("Error in creating table with median elasticities: %s", str(e))
            raise e
            
    def run_process(self):
        """
        This method initiates the price elasticity pipeline by calling various private methods sequentially.
        """

        self.logger.info("Process of calculating elasticities has started...")
        self.calculate_demand_curve()
        self.calculate_elasticities()
        self.calculate_elasticities_per_category()
        self.beautify_table()
        self.show_insights()
        self.logger.info("Process of calculating elasticities is finished")
        
    def save_to_s3(self):
        self.merged_elasticities.to_parquet(self.output_to_s3)
        
    def save_to_s3_for_dashboards(self):
        
        if self.client_name == 'spar':
                
            path = os.path.join('s3://prime-rel-spar/prime/platform/temp/price_elasticity_dataset/'+ self.priceline_name + '_' + self.name + '_'  + self.config_s3['name'])
                
        else: 
            path = os.path.join('s3a://prime-rel-hoogvliet/prime/platform/solutions/price_elasticity/results/'+ '_' + self.name + '_' + self.config_s3['name'])
            
        self.merged_elasticities.to_parquet(path)
            
            
            
    def merge_all_pricelines_for_dashboards(self):
       
        # Initialize a session using Amazon S3
        s3 = boto3.client('s3')
        # Define the bucket name and directory (prefix)
        bucket_name = self.config_s3['bucket'][:-1]
        directory = self.config_s3['output_prefix']  # Ensure the directory ends with '/'

        # List files in the specified S3 bucket and directory
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=directory)

        # Extract the file names
        if 'Contents' in response:
            files = [item['Key'] for item in response['Contents']]
            filtered_files = [file for file in files if any(item in file for item in self.pricelines)]
            dataframe = []
            for item in filtered_files:
                path = os.path.join(self.bucket, item)
                data = pd.read_parquet(path)
                dataframe.append(data)
                
            #merge pricelines  
            merged_df = pd.DataFrame()
            for df in dataframe:
                merged_df = pd.concat([merged_df, df], axis = 0)
                merged_df.sort_values(self.category_level, inplace = True)
            
            merged_df.reset_index(drop=True, inplace = True)
            merged_df.to_parquet(os.path.join(self.bucket, self.config_s3['output_prefix'], '_' + self.name + '_' + 'all_pricelines.parquet'))
            merged_df.to_parquet(os.path.join('s3://prime-rel-spar/prime/platform/temp/price_elasticity_dataset/',  '_' + self.name + '_' + 'all_pricelines.parquet'))
            return(merged_df)
        else:
            print("No files found in the specified directory.")

        
    def merge_clients_for_dashboards(self):
        
        spar_data = pd.read_parquet('s3://prime-rel-ml/data-analytics/price-elasticity/spar/pe_results/Buurt_category_elasticities.parquet')
        hoogvliet_data = pd.read_parquet('s3://prime-rel-ml/data-analytics/price-elasticity/hoogvliet/pe_results/elasticities.parquet')
        spar_data.drop('priceline', axis = 1, inplace = True)
        spar_data['client'] = 'spar'
        hoogvliet_data['client'] = 'hoogvliet'
        merged_data = pd.concat([spar_data, hoogvliet_data], axis = 0, ignore_index=True)
        return(merged_data)
    
    def plot_results(self):
        
        data = self.merge_all_pricelines_for_dashboards()
        data = data[data.category_name_level2.isin(data.category_name_level2.unique()[:6])].reset_index(drop=True)
        fig = px.box(data, x="category_name_level2", y="elasticity", 
                        color="category_name_level2", 
                        color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"])

        # Increase plot size, add title, and change theme
        fig.update_layout(
            width=1000,  # Increase the width of the plot
            height=600,  # Increase the height of the plot
            template="ggplot2"  # Change the theme to a dark theme
        )
        fig.update_layout(
            title={
                'text': "Elasticity Distribution by Category",
                'y': 0.94,  # Adjust this value if you want to change the vertical position of the title
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',  # Align the title's x-position to the center
                'yanchor': 'top'  # Align the title's y-position to the top
            }
        )
        # Display the plot
        fig.show()

        # Display the plot
        fig.show()


        



        
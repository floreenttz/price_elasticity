import os
import json
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import logging
import yaml
import time
import datetime
from pandarallel import pandarallel
from datetime import datetime
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from .functions import *
import joblib
import pickle
import boto3
import s3fs

import warnings
warnings.filterwarnings("ignore")

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

def get_logger():
    
    os.makedirs('logs', exist_ok=True)
    current_time = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
    log_file = os.path.join('logs', f"modelling_{current_time}.log")
  
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


class GradientBoostingModel:
    """
    This class serves as a wrapper around the XGB model, specifically designed
    for handling timeseries data in Pandas. It includes functionality for
    creating features based on provided lags, aggregation functions,
    and supports cross-validation and grid-search via Optuna.
    """

    def __init__(self, config_file, client_name=None, priceline_name = None, logger=None, optuna=False, params=None, debug=False, frequency = 'daily'):

        if logger is None:
            self.logger = get_logger()
        else:
            self.logger = logger
            
        self.debug = debug
        self.client_name = client_name
        self.config_file = config_file
        self.optuna = optuna
        self.s3 = boto3.client('s3')
        self.config = self._load_config()
        self.best_params = params
        self.frequency = frequency
        self.min_grid_perc = self.config['min_grid_perc']
        self.max_grid_perc = self.config['max_grid_perc']
        # TODO: Move config file to separate method
        self.config_s3 = self.config['s3_dir']
        self.price_column = self.config['price_column']
        self.client_name = self.config['client_name']
        
        if self.client_name == 'spar':
            self.priceline_name = self.config['priceline_name']
            
        self.bucket = os.path.join(self.config_s3['s3'], self.config_s3['bucket'])
        self.data_prefix = os.path.join(self.bucket, self.config_s3['data_prefix'])
        
        if self.client_name == 'spar':
            
            if self.frequency=='daily':
                self.data_from_s3 = os.path.join(self.data_prefix, self.priceline_name + '_' + self.config_s3['feature_data'])
                self.estimations_to_s3 = os.path.join(self.data_prefix, self.priceline_name + '_' + self.config_s3['estimations'])
            else:
                self.data_from_s3 = os.path.join(self.data_prefix, self.priceline_name + '_' + self.config_s3['feature_data_weekly'])
                self.estimations_to_s3 = os.path.join(self.data_prefix, self.priceline_name + '_' + self.config_s3['estimations_weekly'])
        else:
            
            if self.frequency=='daily':
                self.data_from_s3 = os.path.join(self.data_prefix, self.config_s3['feature_data'])
                self.estimations_to_s3 = os.path.join(self.data_prefix, self.config_s3['estimations'])
            else:
                self.data_from_s3 = os.path.join(self.data_prefix, self.config_s3['feature_data_weekly'])
                self.estimations_to_s3 = os.path.join(self.data_prefix, self.config_s3['estimations_weekly'])

        self.grid_from_s3 = os.path.join(self.data_prefix, self.config_s3['price_grid'])
        self.model_to_s3 = os.path.join(self.config_s3['model_prefix'], self.config_s3['model'])
        
        self.target = self.config['target']
        self.price_lags = self.config['price_lags']
        self.n_splits = self.config["n_splits"]
        self.price_grid_freq = self.config['price_grid_freq']
        self.competitors = self.config['competitors']
        if self.frequency=='daily':
            self.test_size = self.config["test_size"]

        else:
            self.test_size = 10000

        self.n_trials = self.config["optuna_trials"]
        self.optuna = self.config['optuna']
        self.price_change_columns = self.config['price_change_columns']
        self.live_products = self.config['filters']['filter_live_products']
        
        self.get_data_from_s3()
        
        self.features = self.feature_results.columns.tolist()
        self.features.remove(self.target)
        self.features.remove('date')

        # Ensure that all string columns are set to category
        for col in self.feature_results.select_dtypes(include=['object']).columns:
            self.feature_results[col] = self.feature_results[col].astype('category')

        self.train_test_split()
        self.execute_model()
        self.save_to_s3()
    
    def get_data_from_s3(self):
        
        self.feature_results = pd.read_parquet(self.data_from_s3)
        self.feature_results['date'] = pd.to_datetime(self.feature_results['date'])
        with s3fs.S3FileSystem().open(self.grid_from_s3, 'rb') as f:
            pickle_data = f.read()
            self.price_grid = pickle.loads(pickle_data)

    @log_decorator
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
            self.logger.info("Loading configuration...")
            path = self.config_file
            with open(path, "r") as stream:
                config = yaml.safe_load(stream)
            self.logger.info("Configuration loaded successfully.")
            return config
        except Exception as e:
            self.logger.error("Error loading configuration: %s", str(e))
            raise e
            
    @log_decorator
    # Function to expand test_data for each product code
    def expand_test_data(self, product_data):

        try:
            full_expanded_data = pd.DataFrame()
            product_data = product_data.reset_index(drop=True)
            product_prices = self.price_grid.get(str(product_data["product_code"].iloc[0]), [])

            for i, price_set in enumerate(product_prices):
                expanded_data = pd.DataFrame([product_data.iloc[i]] * len(price_set))
                expanded_data[self.price_column] = price_set
                expanded_data['sequence_number'] = i
                full_expanded_data = pd.concat([full_expanded_data, expanded_data], axis=0)
        except Exception as e:
            pass
            # print(len(product_data))
            # print(f"Product data for product code {product_code} is missing one or more dates so the expanded test data is incomplete.")

        return full_expanded_data
               
    def update_test_data(self):
        """Function to update test (prediction) data with the right price changes and real prices after the addition of the price grid"""
        
        # TO DO: Rewrite solution to be a) flexible given the range of price grid b) if possible find a more elegant solution
        # Add calculated_price to the price grid
        # Function to add the corresponding index to each row
        def add_indices(group, indices):
            group_size = len(group)
            group['idx'] = indices[:group_size]
            return group
        
        self.predict_df['idx'] = 1
        grouped = self.predict_df.groupby(['product_code'], observed=True)
        self.num_prices = abs(int(self.min_grid_perc)) + abs(int(self.max_grid_perc)) + 1  # In this case, 7.5 + 7.5 = 15, so num_prices will be 16

        # Generate list of indices using the adjusted percentage range
        indices_list = np.linspace(self.min_grid_perc + 100, self.max_grid_perc + 100, self.num_prices).tolist() * self.price_grid_freq
        
        # self.predict_df['idx'] = 1
        # grouped = self.predict_df.groupby(['product_code'], observed=True)
        # indices_list = np.linspace(92.5, 107.5, 21).tolist() * self.price_grid_freq

        # Apply function to each group
        self.predict_df = grouped.apply(lambda x: add_indices(x, indices_list)).reset_index(drop=True)
        # self.predict_df["adjusted_price"] = self.predict_df.apply(lambda x: x["adjusted_price"] * (x.idx/100), axis=1)
        # Ensure adjusted_price exists (required downstream)
        if "adjusted_price" not in self.predict_df.columns:
            # fallback: treat price_column as adjusted_price
            self.predict_df["adjusted_price"] = self.predict_df[self.price_column]

        # Scale adjusted_price according to idx
        self.predict_df["adjusted_price"] = self.predict_df["adjusted_price"] * (self.predict_df["idx"] / 100.0)
        # if self.client_name == 'spar':
        #     self.predict_df["adjusted_price_category"] = self.predict_df.apply(lambda x: x["adjusted_price_category"] * (x.idx/100), axis=1)
        self.predict_df.drop(columns=['idx'], inplace=True)
                
        for column in self.price_change_columns:
            for lag in self.price_lags:
                change_column = f"price_change_{column}_lag_{lag}"
                lagged_prices = self.predict_df[f"{column}_lag_{lag}"]
                self.predict_df[change_column] = ((self.predict_df[column] - lagged_prices) / lagged_prices) * 100
                self.predict_df[change_column].replace([np.inf, -np.inf], np.nan, inplace=True)
                self.predict_df.drop(columns=[f"{column}_lag_{lag}"], inplace=True)
                self.x_train.drop(columns=[f"{column}_lag_{lag}"], inplace=True)
                
        for competitor in self.competitors:
            if competitor + "_price" in self.predict_df.columns:
                self.predict_df[f"price_distance_{competitor}"] = (self.predict_df[self.price_column] - self.predict_df[f"{competitor}_price"]) / ((self.predict_df[self.price_column] + self.predict_df[f"{competitor}_price"]) / 2)

        if self.frequency == 'weekly':
            self.predict_df = self.predict_df.groupby('product_code').apply(lambda g: fill_except_median(g, 'days_since_last_change')).reset_index(drop=True)
        
            
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
            self.logger.info(f"Start: {self.feature_results.product_code.nunique()}")
            
            self.cutoff_dates = self.feature_results.date.max() - pd.to_timedelta(self.price_grid_freq - 1, unit='d')
            self.date_range = pd.date_range(start=self.cutoff_dates, end=self.feature_results.date.max())
            self.prods_max_dates = self.feature_results[self.feature_results.date.isin(self.date_range)].product_code.unique()
            
            self.logger.info("Retrieval of live products completed successfully...")
            self.logger.info(self.feature_results.product_code.nunique())
            self.logger.info(f"End: {self.feature_results.product_code.nunique()}")
        except Exception as e:
            self.logger.error("Error in retrieval of live products: %s", str(e))
            raise e
            
    @log_decorator
    def train_test_split(self):
        
        try:
            # Identify the last day for each product
            self.logger.info("Data splitting has started...")
            
            # Calculate the last date for each product_code
            self.last_dates = self.feature_results.groupby("product_code",  observed=True)["date"].transform("max")
            self.cutoff_dates = self.last_dates - pd.to_timedelta(self.price_grid_freq-1, unit='d')
            last_x_days_mask = self.feature_results["date"] >= self.cutoff_dates

            self.logger.info(f"Number of products before live filter: {self.feature_results.product_code.nunique()}")
            
            # Get live products for testing
            if self.live_products:
                self._filter_live_products()
            
            # Create training and testing sets
            train_filter = ~last_x_days_mask
            test_filter = last_x_days_mask
            
            # Apply the live product filter if configured
            if self.live_products:
                train_filter &= self.feature_results['product_code'].isin(self.prods_max_dates)
                test_filter &= self.feature_results['product_code'].isin(self.prods_max_dates)

            # Apply the filters
            self.x_train = self.feature_results[train_filter][self.features]
            #print(self.x_train.columns)
            self.y_train = self.feature_results[train_filter][self.target]
            self.x_test = self.feature_results[test_filter][self.features]
            self.y_test = self.feature_results[test_filter][self.target]
                        
            # Expand test data with price grid
            self.predict_df = (self.x_test.groupby("product_code", observed=True)
                                          .apply(lambda x: self.expand_test_data(product_data=x))
                                          .reset_index(drop=True))
                
            self.predict_df['promotion_indicator'] = False
             
            self.logger.info(f"Number of products after live filter: {self.predict_df.product_code.nunique()}")

            for col in self.predict_df.select_dtypes(include=['object']).columns:
                self.predict_df[col] = self.predict_df[col].astype('category')

            # Update prediction df after expanding the test data
            self.update_test_data()
 
            self.logger.info("Data splitting has been completed...")
        except Exception as e:
            self.logger.error("Error splitting the data: %s", str(e))
            raise e

    def optuna_suggest_param(self, name, details, trial):
        # Loads parameters from the dictionary into the Optuna format
        if "step" in details and isinstance(details["step"], int):
            return trial.suggest_int(
                name, int(details["min"]), int(details["max"]), step=details["step"]
            )
        elif "step" in details:
            return trial.suggest_float(
                name, details["min"], details["max"], step=details["step"]
            )

    def optuna_objective(self, trial):

        # Load optuna Parameters from s3
        s3 = boto3.client('s3')
        bucket_name = self.config_s3['bucket'][:-1]
        object_key = os.path.join(self.config_s3['data_prefix'], 'optuna_parameters_test.json')
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        content = response['Body'].read().decode('utf-8')
        p = json.loads(content)

        param = {
            name: self.optuna_suggest_param(name, details, trial)
            for name, details in p.items()
            if name != "objective"
        }
        param["objective"] = p["objective"]
        
        mse = self.cross_validate(param)
        return mse

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
            self.cutoff_dates = self.feature_results.date.max() - pd.to_timedelta(self.price_grid_freq - 1, unit='d')
            self.date_range = pd.date_range(start=self.cutoff_dates, end=self.feature_results.date.max())
            self.prods_max_dates = self.feature_results[self.feature_results.date.isin(self.date_range)].product_code.unique()        
            self.logger.info("Retrieval of live products completed successfully...")
        except Exception as e:
            self.logger.error("Error in retrieval of live products: %s", str(e))
            raise e
            
    @log_decorator
    def execute_model(self):
        
        try:
            self.logger.info("Model execution has started...")
            
            if self.optuna:
                self.study = optuna.create_study(direction="minimize", study_name="Hyperparameter tuning of XGBoost")
                self.study.optimize(self.optuna_objective, n_trials=self.n_trials, n_jobs=-1)
                print("Best Hyperparameters:", self.study.best_params)
                # self.optuna_visualization()
                self.best_params = self.study.best_params
            else:
                self.best_params = self.best_params
            
            # if self.frequency=='daily':
            #     self.model = xgb.XGBRegressor(
            #         **self.best_params, enable_categorical=True
            #     )
            # else:
            #     self.model = xgb.XGBRegressor(enable_categorical=True)
            self.model = xgb.XGBRegressor(
                    **self.best_params, enable_categorical=True
                )
            self.y_train_transformed = np.log1p(self.y_train)

#             self.x_train['product_selling_price'] = np.log(self.x_train['product_selling_price'])
#             self.predict_df['product_selling_price'] = np.log(self.predict_df['product_selling_price'])

            self.model.fit(self.x_train, self.y_train_transformed) #.drop(columns=['product_selling_price'])

            #self.model_explainability()
            self.feature_importance()    

            self.y_pred_transformed = self.model.predict(self.predict_df.drop(columns=['sequence_number'])) #, 'product_selling_price'
            self.predictions = np.expm1(self.y_pred_transformed)
            self.predictions = np.maximum(self.predictions, 0)

            if self.optuna:
                # Plot losses after cross-validation
                self.plot_losses()

            self.predictions_df = pd.DataFrame(
                {
                    "product_code": self.predict_df["product_code"],
                    "predicted_quantity": self.predictions,
                }
            )

            # Extract the last n prices for each product_code
            self.last_days_df = (
                self.feature_results.sort_values(by=["product_code", "date"])
                .groupby("product_code",  observed=True)
                .tail(self.price_grid_freq)
            ).reset_index(drop=True)
            
            # self.last_days_df[self.price_column] = np.round(self.last_days_df[self.price_column], 3)
            
            #remove zero prices
            zero_prices = self.last_days_df[self.last_days_df[self.price_column] <0.1]
            if not zero_prices.empty:
                self.last_days_df = self.last_days_df[~(self.last_days_df.product_code.isin(zero_prices.product_code.unique()))]
                self.last_days_df.reset_index(drop=True, inplace=True)
            self.predictions_df[self.price_column] = self.predict_df[
                self.price_column
            ].copy()
            
            num_sequences = self.predict_df["sequence_number"].max()

            # Add matching sequence numbers to last_days_df
            self.last_days_df = add_sequence_numbers_to_last_days(self.last_days_df, num_sequences)

            # Merge using product_code and sequence_number
            self.predictions_df[self.price_column] = self.predict_df[self.price_column].copy()
            self.predictions_df["sequence_number"] = self.predict_df["sequence_number"].copy()

            self.output = self.predictions_df.merge(
                self.last_days_df[["product_code", self.price_column, "sequence_number"]],
                how="left",
                on=["product_code", "sequence_number"],
                suffixes=["_calc", "_real"],
            )
            self.logger.info("Model has been trained and tested successfully...")
        except Exception as e:
            self.logger.error("Error executing the model: %s", str(e))
            raise e
        
    @log_decorator
    def save_to_s3(self):

        try:
            self.logger.info("Saving the data and model to s3...")
            self.output.to_parquet(self.estimations_to_s3)
            # Upload the model to s3
            joblib.dump(self.model, "trained_model.joblib")
            self.s3.upload_file("trained_model.joblib", self.config_s3['bucket'][:-1], self.model_to_s3)
            self.logger.info("Data have been saved successfully.")
        except Exception as e:
            self.logger.error("Error saving the data to s3: %s", str(e))
            raise e
        
    def optuna_visualization(self):
        
        plot_optimization_history(self.study).show()
        plot_param_importances(self.study).show()
    
    def pad_list(self, list_to_pad, length, pad_value=np.nan):
        return list_to_pad + [pad_value] * (length - len(list_to_pad))
        
    def cross_validate(self, param):
        param['eval_metric'] = 'rmse'  # Set eval_metric in the constructor
        xgb_model = xgb.XGBRegressor(**param, enable_categorical=True)
        xgb_model.set_params(early_stopping_rounds=10)
        cv = TimeSeriesSplit(n_splits=self.n_splits)#, test_size=self.test_size)
        
        all_train_losses = []
        all_val_losses = []

        max_length = 0

        for train_idx, val_idx in cv.split(self.x_train):
      
            x_train_fold = self.x_train.iloc[train_idx, :].reset_index(drop=True)
            x_val_fold = self.x_train.iloc[val_idx, :].reset_index(drop=True)
            y_train_fold = self.y_train.iloc[train_idx].reset_index(drop=True)
            y_val_fold = self.y_train.iloc[val_idx].reset_index(drop=True)
        
            eval_set = [(x_train_fold, y_train_fold), (x_val_fold, y_val_fold)]
            xgb_model.fit(
                x_train_fold, y_train_fold,
                eval_set=eval_set,
                verbose=False
            )

            results = xgb_model.evals_result()

            train_rmse = results['validation_0']['rmse']
            val_rmse = results['validation_1']['rmse']

            max_length = max(max_length, len(train_rmse), len(val_rmse))

            all_train_losses.append(train_rmse)
            all_val_losses.append(val_rmse)

        # Pad the lists to the maximum length
        all_train_losses = [self.pad_list(lst, max_length) for lst in all_train_losses]
        all_val_losses = [self.pad_list(lst, max_length) for lst in all_val_losses]

        # Convert to NumPy arrays
        all_train_losses = np.array(all_train_losses)
        all_val_losses = np.array(all_val_losses)

        # Calculate the mean ignoring NaNs
        self.train_losses = np.nanmean(all_train_losses, axis=0)
        self.val_losses = np.nanmean(all_val_losses, axis=0)

        return np.mean([loss[-1] for loss in all_val_losses if not np.isnan(loss[-1])])
    
    def plot_losses(self):
        # Plot training and validation losses
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()
        
    def model_explainability(self):
        
        # Create object that can calculate SHAP values
        explainer = shap.TreeExplainer(model, approximate=True)
        shap_values = explainer.shap_values(self.predict_df)

        # Summary plot for feature importance
        shap.summary_plot(shap_values, self.predict_df, title="Individual contribution of features on Test Set")
        
    def feature_importance(self):
        
        ax = xgb.plot_importance(self.model, title='Feature Importance')
        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        plt.grid(False)
        plt.show()

        
        
        

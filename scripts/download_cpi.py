import requests
import pandas as pd
import boto3
from io import BytesIO

# Function to get data from CBS API with query parameters
def get_cbs_data(endpoint, params=None):
    base_url = "https://opendata.cbs.nl/ODataApi/odata"
    url = f"{base_url}/{endpoint}"
    response = requests.get(url, params=params)
    response.raise_for_status()  # Check for HTTP request errors
    return response.json()

# Function to upload a dataframe to S3 as a parquet file
def upload_to_s3(df, bucket_name, s3_key):
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3_client = boto3.client('s3')
    s3_client.upload_fileobj(buffer, bucket_name, s3_key)
    
def preprocess_cpi(df):
    # Select the relevant columns and rename them
    df = df[['Perioden', 'CPI_1']]
    df.columns = ['month_year', 'cpi']
    
    # Replace "MM" with "-" in the month_year column
    df['month_year'] = df['month_year'].astype(str).str.replace('MM', '-')
    
    # Filter out rows where month_year contains 'JJ'
    df = df[~df['month_year'].str.contains('JJ')]
    
    # Convert month_year to datetime format
    df['month_year'] = pd.to_datetime(df['month_year'], format='%Y-%m')
    
    # Extract year, month, and days_in_month
    df['year'] = df['month_year'].dt.year
    df['month'] = df['month_year'].dt.month
    df['days_in_month'] = df['month_year'].dt.days_in_month
    
    return df

def download_cpi():
    # Example usage: Getting data for Bestedingscategorieen: 000000 from 2020 onwards
    table_endpoint = "83131NED/TypedDataSet"
    params = {
        "$filter": "Perioden ge '2020MM01' and Bestedingscategorieen eq 'T001112  '",
        "$top": 1000  # Adjust the limit as necessary
    }
    data = get_cbs_data(table_endpoint, params=params)
    data_values = data['value']

    df = pd.DataFrame(data_values)
    df = preprocess_cpi(df)
    upload_to_s3(df, 'prime-rel-ml', 'data-analytics/price-elasticity/data/cpi_alle_bestedingen.parquet')

    params = {
        "$filter": "Perioden ge '2020MM01' and Bestedingscategorieen eq 'CPI011000'",
        "$top": 1000  # Adjust the limit as necessary
    }
    data = get_cbs_data(table_endpoint, params=params)
    data_values = data['value']

    df = pd.DataFrame(data_values)
    df = preprocess_cpi(df)
    upload_to_s3(df, 'prime-rel-ml', 'data-analytics/price-elasticity/data/cpi_voedingsmidellen.parquet')

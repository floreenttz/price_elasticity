import io #
import os
import boto3
import pandas as pd


def list_s3_objects(s3client, bucket, folder):
    """
    Lists all objects in a specified folder within an S3 bucket.

    This function uses the AWS S3 'list_objects_v2' API call to paginate through all objects
    in the specified `folder` within the `bucket`. It filters out a specific file type (a success marker file)
    that is not needed in the returned list. This can be useful for processing datasets
    that are stored in S3, especially when working with large numbers of files.

    Parameters:
    - bucket (str): The name of the S3 bucket.
    - folder (str): The prefix (folder path) where the objects are stored in the bucket.

    Returns:
    - list: A list of strings, where each string is the key (path) of an object within the specified folder.
    """ 

    # Get a paginator for the 'list_objects_v2' operation.
    # Paginators are a feature of Boto3 that automatically handle the process of iterating
    # over pages of results from AWS API calls that might return more items than can be returned in a single response.
    paginator = s3client.get_paginator('list_objects_v2')

    # Use the paginator to generate an iterator of pages. Each page is a dictionary that contains a list of objects.
    # We specify the bucket name and a prefix to filter the objects by a specific folder.
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=folder)

    objects = []  # Initialize an empty list to store the keys of the objects found.

    # Iterate through each page provided by the paginator.
    for page in page_iterator:
        # Check if the page contains the 'Contents' key, indicating it has object information.
        if 'Contents' in page:
            # If it does, iterate through each object in the 'Contents' list.
            for obj in page['Contents']:
                # Append the object key (its unique identifier within the bucket) to our list of objects.
                objects.append(obj['Key'])

    # Exclude any objects that have '_SUCCESS' in their key.
    # This is a common pattern where a '_SUCCESS' file is used as a marker to indicate
    # that a certain process or batch job has completed successfully, and it is not usually needed in the data processing.
    objects = [s for s in objects if '_SUCCESS' not in s]

    return objects  # Return the list of object keys.


def load_file_from_s3(s3client, file, bucket, sep=","):
    """
    Loads a file from an S3 bucket and processes it. Supports both CSV and Parquet file formats.

    This function reads a file (either CSV or Parquet) from the specified S3 bucket and key (file name),
    converts certain columns to appropriate data types, and handles missing values. It is designed
    to prepare the 'date', 'skuQty', and 'turnover' columns for further analysis, making it versatile
    for different data handling scenarios.

    Parameters:
    - file (str): The key of the file in the S3 bucket (its path within the bucket).
    - bucket (str): The name of the S3 bucket.

    Returns:
    - pandas.DataFrame: A DataFrame containing the processed data, ready for analysis.
    """
    # Retrieve the object from S3 based on bucket and file key
    obj = s3client.get_object(Bucket=bucket, Key=file)

    # Read the file into a DataFrame based on its extension
    if file.endswith('.csv'):
        # For CSV files, specifying the delimiter and handling of bad lines
        df = pd.read_csv(io.BytesIO(obj['Body'].read()), sep=sep, on_bad_lines='skip')
    elif file.endswith('.parquet'):
        # For Parquet files, direct reading without additional parameters
        df = pd.read_parquet(io.BytesIO(obj['Body'].read()))

    return df


def instantiate_s3client(ID, SECRET, TOKEN):

    s3client = boto3.client(
        service_name='s3',
        region_name='eu-central-1',
        aws_access_key_id=ID,
        aws_secret_access_key=SECRET,
        aws_session_token=TOKEN)

    return s3client
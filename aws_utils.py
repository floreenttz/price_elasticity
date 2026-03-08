"""
AWS utilities with multi-threaded S3 operations.

Works in both environments:
    - SageMaker: Uses IAM role credentials automatically
    - Local: Uses AWS SSO profiles

Local usage:
    aws sso login --profile release-data  # or production-data
    python sales_competitors.py

Profile selection based on bucket name (local only):
    - prime-rel-* buckets -> release-data profile
    - prime-data-lake, prime-ipv-scraping, prime-intergamma -> production-data profile
"""

import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import boto3
import pandas as pd
from botocore.config import Config


def get_profile_for_bucket(bucket: str) -> str:
    """
    Determine which AWS SSO profile to use based on bucket name.

    Rules:
        - prime-rel-* -> release-data (preferred for safety)
        - prime-data-lake, prime-ipv-scraping, prime-intergamma -> production-data
        - prime-{client} (e.g., prime-ijsvogel) -> production-data
    """
    if bucket.startswith("prime-rel-"):
        return "release-data"
    elif bucket in ["prime-data-lake", "prime-ipv-scraping", "prime-intergamma"]:
        return "production-data"
    elif bucket.startswith("prime-"):
        return "production-data"
    else:
        # Default to release-data for safety
        return "release-data"


def is_running_on_sagemaker() -> bool:
    """Check if running on SageMaker (has IAM role credentials)."""
    return os.path.exists("/home/sagemaker-user") or "SM_CHANNEL" in os.environ


def create_s3_client(profile_name: Optional[str] = None, bucket: Optional[str] = None) -> boto3.client:
    """
    Create an S3 client with appropriate authentication.

    On SageMaker: Uses IAM role credentials (no profile needed)
    Locally: Uses AWS SSO profile (inferred from bucket name or specified)

    Args:
        profile_name: AWS SSO profile name (local only). If not provided, inferred from bucket.
        bucket: S3 bucket name used to infer profile if profile_name not provided.

    Returns:
        boto3 S3 client
    """
    # Configure for better performance with many concurrent requests
    config = Config(
        max_pool_connections=50,
        retries={"max_attempts": 3, "mode": "adaptive"}
    )

    # On SageMaker, use default credentials (IAM role)
    if is_running_on_sagemaker():
        print("Running on SageMaker - using IAM role credentials")
        return boto3.client("s3", config=config)

    # Locally, use SSO profile
    if profile_name is None:
        if bucket is None:
            raise ValueError("Either profile_name or bucket must be provided for local execution")
        profile_name = get_profile_for_bucket(bucket)

    print(f"Using AWS SSO profile: {profile_name}")
    session = boto3.Session(profile_name=profile_name)
    return session.client("s3", config=config)


def list_s3_objects(s3_client, bucket: str, prefix: str) -> List[str]:
    """
    List all objects in an S3 bucket with given prefix.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 key prefix to filter objects

    Returns:
        List of S3 object keys
    """
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    return keys


def load_file_from_s3(s3_client, bucket: str, key: str) -> bytes:
    """
    Load a single file from S3.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        File contents as bytes
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def _download_single_parquet(
    s3_client,
    bucket: str,
    key: str,
    index: int,
    total: int
) -> Tuple[int, pd.DataFrame]:
    """
    Download a single parquet file from S3.

    Returns:
        Tuple of (index, DataFrame) for ordering results
    """
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    return (index, df)


def load_multiple_parquets_from_s3(
    s3_client,
    file_keys: List[str],
    bucket: str,
    max_workers: int = 10,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Load and combine multiple parquet files from S3 using multi-threading.

    Args:
        s3_client: boto3 S3 client
        file_keys: List of S3 file keys (paths) to load
        bucket: S3 bucket name
        max_workers: Maximum number of concurrent download threads (default: 10)
        show_progress: Whether to print progress messages

    Returns:
        Combined DataFrame containing data from all parquet files

    Raises:
        ValueError: If file_keys list is empty
    """
    if not file_keys:
        raise ValueError("file_keys list is empty")

    total = len(file_keys)
    if show_progress:
        print(f"Loading {total} parquet file(s) with {max_workers} threads...")

    dfs = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        futures = {
            executor.submit(
                _download_single_parquet,
                s3_client,
                bucket,
                key,
                idx,
                total
            ): key
            for idx, key in enumerate(file_keys)
        }

        # Process completed downloads
        for future in as_completed(futures):
            key = futures[future]
            try:
                idx, df = future.result()
                dfs[idx] = df
                completed += 1
                if show_progress:
                    print(f"Downloaded {completed}/{total}: {key.split('/')[-1]}")
            except Exception as e:
                print(f"Error downloading {key}: {e}")
                raise

    combined_df = pd.concat(dfs, ignore_index=True)
    if show_progress:
        print(f"Combined DataFrame shape: {combined_df.shape}")

    return combined_df


def upload_to_s3(
    s3_client,
    df: pd.DataFrame,
    bucket: str,
    key: str
) -> None:
    """
    Upload a DataFrame to S3 as a parquet file.

    Args:
        s3_client: boto3 S3 client
        df: DataFrame to upload
        bucket: S3 bucket name
        key: S3 object key
    """
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, bucket, key)


def upload_multiple_to_s3(
    s3_client,
    uploads: List[Tuple[pd.DataFrame, str, str]],
    max_workers: int = 5,
    show_progress: bool = True
) -> None:
    """
    Upload multiple DataFrames to S3 using multi-threading.

    Args:
        s3_client: boto3 S3 client
        uploads: List of (DataFrame, bucket, key) tuples
        max_workers: Maximum number of concurrent upload threads
        show_progress: Whether to print progress messages
    """
    total = len(uploads)
    if show_progress:
        print(f"Uploading {total} file(s) with {max_workers} threads...")

    completed = 0

    def _upload_single(df, bucket, key):
        upload_to_s3(s3_client, df, bucket, key)
        return key

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_upload_single, df, bucket, key): key
            for df, bucket, key in uploads
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
                completed += 1
                if show_progress:
                    print(f"Uploaded {completed}/{total}: {key.split('/')[-1]}")
            except Exception as e:
                print(f"Error uploading {key}: {e}")
                raise

    if show_progress:
        print("All uploads completed.")

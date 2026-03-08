"""
S3 storage implementation with AWS SSO support.

Works in both environments:
    - SageMaker: Uses IAM role credentials automatically
    - Local: Uses AWS SSO profiles

Local usage:
    aws sso login --profile release-data  # or production-data
    python -m scripts.run_pipeline --client hoogvliet
"""

import os
import pickle
from typing import Any

import boto3
import pandas as pd
import s3fs
from botocore.config import Config


def get_profile_for_bucket(bucket: str) -> str:
    """
    Determine which AWS SSO profile to use based on bucket name.

    Rules:
        - prime-rel-* -> release-data
        - prime-data-lake, prime-ipv-scraping, prime-intergamma -> production-data
        - prime-{client} -> production-data
    """
    if bucket.startswith("prime-rel-"):
        return "release-data"
    elif bucket in ["prime-data-lake", "prime-ipv-scraping", "prime-intergamma"]:
        return "production-data"
    elif bucket.startswith("prime-"):
        return "production-data"
    else:
        return "release-data"


def is_running_on_sagemaker() -> bool:
    """Check if running on SageMaker (has IAM role credentials)."""
    return os.path.exists("/home/sagemaker-user") or "SM_CHANNEL" in os.environ


class S3Storage:
    """Storage backend for AWS S3 with SSO support."""

    def __init__(self, bucket: str | None = None, profile_name: str | None = None):
        """
        Initialize S3 storage.

        Args:
            bucket: Default bucket name (without s3:// prefix).
                    Can be overridden per-operation via full s3:// paths.
            profile_name: AWS SSO profile name (local only). If not provided,
                         inferred from bucket name.
        """
        self.bucket = bucket

        # Configure boto3 client
        config = Config(
            max_pool_connections=50,
            retries={"max_attempts": 3, "mode": "adaptive"}
        )

        if is_running_on_sagemaker():
            print("Running on SageMaker - using IAM role credentials")
            self.s3_client = boto3.client("s3", config=config)
            self.fs = s3fs.S3FileSystem()
        else:
            # Local execution - use SSO profile
            if profile_name is None and bucket:
                profile_name = get_profile_for_bucket(bucket)
            elif profile_name is None:
                profile_name = "release-data"  # Default

            print(f"Using AWS SSO profile: {profile_name}")
            session = boto3.Session(profile_name=profile_name)
            self.s3_client = session.client("s3", config=config)
            self.fs = s3fs.S3FileSystem(profile=profile_name)

    def _normalize_path(self, path: str) -> str:
        """Ensure path has s3:// prefix."""
        if path.startswith("s3://"):
            return path
        if self.bucket:
            return f"s3://{self.bucket}/{path.lstrip('/')}"
        raise ValueError(f"No bucket specified and path is not absolute: {path}")

    def read_parquet(self, path: str) -> pd.DataFrame:
        """Read a parquet file from S3."""
        full_path = self._normalize_path(path)
        # Remove s3:// prefix for s3fs
        fs_path = full_path[5:] if full_path.startswith("s3://") else full_path
        with self.fs.open(fs_path, "rb") as f:
            return pd.read_parquet(f)

    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        """Write a DataFrame to parquet in S3."""
        full_path = self._normalize_path(path)
        # Remove s3:// prefix for s3fs
        fs_path = full_path[5:] if full_path.startswith("s3://") else full_path
        with self.fs.open(fs_path, "wb") as f:
            df.to_parquet(f, index=False)

    def read_pickle(self, path: str) -> Any:
        """Read a pickle file from S3."""
        full_path = self._normalize_path(path)
        with self.fs.open(full_path, "rb") as f:
            return pickle.load(f)

    def write_pickle(self, obj: Any, path: str) -> None:
        """Write an object to pickle in S3."""
        full_path = self._normalize_path(path)
        with self.fs.open(full_path, "wb") as f:
            pickle.dump(obj, f)

    def glob(self, pattern: str) -> list[str]:
        """Find files matching a glob pattern in S3."""
        # Remove s3:// prefix if present for glob
        if pattern.startswith("s3://"):
            pattern = pattern[5:]
        matches = self.fs.glob(pattern)
        return [f"s3://{m}" for m in matches]

    def exists(self, path: str) -> bool:
        """Check if a path exists in S3."""
        full_path = self._normalize_path(path)
        if full_path.startswith("s3://"):
            full_path = full_path[5:]
        return self.fs.exists(full_path)

"""
Sales + Competitors data processing pipeline (Polars version).

This script:
1. Loads sales data from S3 (multi-threaded)
2. Loads competitor prices from local parquet
3. Filters price outliers using 6-month rolling median:
   - High outliers: > 2x median (catches €249 instead of €2.49)
   - Low outliers: < 0.3x median (catches €0.01 instead of €1.00)
4. Deduplicates remaining prices using median
5. Merges competitor prices with sales data
6. Uploads combined data partitioned by year_week (multi-threaded)

Usage:
    # First, login with AWS SSO:
    aws sso login --profile release-data

    # Then run (uses defaults: high=2.0x, low=0.3x, window=180 days):
    python sales_competitors_polars.py

    # With custom settings:
    python sales_competitors_polars.py --max-workers 20 --chunk-size 1000000

    # Adjust outlier thresholds:
    python sales_competitors_polars.py --outlier-high 1.5 --outlier-low 0.5

    # Disable low outlier filtering (if promotions are in data):
    python sales_competitors_polars.py --outlier-low 0

    # Disable all outlier filtering:
    python sales_competitors_polars.py --no-outlier-filter
"""

import argparse
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import List, Optional, Tuple

import polars as pl

from aws_utils import (
    create_s3_client,
    list_s3_objects,
)


# Configuration
SALES_BUCKET = "prime-rel-hoogvliet"
SALES_FOLDER = "prime/platform/solutions/price_elasticity/inputs/total/"

OUT_BUCKET = "prime-rel-hoogvliet"
OUT_FOLDER = "prime/platform/temp/daily_sales/sales_with_competitors/"

# Default local path for competitor data
DEFAULT_COMPETITOR_PATH = "competitor_prices (1).parquet"


def filter_price_outliers(
    df: pl.DataFrame,
    window_days: int = 180,
    high_threshold: float = 2.0,
    low_threshold: float = 0.3,
    min_observations: int = 5
) -> pl.DataFrame:
    """
    Filter out price outliers using a rolling window median.

    Args:
        df: DataFrame with product_code, competitor_name, price_date, normalized_price
        window_days: Rolling window size in days (default: 180 = 6 months)
        high_threshold: Filter prices > threshold * rolling median (default: 2.0)
        low_threshold: Filter prices < threshold * rolling median (default: 0.3)
                       Set to None to disable low outlier filtering.
        min_observations: Minimum observations before outlier detection kicks in (default: 5)
                          For earlier observations, uses forward-looking median from first N obs.

    Returns:
        DataFrame with outliers removed
    """
    threshold_desc = f"high={high_threshold}x"
    if low_threshold is not None:
        threshold_desc += f", low={low_threshold}x"
    print(f"Filtering price outliers (window={window_days} days, {threshold_desc})...")

    # Sort by group keys and date
    df = df.sort(["product_code", "competitor_name", "price_date"])

    # Add row number within each group
    df = df.with_columns(
        pl.col("normalized_price").cum_count().over(["product_code", "competitor_name"]).alias("obs_num")
    )

    # Compute forward-looking baseline: median of FIRST N observations per group
    # This catches early outliers without using future data
    df = df.with_columns(
        pl.when(pl.col("obs_num") <= min_observations)
        .then(pl.col("normalized_price"))
        .otherwise(None)
        .median()
        .over(["product_code", "competitor_name"])
        .alias("baseline_median")
    )

    # Compute rolling median per group using rolling with index
    df = df.with_columns(
        pl.col("normalized_price")
        .rolling_median_by(
            by="price_date",
            window_size=f"{window_days}d",
            min_periods=min_observations,
            closed="right",
        )
        .over(["product_code", "competitor_name"])
        .alias("rolling_median")
    )

    # For early observations (before min_observations), use early-period median as baseline
    df = df.with_columns(
        pl.when(pl.col("rolling_median").is_null())
        .then(pl.col("baseline_median"))
        .otherwise(pl.col("rolling_median"))
        .alias("rolling_median")
    ).drop(["baseline_median", "obs_num"])

    # Flag high outliers
    df = df.with_columns(
        (pl.col("normalized_price") > high_threshold * pl.col("rolling_median")).alias("is_high_outlier")
    )

    # Flag low outliers if threshold is set
    if low_threshold is not None:
        df = df.with_columns(
            (pl.col("normalized_price") < low_threshold * pl.col("rolling_median")).alias("is_low_outlier")
        )
    else:
        df = df.with_columns(pl.lit(False).alias("is_low_outlier"))

    # Combine outlier flags
    df = df.with_columns(
        (pl.col("is_high_outlier") | pl.col("is_low_outlier")).alias("is_outlier")
    )

    high_count = df.filter(pl.col("is_high_outlier")).height
    low_count = df.filter(pl.col("is_low_outlier")).height
    total_count = df.filter(pl.col("is_outlier")).height

    print(f"  High outliers: {high_count:,}")
    if low_threshold is not None:
        print(f"  Low outliers: {low_count:,}")
    print(f"  Total outliers: {total_count:,} ({total_count/df.height*100:.3f}%)")

    # Remove outliers and cleanup temp columns
    clean_df = df.filter(~pl.col("is_outlier")).drop(
        ["rolling_median", "is_high_outlier", "is_low_outlier", "is_outlier"]
    )

    return clean_df


def load_competitor_data(
    path: str,
    filter_outliers: bool = True,
    window_days: int = 180,
    high_threshold: float = 2.0,
    low_threshold: float = 0.3
) -> pl.DataFrame:
    """
    Load and prepare competitor price data.

    Args:
        path: Path to competitor prices parquet file
        filter_outliers: Whether to filter price outliers (default: True)
        window_days: Rolling window for outlier detection (default: 180 days)
        high_threshold: Filter prices > threshold * rolling median (default: 2.0)
        low_threshold: Filter prices < threshold * rolling median (default: 0.3)
    """
    print(f"Loading competitor data from {path}...")
    # Use pyarrow backend for consistency with pandas parquet reading
    comp_df = pl.read_parquet(path, use_pyarrow=True)

    # Convert types
    comp_df = comp_df.with_columns([
        pl.col("price_date").cast(pl.Date),
        pl.col("product_code").cast(pl.Utf8),
    ])

    print(f"  Raw data shape: {comp_df.shape}")

    # Filter outliers before grouping
    if filter_outliers:
        comp_df = filter_price_outliers(
            comp_df,
            window_days=window_days,
            high_threshold=high_threshold,
            low_threshold=low_threshold
        )
        print(f"  After outlier filtering: {comp_df.shape}")

    # Handle remaining duplicates by taking median price
    comp_grouped = (
        comp_df
        .group_by(["product_code", "price_date", "competitor_name"])
        .agg(pl.col("normalized_price").median())
    )

    print(f"  After deduplication: {comp_grouped.shape}")

    return comp_grouped


def pivot_competitor_data(comp_grouped: pl.DataFrame) -> pl.DataFrame:
    """Pivot competitor data to wide format with one column per competitor."""
    print("Pivoting competitor data...")

    # Pivot: each competitor gets their own price column
    competitor_pivot = comp_grouped.pivot(
        on="competitor_name",
        index=["product_code", "price_date"],
        values="normalized_price"
    )

    # Rename price_date to join_date
    competitor_pivot = competitor_pivot.rename({"price_date": "join_date"})

    # Rename competitor columns to: jumbo_price, albert_heijn_price, etc.
    rename_map = {
        c: f"{c.lower().strip().replace(' ', '_')}_price"
        for c in competitor_pivot.columns
        if c not in ["product_code", "join_date"]
    }
    competitor_pivot = competitor_pivot.rename(rename_map)

    print(f"Pivoted shape: {competitor_pivot.shape}")
    return competitor_pivot


def _download_single_parquet_polars(
    s3_client,
    bucket: str,
    key: str,
    index: int,
    total: int
) -> Tuple[int, pl.DataFrame]:
    """Download a single parquet file from S3 and return as Polars DataFrame."""
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    # Use pyarrow backend for consistency with pandas parquet reading
    df = pl.read_parquet(io.BytesIO(obj["Body"].read()), use_pyarrow=True)
    return (index, df)


def load_multiple_parquets_from_s3_polars(
    s3_client,
    file_keys: List[str],
    bucket: str,
    max_workers: int = 10,
    show_progress: bool = True
) -> pl.DataFrame:
    """
    Load and combine multiple parquet files from S3 using multi-threading.

    Returns:
        Combined Polars DataFrame
    """
    if not file_keys:
        raise ValueError("file_keys list is empty")

    total = len(file_keys)
    if show_progress:
        print(f"Loading {total} parquet file(s) with {max_workers} threads...")

    dfs = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _download_single_parquet_polars,
                s3_client,
                bucket,
                key,
                idx,
                total
            ): key
            for idx, key in enumerate(file_keys)
        }

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

    combined_df = pl.concat(dfs)
    if show_progress:
        print(f"Combined DataFrame shape: {combined_df.shape}")

    return combined_df


def load_sales_data(
    s3_client,
    bucket: str,
    folder: str,
    max_workers: int = 10
) -> pl.DataFrame:
    """Load all sales parquet files from S3 using multi-threading."""
    print(f"Listing files in s3://{bucket}/{folder}...")
    sales_keys = [
        k for k in list_s3_objects(s3_client, bucket, folder)
        if k.endswith(".parquet")
    ]
    sales_keys = sorted(sales_keys)
    print(f"Found {len(sales_keys)} sales files")

    if not sales_keys:
        raise ValueError(f"No parquet files found in s3://{bucket}/{folder}")

    sales_df = load_multiple_parquets_from_s3_polars(
        s3_client,
        sales_keys,
        bucket,
        max_workers=max_workers
    )

    # Prepare join column
    sales_df = sales_df.with_columns([
        pl.col("date_code").cast(pl.Utf8).str.to_date("%Y%m%d").alias("join_date"),
        pl.col("product_code").cast(pl.Utf8),
    ])

    print(f"Sales data shape: {sales_df.shape}")
    return sales_df


def merge_sales_with_competitors(
    sales_df: pl.DataFrame,
    competitor_pivot: pl.DataFrame
) -> pl.DataFrame:
    """Merge sales data with competitor prices."""
    print("Merging sales with competitor prices...")

    # Left join keeps all sales rows, inserts null where competitor prices missing
    final_df = sales_df.join(
        competitor_pivot,
        on=["product_code", "join_date"],
        how="left"
    ).drop("join_date")

    final_df = final_df.with_columns(
        pl.col("yearweek").cast(pl.Utf8).alias("year_week")
    )

    print(f"Final merged shape: {final_df.shape}")
    return final_df


def upload_to_s3_polars(
    s3_client,
    df: pl.DataFrame,
    bucket: str,
    key: str
) -> None:
    """Upload a Polars DataFrame to S3 as a parquet file."""
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, bucket, key)


def upload_multiple_to_s3_polars(
    s3_client,
    uploads: List[Tuple[pl.DataFrame, str, str]],
    max_workers: int = 5,
    show_progress: bool = True
) -> None:
    """Upload multiple Polars DataFrames to S3 using multi-threading."""
    total = len(uploads)
    if show_progress:
        print(f"Uploading {total} file(s) with {max_workers} threads...")

    completed = 0

    def _upload_single(df, bucket, key):
        upload_to_s3_polars(s3_client, df, bucket, key)
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


def upload_partitioned_data(
    s3_client,
    df: pl.DataFrame,
    bucket: str,
    folder: str,
    chunk_size: int = 2_000_000,
    max_workers: int = 10
) -> None:
    """Upload data partitioned by year_week using multi-threading."""
    print(f"Preparing uploads partitioned by year_week (chunk_size={chunk_size:,})...")

    # Collect all uploads first
    uploads = []
    for yw in df.get_column("year_week").unique().sort().to_list():
        part_df = df.filter(pl.col("year_week") == yw)
        num_rows = part_df.height
        num_parts = (num_rows + chunk_size - 1) // chunk_size

        # Format year_week for path
        s = str(yw)
        if s.isdigit() and len(s) == 6:
            yw_fmt = f"{s[:4]}_{s[4:6]}"
        else:
            yw_fmt = s.replace("-", "_")

        for part_idx in range(num_parts):
            start = part_idx * chunk_size
            end = min((part_idx + 1) * chunk_size, num_rows)
            out_key = f"{folder}year_week={yw_fmt}/part-{part_idx:05d}.parquet"
            uploads.append((part_df.slice(start, end - start), bucket, out_key))

    print(f"Uploading {len(uploads)} files with {max_workers} threads...")
    upload_multiple_to_s3_polars(s3_client, uploads, max_workers=max_workers)


def main(
    competitor_path: Optional[str] = None,
    max_workers: int = 10,
    chunk_size: int = 2_000_000,
    dry_run: bool = False,
    filter_outliers: bool = True,
    outlier_window_days: int = 180,
    outlier_high_threshold: float = 2.0,
    outlier_low_threshold: float = 0.3
) -> None:
    """
    Main processing pipeline.

    Args:
        competitor_path: Path to competitor prices parquet file
        max_workers: Number of concurrent download/upload threads
        chunk_size: Rows per output file partition
        dry_run: If True, skip upload step
        filter_outliers: Whether to filter price outliers
        outlier_window_days: Rolling window for outlier detection (days)
        outlier_high_threshold: Filter prices > threshold * rolling median
        outlier_low_threshold: Filter prices < threshold * rolling median
    """
    if competitor_path is None:
        competitor_path = DEFAULT_COMPETITOR_PATH

    # Check competitor file exists
    if not os.path.exists(competitor_path):
        raise FileNotFoundError(
            f"Competitor file not found: {competitor_path}\n"
            f"Expected file in current directory or provide --competitor-path"
        )

    # Create S3 client with SSO
    print(f"Creating S3 client for bucket: {SALES_BUCKET}")
    s3_client = create_s3_client(bucket=SALES_BUCKET)

    # Load and prepare competitor data (with outlier filtering)
    comp_grouped = load_competitor_data(
        competitor_path,
        filter_outliers=filter_outliers,
        window_days=outlier_window_days,
        high_threshold=outlier_high_threshold,
        low_threshold=outlier_low_threshold
    )
    competitor_pivot = pivot_competitor_data(comp_grouped)

    # Load sales data with multi-threading
    sales_df = load_sales_data(
        s3_client,
        SALES_BUCKET,
        SALES_FOLDER,
        max_workers=max_workers
    )

    # Merge datasets
    final_df = merge_sales_with_competitors(sales_df, competitor_pivot)

    # Memory usage info
    estimated_mb = final_df.estimated_size("mb")
    print(f"\nEstimated memory usage: {estimated_mb:.1f} MB")

    if dry_run:
        print("\nDry run - skipping upload")
        print(f"Would upload to: s3://{OUT_BUCKET}/{OUT_FOLDER}")
    else:
        # Upload partitioned results
        upload_partitioned_data(
            s3_client,
            final_df,
            OUT_BUCKET,
            OUT_FOLDER,
            chunk_size=chunk_size,
            max_workers=max_workers
        )

    print("\nDONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process sales data with competitor prices (Polars version)"
    )
    parser.add_argument(
        "--competitor-path",
        type=str,
        default=DEFAULT_COMPETITOR_PATH,
        help="Path to competitor prices parquet file"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of concurrent download/upload threads (default: 10)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Rows per output file partition (default: 2,000,000)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip upload step, just process data"
    )
    parser.add_argument(
        "--no-outlier-filter",
        action="store_true",
        help="Disable outlier filtering"
    )
    parser.add_argument(
        "--outlier-window",
        type=int,
        default=180,
        help="Rolling window for outlier detection in days (default: 180)"
    )
    parser.add_argument(
        "--outlier-high",
        type=float,
        default=2.0,
        help="Filter prices > threshold * rolling median (default: 2.0)"
    )
    parser.add_argument(
        "--outlier-low",
        type=float,
        default=0.3,
        help="Filter prices < threshold * rolling median (default: 0.3). "
             "Set to 0 to disable low outlier filtering."
    )

    args = parser.parse_args()

    main(
        competitor_path=args.competitor_path,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        dry_run=args.dry_run,
        filter_outliers=not args.no_outlier_filter,
        outlier_window_days=args.outlier_window,
        outlier_high_threshold=args.outlier_high,
        outlier_low_threshold=args.outlier_low if args.outlier_low > 0 else None
    )

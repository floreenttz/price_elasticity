"""
Sales + Competitors data processing pipeline.

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
    python sales_competitors.py

    # With custom settings:
    python sales_competitors.py --max-workers 20 --chunk-size 1000000

    # Adjust outlier thresholds:
    python sales_competitors.py --outlier-high 1.5 --outlier-low 0.5

    # Disable low outlier filtering (if promotions are in data):
    python sales_competitors.py --outlier-low 0

    # Disable all outlier filtering:
    python sales_competitors.py --no-outlier-filter
"""

import argparse
import os
from typing import Optional

import pandas as pd

from aws_utils import (
    create_s3_client,
    list_s3_objects,
    load_multiple_parquets_from_s3,
    upload_multiple_to_s3,
)


# Configuration
SALES_BUCKET = "prime-rel-hoogvliet"
SALES_FOLDER = "prime/platform/solutions/price_elasticity/inputs/total/"

OUT_BUCKET = "prime-rel-hoogvliet"
OUT_FOLDER = "prime/platform/temp/daily_sales/sales_with_competitors/"

# Default local path for competitor data
DEFAULT_COMPETITOR_PATH = "competitor_prices (1).parquet"


def filter_price_outliers(
    df: pd.DataFrame,
    window_days: int = 180,
    high_threshold: float = 2.0,
    low_threshold: float = 0.3,
    min_observations: int = 5
) -> pd.DataFrame:
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

    df = df.sort_values(["product_code", "competitor_name", "price_date"]).reset_index(drop=True)

    # Compute forward-looking baseline: median of FIRST N observations per group
    # This catches early outliers without using future data
    early_medians = (
        df.groupby(["product_code", "competitor_name"])["normalized_price"]
        .apply(lambda x: x.head(min_observations).median())
    )
    baseline_median = df.set_index(["product_code", "competitor_name"]).index.map(early_medians)

    # Compute rolling median per group using time-based index
    df = df.set_index("price_date")
    df["rolling_median"] = (
        df.groupby(["product_code", "competitor_name"])["normalized_price"]
        .apply(lambda x: x.sort_index().rolling(f"{window_days}D", min_periods=min_observations).median())
        .droplevel([0, 1])
    )
    df = df.reset_index()

    # For early observations (before min_observations), use early-period median as baseline
    df["rolling_median"] = df["rolling_median"].fillna(pd.Series(baseline_median, index=df.index))

    # Flag high outliers (likely data errors like €249 instead of €2.49)
    df["is_high_outlier"] = df["normalized_price"] > high_threshold * df["rolling_median"]

    # Optionally flag low outliers (use with caution - may be promotions)
    if low_threshold is not None:
        df["is_low_outlier"] = df["normalized_price"] < low_threshold * df["rolling_median"]
    else:
        df["is_low_outlier"] = False

    df["is_outlier"] = df["is_high_outlier"] | df["is_low_outlier"]

    high_count = df["is_high_outlier"].sum()
    low_count = df["is_low_outlier"].sum()
    total_count = df["is_outlier"].sum()

    print(f"  High outliers: {high_count:,}")
    if low_threshold is not None:
        print(f"  Low outliers: {low_count:,}")
    print(f"  Total outliers: {total_count:,} ({total_count/len(df)*100:.3f}%)")

    # Remove outliers and cleanup temp columns
    clean_df = df[~df["is_outlier"]].drop(
        columns=["rolling_median", "is_high_outlier", "is_low_outlier", "is_outlier"]
    )

    return clean_df


def load_competitor_data(
    path: str,
    filter_outliers: bool = True,
    window_days: int = 180,
    high_threshold: float = 2.0,
    low_threshold: float = 0.3
) -> pd.DataFrame:
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
    comp_df = pd.read_parquet(path)
    comp_df["price_date"] = pd.to_datetime(comp_df["price_date"], errors="coerce")
    comp_df["product_code"] = comp_df["product_code"].astype(str)

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
    comp_grouped = comp_df.groupby(
        ["product_code", "price_date", "competitor_name"], as_index=False
    )["normalized_price"].median()

    print(f"  After deduplication: {comp_grouped.shape}")

    return comp_grouped


def pivot_competitor_data(comp_grouped: pd.DataFrame) -> pd.DataFrame:
    """Pivot competitor data to wide format with one column per competitor."""
    print("Pivoting competitor data...")

    # Pivot: each competitor gets their own price column
    competitor_pivot = comp_grouped.pivot(
        index=["product_code", "price_date"],
        columns="competitor_name",
        values="normalized_price"
    ).reset_index()

    competitor_pivot.columns.name = None
    competitor_pivot = competitor_pivot.rename(columns={"price_date": "join_date"})

    # Rename competitor columns to: jumbo_price, albert_heijn_price, etc.
    competitor_pivot = competitor_pivot.rename(
        columns={
            c: f"{str(c).lower().strip().replace(' ', '_')}_price"
            for c in competitor_pivot.columns
            if c not in ["product_code", "join_date"]
        }
    )

    print(f"Pivoted shape: {competitor_pivot.shape}")
    return competitor_pivot


def load_sales_data(
    s3_client,
    bucket: str,
    folder: str,
    max_workers: int = 10
) -> pd.DataFrame:
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

    sales_df = load_multiple_parquets_from_s3(
        s3_client,
        sales_keys,
        bucket,
        max_workers=max_workers
    )

    # Prepare join column
    sales_df["join_date"] = pd.to_datetime(
        sales_df["date_code"].astype(str), format="%Y%m%d", errors="coerce"
    )
    sales_df["product_code"] = sales_df["product_code"].astype(str)

    print(f"Sales data shape: {sales_df.shape}")
    return sales_df


def merge_sales_with_competitors(
    sales_df: pd.DataFrame,
    competitor_pivot: pd.DataFrame
) -> pd.DataFrame:
    """Merge sales data with competitor prices."""
    print("Merging sales with competitor prices...")

    # Left join keeps all sales rows, inserts NaN where competitor prices missing
    final_df = sales_df.merge(
        competitor_pivot,
        on=["product_code", "join_date"],
        how="left"
    ).drop(columns=["join_date"])

    final_df["year_week"] = final_df["yearweek"].astype(str)

    print(f"Final merged shape: {final_df.shape}")
    return final_df


def upload_partitioned_data(
    s3_client,
    df: pd.DataFrame,
    bucket: str,
    folder: str,
    chunk_size: int = 2_000_000,
    max_workers: int = 10
) -> None:
    """Upload data partitioned by year_week using multi-threading."""
    print(f"Preparing uploads partitioned by year_week (chunk_size={chunk_size:,})...")

    # Collect all uploads first
    uploads = []
    for yw, part_df in df.groupby("year_week", sort=True):
        part_df = part_df.reset_index(drop=True)
        num_parts = (len(part_df) + chunk_size - 1) // chunk_size

        # Format year_week for path
        s = str(yw)
        if s.isdigit() and len(s) == 6:
            yw_fmt = f"{s[:4]}_{s[4:6]}"
        else:
            yw_fmt = s.replace("-", "_")

        for part_idx in range(num_parts):
            start = part_idx * chunk_size
            end = min((part_idx + 1) * chunk_size, len(part_df))
            out_key = f"{folder}year_week={yw_fmt}/part-{part_idx:05d}.parquet"
            uploads.append((part_df.iloc[start:end].copy(), bucket, out_key))

    print(f"Uploading {len(uploads)} files with {max_workers} threads...")
    upload_multiple_to_s3(s3_client, uploads, max_workers=max_workers)


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
    sales_mb = sales_df.memory_usage(deep=True).sum() / 1024**2
    final_mb = final_df.memory_usage(deep=True).sum() / 1024**2
    print(f"\nMemory usage:")
    print(f"  Sales: {sales_mb:.1f} MB")
    print(f"  Final: {final_mb:.1f} MB")

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
        description="Process sales data with competitor prices"
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

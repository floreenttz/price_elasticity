"""
Run the price elasticity pipeline for Hoogvliet.

Usage:
    python scripts/run_hoogvliet.py              # Full run
    python scripts/run_hoogvliet.py --subset     # Subset (10% of data, faster)

Works on both SageMaker (uses IAM role) and locally (uses SSO).
"""

import sys
from pathlib import Path

# Ensure scripts directory is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
from scripts.clients.hoogvliet import HoogvlietAdapter
from scripts.storage.s3 import S3Storage
from scripts.pipeline.preprocessing import DataPreprocessor
from scripts.pipeline.features import FeatureEngineer
from scripts.pipeline.modeling import DemandModel
from scripts.pipeline.elasticity import ElasticityCalculator

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pipeline")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Set to True for quick testing with 10% of data
SUBSET = False

# Config file path
CONFIG_PATH = "config_files/config_hoogvliet.yaml"


# =============================================================================
# PIPELINE
# =============================================================================

def run_hoogvliet_pipeline(subset: bool = False, config_path: str = CONFIG_PATH):
    """
    Run the full pipeline for Hoogvliet.

    Args:
        subset: If True, use only 10% of data (faster for testing)
        config_path: Path to config YAML file
    """

    logger.info("Starting pipeline for client: hoogvliet")
    if subset:
        logger.info("*** RUNNING IN SUBSET MODE (10% of data) ***")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize client adapter and storage
    client = HoogvlietAdapter()
    bucket = config["s3_dir"]["bucket"].rstrip("/")
    storage = S3Storage(bucket=bucket)

    # =========================================================================
    # STEP 1: Preprocessing
    # =========================================================================
    logger.info("=" * 50)
    logger.info("STEP 1: Preprocessing")
    logger.info("=" * 50)

    preprocessor = DataPreprocessor(
        client=client,
        config=config,
        storage=storage,
        frequency="daily",
    )
    data = preprocessor.run(subset=subset, overview=True)
    preprocessor.save()

    logger.info(f"Preprocessed: {len(data):,} rows, {data['product_code'].nunique():,} products")

    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    logger.info("=" * 50)
    logger.info("STEP 2: Feature Engineering")
    logger.info("=" * 50)

    feature_engineer = FeatureEngineer(
        client=client,
        config=config,
        storage=storage,
    )
    feature_data = feature_engineer.run(data=data)
    feature_engineer.save(frequency="daily")

    logger.info(f"Features: {len(feature_data):,} rows, {len(feature_data.columns)} columns")

    # =========================================================================
    # STEP 3: Demand Modeling
    # =========================================================================
    logger.info("=" * 50)
    logger.info("STEP 3: Demand Modeling")
    logger.info("=" * 50)

    model = DemandModel(
        client=client,
        config=config,
        storage=storage,
        frequency="daily",
    )
    predictions = model.run(data=feature_data)
    model.save()

    logger.info(f"Predictions: {len(predictions):,} rows, {predictions['product_code'].nunique():,} products")

    # =========================================================================
    # STEP 4: Elasticity Calculation
    # =========================================================================
    logger.info("=" * 50)
    logger.info("STEP 4: Elasticity Calculation")
    logger.info("=" * 50)

    elasticity_calc = ElasticityCalculator(
        client=client,
        config=config,
        storage=storage,
        frequency="daily",
    )
    elasticities = elasticity_calc.run(
        estimations=predictions,
        preprocessed_data=data,
    )
    elasticity_calc.save()

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 50)
    logger.info("Pipeline complete!")
    logger.info("=" * 50)

    # Filter out zero elasticities for summary
    non_zero = elasticities[elasticities['elasticity'] != 0]

    summary = {
        "total_products": len(elasticities),
        "products_with_elasticity": len(non_zero),
        "median_elasticity": non_zero['elasticity'].median(),
        "mean_elasticity": non_zero['elasticity'].mean(),
        "elastic_count": len(non_zero[(non_zero['elasticity'] < -1) & (non_zero['elasticity'] > -5)]),
        "inelastic_count": len(non_zero[(non_zero['elasticity'] >= -1) & (non_zero['elasticity'] < 0)]),
        "positive_count": len(non_zero[non_zero['elasticity'] > 0]),
    }

    for key, value in summary.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value:,}")

    return elasticities


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Hoogvliet price elasticity pipeline")
    parser.add_argument(
        "--subset",
        action="store_true",
        default=SUBSET,
        help="Run with 10%% subset of data (faster for testing)"
    )
    args = parser.parse_args()

    # Run pipeline
    elasticities = run_hoogvliet_pipeline(subset=args.subset)
    print(f"\nDone! Calculated elasticities for {len(elasticities):,} products.")

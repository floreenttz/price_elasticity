"""
Pipeline runner for price elasticity calculation.

This module provides the entry point for running the full pipeline
or individual steps with any client.

Usage:
    # Full pipeline with S3
    python -m scripts.run_pipeline --client hoogvliet

    # Full pipeline for Spar with priceline
    python -m scripts.run_pipeline --client spar --priceline Enjoy

    # Local testing
    python -m scripts.run_pipeline --client hoogvliet --local --base-path ./test_data
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.clients.hoogvliet import HoogvlietAdapter
from scripts.clients.spar import SparAdapter
from scripts.storage.s3 import S3Storage
from scripts.storage.local import LocalStorage
from scripts.pipeline.preprocessing import DataPreprocessor
from scripts.pipeline.features import FeatureEngineer
from scripts.pipeline.modeling import DemandModel
from scripts.pipeline.elasticity import ElasticityCalculator

# Registry mapping client names to factory functions.
CLIENT_REGISTRY = {
    "hoogvliet": lambda priceline: HoogvlietAdapter(),
    "spar": lambda priceline: SparAdapter(priceline=priceline)
}

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_client(client_name: str, priceline: str | None = None):
    """Get the appropriate client adapter."""
    if client_name not in CLIENT_REGISTRY: 
        raise ValueError(f"Unknown client: '{client_name}'. Available: {list(CLIENT_REGISTRY)}")
    if client_name == "spar" and priceline is None: 
        raise ValueError("Spar requires a priceline (e.g. 'Enjoy', 'City')")
    return CLIENT_REGISTRY[client_name](priceline)



def get_storage(local: bool = False, base_path: str = ".", bucket: str | None = None):
    """Get the appropriate storage backend."""
    if local:
        return LocalStorage(base_path=base_path)
    else:
        return S3Storage(bucket=bucket)


def run_pipeline(
    client_name: str,
    config_path: str,
    priceline: str | None = None,
    local: bool = False,
    base_path: str = ".",
    frequency: str = "daily",
    subset: bool = False,
    steps: list[str] | None = None,
    xgb_params: dict | None = None,
) -> dict:
    """
    Run the full price elasticity pipeline.

    Args:
        client_name: Client name ('hoogvliet' or 'spar').
        config_path: Path to config YAML file.
        priceline: Priceline name (required for Spar).
        local: If True, use local storage instead of S3.
        base_path: Base path for local storage.
        frequency: 'daily' or 'weekly'.
        subset: If True, use subset of data for testing.
        steps: List of steps to run. If None, runs all.
               Options: ['preprocess', 'features', 'model', 'elasticity']
        xgb_params: Optional XGBoost hyperparameters.

    Returns:
        Dictionary with results from each step.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("pipeline")

    logger.info(f"Starting pipeline for client: {client_name}")

    # Load config
    config = load_config(config_path)

    # Get client adapter
    client = get_client(client_name, priceline)

    # Get storage
    bucket = config.get("s3_dir", {}).get("bucket", "").rstrip("/")
    storage = get_storage(local=local, base_path=base_path, bucket=bucket)

    # Determine which steps to run
    all_steps = ["preprocess", "features", "model", "elasticity"]
    steps_to_run = steps or all_steps

    results = {}

    # Step 1: Preprocessing
    if "preprocess" in steps_to_run:
        logger.info("=" * 50)
        logger.info("STEP 1: Preprocessing")
        logger.info("=" * 50)

        preprocessor = DataPreprocessor(
            client=client,
            config=config,
            storage=storage,
            frequency=frequency,
        )
        data = preprocessor.run(subset=subset)
        preprocessor.save()

        results["preprocess"] = {
            "rows": len(data),
            "products": data["product_code"].nunique(),
            "price_grid_products": len(preprocessor.price_grid),
        }

    # Step 2: Feature Engineering
    if "features" in steps_to_run:
        logger.info("=" * 50)
        logger.info("STEP 2: Feature Engineering")
        logger.info("=" * 50)

        feature_engineer = FeatureEngineer(
            client=client,
            config=config,
            storage=storage,
        )

        # Pass data from previous step if available
        input_data = results.get("preprocess", {}).get("data")
        feature_data = feature_engineer.run(data=input_data)
        feature_engineer.save(frequency=frequency)

        results["features"] = {
            "rows": len(feature_data),
            "num_features": len(feature_engineer.features) if hasattr(feature_engineer, 'features') else len(feature_data.columns),
        }

    # Step 3: Demand Modeling
    if "model" in steps_to_run:
        logger.info("=" * 50)
        logger.info("STEP 3: Demand Modeling")
        logger.info("=" * 50)

        model = DemandModel(
            client=client,
            config=config,
            storage=storage,
            params=xgb_params,
            frequency=frequency,
        )
        predictions = model.run()
        model.save()

        results["model"] = {
            "predictions": len(predictions),
            "products": predictions["product_code"].nunique() if not predictions.empty else 0,
        }

    # Step 4: Elasticity Calculation
    if "elasticity" in steps_to_run:
        logger.info("=" * 50)
        logger.info("STEP 4: Elasticity Calculation")
        logger.info("=" * 50)

        elasticity_calc = ElasticityCalculator(
            client=client,
            config=config,
            storage=storage,
            frequency=frequency,
        )
        elasticities = elasticity_calc.run()
        elasticity_calc.save()

        summary = elasticity_calc.get_summary()
        results["elasticity"] = summary

    logger.info("=" * 50)
    logger.info("Pipeline complete!")
    logger.info("=" * 50)

    for step, result in results.items():
        logger.info(f"{step}: {result}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run price elasticity pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline for Hoogvliet
    python -m scripts.run_pipeline --client hoogvliet --config config_files/config_hoogvliet.yaml

    # Run for Spar Enjoy priceline
    python -m scripts.run_pipeline --client spar --priceline Enjoy --config config_files/config_spar_Enjoy.yaml

    # Run locally for testing
    python -m scripts.run_pipeline --client hoogvliet --local --base-path ./test_data --config config_files/config_hoogvliet.yaml

    # Run only preprocessing step
    python -m scripts.run_pipeline --client hoogvliet --steps preprocess --config config_files/config_hoogvliet.yaml
        """,
    )

    parser.add_argument(
        "--client",
        type=str,
        required=True,
        choices=list(CLIENT_REGISTRY.keys()),
        help="Client name",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--priceline",
        type=str,
        help="Priceline name (required for Spar)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local storage instead of S3",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=".",
        help="Base path for local storage",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="daily",
        choices=["daily", "weekly"],
        help="Data frequency",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Use subset of data for testing",
    )
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        choices=["preprocess", "features", "model", "elasticity"],
        help="Specific steps to run (default: all)",
    )

    args = parser.parse_args()

    run_pipeline(
        client_name=args.client,
        config_path=args.config,
        priceline=args.priceline,
        local=args.local,
        base_path=args.base_path,
        frequency=args.frequency,
        subset=args.subset,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()

"""
Synthetic data generator for price elasticity model validation.

Generates controlled retail data with known true elasticities per category,
so that estimated elasticities can be compared against ground truth.
"""

import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    """
    Generates synthetic weekly retail sales data with known price elasticities.

    Parameters
    ----------
    n_products : int
        Total number of products to generate.
    n_categories : int
        Number of categories to distribute products across.
    n_weeks : int
        Number of weeks in the time series.
    seed : int
        Random seed for reproducibility.
    price_mean : float
        Mean of log-normal distribution for base prices (log scale).
    price_sigma : float
        Std of log-normal distribution for base prices (log scale).
    quantity_mean : float
        Mean of log-normal distribution for base quantities (log scale).
    quantity_sigma : float
        Std of log-normal distribution for base quantities (log scale).
    eta_range : tuple[float, float]
        (min, max) range for true elasticity per category. Values should be negative.
    baseline_price_noise : float
        Std of weekly Gaussian noise applied to log price.
    promo_frequency : float
        Probability of a promo shock in any given week per product.
    promo_depth : float
        Price discount during promo weeks (e.g. 0.20 = 20% off).
    seasonality_amplitude : float
        Amplitude of the seasonal sine wave on log quantity.
    confounder_promo_effect : float
        Direct demand lift from promotion, independent of price (log scale).
    confounder_shock_std : float
        Std of random weekly demand shocks (weather, events, etc.) on log quantity.
    quantity_noise_sigma : float
        Std of multiplicative log-normal noise on final quantity.
    """

    def __init__(
        self,
        n_products: int = 10_000,
        n_categories: int = 50,
        n_weeks: int = 104,
        seed: int = 42,
        price_mean: float = 2.82,
        price_sigma: float = 1.16,
        quantity_mean: float = 0.91,
        quantity_sigma: float = 0.86,
        eta_range: tuple = (-3.0, -0.5),
        baseline_price_noise: float = 0.02,
        promo_frequency: float = 0.08,
        promo_depth: float = 0.20,
        seasonality_amplitude: float = 0.15,
        confounder_promo_effect: float = 0.10,
        confounder_shock_std: float = 0.05,
        quantity_noise_sigma: float = 0.10,
    ):
        self.n_products = n_products
        self.n_categories = n_categories
        self.n_weeks = n_weeks
        self.seed = seed
        self.price_mean = price_mean
        self.price_sigma = price_sigma
        self.quantity_mean = quantity_mean
        self.quantity_sigma = quantity_sigma
        self.eta_range = eta_range
        self.baseline_price_noise = baseline_price_noise
        self.promo_frequency = promo_frequency
        self.promo_depth = promo_depth
        self.seasonality_amplitude = seasonality_amplitude
        self.confounder_promo_effect = confounder_promo_effect
        self.confounder_shock_std = confounder_shock_std
        self.quantity_noise_sigma = quantity_noise_sigma

        self.rng = np.random.default_rng(seed)

    def _build_product_catalog(self) -> pd.DataFrame:
        """Assign products to categories with base prices and quantities."""
        product_ids = [f"P{i:05d}" for i in range(self.n_products)]
        category_ids = self.rng.integers(0, self.n_categories, size=self.n_products)

        base_prices = self.rng.lognormal(self.price_mean, self.price_sigma, size=self.n_products)
        base_quantities = self.rng.lognormal(self.quantity_mean, self.quantity_sigma, size=self.n_products)

        catalog = pd.DataFrame({
            "product_code": product_ids,
            "category_id": category_ids,
            "base_price": base_prices,
            "base_quantity": base_quantities,
        })

        # Category name columns matching preprocessed schema
        catalog["category_name_level1"] = catalog["category_id"].apply(
            lambda x: f"L1_{x // 10:02d}"
        )
        catalog["category_name_level2"] = catalog["category_id"].apply(
            lambda x: f"L2_{x // 5:02d}"
        )
        catalog["category_name_level3"] = catalog["category_id"].apply(
            lambda x: f"CAT_{x:02d}"
        )

        return catalog

    def _build_category_elasticities(self) -> dict[int, float]:
        """Sample one true elasticity per category."""
        lo, hi = self.eta_range
        etas = self.rng.uniform(lo, hi, size=self.n_categories)
        return {i: etas[i] for i in range(self.n_categories)}

    def _build_seasonality(self) -> np.ndarray:
        """Build a 52-week repeating seasonal curve (sine wave)."""
        weeks = np.arange(self.n_weeks)
        return self.seasonality_amplitude * np.sin(2 * np.pi * weeks / 52)

    def generate(self, start_date: str = "2023-01-02") -> pd.DataFrame:
        """
        Generate the full synthetic dataset.

        Returns
        -------
        pd.DataFrame
            Weekly panel with columns matching the preprocessed schema plus true_eta.
        """
        catalog = self._build_product_catalog()
        eta_map = self._build_category_elasticities()
        seasonality = self._build_seasonality()

        start = pd.Timestamp(start_date)
        dates = [start + pd.Timedelta(weeks=w) for w in range(self.n_weeks)]

        records = []

        for _, product in catalog.iterrows():
            cat_id = int(product["category_id"])
            eta_c = eta_map[cat_id]
            p0 = product["base_price"]
            q0 = product["base_quantity"]

            # Weekly price noise
            price_noise = self.rng.normal(0, self.baseline_price_noise, size=self.n_weeks)

            # Promo shocks: random weeks
            promo_mask = self.rng.random(size=self.n_weeks) < self.promo_frequency
            promo_shock = np.where(promo_mask, -self.promo_depth, 0.0)

            # Log price deviation from base
            delta_log_p = price_noise + np.log1p(promo_shock)

            # Demand confounders
            promo_demand = promo_mask.astype(float) * self.confounder_promo_effect
            demand_shocks = self.rng.normal(0, self.confounder_shock_std, size=self.n_weeks)

            # Quantity noise
            eps = self.rng.normal(0, self.quantity_noise_sigma, size=self.n_weeks)

            # Core formula: log q_t = log q0 + η_c * Δlog p_t + seasonal + confounders + ε
            log_q = (
                np.log(q0)
                + eta_c * delta_log_p
                + seasonality
                + promo_demand
                + demand_shocks
                + eps
            )

            quantity = np.exp(log_q).round().astype(int)
            quantity = np.maximum(quantity, 0)

            selling_price = p0 * np.exp(delta_log_p)

            for w in range(self.n_weeks):
                iso = dates[w].isocalendar()
                records.append({
                    "product_code": product["product_code"],
                    "date": dates[w],
                    "year_week": f"{iso.year}_{iso.week:02d}",
                    "category_name_level1": product["category_name_level1"],
                    "category_name_level2": product["category_name_level2"],
                    "category_name_level3": product["category_name_level3"],
                    "recommended_retail_price": round(p0, 2),
                    "product_selling_price": round(selling_price[w], 2),
                    "quantity_sold": quantity[w],
                    "promotion_indicator": bool(promo_mask[w]),
                    "true_eta": eta_c,
                })

        return pd.DataFrame(records)

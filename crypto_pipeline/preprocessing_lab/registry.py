# preprocessing_lab/registry.py

"""
registry.py
-----------
The ONLY place that maps a config.yaml string to an actual preprocessing
function. main.py never has if/elif chains for method names -- it does:

    func = PREPROCESSING_REGISTRY[config["method"]]
    transformed_df, fit_info = func(df, **config.get("params", {}))

To add a new method: write the function in preprocessing/, add one line
below, add the name to config.yaml. Nothing else changes.
"""

from typing import Callable, Dict

from crypto_pipeline.preprocessing_lab.scalers import (
    apply_standard_scaler,
    apply_minmax_scaler,
    apply_robust_scaler,
    apply_maxabs_scaler,
    apply_quantile_transformer,
    apply_power_transformer,
    apply_normalizer,
    apply_rolling_zscore,
)
from crypto_pipeline.preprocessing_lab.stationarity import (
    apply_fractional_differencing,
    apply_simple_differencing,
)
from crypto_pipeline.preprocessing_lab.distribution import (
    apply_winsorization,
    apply_log_transform,
    apply_gaussian_quantile_transform,
)


def apply_none(df, columns=None):
    """
    Baseline / control group: no preprocessing at all.

    Every other method is judged AGAINST this. Without it, we'd only know
    scalers differ from each other, not whether scaling helps at all.
    """
    return df.copy(), {}


PREPROCESSING_REGISTRY: Dict[str, Callable] = {
    # baseline
    "none": apply_none,

    # scaling -- required 5
    "standard_scaler": apply_standard_scaler,
    "minmax_scaler": apply_minmax_scaler,
    "robust_scaler": apply_robust_scaler,
    "maxabs_scaler": apply_maxabs_scaler,
    "quantile_transformer": apply_quantile_transformer,

    # scaling -- extra 3 (chosen for crypto-specific properties)
    "power_transformer": apply_power_transformer,
    "normalizer": apply_normalizer,
    "rolling_zscore": apply_rolling_zscore,

    # stationarity
    "fractional_differencing": apply_fractional_differencing,
    "simple_differencing": apply_simple_differencing,

    # distribution processing
    "winsorization": apply_winsorization,
    "log_transform": apply_log_transform,
    "gaussian_quantile_transform": apply_gaussian_quantile_transform,
}
# crypto_pipeline/stats/plots.py

"""
plots.py
--------
Used to save quantstats.plots PNGs. Now generates the same underlying
numeric data those plots were drawn from instead, so it can be stored in
JSON (no matplotlib figures, no images on disk).

Kept as a name -> function map (same registry idea as
preprocessing_lab/registry.py) so config.yaml's `plots:` list drives
exactly which series get computed, with no if/elif chain here.

Each entry in PLOT_DATA_REGISTRY takes (returns, equity) and returns
something JSON-safe (dict / list of records) -- callers should still run
the result through utils.to_json_safe before writing it out.
"""

import pandas as pd
import quantstats as qs


def _series_to_records(series: pd.Series) -> dict:
    """{timestamp_str: value} for a datetime-indexed series."""
    return {str(idx): val for idx, val in series.items()}


def _returns_data(returns: pd.Series, equity: pd.Series) -> dict:
    """Plain period returns, plus cumulative compounded returns."""
    cumulative = qs.stats.compsum(returns)
    return {
        "returns": _series_to_records(returns),
        "cumulative_returns": _series_to_records(cumulative),
    }


def _drawdown_data(returns: pd.Series, equity: pd.Series) -> dict:
    drawdown = qs.stats.to_drawdown_series(returns)
    details = qs.stats.drawdown_details(drawdown)
    details_records = (
        details.to_dict(orient="records") if isinstance(details, pd.DataFrame) else []
    )
    return {
        "drawdown_series": _series_to_records(drawdown),
        "drawdown_periods": details_records,
    }


def _rolling_sharpe_data(returns: pd.Series, equity: pd.Series) -> dict:
    rolling = qs.stats.rolling_sharpe(returns)
    return {"rolling_sharpe": _series_to_records(rolling)}


def _rolling_volatility_data(returns: pd.Series, equity: pd.Series) -> dict:
    rolling = qs.stats.rolling_volatility(returns)
    return {"rolling_volatility": _series_to_records(rolling)}


def _monthly_heatmap_data(returns: pd.Series, equity: pd.Series) -> dict:
    monthly = qs.stats.monthly_returns(returns)
    if isinstance(monthly, pd.DataFrame):
        monthly.index = monthly.index.astype(str)
        table = {str(year): row.dropna().to_dict() for year, row in monthly.iterrows()}
    else:
        table = _series_to_records(monthly)
    return {"monthly_returns": table}


def _yearly_returns_data(returns: pd.Series, equity: pd.Series) -> dict:
    yearly = returns.resample("YE").apply(lambda r: (1 + r).prod() - 1)
    yearly.index = yearly.index.year.astype(str)
    return {"yearly_returns": yearly.to_dict()}


def _distribution_data(returns: pd.Series, equity: pd.Series) -> dict:
    return {"distribution": qs.stats.distribution(returns)}


PLOT_DATA_REGISTRY = {
    "returns": _returns_data,
    "drawdown": _drawdown_data,
    "rolling_sharpe": _rolling_sharpe_data,
    "rolling_volatility": _rolling_volatility_data,
    "monthly_heatmap": _monthly_heatmap_data,
    "yearly_returns": _yearly_returns_data,
    "distribution": _distribution_data,
}


def generate_plot_data(returns: pd.Series, equity: pd.Series, plot_names: list) -> dict:
    """
    Computes the numeric data behind each requested "plot" and returns
    {plot_name: {...data...}}. A plot whose data can't be computed on
    this data (e.g. not enough points for a rolling window) is skipped
    rather than aborting the rest.
    """
    result = {}

    for name in plot_names:
        func = PLOT_DATA_REGISTRY.get(name)
        if func is None:
            print(f"  (skip plot '{name}': not in PLOT_DATA_REGISTRY)")
            continue
        try:
            result[name] = func(returns, equity)
        except Exception as e:
            print(f"  (skip plot '{name}': {e})")

    return result
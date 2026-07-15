# crypto_pipeline/stats/plots.py

"""
plots.py
--------
Thin wrappers around quantstats.plots, each saving one .png. Kept as a
name -> function map (same registry idea as preprocessing_lab/registry.py)
so config.yaml's `plots:` list drives exactly which ones run, with no
if/elif chain here.
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless -- no display available when run as a pipeline step

import quantstats as qs


def _save(plot_func, returns, path, **kwargs):
    plot_func(returns, savefig=path, show=False, **kwargs)


PLOT_REGISTRY = {
    "returns": lambda returns, path: _save(qs.plots.returns, returns, path),
    "drawdown": lambda returns, path: _save(qs.plots.drawdown, returns, path),
    "rolling_sharpe": lambda returns, path: _save(qs.plots.rolling_sharpe, returns, path),
    "rolling_volatility": lambda returns, path: _save(qs.plots.rolling_volatility, returns, path),
    "monthly_heatmap": lambda returns, path: _save(qs.plots.monthly_heatmap, returns, path),
    "yearly_returns": lambda returns, path: _save(qs.plots.yearly_returns, returns, path),
    "distribution": lambda returns, path: _save(qs.plots.distribution, returns, path),
}


def generate_plots(returns, out_dir: str, plot_names: list) -> list:
    """
    Generates each requested plot and saves it as <out_dir>/<name>.png.
    A plot that fails on this data (e.g. not enough points for a rolling
    window) is skipped rather than aborting the rest.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    for name in plot_names:
        plot_func = PLOT_REGISTRY.get(name)
        if plot_func is None:
            print(f"  (skip plot '{name}': not in PLOT_REGISTRY)")
            continue
        path = os.path.join(out_dir, f"{name}.png")
        try:
            plot_func(returns, path)
            saved.append(path)
        except Exception as e:
            print(f"  (skip plot '{name}': {e})")

    return saved
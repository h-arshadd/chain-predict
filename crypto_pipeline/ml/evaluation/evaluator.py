# crypto_pipeline/ml/evaluation/evaluator.py

"""
evaluator.py
------------
Model Evaluation stage (PDF heading 10) -- the two-stage evaluation:
    1. Machine Learning Evaluation  (regression_metrics.py / classification_metrics.py)
    2. Trading Strategy Evaluation  (backtest.run_backtest() + stats.calculator.compute_stats())

evaluate_model() runs both stages for one trained model and returns a
single result dict. select_best_model() then picks the best of several
such results using whichever trading metric ml/config.yaml's
evaluation.primary_metric names -- ML metrics are computed and reported,
but per the PDF are explicitly NOT used for model selection.

Backtesting/Stats integration: this module calls the project's own
Backtesting and Statistics modules directly -- run_backtest() from
crypto_pipeline.backtest.backtest and compute_stats() from
crypto_pipeline.stats.calculator -- rather than re-implementing trade
simulation or metric computation here. Signal generation (ml/signals/)
already produced the Buy/Sell/Hold array; this module's only job is to
hand that array plus the 1-minute OHLCV to run_backtest(), then hand
run_backtest()'s result straight to compute_stats(), and read the
"metrics" dict compute_stats() returns.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from crypto_pipeline.backtest.backtest import run_backtest
from crypto_pipeline.stats.calculator import compute_stats
from crypto_pipeline.ml.evaluation.classification_metrics import compute_classification_metrics
from crypto_pipeline.ml.evaluation.regression_metrics import compute_regression_metrics

logger = logging.getLogger(__name__)

# PDF heading 10's "Supported optimization metrics" list, mapped to the
# actual quantstats metric name compute_stats()["metrics"] uses (see
# stats/metrics.py -- these are discovered directly off quantstats.stats,
# e.g. "sharpe" not "sharpe_ratio"). ml/config.yaml's
# evaluation.primary_metric is written using the PDF's own names on the
# left; this is the one place that translates to the stats module's
# actual dict keys.
_METRIC_NAME_MAP = {
    "total_return": "comp",
    "sharpe_ratio": "sharpe",
    "sortino_ratio": "sortino",
    "calmar_ratio": "calmar",
    "max_drawdown": "max_drawdown",
    "profit_factor": "profit_factor",
    "win_rate": "win_rate",
}

# Metrics where LOWER is better (used when ranking models in select_best_model()).
# Every other metric in _METRIC_NAME_MAP is "higher is better".
# max_drawdown from quantstats is reported as a negative number (e.g.
# -0.15 for a 15% drawdown), so "lower is better" is the correct
# direction here (more negative = worse).
_LOWER_IS_BETTER = {"max_drawdown"}


def evaluate_model(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    signals: np.ndarray,
    signal_timestamps: pd.Series,
    ohlcv_1m: pd.DataFrame,
    backtest_config: dict,
    stats_config: dict,
    plot_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    """
    Run both evaluation stages for one trained model.

    Args:
        task_type: "regression", "classification", "timeseries_regression",
            or "timeseries_classification" -- picks which ML metrics
            module runs (timeseries_regression reuses
            compute_regression_metrics(), since a price forecast's
            error is the same MAE/RMSE/etc shape as a return
            prediction's; timeseries_classification reuses
            compute_classification_metrics() the same way. "timeseries"
            is still accepted as an alias for "timeseries_regression",
            for compatibility with any external caller still using it.)
        y_true: true target values/labels for the test set.
        y_pred: predicted values/labels for the test set (e.g.
            prediction_result["predictions"] from predictor.py).
        signals: Buy/Sell/Hold array from regression_signals.py /
            classification_signals.py, same row order as y_true/y_pred.
        signal_timestamps: datetime for each row of `signals` (the test
            set's timestamp column) -- paired with `signals` to build
            the datetime/signal DataFrame run_backtest() expects.
        ohlcv_1m: 1-minute OHLCV DataFrame (datetime, open, high, low,
            close) covering the test period, for backtest execution.
            Signals may be on a coarser timeframe than 1 minute --
            run_backtest() itself aligns them onto the 1-minute grid.
        backtest_config: backtest/config.yaml dict, passed straight to
            run_backtest().
        stats_config: stats/config.yaml dict, passed straight to
            compute_stats().
        plot_dir: optional directory to save quantstats plots into (see
            stats/plots.py) -- skipped if not given.
        run_id: optional identifier, echoed back in the result for
            bookkeeping (e.g. to match against artifact_manager.py's
            run_id when writing results to disk).

    Returns:
        dict:
            run_id: str or None, echoed back
            task_type: str, echoed back
            ml_metrics: dict from compute_regression_metrics() or
                compute_classification_metrics()
            trading_metrics: the "metrics" dict from
                stats.calculator.compute_stats() -- every quantstats
                metric discovered by stats/metrics.py, keyed by its
                quantstats name (e.g. "sharpe", not "sharpe_ratio")
            trade_summary: the "trade_summary" dict from compute_stats()
                (final_balance, total_net_profit, total_trades, win_loss)
            backtest_result: the raw dict run_backtest() returned
                (trade_ledger, equity_curve, etc.) -- kept in case a
                caller wants the full ledger, not just the summary stats
    """
    if task_type == "regression":
        ml_metrics = compute_regression_metrics(y_true, y_pred)
    elif task_type == "classification":
        ml_metrics = compute_classification_metrics(y_true, y_pred)
    elif task_type in ("timeseries", "timeseries_regression"):
        # A price forecast's error (y_true/y_pred both close prices,
        # same length as the forecast horizon) is the same MAE/RMSE/etc
        # shape as a regression return prediction's -- no separate
        # timeseries metrics module needed. "timeseries" (old name) is
        # kept as an accepted alias so any external caller still using
        # it doesn't break.
        ml_metrics = compute_regression_metrics(y_true, y_pred)
    elif task_type == "timeseries_classification":
        # A timeseries classifier's forecasted label vs actual label is
        # the same accuracy/precision/recall shape as a row-wise
        # classifier's -- reuses compute_classification_metrics()
        # directly, same reasoning as the regression case above.
        ml_metrics = compute_classification_metrics(y_true, y_pred)
    else:
        raise ValueError(
            f"task_type must be 'regression', 'classification', 'timeseries_regression', "
            f"or 'timeseries_classification', got '{task_type}'"
        )

    logger.info(f"ML evaluation ({task_type}): {ml_metrics}")

    # Heading 9 -> 10 handoff: signals + their timestamps become the
    # datetime/signal DataFrame run_backtest() expects. "Buy"/"Sell"/"Hold"
    # (ml/signals' string labels) map onto run_backtest()'s numeric
    # convention (1=Buy, -1=Sell, 0=no signal / Hold).
    signal_df = _to_signal_dataframe(signals, signal_timestamps)

    # Heading 10: "The Machine Learning module shall invoke the
    # Backtesting module after signal generation."
    backtest_result = run_backtest(ohlcv_1m, signal_df, backtest_config)
    logger.info(
        f"Backtest complete: {backtest_result['total_trades']} trades, "
        f"final balance {backtest_result['final_balance']:.2f}"
    )

    # Trading Strategy Evaluation: hand the backtest result straight to
    # the Statistics module -- compute_stats() derives returns from
    # equity_curve and computes every quantstats metric itself.
    stats_result = compute_stats(backtest_result, stats_config, plot_dir=plot_dir)
    trading_metrics = stats_result["metrics"]

    # trading_metrics (== stats_result["metrics"]) is compute_stats()'s
    # full quantstats output -- dozens of keys, several of them
    # (rolling_sharpe, pct_rank, implied_volatility, remove_outliers,
    # outliers, ...) are per-day series covering the whole backtest
    # period, not scalars. Logging the dict directly here used to dump
    # every one of those series to the log/terminal on every single
    # algorithm run. Log a short, scalar-only summary instead, using
    # the same key names _METRIC_NAME_MAP above already establishes as
    # the supported/known-scalar ones -- anyone who wants the full dict
    # still has it via this function's return value
    # (result["trading_metrics"]) or stats_result itself.
    metrics_summary = {k: trading_metrics.get(k) for k in _METRIC_NAME_MAP.values() if k in trading_metrics}
    logger.info(f"Trading strategy evaluation: {metrics_summary}")

    return {
        "run_id": run_id,
        "task_type": task_type,
        "ml_metrics": ml_metrics,
        "trading_metrics": trading_metrics,
        "trade_summary": stats_result["trade_summary"],
        "backtest_result": backtest_result,
    }


def select_best_model(evaluation_results: List[dict], ml_config: dict) -> dict:
    """
    Pick the best model out of several evaluate_model() results, using
    ml/config.yaml's evaluation.primary_metric (PDF heading 10:
    "The metric used to determine the 'best' model shall be configurable").

    ML metrics are never used here, only trading_metrics[primary_metric]
    -- this is the literal enforcement of "ML metrics ... shall not be
    considered the primary criterion for model selection."

    Args:
        evaluation_results: list of dicts, each from evaluate_model().
        ml_config: ml/config.yaml dict. Expects:

            evaluation:
              primary_metric: sharpe_ratio

    Returns:
        The single dict from evaluation_results with the best
        primary_metric value (highest, except max_drawdown which is
        ranked lowest-is-best).
    """
    if not evaluation_results:
        raise ValueError("evaluation_results is empty -- nothing to select from")

    primary_metric = ml_config.get("evaluation", {}).get("primary_metric")
    if not primary_metric:
        raise ValueError("ml/config.yaml must set evaluation.primary_metric (e.g. 'sharpe_ratio')")
    if primary_metric not in _METRIC_NAME_MAP:
        raise ValueError(
            f"evaluation.primary_metric='{primary_metric}' is not a supported metric. "
            f"Supported: {sorted(_METRIC_NAME_MAP.keys())}"
        )

    stats_key = _METRIC_NAME_MAP[primary_metric]

    for result in evaluation_results:
        if stats_key not in result["trading_metrics"]:
            raise ValueError(
                f"Result for run_id={result.get('run_id')} is missing trading_metrics"
                f"['{stats_key}'] (for primary_metric='{primary_metric}') -- "
                f"compute_stats() did not report this metric, or quantstats couldn't "
                f"compute it on this data."
            )
        if result["trading_metrics"][stats_key] is None:
            raise ValueError(
                f"Result for run_id={result.get('run_id')} has trading_metrics"
                f"['{stats_key}']=None -- quantstats failed to compute this metric on "
                f"this run's data (see stats/metrics.py, errors are recorded as None "
                f"rather than raised)."
            )

    lower_is_better = primary_metric in _LOWER_IS_BETTER
    best = min(
        evaluation_results,
        key=lambda r: r["trading_metrics"][stats_key],
    ) if lower_is_better else max(
        evaluation_results,
        key=lambda r: r["trading_metrics"][stats_key],
    )

    logger.info(
        f"Best model selected by {primary_metric} ('{stats_key}' in trading_metrics)"
        f"{' (lower is better)' if lower_is_better else ''}: "
        f"run_id={best.get('run_id')}, value={best['trading_metrics'][stats_key]}"
    )
    return best


def _to_signal_dataframe(signals: np.ndarray, signal_timestamps: pd.Series) -> pd.DataFrame:
    """
    Converts ml/signals' string labels ("Buy"/"Sell"/"Hold") into the
    datetime/signal DataFrame run_backtest() expects (signal column:
    1=Buy, -1=Sell, 0=no signal). "Hold" and any unrecognized label both
    map to 0 -- run_backtest() only ever acts on nonzero signal rows.
    """
    label_to_numeric = {"Buy": 1, "Sell": -1, "Hold": 0}
    numeric_signals = pd.Series(signals).map(label_to_numeric).fillna(0).astype(int)

    return pd.DataFrame({
        "datetime": pd.Series(signal_timestamps).reset_index(drop=True),
        "signal": numeric_signals.reset_index(drop=True),
    })
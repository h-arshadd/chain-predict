# crypto_pipeline/validation/threshold_analysis.py

"""
threshold_analysis.py
----------------------
Answers: "is upper_threshold/lower_threshold (currently +-0.001 in
ml_module/config.yaml) set correctly, or should it go up/down?"

For BOTH regression and classification it:
  1. Prints how many LONG / SHORT / FLAT signals get generated.
  2. Prints the actual return distribution (min/mean/median/max) inside
     each bucket.
  3. Flags how many signals are "borderline" -- i.e. their actual return
     barely clears the threshold -- since a threshold surrounded by mostly
     borderline trades is usually the sign it needs to move.

Regression target IS the return, so it's used directly.
Classification target is discrete (-1/0/1, Triple Barrier Labeling), so to
compare it against the threshold we also compute the plain forward log
return over the same horizon -- that's the number that actually decided
whether a barrier got hit, and how comfortably.

Both analyses run off ONE shared data pull (collect_market_data +
engineer_features called once) so you don't fetch market data twice.

Usage:
    python -m crypto_pipeline.validation.threshold_analysis
or:
    python validation/threshold_analysis.py
"""

import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_pipeline.ml_module.ml_utils import load_config_yaml
from crypto_pipeline.ml_module.data_pipeline import collect_market_data
from crypto_pipeline.ml_module.feature_pipeline import engineer_features
from crypto_pipeline.ml_module.target_pipeline import generate_target


def _load_base_df(ml_config_path: str):
    """Fetch market data + features ONCE, shared by both analyses below."""
    config = load_config_yaml(ml_config_path)

    df = collect_market_data(config)
    if config.get("features", {}).get("enabled", False):
        df = engineer_features(df, config)

    return df, config


def analyze_regression(df: pd.DataFrame, config: dict,
                        upper_threshold: float, lower_threshold: float) -> pd.DataFrame:
    """Build the regression target (log return) and bucket it by threshold."""

    cfg = dict(config)
    cfg["model_type"] = "regression"
    target_cfg = dict(cfg.get("target", {}))
    # Turn off noise filtering here so we see the FULL return distribution,
    # not a pre-filtered one -- we want to judge the threshold on everything.
    target_cfg["filter_noise"] = False
    cfg["target"] = target_cfg

    reg_df = generate_target(df.copy(), cfg)
    returns = reg_df["target"]

    long_mask = returns > upper_threshold
    short_mask = returns < lower_threshold
    flat_mask = ~(long_mask | short_mask)

    report = _print_report("REGRESSION", returns, long_mask, short_mask, flat_mask,
                            upper_threshold, lower_threshold)

    return reg_df, report


def analyze_classification(df: pd.DataFrame, config: dict,
                            upper_threshold: float, lower_threshold: float) -> pd.DataFrame:
    """Build classification target (Triple Barrier Labeling) and analyze
    the actual forward returns that triggered each label.
    """

    cfg = dict(config)
    cfg["model_type"] = "classification"

    target_cfg = dict(cfg.get("target", {}))
    target_cfg["upper_threshold"] = upper_threshold
    target_cfg["lower_threshold"] = lower_threshold
    cfg["target"] = target_cfg

    horizon = target_cfg.get("horizon", 1)
    cls_df = generate_target(df.copy(), cfg)

    # Triple barrier labels
    labels = cls_df["target"]

    # Compute the forward log returns that actually decided the barrier hit
    close = cls_df["close"].values
    forward_returns = np.log(np.roll(close, -horizon) / close)
    # Rows at the end (within horizon of end) will have invalid returns from roll
    # Mark them as NaN
    forward_returns[-horizon:] = np.nan
    cls_df["forward_return"] = forward_returns

    long_mask = labels == 1
    short_mask = labels == -1
    flat_mask = labels == 0

    # For classification, we bucket by the LABEL, but report the forward returns
    # that led to each label. This shows how comfortably each barrier was hit.
    report = _print_report("CLASSIFICATION", cls_df["forward_return"], 
                            long_mask, short_mask, flat_mask,
                            upper_threshold, lower_threshold)

    return cls_df, report


def _print_report(label: str, returns: pd.Series,
                   long_mask: pd.Series, short_mask: pd.Series, flat_mask: pd.Series,
                   upper_threshold: float, lower_threshold: float,
                   near_pct: float = 0.20) -> dict:
    """
    near_pct: how close (as a fraction of the threshold itself) a return has
    to be to count as "borderline". 0.20 = within 20% of the threshold value.

    Returns a plain dict of everything printed, plus the exact text under
    report["report_text"], so the caller can persist it (see save_report
    below) as a human-readable .txt -- not just structured JSON/CSV.
    """

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    total = len(returns)
    n_long, n_short, n_flat = int(long_mask.sum()), int(short_mask.sum()), int(flat_mask.sum())

    emit("=" * 70)
    emit(f"{label} -- threshold report  (upper={upper_threshold}, lower={lower_threshold})")
    emit("=" * 70)
    emit(f"Total rows:   {total}")
    emit(f"Long  (1):    {n_long:>6} ({n_long/total*100:5.1f}%)")
    emit(f"Short (-1):   {n_short:>6} ({n_short/total*100:5.1f}%)")
    emit(f"Flat  (0):    {n_flat:>6} ({n_flat/total*100:5.1f}%)")
    emit()

    report = {
        "label": label,
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
        "near_pct": near_pct,
        "total_rows": total,
        "counts": {"long": n_long, "short": n_short, "flat": n_flat},
        "buckets": {},
    }

    borderline_total = 0
    signal_total = n_long + n_short

    for name, mask, threshold in [("long", long_mask, upper_threshold),
                                   ("short", short_mask, lower_threshold)]:
        vals = returns[mask].dropna()
        if len(vals) == 0:
            emit(f"{name.upper()}: no signals generated\n")
            report["buckets"][name] = None
            continue

        # distance from threshold, as a multiple of the threshold itself.
        # 0.0 = sitting right on the line, 1.0 = return is double the threshold.
        distance = (vals - threshold).abs() / abs(threshold)
        near = distance <= near_pct
        borderline_total += int(near.sum())

        emit(f"{name.upper()} signals ({len(vals)}):")
        emit(f"  return min/mean/median/max: "
             f"{vals.min():.5f} / {vals.mean():.5f} / {vals.median():.5f} / {vals.max():.5f}")
        emit(f"  within {near_pct*100:.0f}% of threshold: {int(near.sum())} "
             f"({near.sum()/len(vals)*100:.1f}%)  <- borderline trades")
        emit()

        report["buckets"][name] = {
            "count": int(len(vals)),
            "min": float(vals.min()),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "max": float(vals.max()),
            "borderline_count": int(near.sum()),
            "borderline_pct": float(near.sum() / len(vals) * 100),
        }

    verdict = None
    if signal_total > 0:
        borderline_pct = borderline_total / signal_total * 100
        emit(f"Borderline signals overall: {borderline_pct:.1f}% of all long/short signals")
        if borderline_pct > 40:
            verdict = "RAISE threshold -- most signals barely clear it (low conviction)."
        elif borderline_pct < 10:
            verdict = ("Signals clear threshold comfortably -- could LOWER it to catch more "
                       "real moves, if win-rate/ROI support it.")
        else:
            verdict = "Mixed -- reasonably calibrated, cross-check against ROI/win-rate too."
        emit(f"-> {verdict}")

    report["borderline_pct_overall"] = (
        borderline_total / signal_total * 100 if signal_total > 0 else None
    )
    report["verdict"] = verdict
    emit()

    report["report_text"] = "\n".join(lines)
    return report


def save_report(reg_report: dict, cls_report: dict, reg_df: pd.DataFrame, cls_df: pd.DataFrame,
                 output_dir: str = None) -> None:
    """
    Persist results, mirroring validate_targets.py's save_results() convention:
      - threshold_report_{timestamp}.txt  : the exact human-readable console output
      - threshold_report_{timestamp}.json : same info, structured, for other scripts
      - regression_rows_{timestamp}.csv   : datetime, target(=return), signal
      - classification_rows_{timestamp}.csv : datetime, target(-1/0/1), forward_return, signal
    """

    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs" / "threshold_analysis"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Human-readable .txt -- same text you saw printed to the console.
    txt_path = os.path.join(output_dir, f"threshold_report_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(reg_report["report_text"])
        f.write("\n\n")
        f.write(cls_report["report_text"])
    print(f"✓ Text report saved: {txt_path}")

    # Structured JSON -- same info, for scripts/other tools to read back in.
    # (strip report_text out of the JSON copy so it isn't duplicated twice)
    reg_json = {k: v for k, v in reg_report.items() if k != "report_text"}
    cls_json = {k: v for k, v in cls_report.items() if k != "report_text"}
    summary_path = os.path.join(output_dir, f"threshold_report_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump({"regression": reg_json, "classification": cls_json}, f, indent=2, default=str)
    print(f"✓ JSON report saved: {summary_path}")

    reg_out = reg_df[["datetime", "target"]].copy()
    reg_out["signal"] = np.where(reg_out["target"] > reg_report["upper_threshold"], 1,
                          np.where(reg_out["target"] < reg_report["lower_threshold"], -1, 0))
    reg_path = os.path.join(output_dir, f"regression_rows_{timestamp}.csv")
    reg_out.to_csv(reg_path, index=False)
    print(f"✓ Regression rows saved: {reg_path}")

    # For classification, export both the label and the forward return that triggered it
    cls_out = cls_df[["datetime", "target", "forward_return"]].copy()
    cls_out = cls_out.rename(columns={"target": "signal"})
    cls_path = os.path.join(output_dir, f"classification_rows_{timestamp}.csv")
    cls_out.to_csv(cls_path, index=False)
    print(f"✓ Classification rows saved: {cls_path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ml_config_path = os.path.join(here, "..", "ml_module", "config.yaml")

    base_df, base_config = _load_base_df(ml_config_path)

    target_cfg = base_config.get("target", {})
    upper_threshold = target_cfg.get("upper_threshold", 0.001)
    lower_threshold = target_cfg.get("lower_threshold", -0.001)

    reg_df, reg_report = analyze_regression(base_df, base_config, upper_threshold, lower_threshold)
    cls_df, cls_report = analyze_classification(base_df, base_config, upper_threshold, lower_threshold)

    save_report(reg_report, cls_report, reg_df, cls_df)
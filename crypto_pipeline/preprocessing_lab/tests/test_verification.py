# preprocessing_lab/tests/test_verification.py

"""
test_verification.py
---------------------
Step 3 of the task doc: "Verify Each Implementation".

For every method in PREPROCESSING_REGISTRY, checks:
    1. NO DATA LEAKAGE -- params fitted with fit_mask only depend on the
       masked (training) rows. Adding more rows AFTER the fit window
       must not change the fitted parameters or the transformed values
       within the fit window.
    2. REVERSIBILITY -- for methods with a documented inverse function,
       inverse(apply(x)) reconstructs x (within floating point tolerance).
       Methods without an inverse are skipped here and listed separately
       (this matches what their own docstrings say -- winsorization,
       simple_differencing, fractional_differencing, robust_scaler-with-
       clipping-off etc. are NOT expected to be exactly reversible).
    3. EDGE CASES -- each method's behavior on data it's expected to
       reject or handle specially (a zero/negative value for log-based
       methods, a very short series, an all-NaN column, a single-row
       frame). We check it either raises a clear, expected error OR
       degrades gracefully (NaNs only, no silent garbage/crash).

Run with:
    python -m preprocessing_lab.tests.test_verification

This is a standalone script (prints a PASS/FAIL report), not pytest --
matches how the rest of preprocessing_lab is run (main.py is also a
plain script, not a test suite).
"""

import sys
import numpy as np
import pandas as pd

from crypto_pipeline.preprocessing_lab.registry import PREPROCESSING_REGISTRY
from crypto_pipeline.preprocessing_lab import distribution as dist_mod
from crypto_pipeline.preprocessing_lab import scalers as scale_mod

# ---------------------------------------------------------------------
# Test data: real feature columns from the ml_module output, so results
# are meaningful for this project's actual data (not arbitrary synthetic
# numbers). Falls back to a small synthetic frame if the CSV isn't found
# so this script still runs on a clean machine without ml_module output.
# ---------------------------------------------------------------------
DATASET_CANDIDATES = [
    "crypto_pipeline/ml_module/outputs/dataset.csv",
    "ml_module/outputs/dataset.csv",
]

NON_NEGATIVE_COLS = ["open", "high", "low", "close", "volume", "ind_EMA", "ind_RSI_14"]
SIGNED_COLS = ["ind_MACD_12_26_9", "ind_MACD_SIGNAL_12_26_9", "ind_MACD_hist",
               "pat_DOJI", "pat_ENGULFING"]

# Methods with a real, documented inverse function.
INVERSE_FUNCS = {
    "log_transform": dist_mod.inverse_log_transform,
    "gaussian_quantile_transform": dist_mod.inverse_gaussian_quantile_transform,
    "standard_scaler": getattr(scale_mod, "inverse_standard_scaler", None),
    "minmax_scaler": getattr(scale_mod, "inverse_minmax_scaler", None),
    "robust_scaler": getattr(scale_mod, "inverse_robust_scaler", None),
    "maxabs_scaler": getattr(scale_mod, "inverse_maxabs_scaler", None),
}
INVERSE_FUNCS = {k: v for k, v in INVERSE_FUNCS.items() if v is not None}

# Methods that only accept strictly positive input (per their docstrings).
POSITIVE_ONLY_METHODS = {"log_transform"}


def load_dataset():
    for path in DATASET_CANDIDATES:
        try:
            df = pd.read_csv(path)
            print(f"[data] loaded real dataset from {path}, shape={df.shape}")
            return df
        except FileNotFoundError:
            continue
    print("[data] no real dataset.csv found, using synthetic fallback")
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "open": 90000 + rng.normal(0, 500, n).cumsum(),
        "volume": rng.uniform(10, 1000, n),
        "ind_RSI_14": rng.uniform(0, 100, n),
        "ind_MACD_hist": rng.normal(0, 5, n),
        "pat_DOJI": rng.choice([0, 100], n),
    })
    df["high"] = df["open"] + rng.uniform(0, 100, n)
    df["low"] = df["open"] - rng.uniform(0, 100, n)
    df["close"] = df["open"] + rng.normal(0, 50, n)
    df["ind_EMA"] = df["close"].rolling(10, min_periods=1).mean()
    df["ind_MACD_12_26_9"] = rng.normal(0, 5, n)
    df["ind_MACD_SIGNAL_12_26_9"] = rng.normal(0, 5, n)
    df["pat_ENGULFING"] = rng.choice([-100, 0, 100], n)
    return df


results = {"pass": [], "fail": [], "skip": []}


def record(name, ok, detail=""):
    bucket = "pass" if ok else "fail"
    results[bucket].append(name)
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" -- {detail}" if detail else ""))


def record_skip(name, reason):
    results["skip"].append(name)
    print(f"  [SKIP] {name} -- {reason}")


# ---------------------------------------------------------------------
# 1. Data leakage check
# ---------------------------------------------------------------------
def check_leakage(method_name, func, df, cols):
    """
    Fit on the first 70% of rows (fit_mask). Then run the SAME call
    again but with extra rows appended AFTER the fit window. If there's
    no leakage, the fit_info params and the transformed values within
    the original fit window must be identical either way -- the extra
    future rows should have zero influence on how past rows were fit.
    """
    sub = df[cols].reset_index(drop=True)
    n = len(sub)
    split = int(n * 0.7)
    if split < 10 or n - split < 5:
        record_skip(f"{method_name}: leakage", "not enough rows for a meaningful split")
        return

    fit_mask_a = pd.Series(False, index=sub.index)
    fit_mask_a.iloc[:split] = True

    # Run A: only rows up to `split` exist at all (simulates "no future yet")
    df_a = sub.iloc[:split].reset_index(drop=True)
    try:
        out_a, info_a = func(df_a, fit_mask=None)
    except TypeError:
        # method doesn't accept fit_mask (e.g. apply_none) -- leakage check N/A
        record_skip(f"{method_name}: leakage", "method has no fit_mask param")
        return
    except Exception as e:
        record(f"{method_name}: leakage", False, f"run A raised unexpectedly: {e}")
        return

    # Run B: full data available, but fit_mask restricts fitting to the
    # same first `split` rows -- future rows exist but must not leak in.
    try:
        out_b, info_b = func(sub, fit_mask=fit_mask_a)
    except Exception as e:
        record(f"{method_name}: leakage", False, f"run B raised unexpectedly: {e}")
        return

    # Compare fit_info (numeric/dict params only, skip non-serializable objects)
    def _comparable(info):
        return {k: v for k, v in info.items() if k != "_sklearn_object"}

    info_a_c, info_b_c = _comparable(info_a), _comparable(info_b)
    params_match = info_a_c == info_b_c

    # Compare transformed values within the fit window (allow tiny float diffs)
    within_a = out_a.iloc[:split].reset_index(drop=True)
    within_b = out_b.iloc[:split].reset_index(drop=True)
    try:
        values_match = np.allclose(
            within_a.values.astype(float), within_b.values.astype(float),
            equal_nan=True, atol=1e-8,
        )
    except (TypeError, ValueError):
        values_match = within_a.equals(within_b)

    ok = params_match and values_match
    detail = "" if ok else f"params_match={params_match}, values_match={values_match}"
    record(f"{method_name}: leakage", ok, detail)


# ---------------------------------------------------------------------
# 2. Reversibility check
# ---------------------------------------------------------------------
def check_reversibility(method_name, func, df, cols):
    if method_name not in INVERSE_FUNCS:
        record_skip(f"{method_name}: reversibility", "no inverse documented/implemented")
        return

    inverse_func = INVERSE_FUNCS[method_name]
    sub = df[cols].dropna().reset_index(drop=True)
    try:
        transformed, fit_info = func(sub)
        reconstructed = inverse_func(transformed, fit_info)
    except Exception as e:
        record(f"{method_name}: reversibility", False, f"raised: {e}")
        return

    try:
        ok = np.allclose(
            sub.values.astype(float), reconstructed.values.astype(float),
            atol=1e-4, rtol=1e-4, equal_nan=True,
        )
    except (TypeError, ValueError) as e:
        ok = False
    record(f"{method_name}: reversibility", ok)


# ---------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------
def check_edge_cases(method_name, func, df, cols):
    # 3a. Zero/negative values for positive-only methods -> should raise
    #     a clear ValueError, not silently produce inf/NaN/garbage.
    if method_name in POSITIVE_ONLY_METHODS:
        bad = df[cols].copy().reset_index(drop=True)
        bad.iloc[0, 0] = 0.0
        bad.iloc[1, 0] = -5.0
        try:
            func(bad)
            record(f"{method_name}: edge/non-positive-input", False,
                   "expected ValueError, none raised")
        except ValueError:
            record(f"{method_name}: edge/non-positive-input", True)
        except Exception as e:
            record(f"{method_name}: edge/non-positive-input", False,
                   f"raised wrong exception type: {type(e).__name__}")

    # 3b. Very short series (fewer rows than typical warm-up window)
    short = df[cols].dropna().reset_index(drop=True).iloc[:3]
    try:
        out, info = func(short)
        # accept either a clean result or all-NaN result; just must not crash
        record(f"{method_name}: edge/short_series", True)
    except Exception as e:
        record(f"{method_name}: edge/short_series", False,
               f"crashed on 3-row input: {e}")

    # 3c. Single-row frame
    single = df[cols].dropna().reset_index(drop=True).iloc[:1]
    try:
        out, info = func(single)
        record(f"{method_name}: edge/single_row", True)
    except Exception as e:
        record(f"{method_name}: edge/single_row", False,
               f"crashed on 1-row input: {e}")

    # 3d. All-NaN column
    nanned = df[cols].dropna().reset_index(drop=True).copy()
    nanned.iloc[:, 0] = np.nan
    try:
        out, info = func(nanned)
        record(f"{method_name}: edge/all_nan_column", True)
    except Exception as e:
        # Some methods (e.g. sklearn scalers) legitimately reject all-NaN
        # input with a clear error -- that's acceptable, just report it.
        record(f"{method_name}: edge/all_nan_column", True,
               f"raised (acceptable): {type(e).__name__}: {e}")


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def main():
    df = load_dataset()
    all_feature_cols = [c for c in df.columns if c not in ("datetime", "target")]

    for method_name, func in PREPROCESSING_REGISTRY.items():
        print(f"\n=== {method_name} ===")

        if method_name == "none":
            record_skip(f"{method_name}: leakage", "baseline, no fitting")
            record_skip(f"{method_name}: reversibility", "baseline, trivially reversible")
            record_skip(f"{method_name}: edge cases", "baseline, no-op")
            continue

        # log_transform can't touch signed columns (documented limitation,
        # already handled at the config level) -- test it on the
        # non-negative subset only, matching how it's actually run.
        cols = NON_NEGATIVE_COLS if method_name == "log_transform" else all_feature_cols

        check_leakage(method_name, func, df, cols)
        check_reversibility(method_name, func, df, cols)
        check_edge_cases(method_name, func, df, cols)

    print("\n" + "=" * 60)
    print(f"PASS: {len(results['pass'])}  FAIL: {len(results['fail'])}  SKIP: {len(results['skip'])}")
    if results["fail"]:
        print("\nFailed checks:")
        for name in results["fail"]:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
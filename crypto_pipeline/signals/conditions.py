"""
conditions.py (CORRECTED)

Evaluates every strategy condition and returns a DataFrame
containing one boolean column for each condition.

LOOK-AHEAD BIAS — SINGLE CENTRALIZED SHIFT:
- talib_indicators.py now returns RAW, unshifted indicator values.
  ind_X[N] is computed using data through candle N (inclusive) — same
  convention as talib itself, and same convention as the raw OHLCV columns.
- Because everything (prices AND indicators) is unshifted and on the same
  clock, conditions are evaluated here exactly as if you were live at the
  close of each candle — no per-operand shifting needed in resolve_operand().
- The ONE shift that prevents look-ahead is applied once, at the very end
  of evaluate_conditions(), to the whole condition_df at once. This moves
  every condition (and therefore every signal built from it) forward by
  one bar, so a condition that becomes true using candle N's data is only
  actionable starting at candle N+1 — exactly the anti-look-ahead
  guarantee, but from a single place instead of scattered across two files.
- Do NOT re-add per-column shifting in resolve_operand(), and do NOT add
  .shift(1) back into talib_indicators.py — either one would double-shift
  on top of the shift already applied here.
"""

import pandas as pd
import numpy as np


# ==========================================================
# Cross Operators
# ==========================================================

def cross_above(left: pd.Series, right: pd.Series) -> pd.Series:
    """
    True only on the bar where left crosses above right.

    Operates on raw, unshifted series — the anti-look-ahead shift is
    applied once, centrally, in evaluate_conditions().
    """
    return (left > right) & (left.shift(1) <= right.shift(1))


def cross_below(left: pd.Series, right: pd.Series) -> pd.Series:
    """
    True only on the bar where left crosses below right.
    
    Assumes left and right are pre-aligned.
    """
    return (left < right) & (left.shift(1) >= right.shift(1))


# ==========================================================
# Persist Bars
# ==========================================================

def apply_persist(condition: pd.Series, bars: int) -> pd.Series:
    """
    If a condition becomes True,
    keep it True for the next N bars.

    persist_bars = 0
    means no persistence.
    """

    condition = condition.fillna(False).astype(bool)

    if bars <= 0:
        return condition

    result = condition.copy()

    for i in range(1, bars + 1):
        result |= condition.shift(i, fill_value=False)

    return result


# ==========================================================
# Value Resolver (FIXED)
# ==========================================================

def resolve_operand(df: pd.DataFrame, operand):
    """
    Resolve operand to either a Series or a constant.

    Operand can be:
        ind_EMA_20          → indicator column (raw, unshifted)
        close, open, etc    → price column (raw, unshifted)
        30, 70              → numeric constant
        True                → boolean constant

    No shifting happens here. Both indicators and prices are on the same,
    unshifted clock, so they're already timely aligned with each other.
    The single anti-look-ahead shift is applied once, to the finished
    condition_df, at the end of evaluate_conditions().
    """

    if isinstance(operand, str):

        if operand in df.columns:
            return df[operand]

    # Return constant (number, boolean, etc)
    return operand


# ==========================================================
# Operator Dispatcher
# ==========================================================

def evaluate_operator(
    df: pd.DataFrame,
    left,
    operator,
    right,
):
    """
    Evaluates one condition, using each operand's raw (unshifted) values.
    The anti-look-ahead shift is applied once, centrally, at the end of
    evaluate_conditions() — not here.
    """

    left_value = resolve_operand(df, left)
    right_value = resolve_operand(df, right)

    # -------------------------------
    # Comparison Operators
    # -------------------------------

    if operator == ">":
        return left_value > right_value

    if operator == ">=":
        return left_value >= right_value

    if operator == "<":
        return left_value < right_value

    if operator == "<=":
        return left_value <= right_value

    if operator == "==":
        return left_value == right_value

    if operator == "!=":
        return left_value != right_value

    # -------------------------------
    # Cross Operators
    # -------------------------------

    if operator == "cross_above":
        return cross_above(left_value, right_value)

    if operator == "cross_below":
        return cross_below(left_value, right_value)

    # -------------------------------
    # Price Operators
    # (Same raw, unshifted convention as everything else here)
    # -------------------------------

    if operator == "close_above":
        return df["close"] > right_value

    if operator == "close_below":
        return df["close"] < right_value

    if operator == "open_above":
        return df["open"] > right_value

    if operator == "open_below":
        return df["open"] < right_value

    if operator == "high_above":
        return df["high"] > right_value

    if operator == "high_below":
        return df["high"] < right_value

    if operator == "low_above":
        return df["low"] > right_value

    if operator == "low_below":
        return df["low"] < right_value

    # -------------------------------
    # Pattern Match
    # -------------------------------

    if operator == "pattern_match":

        if isinstance(left_value, pd.Series):
            return left_value.astype(bool)

        return pd.Series(False, index=df.index)

    raise ValueError(
        f"Unsupported operator: {operator}"
    )

# ==========================================================
# Main Evaluation Function
# ==========================================================

def evaluate_conditions(
    df: pd.DataFrame,
    strategy_config: dict
) -> pd.DataFrame:
    """
    Evaluate every strategy condition.

    Returns
    -------
    DataFrame

        long_cond_1
        long_cond_2
        ...
        short_cond_1
        short_cond_2
        ...
    """

    result = pd.DataFrame(index=df.index)

    # =====================================================
    # LONG CONDITIONS
    # =====================================================

    if "long" in strategy_config:

        long_conditions = strategy_config["long"].get(
            "conditions",
            []
        )

        for idx, condition in enumerate(long_conditions, start=1):

            series = evaluate_operator(

                df=df,

                left=condition["left"],

                operator=condition["operator"],

                right=condition["right"]

            )

            persist = condition.get(
                "persist_bars",
                0
            )

            series = apply_persist(
                series,
                persist
            )

            result[f"long_cond_{idx}"] = (
                series
                .fillna(False)
                .astype(bool)
            )

    # =====================================================
    # SHORT CONDITIONS
    # =====================================================

    if "short" in strategy_config:

        short_conditions = strategy_config["short"].get(
            "conditions",
            []
        )

        for idx, condition in enumerate(short_conditions, start=1):

            series = evaluate_operator(

                df=df,

                left=condition["left"],

                operator=condition["operator"],

                right=condition["right"]

            )

            persist = condition.get(
                "persist_bars",
                0
            )

            series = apply_persist(
                series,
                persist
            )

            result[f"short_cond_{idx}"] = (
                series
                .fillna(False)
                .astype(bool)
            )

    # =====================================================
    # SINGLE, CENTRALIZED ANTI-LOOK-AHEAD SHIFT
    # =====================================================
    # Every condition above was evaluated using each row's OWN, unshifted
    # data (raw indicators, raw prices) — i.e. "as if live at the close of
    # that candle". Shifting the whole result forward by one bar here means
    # a condition that became true using candle N's data is only reflected
    # as True starting at candle N+1, which is the one and only shift this
    # pipeline applies. Do not shift anywhere else (talib_indicators.py,
    # resolve_operand, or main.py's OHLCV) or you will double-shift.
    #
    # shift(1) introduces NaN into these boolean columns, which upcasts them
    # to object dtype; infer_objects(copy=False) lets fillna+astype cleanly
    # downcast back to bool without pandas' FutureWarning.
    result = result.shift(1).fillna(False).infer_objects(copy=False).astype(bool)

    return result
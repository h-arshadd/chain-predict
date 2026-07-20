"""
conditions.py

Evaluates every strategy condition and returns a DataFrame
containing one boolean column for each condition.
"""

import pandas as pd
import numpy as np

# pandas' own fix for the "Downcasting object dtype arrays on .fillna,
# .ffill, .bfill is deprecated" FutureWarning: opt in to the future
# behavior (fillna no longer silently downcasts object dtype) rather than
# trying to call .infer_objects() after the fact -- the warning fires
# inside .fillna() itself, so nothing chained after it can suppress it.
# Safe to set unconditionally: every .fillna(False) in this file is
# immediately followed by .astype(bool) anyway, so we always wanted the
# downcast to happen -- this just stops pandas from warning about doing
# the thing we already explicitly ask for two lines later.
pd.set_option("future.no_silent_downcasting", True)


# ==========================================================
# Cross Operators
# ==========================================================

def cross_above(left: pd.Series, right: pd.Series) -> pd.Series:
    """
    True only on the bar where left crosses above right.
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

    condition = condition.fillna(False).infer_objects(copy=False).astype(bool)

    if bars <= 0:
        return condition

    result = condition.copy()

    for i in range(1, bars + 1):
        result |= condition.shift(i, fill_value=False)

    return result


# ==========================================================
# Value Resolver
# ==========================================================

def resolve_operand(df: pd.DataFrame, operand):
    """
    Resolve operand to either a Series or a constant.

    Operand can be:
        ind_EMA_20          → indicator column
        close, open, etc    → price column
        30, 70              → numeric constant
        True                → boolean constant
    """

    if isinstance(operand, str):

        if operand in df.columns:
            return df[operand]

        # A string operand that isn't a real column is almost always a
        # typo'd/missing column reference (e.g. an indicator alias that
        # was never created -- see calculate_indicators' KeyError for the
        # usual root cause), not an intentional string constant -- this
        # config format has no legitimate use for comparing against a
        # literal string. Raise here with the actual operand name rather
        # than silently falling through to "return operand" (the literal
        # string), which previously surfaced many conditions later as an
        # opaque "'>' not supported between instances of 'str' and 'int'"
        # with no indication of which column was missing.
        if operand not in ("open", "high", "low", "close", "volume"):
            raise KeyError(
                f"Condition references '{operand}', but that column doesn't "
                f"exist in the indicator DataFrame. Available columns: "
                f"{sorted(df.columns)}. Check this strategy's 'left'/'right' "
                f"values and its indicator aliases for a typo or a missing "
                f"indicator block."
            )

    # Return constant (number, boolean, etc)
    return operand


def _coerce_numeric(value):
    """
    Coerce a resolved operand to a numeric dtype before it's fed into a
    numeric comparison operator (>, >=, <, <=).

    Some indicator columns can come back as object dtype -- e.g. a warm-up
    period filled with None/NaN placeholders from the underlying talib
    wrapper, or a mix of numeric and non-numeric values -- and pandas
    raises "'>' not supported between instances of 'str' and 'int'" the
    moment that object-dtype Series meets an int/float on the other side
    of the comparison. That's a crash for the *entire* run (every
    exchange/symbol/strategy after it in the loop), not just a bad row.

    Coercing here turns anything that isn't actually numeric into NaN
    instead (via pd.to_numeric(errors="coerce")), so the comparison
    produces NaN/False for that row -- exactly what apply_persist's
    fillna(False) and evaluate_conditions' fillna(False) already assume --
    rather than blowing up the whole pipeline over one warm-up row or one
    bad indicator value.
    """
    if isinstance(value, pd.Series) and value.dtype == object:
        return pd.to_numeric(value, errors="coerce")
    return value


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
    Evaluates one condition.
    """

    left_value = resolve_operand(df, left)
    right_value = resolve_operand(df, right)

    # -------------------------------
    # Comparison Operators
    # -------------------------------

    if operator == ">":
        return _coerce_numeric(left_value) > _coerce_numeric(right_value)

    if operator == ">=":
        return _coerce_numeric(left_value) >= _coerce_numeric(right_value)

    if operator == "<":
        return _coerce_numeric(left_value) < _coerce_numeric(right_value)

    if operator == "<=":
        return _coerce_numeric(left_value) <= _coerce_numeric(right_value)

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
                .infer_objects(copy=False)
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
                .infer_objects(copy=False)
                .astype(bool)
            )

    # =====================================================
    # CENTRALIZED ANTI-LOOK-AHEAD SHIFT
    # =====================================================
    result = result.shift(1).fillna(False).infer_objects(copy=False).astype(bool)

    return result
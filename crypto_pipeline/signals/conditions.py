"""
conditions.py

Evaluates every strategy condition and returns a DataFrame
containing one boolean column for each condition.
"""

import pandas as pd
import numpy as np


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
# Value Resolver
# ==========================================================

def resolve_operand(df: pd.DataFrame, operand):
    """
    Operand can be

        ind_EMA_20
        close
        high
        30
        70
        True

    If operand exists as a dataframe column,
    return the Series.

    Otherwise return the constant.
    """

    if isinstance(operand, str):

        if operand in df.columns:
            return df[operand]

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
    Evaluates one condition.
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

    return result
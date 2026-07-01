"""
rules.py

Combines evaluated conditions into final trading signals.

Output
------
 1  -> Buy
-1  -> Sell
 0  -> No Signal
"""

import numpy as np
import pandas as pd


# ==========================================================
# Rule Engines
# ==========================================================

def and_rule(df: pd.DataFrame) -> pd.Series:
    """All conditions must be True."""
    return df.all(axis=1)


def or_rule(df: pd.DataFrame) -> pd.Series:
    """At least one condition must be True."""
    return df.any(axis=1)


def majority_rule(df: pd.DataFrame) -> pd.Series:
    """
    More than half of the conditions must be True.
    """

    votes = df.sum(axis=1)

    required = (df.shape[1] // 2) + 1

    return votes >= required


def weighted_rule(
    df: pd.DataFrame,
    weights: list,
    threshold: float
) -> pd.Series:
    """
    Weighted voting.

    Example
    -------
    weights = [0.5, 0.3, 0.2]

    threshold = 0.7
    """

    if len(weights) != df.shape[1]:
        raise ValueError(
            "Number of weights must equal number of conditions."
        )

    weight_array = np.asarray(weights)

    score = df.astype(float).values @ weight_array

    return pd.Series(
        score >= threshold,
        index=df.index
    )


# ==========================================================
# Apply Selected Rule
# ==========================================================

def evaluate_group(
    condition_df: pd.DataFrame,
    group_config: dict,
    prefix: str
) -> pd.Series:
    """
    Evaluate either LONG or SHORT group.
    """

    columns = [
        col
        for col in condition_df.columns
        if col.startswith(prefix)
    ]

    if len(columns) == 0:
        return pd.Series(
            False,
            index=condition_df.index
        )

    df = condition_df[columns]

    rule = group_config["rule"].upper()

    if rule == "AND":
        return and_rule(df)

    if rule == "OR":
        return or_rule(df)

    if rule == "MAJORITY":
        return majority_rule(df)

    if rule == "WEIGHTED":

        weights = group_config.get(
            "weights",
            [1] * len(columns)
        )

        threshold = group_config.get(
            "threshold",
            sum(weights)
        )

        return weighted_rule(
            df,
            weights,
            threshold
        )

    raise ValueError(
        f"Unsupported rule: {rule}"
    )


# ==========================================================
# Main Rule Engine
# ==========================================================

def apply_rules(
    condition_df: pd.DataFrame,
    strategy_config: dict
) -> pd.Series:
    """
    Combine condition outputs into final signals.

    Returns
    -------
    pd.Series

        1  Buy
        0  Neutral
       -1 Sell
    """

    long_signal = evaluate_group(
        condition_df,
        strategy_config["long"],
        "long"
    )

    short_signal = evaluate_group(
        condition_df,
        strategy_config["short"],
        "short"
    )

    signal = pd.Series(
        0,
        index=condition_df.index,
        dtype=int
    )

    signal.loc[long_signal] = 1

    signal.loc[short_signal] = -1

    # If both are True simultaneously,
    # treat it as no signal.

    both = long_signal & short_signal

    signal.loc[both] = 0

    return signal
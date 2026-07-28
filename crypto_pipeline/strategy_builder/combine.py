"""
combine.py

Combines N already-resolved WHOLE-STRATEGY signals (each a 1 / 0 / -1
pd.Series produced by signals/main.py's generate_signals(), one per
selected playbook entry and/or ML model) into a single final signal.

This is a different layer than signals/rules.py: rules.py's AND / OR /
MAJORITY / WEIGHTED combine boolean CONDITIONS inside one strategy's own
long/short block. This file combines multiple strategies' finished
1/0/-1 outputs -- the real structural gap identified for the Strategy
Builder (see STRATEGY_BUILDER_SPEC.md Section 2.1 / 5.4). Same rule
vocabulary as rules.py for consistency, but operating one level up.

Output convention, same as rules.py everywhere else in this codebase:
 1  -> Buy / Long
-1  -> Sell / Short
 0  -> No Signal
"""

import numpy as np
import pandas as pd


# ==========================================================
# Whole-Strategy Persist
# ==========================================================

def apply_whole_strategy_persist(signal: pd.Series, bars: int) -> pd.Series:
    """
    The builder's whole-strategy persist_bars (Strategy_Builder_Module.pdf's
    "EMA Trend, Persist Bars = 5" concept) -- applied to one playbook
    entry's already-resolved 1/0/-1 signal, on top of (not instead of)
    whatever per-condition persist_bars is already baked into that entry's
    own strategy_config.

    Same idea as signals/conditions.apply_persist, but that function
    persists a boolean condition; this persists a signed signal (1/0/-1),
    so it has to track direction, not just True/False -- once a bar fires
    1 or -1, that same direction keeps repeating for the next `bars` bars
    (only where the signal is currently 0 -- an opposite signal appearing
    during the persist window is real information and is never
    overwritten).

    persist_bars = 0 means no persistence (signal returned unchanged).
    """
    signal = signal.fillna(0).astype(int)

    if bars <= 0:
        return signal

    result = signal.copy()

    for i in range(1, bars + 1):
        carried = signal.shift(i, fill_value=0)
        # Only fill bars that are currently neutral -- never overwrite an
        # already-nonzero signal (either the entry's own fresh signal, or
        # a carried value from an earlier, closer shift).
        fillable = (result == 0) & (carried != 0)
        result.loc[fillable] = carried.loc[fillable]

    return result


# ==========================================================
# Combine Rule Engines
# ==========================================================
# Each takes a DataFrame of aligned 1/0/-1 signal columns (one column per
# playbook entry / ML model being combined) and returns one combined
# 1/0/-1 pd.Series.

def _and_rule(df: pd.DataFrame) -> pd.Series:
    """
    Every input must agree on the SAME nonzero direction. If any input is
    neutral (0), or inputs disagree (some 1, some -1), the combined result
    is 0 -- "AND" means unanimous agreement, not "majority agreement".
    """
    all_long = (df == 1).all(axis=1)
    all_short = (df == -1).all(axis=1)

    result = pd.Series(0, index=df.index, dtype=int)
    result.loc[all_long] = 1
    result.loc[all_short] = -1
    return result


def _or_rule(df: pd.DataFrame) -> pd.Series:
    """
    Any input signaling is enough to trigger. Tie-break rule (documented
    here since rules.py has no equivalent case to copy from): if inputs
    disagree on direction on the same bar (some long, some short), that's
    treated as conflicting information, not a trade -- combined result is
    0, same "conflict cancels out" convention rules.apply_rules() already
    uses when a single strategy's own long and short both fire at once.
    """
    any_long = (df == 1).any(axis=1)
    any_short = (df == -1).any(axis=1)

    result = pd.Series(0, index=df.index, dtype=int)
    result.loc[any_long & ~any_short] = 1
    result.loc[any_short & ~any_long] = -1
    # any_long & any_short (conflicting) -> stays 0, same as neither firing.
    return result


def _majority_rule(df: pd.DataFrame) -> pd.Series:
    """
    More than half of the inputs must agree on the same direction.
    """
    n = df.shape[1]
    required = (n // 2) + 1

    long_votes = (df == 1).sum(axis=1)
    short_votes = (df == -1).sum(axis=1)

    result = pd.Series(0, index=df.index, dtype=int)
    result.loc[long_votes >= required] = 1
    result.loc[short_votes >= required] = -1
    return result


def _weighted_rule(df: pd.DataFrame, weights: list, threshold: float) -> pd.Series:
    """
    Weighted voting, same shape as rules.weighted_rule but signed: each
    input contributes +weight for a long signal, -weight for a short
    signal, 0 for neutral. Combined result is long if the summed score
    clears +threshold, short if it clears -threshold, else neutral.

    Example
    -------
    weights = [0.5, 0.3, 0.2], threshold = 0.6
    """
    if len(weights) != df.shape[1]:
        raise ValueError("Number of weights must equal number of signal columns.")

    weight_array = np.asarray(weights, dtype=float)
    score = df.astype(float).values @ weight_array

    result = pd.Series(0, index=df.index, dtype=int)
    result.loc[score >= threshold] = 1
    result.loc[score <= -threshold] = -1
    return result


# ==========================================================
# Main Combine Entry Point
# ==========================================================

def combine_signals(
    signals: list,
    rule: str,
    weights: list = None,
    threshold: float = None,
) -> pd.Series:
    """
    Combine N already-resolved whole-strategy signals (each a 1/0/-1
    pd.Series, same index/alignment -- callers are responsible for
    aligning them, e.g. via pd.concat(..., axis=1) upstream having already
    dropped any mismatched bars) into one final signal, using the same
    AND / OR / MAJORITY / WEIGHTED vocabulary as signals/rules.py.

    Parameters
    ----------
    signals : list of pd.Series
        One 1/0/-1 series per playbook entry (post whole-strategy persist,
        if any -- see apply_whole_strategy_persist) and/or ML model.
    rule : str
        "AND", "OR", "MAJORITY", or "WEIGHTED" (case-insensitive).
    weights, threshold : only used when rule == "WEIGHTED".

    Returns
    -------
    pd.Series of 1 / 0 / -1, same index as the input signals.
    """
    if len(signals) == 0:
        raise ValueError("combine_signals() requires at least one signal.")

    if len(signals) == 1:
        # Nothing to combine -- a single selected playbook entry passes
        # through unchanged, same as the PDF's "user may select one or
        # multiple strategies" allowing exactly one.
        return signals[0].fillna(0).astype(int)

    df = pd.concat(signals, axis=1)
    df.columns = range(df.shape[1])
    df = df.fillna(0).astype(int)

    rule = rule.upper()

    if rule == "AND":
        return _and_rule(df)

    if rule == "OR":
        return _or_rule(df)

    if rule == "MAJORITY":
        return _majority_rule(df)

    if rule == "WEIGHTED":
        if weights is None:
            weights = [1] * df.shape[1]
        if threshold is None:
            threshold = sum(w for w in weights if w > 0)
        return _weighted_rule(df, weights, threshold)

    raise ValueError(f"Unsupported combine rule: {rule}")
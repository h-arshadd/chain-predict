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
# Each takes a DataFrame of aligned signal columns (one column per
# playbook entry / ML model being combined) and returns one combined
# 1/0/-1 pd.Series.
#
# Columns may contain NaN, not just 1/0/-1: resolve_component_signal()
# leaves an ml_model component's bar as NaN when ml.model_signals has no
# row for it yet (no data), as opposed to 0 (model genuinely predicted
# Hold) -- see that function's docstring.
#
# NaN handling (applies to every rule below): a NaN component at some
# bar simply does not participate in that bar's vote -- the rule is
# computed only over whichever components DO have a real signal there,
# same rule, fewer voters. This is not "vote 0" (an AND would wrongly
# lose agreement) and not "poison the whole bar to NaN" (one still-
# warming-up ML model shouldn't block two playbook strategies that
# already agree). A bar where every component is NaN has no voters and
# stays neutral (0), same as a genuine unanimous "no signal" would.

def _and_rule(df: pd.DataFrame) -> pd.Series:
    """
    Every input WITH A REAL SIGNAL at that bar must agree on the same
    nonzero direction (a component with no signal yet at that bar
    doesn't count toward or against agreement -- see module docstring
    above). If any voting input is neutral (0), or voting inputs
    disagree (some 1, some -1), the combined result is 0. A bar with no
    voters at all also stays 0.
    """
    present = df.notna()
    all_long = ((df == 1) | ~present).all(axis=1) & present.any(axis=1)
    all_short = ((df == -1) | ~present).all(axis=1) & present.any(axis=1)

    result = pd.Series(0, index=df.index, dtype=int)
    result.loc[all_long] = 1
    result.loc[all_short] = -1
    return result


def _or_rule(df: pd.DataFrame) -> pd.Series:
    """
    Any input WITH A REAL SIGNAL at that bar signaling is enough to
    trigger (a component with no signal yet doesn't participate either
    way). Tie-break rule (documented here since rules.py has no
    equivalent case to copy from): if voting inputs disagree on
    direction on the same bar (some long, some short), that's treated
    as conflicting information, not a trade -- combined result is 0,
    same "conflict cancels out" convention rules.apply_rules() already
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
    More than half of the inputs WITH A REAL SIGNAL at that bar must
    agree on the same direction -- required is computed against how
    many components actually have a signal at that bar, not the total
    component count, so a component with no signal yet doesn't shift
    what counts as a majority of the others.
    """
    present = df.notna()
    n_voting = present.sum(axis=1)
    required = (n_voting // 2) + 1

    long_votes = (df == 1).sum(axis=1)
    short_votes = (df == -1).sum(axis=1)

    result = pd.Series(0, index=df.index, dtype=int)
    has_voters = n_voting > 0
    result.loc[has_voters & (long_votes >= required)] = 1
    result.loc[has_voters & (short_votes >= required)] = -1
    return result


def _weighted_rule(df: pd.DataFrame, weights: list, threshold: float) -> pd.Series:
    """
    Weighted voting, same shape as rules.weighted_rule but signed: each
    input WITH A REAL SIGNAL at that bar contributes +weight for a long
    signal, -weight for a short signal, 0 for neutral. A component with
    no signal yet at that bar contributes nothing to the score (it's
    excluded from the sum entirely for that bar, not counted as 0-for-
    neutral). Combined result is long if the summed score clears
    +threshold, short if it clears -threshold, else neutral.

    Example
    -------
    weights = [0.5, 0.3, 0.2], threshold = 0.6
    """
    if len(weights) != df.shape[1]:
        raise ValueError("Number of weights must equal number of signal columns.")

    weight_array = np.asarray(weights, dtype=float)
    # NaN * weight = NaN, so a missing component's contribution is NaN
    # per-cell, then excluded (not summed as 0) via nansum below.
    contributions = df.values * weight_array
    score = np.nansum(contributions, axis=1)
    # A bar where every component is missing has no real score --
    # np.nansum would silently return 0.0 for an all-NaN row, which
    # reads as "unanimously neutral" rather than "no data at all".
    # Keep it explicitly at 0 either way since combine_signals() with
    # no components voting has nothing better to fall back to, but
    # this comment flags it as the one place that ambiguity is
    # unavoidable rather than a missed case.

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
    pd.Series, possibly with NaN bars for an ml_model component with no
    signal yet -- see resolve_component_signal()'s docstring -- same
    index/alignment; callers are responsible for aligning them, e.g. via
    pd.concat(..., axis=1) upstream having already dropped any mismatched
    bars) into one final signal, using the same AND / OR / MAJORITY /
    WEIGHTED vocabulary as signals/rules.py.

    Parameters
    ----------
    signals : list of pd.Series
        One 1/0/-1(/NaN) series per playbook entry (post whole-strategy
        persist, if any -- see apply_whole_strategy_persist) and/or ML
        model.
    rule : str
        "AND", "OR", "MAJORITY", or "WEIGHTED" (case-insensitive).
    weights, threshold : only used when rule == "WEIGHTED".

    Returns
    -------
    pd.Series of 1 / 0 / -1, same index as the input signals. A
    component with no real signal at some bar (NaN) simply doesn't
    participate in that bar's vote -- see each _*_rule's docstring.
    """
    if len(signals) == 0:
        raise ValueError("combine_signals() requires at least one signal.")

    if len(signals) == 1:
        # Nothing to combine -- a single selected playbook entry/model
        # passes through unchanged, same as the PDF's "user may select
        # one or multiple strategies" allowing exactly one. NOT filled
        # to 0 -- an ml_model component with no signal yet should stay
        # NaN here too, same as it would if it were one voter among
        # several (see resolve_component_signal()'s docstring for why
        # faking 0 is wrong).
        return signals[0]

    df = pd.concat(signals, axis=1)
    df.columns = range(df.shape[1])
    # Deliberately NOT df.fillna(0) here -- every rule below needs to
    # see the real NaN to tell "this component has no signal yet" apart
    # from "this component voted neutral" (see module docstring above).

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
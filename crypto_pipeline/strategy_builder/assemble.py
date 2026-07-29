"""
assemble.py

Turns N selected playbook entries (+ optional ML model run_ids) into a
runnable metadata.strategy row -- the Strategy Builder's "Final Strategy"
step (Strategy_Builder_Module.pdf). This is the only place a playbook
template gets bound to exchange/coin/TP/SL/time_horizon; playbook itself
stays pair-agnostic (see metadata_utils.create_playbook_table).

Everything actually generating/combining signals lives in combine.py and
signals/main.py -- this file's job is orchestration: pull each
component's resolved signal, apply whole-strategy persist, combine them,
and package the result into the exact shape insert_strategy() expects.

Bybit only, for now (per current instructions) -- exchange defaults to
"bybit" everywhere in this module rather than being a free-text field the
frontend has to supply.
"""

import pandas as pd

from crypto_pipeline.signals.main import generate_signals
from crypto_pipeline.strategy_builder.combine import (
    combine_signals,
    apply_whole_strategy_persist,
)

DEFAULT_EXCHANGE = "bybit"


def resolve_component_signal(component: dict, ohlcv: pd.DataFrame, get_model_signals=None) -> pd.Series:
    """
    Resolve ONE selected builder component (a playbook entry OR an ML
    model reference) down to a single 1/0/-1(/NaN) pd.Series aligned to
    ohlcv's index, with that component's own whole-strategy persist_bars
    already applied.

    component shape (one of):
        {"kind": "playbook", "strategy_config": {...}, "persist_bars": 5}
        {"kind": "ml_model", "run_id": "...", "persist_bars": 0}

    get_model_signals: injected callable (conn, run_id, start, end) ->
    pd.Series, so this module doesn't import db_utils/ml_repo directly --
    keeps strategy_builder/ decoupled from the DB layer per the "combine
    resolved signals" scope decided in STRATEGY_BUILDER_SPEC.md Section
    5.4. Callers (the API repo) pass in the real function.

    NaN vs 0, and why ml_model does NOT get fillna(0) here: a playbook
    entry's signal is computed live against this exact ohlcv, so a NaN
    there only ever means genuine indicator warm-up with no real signal
    -- 0/neutral is the correct, honest value. An ml_model's signal is
    read back from ml.model_signals, a separately-populated table that
    can legitimately have no row yet for a given bar (inference hasn't
    caught up, a gap, etc.) -- reindexing that onto ohlcv and forcing
    missing bars to 0 would silently claim "the model predicted Hold"
    when the truth is "we don't know what the model would have said".
    That corrupts every combine rule differently (AND treats a missing
    model as active disagreement, WEIGHTED dilutes the score, etc.), so
    ml_model bars with no real row stay NaN and combine_signals() is
    responsible for treating NaN as this component abstaining, not
    voting neutral.
    """
    persist_bars = component.get("persist_bars", 0) or 0

    if component["kind"] == "playbook":
        _, _, signal = generate_signals(ohlcv, config_dict=component["strategy_config"])
        signal = signal.reindex(ohlcv.index).fillna(0).astype(int)
        signal = apply_whole_strategy_persist(signal, persist_bars)

    elif component["kind"] == "ml_model":
        if get_model_signals is None:
            raise ValueError("get_model_signals callable required to resolve an ml_model component.")
        signal = get_model_signals(component["run_id"])
        signal = signal.reindex(ohlcv.index)  # NaN where this run_id has no row for that bar -- not filled.
        # apply_whole_strategy_persist requires an int series (it does
        # signal.fillna(0).astype(int) internally), which would re-
        # introduce the exact fake-0 problem this function exists to
        # avoid. Persist bars only where we have a real signal, and
        # restore the true NaN gaps afterward so they still propagate
        # to combine_signals() as "no data" rather than "neutral".
        missing_mask = signal.isna()
        if persist_bars > 0:
            persisted = apply_whole_strategy_persist(signal.fillna(0).astype(int), persist_bars)
            signal = persisted.astype(float)
            signal[missing_mask] = float("nan")
        return signal

    else:
        raise ValueError(f"Unknown component kind: {component['kind']!r}")

    return signal


def build_combined_signal(components: list, ohlcv: pd.DataFrame, combine_rule: str,
                           weights: list = None, threshold: float = None,
                           get_model_signals=None) -> pd.Series:
    """
    Resolve every component to its own signal, then combine them into one
    final 1/0/-1 series via combine_signals(). This is what the builder's
    "Run Backtest" step calls right before handing the result to the
    existing backtest flow, and it's also what a "preview before saving"
    endpoint would call.
    """
    signals = [
        resolve_component_signal(c, ohlcv, get_model_signals=get_model_signals)
        for c in components
    ]
    return combine_signals(signals, rule=combine_rule, weights=weights, threshold=threshold)


def assemble_strategy_config(components: list, combine_rule: str,
                              weights: list = None, threshold: float = None) -> dict:
    """
    Build the strategy_config JSON to save into metadata.strategy for a
    combined strategy -- NOT the resolved signal itself (that's
    build_combined_signal(), re-run live at backtest time per Section 4
    decision 5: "recompute live, no new signal-cache table").

    Stores enough to reconstruct exactly what was combined -- which
    playbook entries / ML run_ids, each one's whole-strategy persist_bars,
    and the combine rule/weights/threshold -- so a saved combined strategy
    stays inspectable/editable later instead of being a flattened opaque
    blob. Shape:

        {
          "builder": {
            "components": [
              {"kind": "playbook", "playbook_id": 3, "strategy_name": "...",
               "persist_bars": 5},
              {"kind": "ml_model", "run_id": "...", "persist_bars": 0}
            ],
            "combine_rule": "AND",
            "weights": null,
            "threshold": null
          }
        }

    This dict is what gets stored in metadata.strategy.strategy_config --
    generate_signals() itself never reads this shape directly (a combined
    strategy's signal always comes from build_combined_signal(), not from
    re-parsing this JSON as if it were a single-strategy config) -- kept
    separate and explicit so there's no ambiguity about which code path
    produces a combined strategy's actual signal.
    """
    builder_components = []
    for c in components:
        entry = {"kind": c["kind"], "persist_bars": c.get("persist_bars", 0) or 0}
        if c["kind"] == "playbook":
            entry["playbook_id"] = c.get("playbook_id")
            entry["strategy_name"] = c.get("strategy_name")
        elif c["kind"] == "ml_model":
            entry["run_id"] = c.get("run_id")
        builder_components.append(entry)

    return {
        "builder": {
            "components": builder_components,
            "combine_rule": combine_rule.upper(),
            "weights": weights,
            "threshold": threshold,
        }
    }


def default_strategy_name(components: list) -> str | None:
    """
    A strategy built from exactly one playbook entry (no ML models, no
    combination) can default its name to that entry's own strategy_name
    (still editable by the user before saving) -- per Section 4 decision
    4. Returns None for anything else (2+ components, or a single ML
    model with no playbook entry), since there's no sensible single name
    to default to -- the caller/frontend must ask for one.
    """
    if len(components) == 1 and components[0]["kind"] == "playbook":
        return components[0].get("strategy_name")
    return None
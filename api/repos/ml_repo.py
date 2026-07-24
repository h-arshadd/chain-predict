"""
repos/ml_repo.py
------------------
Read access for the ML Models (list) + Model Details pages.

Unlike every other repo in this codebase, this one does NOT talk to
Postgres. Trained runs are not stored in the DB at all -- they live on
disk as one run_config.json per run, written by
crypto_pipeline.ml.persistence.artifact_manager.save_run():

    crypto_pipeline/ml/artifacts/configs/{run_id}/run_config.json

This module scans that folder fresh on every call. Nothing here is
hardcoded to a fixed list of run_ids or algorithms -- train a new model,
its run_config.json shows up under artifacts/configs/, and it appears
here on the next request with zero code changes. Delete a folder, it
disappears the same way.

model_type ("regression" / "classification") and whether a run is deep
learning are both read straight off each run's own run_summary /
model.model_type -- see _is_deep_learning() below. "timeseries" runs
(model_type == "timeseries" or "timeseries_classifier") are filtered out
everywhere in this module: the person's UI has no timeseries option, and
crypto_pipeline.ml.timeseries is a separate pipeline (darts-backed, own
registry) that this module intentionally does not surface yet.

ARTIFACTS_DIR/CONFIGS_SUBDIR/RUN_CONFIG_FILENAME are imported from
artifact_manager.py itself rather than re-declared here, so this stays
correct if that layout ever changes -- one source of truth for where
runs live, same as model_loader.py already does it.
"""

import json
import logging
import os

from crypto_pipeline.ml.persistence.artifact_manager import (
    ARTIFACTS_DIR, CONFIGS_SUBDIR, RUN_CONFIG_FILENAME,
)

logger = logging.getLogger(__name__)

_CONFIGS_ROOT = os.path.join(ARTIFACTS_DIR, CONFIGS_SUBDIR)

# model_kind values (run_summary["model_kind"] / model["model_type"])
# that count as "deep learning" for the include_deep_learning filter.
_DEEP_LEARNING_KINDS = {"deep_learning_regressor", "deep_learning_classifier"}

# model_kind values this module surfaces at all. Timeseries runs are
# read off disk like everything else here but dropped before they ever
# reach the router -- no timeseries option exists in this UI.
_SUPPORTED_KINDS = {"regressor", "classifier"} | _DEEP_LEARNING_KINDS


def _is_deep_learning(model_kind: str) -> bool:
    return model_kind in _DEEP_LEARNING_KINDS


def _read_run_config(run_id: str) -> dict | None:
    """
    Read one run's run_config.json off disk. Returns None (not raises)
    if the folder/file is missing or unreadable -- a run can vanish
    between listing the directory and reading it (deleted mid-request,
    partially written), and one bad/missing folder shouldn't 500 the
    whole list endpoint.
    """
    config_path = os.path.join(_CONFIGS_ROOT, run_id, RUN_CONFIG_FILENAME)
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Skipping unreadable run_config.json for run_id='{run_id}': {e}")
        return None


def _list_run_ids() -> list[str]:
    """Every subfolder under artifacts/configs/ that has a run_config.json."""
    if not os.path.isdir(_CONFIGS_ROOT):
        return []
    return sorted(
        name for name in os.listdir(_CONFIGS_ROOT)
        if os.path.isfile(os.path.join(_CONFIGS_ROOT, name, RUN_CONFIG_FILENAME))
    )


def _summary_row(run_id: str, config: dict) -> dict | None:
    """
    Build one list-view row from a run's config. Returns None for a run
    whose model_kind isn't supported here (timeseries, or anything the
    model_loader.py registry map doesn't recognise) so the router never
    has to know about that filtering itself.
    """
    run_summary = config.get("run_summary", {}) or {}
    model = config.get("model", {}) or {}
    evaluation = config.get("evaluation", {}) or {}

    model_kind = run_summary.get("model_kind") or model.get("model_type")
    if model_kind not in _SUPPORTED_KINDS:
        return None

    trading = evaluation.get("trading_metrics_summary", {}) or {}
    ml_metrics = evaluation.get("ml_metrics", {}) or {}

    return {
        "run_id": run_id,
        "algorithm": run_summary.get("algorithm"),
        "model_kind": model_kind,
        "is_deep_learning": _is_deep_learning(model_kind),
        "model_type": run_summary.get("pipeline_type"),  # "regression" | "classification"
        "symbol": run_summary.get("symbol"),
        "exchange": run_summary.get("exchange"),
        "timeframe": run_summary.get("timeframe"),
        "horizon": run_summary.get("horizon"),
        "start_date": run_summary.get("start_date"),
        "end_date": run_summary.get("end_date"),
        # Headline metrics for the list table -- sharpe/win_rate always
        # come from trading_metrics_summary regardless of model_type
        # (both regression and classification runs get signal-converted
        # and backtested the same way, per evaluator.py). ml_metrics'
        # actual keys differ by model_type (mae/rmse vs accuracy/f1),
        # so we surface both blocks whole rather than picking one field
        # and hardcoding a name that only exists for one of the two.
        "sharpe": trading.get("sharpe"),
        "win_rate": trading.get("win_rate"),
        "ml_metrics": ml_metrics,
    }


def list_runs(model_type: str | None = None, include_deep_learning: bool = True) -> list[dict]:
    """
    List every trained run currently on disk, newest-folder-name-last
    (run_id has no timestamp by design -- see artifact_manager.py's
    make_run_id() docstring -- so this is alphabetical, not chronological;
    the frontend can re-sort by any column it exposes).

    Args:
        model_type: "regression" | "classification" | None (no filter).
            Matches run_summary["pipeline_type"] exactly as written by
            metadata.py -- never partial/case-insensitive matched, since
            the pipeline only ever writes these two exact strings (plus
            "timeseries", which this module excludes outright above).
        include_deep_learning: if False, drops mlp/lstm/gru runs
            (model_kind in {"deep_learning_regressor",
            "deep_learning_classifier"}), leaving only the traditional
            sklearn/xgboost/lightgbm/catboost runs. Default True (show
            everything supported).
    """
    rows = []
    for run_id in _list_run_ids():
        config = _read_run_config(run_id)
        if config is None:
            continue
        row = _summary_row(run_id, config)
        if row is None:
            continue
        if model_type is not None and row["model_type"] != model_type:
            continue
        if not include_deep_learning and row["is_deep_learning"]:
            continue
        rows.append(row)
    return rows


def get_run_detail(run_id: str) -> dict | None:
    """
    Full detail for one run: the summary row fields plus every section
    of run_config.json (data_prep, split, preprocessing, model,
    evaluation) untouched, so the detail page can show the complete
    experiment record -- feature list, train/test/val split, exact
    preprocessing steps, full hyperparameters/architecture, and the full
    evaluation block (ml_metrics + trading_metrics_summary + trade_summary
    + signal_counts) -- without this repo re-shaping or re-guessing field
    names the pipeline already wrote.

    Returns None if run_id doesn't exist, its config can't be read, or
    its model_kind isn't one this module supports (timeseries).
    """
    config = _read_run_config(run_id)
    if config is None:
        return None
    summary = _summary_row(run_id, config)
    if summary is None:
        return None

    return {
        **summary,
        "data_prep": config.get("data_prep", {}),
        "split": config.get("split", {}),
        "preprocessing": config.get("preprocessing", {}),
        "model": config.get("model", {}),
        "evaluation": config.get("evaluation", {}),
    }
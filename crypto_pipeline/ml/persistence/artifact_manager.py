# crypto_pipeline/ml/persistence/artifact_manager.py

"""
artifact_manager.py
--------------------
Owns the artifacts/ and models/ folder layout (PDF heading 11 + your
lead's requirement: "one artifact folder and configs folder in it"),
with trained model files kept in their own top-level models/ folder
instead of alongside the configs.

Layout, one folder per trained run:

    artifacts/
      configs/
        {run_id}/
          run_config.json         <- one file, all five
                                      metadata.build_*_metadata() outputs
                                      merged under top-level keys:
                                      data_prep / split / preprocessing /
                                      model / evaluation

    models/
      {run_id}/
        model.joblib           <- traditional models (regressors/classifiers)
          -or-
        model.pt                <- deep learning models (BaseNetwork/BaseClassifierNetwork)
        preprocessing.joblib     <- fitted scaler/transform objects (fit_objects)

One combined JSON per run instead of one file per stage: everything
about the run (which coin/exchange/timeframe/dates it used, the split,
the preprocessing steps, the model + hyperparams, the eval metrics) is
readable at a glance in a single file. model_saver.py / model_loader.py
own reading/writing model.joblib / model.pt; this file only decides
WHERE things go and writes the config JSON, since config writing
doesn't depend on which serialization format the model itself uses.

run_id is a plain timestamp + algorithm slug by default (sortable,
collision-resistant enough for a single-user pipeline) -- pass your own
if you want a specific name. The same run_id is used as the folder name
under both artifacts/configs/ and models/, so the two stay linked.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import joblib

logger = logging.getLogger(__name__)

# Anchored to the ml/ package directory (persistence/ -> ml/), not to
# whatever directory the process happens to be launched from. Plain
# relative strings ("artifacts", "models") resolve against the current
# working directory, so running `python -m crypto_pipeline.ml.main`
# from the repo root instead of ml/ silently wrote artifacts/models/
# into the repo root -- outside the ml project entirely. Anchoring
# here means every caller that imports these constants (main.py, and
# pipeline/regression_pipeline.py / classification_pipeline.py /
# timeseries_pipeline.py, which all default to these same constants)
# gets ml/artifacts and ml/models regardless of CWD.
_ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(_ML_DIR, "artifacts")
MODELS_DIR = os.path.join(_ML_DIR, "models")
CONFIGS_SUBDIR = "configs"

# Single combined config file, written under configs/{run_id}/. Its
# top-level keys match the metadata dict passed into save_run() ("data_prep",
# "split", "preprocessing", "model", "evaluation").
RUN_CONFIG_FILENAME = "run_config.json"


def make_run_id(algorithm: str, symbol: str = None, exchange: str = None,
                 model_type: str = None, horizon=None) -> str:
    """
    Default run_id, used as the folder name for both models/{run_id}/
    and artifacts/configs/{run_id}/:

        {exchange}_{symbol}_{model_type}_h{horizon}_{algorithm}

    e.g. 'binance_BTCUSDT_classification_h4_xgboost'

    No timestamp -- by design, so folder names stay short and readable.
    This means running the exact same algorithm/symbol/exchange/
    model_type/horizon combo twice OVERWRITES the previous run's folder
    (same name both times) -- both models/{run_id}/ and
    artifacts/configs/{run_id}/. If you want to keep multiple runs of
    the same config side by side, pass your own run_id instead of
    relying on this default.

    Any piece that's missing (None) is just skipped rather than writing
    "None" into the folder name, so this still works if called with
    only `algorithm` (old call sites don't break).

    Args:
        algorithm: e.g. "xgboost", "random_forest", "lstm"
        symbol: e.g. "BTCUSDT" -- ml_config["data"]["symbol"]
        exchange: e.g. "binance" -- ml_config["data"]["exchange"]
        model_type: "regression" | "classification" | "timeseries" --
            ml_config["model_type"]
        horizon: e.g. 4 -- ml_config["target"]["horizon"]
    """
    parts = []
    if exchange:
        parts.append(str(exchange))
    if symbol:
        parts.append(str(symbol))
    if model_type:
        parts.append(str(model_type))
    if horizon is not None:
        parts.append(f"h{horizon}")
    parts.append(algorithm)
    return "_".join(parts)


def _build_run_summary(run_id: str, metadata: dict, trained_at: str) -> dict:
    """
    Flat "what is this run" block written at the very TOP of
    run_config.json, above the detailed data_prep/split/preprocessing/
    model/evaluation sections. Everything in here is just read off of
    those sections -- no new inputs needed -- it exists purely so the
    run/algorithm/symbol/timeframe/dates don't require digging into
    nested sections to find.

    trained_at is the one exception -- it's not derived from `metadata`
    at all, it's the wall-clock moment save_run() actually wrote this
    config (passed in from there, see save_run()'s docstring for why).
    Distinct from data_prep.data.start_date/end_date, which is the date
    RANGE OF THE MARKET DATA the model trained on, not when training
    itself happened -- the two get conflated easily since both are
    "dates on a training run," so they're kept clearly separate here
    and in the frontend (data range stays under "Dataset Information",
    trained_at is its own field).
    """
    data_prep = metadata.get("data_prep", {}) or {}
    data = data_prep.get("data", {}) or {}
    target = data_prep.get("target", {}) or {}
    model = metadata.get("model", {}) or {}

    return {
        "run_id": run_id,
        "algorithm": model.get("algorithm"),
        "model_kind": model.get("model_type"),      # regressor / classifier / deep_learning_regressor / etc.
        "pipeline_type": data_prep.get("model_type"),  # regression / classification / timeseries
        "symbol": data.get("symbol"),
        "exchange": data.get("exchange"),
        "timeframe": data.get("timeframe"),
        "start_date": data.get("start_date"),
        "end_date": data.get("end_date"),
        "horizon": target.get("horizon"),
        "trained_at": trained_at,
    }


def save_run(
    run_id: str,
    metadata: dict,
    fit_objects: list,
    base_dir: str = ARTIFACTS_DIR,
    models_dir: str = MODELS_DIR,
    model_save_fn=None,
) -> dict:
    """
    Write one complete run's artifacts to disk: a single combined config
    JSON under {base_dir}/configs/{run_id}/run_config.json, and the
    fitted preprocessing objects + (if model_save_fn is given) the model
    itself under a separate {models_dir}/{run_id}/.

    Args:
        run_id: folder name for this run, e.g. from make_run_id()
        metadata: dict with keys "data_prep", "split", "preprocessing",
            "model", "evaluation" -- each value is that stage's dict
            from the matching metadata.build_*_metadata() function.
            Written out as-is, merged under those same top-level keys
            in one JSON file, with an extra "run_summary" key at the
            top (see _build_run_summary()) pulling the
            algorithm/symbol/timeframe/dates/trained_at together in one
            flat spot.
        fit_objects: list from preprocessing_pipeline.run_preprocessing()
            (fitted scalers/transforms -- persisted here so inference can
            exactly replay the same preprocessing chain, per PDF heading
            11's "exact preprocessing sequence must be recoverable")
        base_dir: root artifacts folder (configs live here), defaults to "artifacts"
        models_dir: root models folder (model + preprocessing files live
            here, kept separate from configs), defaults to "models"
        model_save_fn: optional callable(path: str) -> None that saves
            the trained model to `path` (e.g. `model.save`, since every
            BaseRegressor/BaseClassifier/BaseNetwork/BaseClassifierNetwork
            already exposes .save(path)). The correct file extension
            (.joblib / .pt / .darts) is chosen from metadata["model"]["serialization_format"].
            If None, only the config + preprocessing objects are written
            (useful for a dry run, or if the caller wants to call
            model.save() itself afterward).

    Returns:
        dict with the paths written:
            run_dir: str (under models_dir -- model + preprocessing files)
            config_dir: str (under base_dir -- config json)
            config_path: str, path to run_config.json
            preprocessing_path: str
            model_path: str or None (None if model_save_fn was not given)
    """
    run_dir = os.path.join(models_dir, run_id)
    config_dir = os.path.join(base_dir, CONFIGS_SUBDIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # Captured here, at the moment this run's config is actually written
    # -- NOT in main.py/the pipelines, and not read back off the
    # filesystem later (folder mtime is unreliable: make_run_id()'s own
    # docstring says re-running the same algorithm/symbol/exchange/
    # model_type/horizon combo OVERWRITES the previous run's folder, so
    # an mtime read after the fact could only ever reflect the LATEST
    # training, never the run history). UTC, ISO 8601, so it sorts as a
    # plain string and needs no timezone guessing on the read side
    # (ml_repo.py / Models.jsx).
    trained_at = datetime.now(timezone.utc).isoformat()

    config_path = os.path.join(config_dir, RUN_CONFIG_FILENAME)
    run_config = {"run_summary": _build_run_summary(run_id, metadata, trained_at)}
    run_config.update({stage: metadata.get(stage, {}) for stage in
                        ("data_prep", "split", "preprocessing", "model", "evaluation")})
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    logger.info(f"Config written: {config_path}")

    preprocessing_path = os.path.join(run_dir, "preprocessing.joblib")
    joblib.dump(fit_objects, preprocessing_path)
    logger.info(f"Preprocessing objects written: {preprocessing_path}")

    model_path: Optional[str] = None
    if model_save_fn is not None:
        serialization_format = metadata.get("model", {}).get("serialization_format", "joblib")
        _EXTENSIONS = {"pytorch_checkpoint": "pt", "darts_checkpoint": "darts"}
        extension = _EXTENSIONS.get(serialization_format, "joblib")
        model_path = os.path.join(run_dir, f"model.{extension}")
        model_save_fn(model_path)
        logger.info(f"Model written: {model_path}")

    return {
        "run_dir": run_dir,
        "config_dir": config_dir,
        "config_path": config_path,
        "preprocessing_path": preprocessing_path,
        "model_path": model_path,
    }
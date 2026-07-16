# crypto_pipeline/ml/persistence/artifact_manager.py

"""
artifact_manager.py
--------------------
Owns the artifacts/ folder layout (PDF heading 11 + your lead's
requirement: "one artifact folder and configs folder in it").

Layout, one folder per trained run:

    artifacts/
      configs/
        {run_id}.yaml         <- metadata.build_metadata()'s output
      {run_id}/
        model.joblib           <- traditional models (regressors/classifiers)
          -or-
        model.pt                <- deep learning models (BaseNetwork/BaseClassifierNetwork)
        preprocessing.joblib     <- fitted scaler/transform objects (fit_objects)

`configs/` is intentionally flat and separate from the per-run model
folders: it's meant to be quickly grep-able/diffable (every experiment's
full config in one place) without wading into each run's binary
artifacts. model_saver.py / model_loader.py (not built yet) will own
reading/writing model.joblib / model.pt; this file only decides WHERE
things go and writes the config YAML, since config writing doesn't
depend on which serialization format the model itself uses.

run_id is a plain timestamp + algorithm slug by default (sortable,
collision-resistant enough for a single-user pipeline) -- pass your own
if you want a specific name.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import joblib
import yaml

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "artifacts"
CONFIGS_SUBDIR = "configs"


def make_run_id(algorithm: str) -> str:
    """Default run_id: {UTC timestamp}_{algorithm}, e.g. '20260716_142530_random_forest'."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{algorithm}"


def save_run(
    run_id: str,
    metadata: dict,
    fit_objects: list,
    base_dir: str = ARTIFACTS_DIR,
    model_save_fn=None,
) -> dict:
    """
    Write one complete run's artifacts to disk: the config YAML under
    configs/, the fitted preprocessing objects, and (if model_save_fn is
    given) the model itself.

    Args:
        run_id: folder/file name for this run, e.g. from make_run_id()
        metadata: dict from metadata.build_metadata()
        fit_objects: list from preprocessing_pipeline.run_preprocessing()
            (fitted scalers/transforms -- persisted here so inference can
            exactly replay the same preprocessing chain, per PDF heading
            11's "exact preprocessing sequence must be recoverable")
        base_dir: root artifacts folder, defaults to "artifacts"
        model_save_fn: optional callable(path: str) -> None that saves
            the trained model to `path` (e.g. `model.save`, since every
            BaseRegressor/BaseClassifier/BaseNetwork/BaseClassifierNetwork
            already exposes .save(path)). The correct file extension
            (.joblib vs .pt) is chosen from metadata["model"]["serialization_format"].
            If None, only the config + preprocessing objects are written
            (useful for a dry run, or if the caller wants to call
            model.save() itself afterward).

    Returns:
        dict with the paths written:
            run_dir: str
            config_path: str
            preprocessing_path: str
            model_path: str or None (None if model_save_fn was not given)
    """
    run_dir = os.path.join(base_dir, run_id)
    configs_dir = os.path.join(base_dir, CONFIGS_SUBDIR)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)

    config_path = os.path.join(configs_dir, f"{run_id}.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False, default_flow_style=False)
    logger.info(f"Config written: {config_path}")

    preprocessing_path = os.path.join(run_dir, "preprocessing.joblib")
    joblib.dump(fit_objects, preprocessing_path)
    logger.info(f"Preprocessing objects written: {preprocessing_path}")

    model_path: Optional[str] = None
    if model_save_fn is not None:
        serialization_format = metadata.get("model", {}).get("serialization_format", "joblib")
        extension = "pt" if serialization_format == "pytorch_checkpoint" else "joblib"
        model_path = os.path.join(run_dir, f"model.{extension}")
        model_save_fn(model_path)
        logger.info(f"Model written: {model_path}")

    return {
        "run_dir": run_dir,
        "config_path": config_path,
        "preprocessing_path": preprocessing_path,
        "model_path": model_path,
    }
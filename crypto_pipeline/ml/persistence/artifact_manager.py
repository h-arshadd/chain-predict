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
          data_prep.yaml        <- metadata.build_data_prep_metadata()'s output
          split.yaml             <- metadata.build_split_metadata()'s output
          preprocessing.yaml     <- metadata.build_preprocessing_metadata()'s output
          model.yaml              <- metadata.build_model_metadata()'s output
          evaluation.yaml         <- metadata.build_evaluation_metadata()'s output

    models/
      {run_id}/
        model.joblib           <- traditional models (regressors/classifiers)
          -or-
        model.pt                <- deep learning models (BaseNetwork/BaseClassifierNetwork)
        preprocessing.joblib     <- fitted scaler/transform objects (fit_objects)

Each stage writes its own yaml rather than one combined config file:
data_prep's config never touches the model's, so a run can be inspected
stage by stage, and a new field in one stage never risks colliding with
another. model_saver.py / model_loader.py own reading/writing
model.joblib / model.pt; this file only decides WHERE things go and
writes the config YAMLs, since config writing doesn't depend on which
serialization format the model itself uses.

run_id is a plain timestamp + algorithm slug by default (sortable,
collision-resistant enough for a single-user pipeline) -- pass your own
if you want a specific name. The same run_id is used as the folder name
under both artifacts/configs/ and models/, so the two stay linked.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import joblib
import yaml

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "artifacts"
MODELS_DIR = "models"
CONFIGS_SUBDIR = "configs"

# Filenames for each per-stage config yaml, written under
# configs/{run_id}/. Keys match the metadata dict passed into save_run().
CONFIG_FILENAMES = {
    "data_prep": "data_prep.yaml",
    "split": "split.yaml",
    "preprocessing": "preprocessing.yaml",
    "model": "model.yaml",
    "evaluation": "evaluation.yaml",
}


def make_run_id(algorithm: str) -> str:
    """Default run_id: {UTC timestamp}_{algorithm}, e.g. '20260716_142530_random_forest'."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{algorithm}"


def save_run(
    run_id: str,
    metadata: dict,
    fit_objects: list,
    base_dir: str = ARTIFACTS_DIR,
    models_dir: str = MODELS_DIR,
    model_save_fn=None,
) -> dict:
    """
    Write one complete run's artifacts to disk: one config yaml per
    pipeline stage under {base_dir}/configs/{run_id}/, and the fitted
    preprocessing objects + (if model_save_fn is given) the model itself
    under a separate {models_dir}/{run_id}/.

    Args:
        run_id: folder name for this run, e.g. from make_run_id()
        metadata: dict with keys "data_prep", "split", "preprocessing",
            "model", "evaluation" -- each value is that stage's dict
            from the matching metadata.build_*_metadata() function.
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
            If None, only the configs + preprocessing objects are written
            (useful for a dry run, or if the caller wants to call
            model.save() itself afterward).

    Returns:
        dict with the paths written:
            run_dir: str (under models_dir -- model + preprocessing files)
            config_dir: str (under base_dir -- config yamls)
            config_paths: dict, one path per stage (same keys as CONFIG_FILENAMES)
            preprocessing_path: str
            model_path: str or None (None if model_save_fn was not given)
    """
    run_dir = os.path.join(models_dir, run_id)
    config_dir = os.path.join(base_dir, CONFIGS_SUBDIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    config_paths = {}
    for stage, filename in CONFIG_FILENAMES.items():
        stage_metadata = metadata.get(stage, {})
        stage_path = os.path.join(config_dir, filename)
        with open(stage_path, "w") as f:
            yaml.safe_dump(stage_metadata, f, sort_keys=False, default_flow_style=False)
        config_paths[stage] = stage_path
        logger.info(f"Config written: {stage_path}")

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
        "config_paths": config_paths,
        "preprocessing_path": preprocessing_path,
        "model_path": model_path,
    }
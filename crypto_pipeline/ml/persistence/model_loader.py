# crypto_pipeline/ml/persistence/model_loader.py

"""
model_loader.py
----------------
The read side of PDF heading 11: "Every trained model shall be
completely reproducible." artifact_manager.py + metadata.py write
everything a run needs; this file is what turns that back into a ready-
to-predict model, given nothing but a run_id.

    from crypto_pipeline.ml.persistence.model_loader import load_run

    run = load_run("20260716_142530_random_forest")
    predictions = run["model"].predict(X_new[run["metadata"]["features"]["feature_columns"]])

Deliberately does NOT re-run preprocessing on X_new itself -- every
fitted preprocessing object is already sitting in run["fit_objects"],
and how they're re-applied (which columns, what order) is
preprocessing_pipeline.py's job, not this file's. This stays a plain
loader: read the artifacts back off disk (the single combined
run_config.json under configs/{run_id}/, plus the model and
preprocessing objects), hand them to the caller unchanged.
"""

import json
import logging
import os

import joblib

from crypto_pipeline.ml.persistence.artifact_manager import (
    ARTIFACTS_DIR, MODELS_DIR, CONFIGS_SUBDIR, RUN_CONFIG_FILENAME,
)
from crypto_pipeline.ml.regressors.registry import REGRESSORS
from crypto_pipeline.ml.classifiers.registry import CLASSIFIERS
from crypto_pipeline.ml.deep_learning.registry import DL_REGRESSORS, DL_CLASSIFIERS
from crypto_pipeline.ml.timeseries.registry import TS_MODELS

logger = logging.getLogger(__name__)

# model_kind (as written by metadata.py) -> which registry + file extension
# to use for reconstruction. Mirrors metadata._MODEL_INFO_BUILDERS's own
# five-way split -- one place per model kind, no if/elif chain here either.
_MODEL_KIND_REGISTRIES = {
    "regressor": (REGRESSORS, "joblib"),
    "classifier": (CLASSIFIERS, "joblib"),
    "deep_learning_regressor": (DL_REGRESSORS, "pt"),
    "deep_learning_classifier": (DL_CLASSIFIERS, "pt"),
    "timeseries": (TS_MODELS, "darts"),
}


def load_run(run_id: str, base_dir: str = ARTIFACTS_DIR, models_dir: str = MODELS_DIR) -> dict:
    """
    Load everything needed to run inference for one trained run.

    Args:
        run_id: the run folder/config name, e.g. from
            artifact_manager.make_run_id() or save_run()'s return value.
        base_dir: root artifacts folder (configs live here), defaults to
            "artifacts" (must match whatever base_dir save_run() was
            originally called with).
        models_dir: root models folder (model + preprocessing files live
            here), defaults to "models" (must match whatever models_dir
            save_run() was originally called with).

    Returns:
        dict:
            metadata: dict with keys "data_prep", "split", "preprocessing",
                "model", "evaluation" -- the 5 per-stage config dicts
                written by metadata.build_*_metadata(), read back from
                the single configs/{run_id}/run_config.json
            model: a trained, ready-to-.predict() model instance -- the
                exact class + algorithm metadata["model"] says was used,
                loaded via that class's own .load() (per PDF heading
                5/6/7's required interface)
            fit_objects: list of {method, fit_info} from
                preprocessing_pipeline.run_preprocessing(), in the exact
                order they were originally applied (heading 4/11's
                "exact preprocessing sequence must be recoverable")
            feature_columns: list[str], the exact order inference must
                feed features in (heading 11's explicit requirement)
    """
    config_dir = os.path.join(base_dir, CONFIGS_SUBDIR, run_id)
    run_dir = os.path.join(models_dir, run_id)

    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"No config folder found for run_id='{run_id}' at {config_dir}")

    config_path = os.path.join(config_dir, RUN_CONFIG_FILENAME)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No run_config.json found for run_id='{run_id}' at {config_path}")
    with open(config_path, "r") as f:
        metadata = json.load(f)

    model_kind = metadata["model"].get("model_type")
    # metadata.py's model builders write model_type as "regressor" /
    # "classifier" / "deep_learning_regressor" / "deep_learning_classifier"
    # -- same values _MODEL_KIND_REGISTRIES keys on below.
    if model_kind not in _MODEL_KIND_REGISTRIES:
        raise ValueError(
            f"Unknown model_kind '{model_kind}' in {config_path}. "
            f"Expected one of: {sorted(_MODEL_KIND_REGISTRIES.keys())}"
        )

    registry, extension = _MODEL_KIND_REGISTRIES[model_kind]
    algorithm = metadata["model"]["algorithm"]
    if algorithm not in registry:
        raise ValueError(
            f"Run '{run_id}' was trained with algorithm='{algorithm}', which isn't in the "
            f"current {model_kind} registry ({sorted(registry.keys())}). Code may have "
            f"changed since this run was trained."
        )

    model_path = os.path.join(run_dir, f"model.{extension}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found for run_id='{run_id}' at {model_path}")

    # Reconstruct the exact class the run used, then use ITS OWN .load()
    # (every BaseRegressor/BaseClassifier/BaseNetwork/BaseClassifierNetwork
    # implements this per headings 5/6/7's required interface) rather than
    # this file knowing joblib vs torch.load details itself.
    model_cls = registry[algorithm]
    model = model_cls().load(model_path)
    logger.info(f"Loaded model: run_id='{run_id}', model_kind='{model_kind}', algorithm='{algorithm}'")

    preprocessing_path = os.path.join(run_dir, "preprocessing.joblib")
    fit_objects = joblib.load(preprocessing_path) if os.path.exists(preprocessing_path) else []

    feature_columns = metadata["preprocessing"].get("feature_columns", [])

    return {
        "metadata": metadata,
        "model": model,
        "fit_objects": fit_objects,
        "feature_columns": feature_columns,
    }
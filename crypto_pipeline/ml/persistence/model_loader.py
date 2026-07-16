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
loader: read the three artifacts back off disk, hand them to the
caller unchanged.
"""

import logging
import os

import joblib
import yaml

from crypto_pipeline.ml.persistence.artifact_manager import ARTIFACTS_DIR, CONFIGS_SUBDIR
from crypto_pipeline.ml.regressors.registry import REGRESSORS
from crypto_pipeline.ml.classifiers.registry import CLASSIFIERS
from crypto_pipeline.ml.deep_learning.registry import DL_REGRESSORS, DL_CLASSIFIERS

logger = logging.getLogger(__name__)

# model_kind (as written by metadata.py) -> which registry + file extension
# to use for reconstruction. Mirrors metadata._MODEL_INFO_BUILDERS's own
# four-way split -- one place per model kind, no if/elif chain here either.
_MODEL_KIND_REGISTRIES = {
    "regressor": (REGRESSORS, "joblib"),
    "classifier": (CLASSIFIERS, "joblib"),
    "deep_learning_regressor": (DL_REGRESSORS, "pt"),
    "deep_learning_classifier": (DL_CLASSIFIERS, "pt"),
}


def load_run(run_id: str, base_dir: str = ARTIFACTS_DIR) -> dict:
    """
    Load everything needed to run inference for one trained run.

    Args:
        run_id: the run folder/config name, e.g. from
            artifact_manager.make_run_id() or save_run()'s return value.
        base_dir: root artifacts folder, defaults to "artifacts" (must
            match whatever base_dir save_run() was originally called with).

    Returns:
        dict:
            metadata: the full config dict written by metadata.build_metadata()
                (dataset/features/data_split/preprocessing/model/evaluation,
                plus run_summary)
            model: a trained, ready-to-.predict() model instance -- the
                exact class + algorithm the run_summary/model_info
                section says was used, loaded via that class's own
                .load() (per PDF heading 5/6/7's required interface)
            fit_objects: list of {method, fit_info} from
                preprocessing_pipeline.run_preprocessing(), in the exact
                order they were originally applied (heading 4/11's
                "exact preprocessing sequence must be recoverable")
            feature_columns: list[str], the exact order inference must
                feed features in (heading 11's explicit requirement)
    """
    configs_dir = os.path.join(base_dir, CONFIGS_SUBDIR)
    config_path = os.path.join(configs_dir, f"{run_id}.yaml")
    run_dir = os.path.join(base_dir, run_id)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config found for run_id='{run_id}' at {config_path}")

    with open(config_path, "r") as f:
        metadata = yaml.safe_load(f)

    model_kind = metadata.get("model_kind")
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

    feature_columns = metadata.get("features", {}).get("feature_columns", [])

    return {
        "metadata": metadata,
        "model": model,
        "fit_objects": fit_objects,
        "feature_columns": feature_columns,
    }
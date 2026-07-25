"""
schemas/ml.py
--------------
Request/response models for /api/ml-models.

Maps to one row per trained run in Postgres (ml.run_configs, see
crypto_pipeline.utils.db_utils.save_ml_run_config/get_ml_run_config),
written by crypto_pipeline.ml.persistence.artifact_manager.save_run()
(see crypto_pipeline.ml.persistence.metadata for how each section is
built). See ml_repo.py for the read side.

model_type here is always "regression" or "classification" -- timeseries
runs are filtered out before they ever reach these schemas (see
ml_repo.py's _SUPPORTED_KINDS). model_kind is the finer-grained value
straight off the run's own config ("regressor" / "classifier" /
"deep_learning_regressor" / "deep_learning_classifier"); is_deep_learning
is just that collapsed to a bool for the frontend's toggle, so the UI
doesn't need to know all four literal strings.

ml_metrics' actual keys differ by model_type (mae/mse/rmse for
regression vs accuracy/precision/recall/f1 for classification, per
crypto_pipeline.ml.evaluation.regression_metrics /
classification_metrics) -- left as an open dict rather than two separate
schemas with optional fields, since the pipeline itself doesn't fix that
key set anywhere either.
"""

from typing import Optional
from pydantic import BaseModel


class ModelRunSummary(BaseModel):
    """One row in the ML Models table."""
    run_id: str
    algorithm: str
    model_kind: str
    is_deep_learning: bool
    model_type: Optional[str] = None  # "regression" | "classification"
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    timeframe: Optional[str] = None
    horizon: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    # UTC ISO 8601 timestamp of when this run was actually trained --
    # None for runs trained before this field was added (older
    # run_config.json files on disk won't have it; not backfilled).
    trained_at: Optional[str] = None
    sharpe: Optional[float] = None
    win_rate: Optional[float] = None
    ml_metrics: dict = {}


class ModelRunDetail(ModelRunSummary):
    # Written as-is from run_config.json's matching top-level section --
    # see metadata.py's build_*_metadata() docstrings for exactly what
    # each contains (feature/target config + row counts; train/test/val
    # date ranges + row counts; feature_columns + preprocessing steps;
    # full hyperparameters/architecture; full evaluation block).
    data_prep: dict = {}
    split: dict = {}
    preprocessing: dict = {}
    model: dict = {}
    evaluation: dict = {}
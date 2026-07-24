"""
routers/ml.py
--------------
/api/ml-models -- ML Models (list) + Model Details pages.

No `conn=Depends(get_conn)` here, unlike every other router in this
codebase -- trained runs live on disk (artifacts/configs/{run_id}/
run_config.json), not in Postgres. See ml_repo.py.

Filters mirror the person's actual dropdown: model_type is Regression /
Classification only (timeseries is never surfaced here at all -- see
ml_repo.py's _SUPPORTED_KINDS), and include_deep_learning is a single
on/off toggle within whichever model_type is selected, not a separate
three-way split -- mlp/lstm/gru are just more algorithm options inside
regression or classification, same as xgboost/random_forest/etc. are.
"""

from fastapi import APIRouter, HTTPException, Query

from api.core.responses import item, list_response
from api.schemas.ml import ModelRunSummary, ModelRunDetail
from api.repos import ml_repo

router = APIRouter(prefix="/api/ml-models", tags=["ml-models"])

_VALID_MODEL_TYPES = {"regression", "classification"}


@router.get("")
def list_models(
    limit: int = 50,
    offset: int = 0,
    model_type: str | None = Query(default=None, description="regression | classification"),
    include_deep_learning: bool = True,
    algorithm: str | None = None,
    symbol: str | None = None,
):
    if model_type is not None and model_type not in _VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"model_type must be one of {sorted(_VALID_MODEL_TYPES)}, got '{model_type}'",
        )

    rows = ml_repo.list_runs(model_type=model_type, include_deep_learning=include_deep_learning)

    if algorithm is not None:
        rows = [r for r in rows if r["algorithm"] == algorithm]
    if symbol is not None:
        rows = [r for r in rows if r["symbol"] == symbol]

    total = len(rows)
    page = rows[offset: offset + limit]
    summaries = [ModelRunSummary(**r).model_dump() for r in page]
    return list_response(summaries, total, limit, offset)


@router.get("/{run_id}")
def get_model(run_id: str):
    detail = ml_repo.get_run_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Model run '{run_id}' not found")
    return item(ModelRunDetail(**detail).model_dump())
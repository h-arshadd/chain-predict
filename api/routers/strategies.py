"""
routers/strategies.py
-----------------------
/api/strategies -- Strategies (list) + Strategy Details pages (spec
sections 1's strategy table and section 2).

Per instructions (see PROJECT_SUMMARY.md section 8.2, decided before
this module was built):
  - No fake "Active/Paused/Stopped" status. Two real states only:
    execution_enabled True/False. The frontend renders this as a toggle
    switch, same visual pattern as Wallets' enable/disable switch, not a
    3-option dropdown.
  - Only ONE strategy per (exchange, coin) pair may be execution_enabled
    at a time -- enforced here (strategies_repo.set_execution_enabled),
    since neither the DB nor metadata_utils.set_strategy_enabled()
    enforce it. Turning one strategy ON atomically turns off whatever
    else was on for that pair, in the same request.
  - simulator_enabled is read-only here for now (shown, not toggleable)
    -- its own PATCH lands with the Simulator module.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.core.db import get_conn
from api.core.responses import item, list_response
from api.schemas.strategies import StrategyEnabledUpdate, StrategySummary, StrategyDetail
from api.repos import strategies_repo

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


@router.get("")
def list_strategies(
    limit: int = 50,
    offset: int = 0,
    exchange: str | None = None,
    coin: str | None = None,
    execution_enabled: bool | None = None,
    conn=Depends(get_conn),
):
    rows = strategies_repo.list_strategies(conn)

    if exchange is not None:
        rows = [r for r in rows if r["exchange"] == exchange]
    if coin is not None:
        rows = [r for r in rows if r["coin"] == coin]
    if execution_enabled is not None:
        rows = [r for r in rows if r["execution_enabled"] == execution_enabled]

    total = len(rows)
    page = rows[offset: offset + limit]
    summaries = [StrategySummary(**r).model_dump() for r in page]
    return list_response(summaries, total, limit, offset)


@router.get("/{strategy_id}")
def get_strategy(strategy_id: int, conn=Depends(get_conn)):
    detail = strategies_repo.get_strategy_detail(conn, strategy_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
    return item(StrategyDetail(**detail).model_dump())


@router.patch("/{strategy_id}/enabled")
def set_execution_enabled(strategy_id: int, body: StrategyEnabledUpdate, conn=Depends(get_conn)):
    row = strategies_repo.set_execution_enabled(conn, strategy_id, body.execution_enabled)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

    detail = strategies_repo.get_strategy_detail(conn, strategy_id)
    return item(StrategyDetail(**detail).model_dump())
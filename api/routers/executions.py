"""
routers/executions.py
-----------------------
/api/executions -- backs BOTH frontend pages that read this same data
(see schemas/executions.py's module docstring):
  - GET   /api/executions                    -> Strategy Deployment page (list)
  - GET   /api/executions/{ex}/{sym}         -> Execution Details page (drill-down)
  - PATCH /api/executions/{ex}/{sym}/wallet  -> assign/reassign/clear which
                                                 wallet places real orders
                                                 for this pair

Deliberately does NOT expose any start/stop/pause action here. Whether a
pair trades is controlled by two existing things, both already real:
  - execution.config having an account_name assigned (wallet routing --
    the PATCH .../wallet endpoint below)
  - that wallet's `enabled` flag (see /api/wallets PATCH .../enabled)
Pausing = disable the wallet. There's no separate "pause this execution"
concept in the DB, so this router doesn't invent one. Which strategy is
execution_enabled belongs to a future /api/strategies -- not duplicated
here. Wallet *assignment* lives here rather than under /api/wallets
because it's keyed by (exchange, symbol) -- the same pair-scoped resource
every other endpoint in this router already addresses -- not by
account_name.

path params are :path-typed because symbol/exchange values in this
codebase are plain lowercase strings (e.g. "bybit", "btc"), never
containing slashes, so the default str converter is fine -- kept
explicit here only for readability, not because it's structurally
required.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.core.db import get_conn
from api.core.responses import item, list_response
from api.schemas.executions import ExecutionSummary, ExecutionDetail, ExecutionWalletAssign
from api.repos import executions_repo

router = APIRouter(prefix="/api/executions", tags=["executions"])


@router.get("")
def list_executions(limit: int = 50, offset: int = 0, conn=Depends(get_conn)):
    rows = executions_repo.list_executions(conn)
    total = len(rows)
    page = rows[offset: offset + limit]
    summaries = [ExecutionSummary(**r).model_dump() for r in page]
    return list_response(summaries, total, limit, offset)


@router.get("/{exchange}/{symbol}")
def get_execution(exchange: str, symbol: str, conn=Depends(get_conn)):
    row = executions_repo.get_execution_detail(conn, exchange, symbol)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No execution.config entry for {exchange}/{symbol} -- pair not deployed",
        )
    return item(ExecutionDetail(**row).model_dump())


@router.patch("/{exchange}/{symbol}/wallet")
def assign_wallet(exchange: str, symbol: str, body: ExecutionWalletAssign, conn=Depends(get_conn)):
    try:
        config = executions_repo.assign_wallet(conn, exchange, symbol, body.account_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"No execution.config entry for {exchange}/{symbol} -- pair not deployed",
        )

    row = executions_repo.get_execution_detail(conn, exchange, symbol)
    return item(ExecutionDetail(**row).model_dump())
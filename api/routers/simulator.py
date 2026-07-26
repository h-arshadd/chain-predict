"""
routers/simulator.py
----------------------
/api/simulator -- backs the Dashboard's Simulation Details page.

  GET /api/simulator                              -> list (real simulator activity)
  GET /api/simulator/{exchange}/{symbol}/{strategy_name} -> Simulation Details drill-down

Keyed on (exchange, symbol, strategy_name) rather than just
(exchange, symbol) -- unlike execution, simulator has no exclusivity
rule, so more than one strategy can be actively running per pair at
once. strategy_name is quoted/unquoted by the frontend when building/
reading this URL since it can contain characters that need encoding
(e.g. underscores are fine, but keeping this explicit for clarity).
"""

from fastapi import APIRouter, Depends, HTTPException

from api.core.db import get_conn
from api.core.responses import item, list_response
from api.schemas.simulator import SimulationSummary, SimulationDetail
from api.repos import simulator_repo

router = APIRouter(prefix="/api/simulator", tags=["simulator"])


@router.get("")
def list_simulations(limit: int = 500, offset: int = 0, conn=Depends(get_conn)):
    rows = simulator_repo.list_simulations(conn)
    total = len(rows)
    page = rows[offset: offset + limit]
    summaries = [SimulationSummary(**r).model_dump() for r in page]
    return list_response(summaries, total, limit, offset)


@router.get("/{exchange}/{symbol}/{strategy_name}")
def get_simulation(exchange: str, symbol: str, strategy_name: str, conn=Depends(get_conn)):
    row = simulator_repo.get_simulation_detail(conn, exchange, symbol, strategy_name)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No simulator.config entry for {exchange}/{symbol}, or no matching strategy '{strategy_name}' for that pair",
        )
    return item(SimulationDetail(**row).model_dump())
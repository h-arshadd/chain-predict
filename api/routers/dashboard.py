"""
routers/dashboard.py
----------------------
/api/dashboard -- Dashboard (landing page), spec section 1.

Two endpoints, matching dashboard_repo.py's two functions:

  GET /api/dashboard            -- the stat-card summary strip (both
      simulator and execution/live-account data).
  GET /api/dashboard/strategies -- the strategy table. Same shape and
      same real execution-first-fallback-to-simulator numbers as
      /api/strategies (see dashboard_repo.list_strategies), just with
      its own search/coin filtering applied here for the dashboard's
      own query params, consistent with /api/strategies' pattern.
"""

from fastapi import APIRouter, Depends

from api.core.db import get_conn
from api.core.responses import item, list_response
from api.schemas.dashboard import DashboardSummary
from api.schemas.strategies import StrategySummary
from api.repos import dashboard_repo

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("")
def get_dashboard_summary(conn=Depends(get_conn)):
    summary = dashboard_repo.get_summary(conn)
    return item(DashboardSummary(**summary).model_dump())


@router.get("/strategies")
def list_dashboard_strategies(
    limit: int = 500,
    offset: int = 0,
    coin: str | None = None,
    search: str | None = None,
    conn=Depends(get_conn),
):
    rows = dashboard_repo.list_strategies(conn)

    if coin is not None:
        rows = [r for r in rows if r["coin"] == coin]
    if search is not None:
        q = search.lower()
        rows = [r for r in rows if q in r["strategy_name"].lower()]

    total = len(rows)
    page = rows[offset: offset + limit]
    summaries = [StrategySummary(**r).model_dump() for r in page]
    return list_response(summaries, total, limit, offset)
"""
routers/backtests.py
-----------------------
/api/backtests -- Backtest Requests + Backtest Details pages.

  POST /api/backtests               -> submit a new backtest request.
      Inserts the metadata.backtest row (status 'pending') and returns
      it immediately, then schedules the actual run
      (backtests_repo.run_backtest_job) as a FastAPI BackgroundTask --
      the response comes back right away with a pending row to show;
      the real work (pulling data, generating signals, running the
      vectorized engine, computing stats) happens after the response is
      sent. The frontend polls GET /api/backtests/{id} to watch it move
      pending -> running -> completed/failed.
  GET  /api/backtests                -> Backtest Requests list (all
      four status buckets -- the frontend splits by `status`).
  GET  /api/backtests/{backtest_id}  -> Backtest Details page.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from api.core.db import get_conn
from api.core.responses import item, list_response
from api.schemas.backtests import BacktestRequestIn, BacktestSummary, BacktestDetail
from api.repos import backtests_repo

router = APIRouter(prefix="/api/backtests", tags=["backtests"])


@router.post("")
def create_backtest(payload: BacktestRequestIn, background_tasks: BackgroundTasks, conn=Depends(get_conn)):
    overrides = payload.model_dump(exclude={"strategy_id"})
    try:
        row = backtests_repo.create_backtest_request(conn, payload.strategy_id, overrides)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    background_tasks.add_task(backtests_repo.run_backtest_job, row["backtest_id"])
    return item(BacktestSummary(**row).model_dump())


@router.get("")
def list_backtests(limit: int = 100, offset: int = 0, status: str | None = None, conn=Depends(get_conn)):
    rows = backtests_repo.list_backtests(conn)
    if status is not None:
        rows = [r for r in rows if r["status"] == status]

    total = len(rows)
    page = rows[offset: offset + limit]
    summaries = [BacktestSummary(**r).model_dump() for r in page]
    return list_response(summaries, total, limit, offset)


@router.get("/{backtest_id}")
def get_backtest(backtest_id: int, conn=Depends(get_conn)):
    detail = backtests_repo.get_backtest_detail(conn, backtest_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return item(BacktestDetail(**detail).model_dump())
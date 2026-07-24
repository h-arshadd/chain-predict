"""
routers/wallets.py
-------------------
/api/wallets -- Wallet Management page (spec section 3).

Per instructions:
  - no /api/users, /api/auth anywhere -- this app has no auth layer
  - no "API Status" field -- dropped entirely, not even computed
  - add/edit/remove wallets, enable/disable toggle that actually means
    something (execution module will check `enabled` before opening a
    new trade -- not built yet, this just stores the flag)
  - per-wallet strategies/positions/open orders/executions for the
    expandable row

"Total PnL" = accounts.stats.net_realized_pnl (from accounts.history,
Bybit's own fill record -- see accounts_utils.py). "Balance" and
"Unrealized PnL" are live-only, fetched from Bybit per request since
they aren't stored anywhere (see repos/wallet_live.py).

Strategies/positions/open orders/executions per wallet are now a real
join against /api/executions's own data (executions_repo.list_executions),
filtered down to whatever pairs have this account_name assigned in
execution.config -- see _wallet_expandable_row() below. No new tables:
same execution.config/positions/*_trades + metadata.strategy this router
already re-derives for the Strategy Deployment page, just filtered to
one wallet.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.core.db import get_conn
from api.core.responses import item, list_response
from api.schemas.wallets import (
    WalletCreate, WalletUpdate, WalletEnabledUpdate,
    WalletSummary, WalletDetail,
)
from api.repos import wallets_repo, wallets_live as wallet_live, executions_repo
from crypto_pipeline.accounts.accounts_utils import get_account_stats

router = APIRouter(prefix="/api/wallets", tags=["wallets"])


def _mask(api_key: str) -> str:
    if len(api_key) <= 8:
        return "••••"
    return f"{api_key[:4]}...{api_key[-4:]}"


def _total_pnl(conn, account_name: str) -> float | None:
    stats = get_account_stats(conn, account_name)
    if not stats:
        return None
    return stats.get("net_realized_pnl")


def _to_summary(conn, row: dict) -> WalletSummary:
    # row must include api_secret -- always call this with the full
    # record from wallets_repo.get_wallet(), never the masked list() row.
    balance_data = wallet_live.fetch_balance(row["api_key"], row["api_secret"], row["demo"])

    return WalletSummary(
        account_name=row["account_name"],
        exchange=row["exchange"],
        demo=row["demo"],
        enabled=row["enabled"],
        api_key_masked=_mask(row["api_key"]),
        balance=balance_data.get("balance"),
        unrealized_pnl=balance_data.get("unrealized_pnl"),
        total_pnl=_total_pnl(conn, row["account_name"]),
        balance_error=balance_data.get("error"),
        updated_at=row.get("updated_at"),
    )


@router.get("")
def list_wallets(limit: int = 50, offset: int = 0, conn=Depends(get_conn)):
    rows = wallets_repo.list_wallets(conn)
    total = len(rows)
    page = rows[offset: offset + limit]

    # list_wallets() doesn't return api_secret (by design -- it's not
    # needed for the table view's masked key column), so fetch each
    # wallet's full record for the live balance call.
    summaries = []
    for row in page:
        full = wallets_repo.get_wallet(conn, row["account_name"])
        summaries.append(_to_summary(conn, full))

    return list_response([s.model_dump() for s in summaries], total, limit, offset)


def _time_ago(dt) -> str:
    if dt is None:
        return "—"
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    seconds = (now - dt).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _wallet_expandable_row(conn, account_name: str) -> dict:
    """
    Real strategies/positions/executions for this wallet, filtered from
    executions_repo.list_executions() down to whichever pairs have this
    account_name assigned in execution.config.

    "Open Orders" stays empty -- this codebase places TP/SL natively on
    the exchange-side position (see bybit_client.set_trading_stop),
    there is no separate resting "pending order" concept to list. What
    WAS missing here wasn't orders, it was visibility into the open
    positions that already exist: those now carry a real "Open" status
    plus a live mark price, fetched via executions_repo._live_bybit_position
    -- one Bybit call per open position ON THIS WALLET ONLY. Expanding a
    wallet row is a single-wallet, user-initiated action (not part of
    the list endpoint), so this doesn't multiply into N calls across
    every wallet the way it would if done in list_executions/list_wallets.
    """
    all_executions = [e for e in executions_repo.list_executions(conn) if e.get("account_name") == account_name]

    strategies = [
        {
            "name": e["strategy_name"],
            "symbol": e["symbol"],
            "status": "Active" if e["status"] == "running" else "Inactive",
        }
        for e in all_executions if e["strategy_name"] != "—"
    ]

    positions = []
    for e in all_executions:
        if e.get("position") is None:
            continue
        entry_price = e["position"]["entry_price"] or 0.0
        # Real live position + mark price for this specific pair -- same
        # live fetch ExecutionDetail's live_position already makes,
        # reused here rather than duplicated. None if Bybit errors, the
        # wallet has no credentials, or it's actually flat (e.g. closed
        # since execution.positions was last written) -- entry price is
        # still shown as the fallback mark so the row never blanks out.
        live = executions_repo._live_bybit_position(account_name, conn, e["symbol"])
        mark_price = live["mark_price"] if live and live.get("mark_price") is not None else entry_price
        positions.append({
            "symbol": e["symbol"],
            "side": "Long" if e["position"]["direction"] == "long" else "Short",
            "size": e["position"]["quantity"] or 0.0,
            "entry": entry_price,
            "mark": mark_price,
            "pnl": e["cumulative_pnl"] or 0.0,
            # "Open" = confirmed still open on Bybit right now (live is
            # not None); "Open (unconfirmed)" = our DB thinks it's open
            # but the live Bybit check didn't confirm it (error, no
            # wallet creds, or it already closed since the last DB
            # write) -- surfaces the gap instead of silently claiming a
            # position is live when we couldn't actually verify it.
            "status": "Open" if live is not None else "Open (unconfirmed)",
        })

    executions = [
        {
            "strategy": e["strategy_name"],
            "symbol": e["symbol"],
            "status": "Running" if e["status"] == "running" else e["status"].replace("_", " ").title(),
            "uptime": _time_ago(e.get("last_processed")),
        }
        for e in all_executions if e["strategy_name"] != "—"
    ]

    return {"strategies": strategies, "positions": positions, "open_orders": [], "executions": executions}


@router.get("/{account_name}")
def get_wallet(account_name: str, conn=Depends(get_conn)):
    row = wallets_repo.get_wallet(conn, account_name)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Wallet '{account_name}' not found")

    summary = _to_summary(conn, row)
    expandable = _wallet_expandable_row(conn, account_name)
    detail = WalletDetail(**summary.model_dump(), **expandable)
    return item(detail.model_dump())


@router.post("", status_code=201)
def create_wallet(body: WalletCreate, conn=Depends(get_conn)):
    existing = wallets_repo.get_wallet(conn, body.account_name)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Wallet '{body.account_name}' already exists")

    row = wallets_repo.create_wallet(
        conn, body.account_name, body.exchange, body.api_key, body.api_secret, body.demo
    )
    return item(_to_summary(conn, row).model_dump())


@router.put("/{account_name}")
def update_wallet(account_name: str, body: WalletUpdate, conn=Depends(get_conn)):
    row = wallets_repo.update_wallet(
        conn, account_name, body.exchange, body.api_key, body.api_secret, body.demo
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Wallet '{account_name}' not found")
    return item(_to_summary(conn, row).model_dump())


@router.delete("/{account_name}", status_code=204)
def delete_wallet(account_name: str, conn=Depends(get_conn)):
    deleted = wallets_repo.delete_wallet(conn, account_name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Wallet '{account_name}' not found")
    return None


@router.patch("/{account_name}/enabled")
def set_enabled(account_name: str, body: WalletEnabledUpdate, conn=Depends(get_conn)):
    row = wallets_repo.set_enabled(conn, account_name, body.enabled)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Wallet '{account_name}' not found")
    return item(_to_summary(conn, row).model_dump())
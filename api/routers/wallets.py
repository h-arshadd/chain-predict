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

Strategies/positions/open orders/executions per wallet are stubbed as
empty lists for now -- that join lands with the Strategy Deployment
module, not here. Wallets works standalone today; nothing here needs to
change when that module is built, only the stub gets replaced.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.core.db import get_conn
from api.core.responses import item, list_response
from api.schemas.wallets import (
    WalletCreate, WalletUpdate, WalletEnabledUpdate,
    WalletSummary, WalletDetail,
)
from api.repos import wallets_repo, wallet_live
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


@router.get("/{account_name}")
def get_wallet(account_name: str, conn=Depends(get_conn)):
    row = wallets_repo.get_wallet(conn, account_name)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Wallet '{account_name}' not found")

    summary = _to_summary(conn, row)
    detail = WalletDetail(
        **summary.model_dump(),
        strategies=[],
        positions=[],
        open_orders=[],
        executions=[],
    )
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
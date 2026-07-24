"""
schemas/wallets.py
-------------------
Request/response models for /api/wallets.

Maps to accounts.api_keys (account_name, exchange, api_key, api_secret,
demo, updated_at, enabled) plus live balance/PnL pulled from Bybit at
request time (not stored -- there's no "balance" column anywhere in the
DB, it only exists on the exchange).

No "apiStatus" field anywhere here on purpose -- per instructions, that
column is dropped. "enabled" is the only connection-state flag, and it's
ours (blocks new executions), not Bybit's.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class WalletCreate(BaseModel):
    account_name: str = Field(..., min_length=1, max_length=100, description="Your own label for this account, must be unique")
    exchange: str = Field(default="bybit")
    api_key: str = Field(..., min_length=1)
    api_secret: str = Field(..., min_length=1)
    demo: bool = Field(default=True)


class WalletUpdate(BaseModel):
    # Everything optional: PUT allows partial edits (e.g. relabel without
    # touching keys). Blank/omitted api_key or api_secret means "keep
    # current", matching the frontend's existing "leave blank to keep
    # current key" copy in the Edit Wallet modal.
    exchange: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    demo: Optional[bool] = None


class WalletEnabledUpdate(BaseModel):
    enabled: bool


class WalletSummary(BaseModel):
    """One row in the Wallets table."""
    account_name: str
    exchange: str
    demo: bool
    enabled: bool
    api_key_masked: str
    balance: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    total_pnl: Optional[float] = None
    balance_error: Optional[str] = None  # set instead of failing the whole list if the live call fails
    updated_at: Optional[datetime] = None


class WalletStrategyAssignment(BaseModel):
    name: str
    symbol: str
    status: str


class WalletPosition(BaseModel):
    """
    One open position for this wallet, derived from execution.positions
    (see routers/wallets.py's _wallet_expandable_row). `mark` is a real
    live mark price fetched from Bybit for this specific pair when the
    wallet row is expanded (falls back to `entry` only if that live call
    errors or returns nothing). `status` is "Open" when Bybit confirms
    the position is still live right now, or "Open (unconfirmed)" if our
    DB thinks it's open but the live check couldn't verify it (no
    wallet credentials, a Bybit error, or it already closed since
    execution.positions was last written). `pnl` is the real,
    already-computed cumulative PnL for that pair
    (execution.positions.cumulative_pnl).
    """
    symbol: str
    side: str
    size: float
    entry: float
    mark: float
    pnl: float
    status: str


class WalletOpenOrder(BaseModel):
    """
    Always [] today -- this codebase has no "pending order" concept per
    wallet (TP/SL is attached natively to the exchange-side position via
    bybit_client.set_trading_stop, not placed as separate resting
    orders). Kept as a real field/shape for forward-compatibility rather
    than removed outright, in case a genuine pending-order feature is
    added later.
    """
    symbol: str
    side: str
    type: str
    price: float
    qty: float


class WalletExecution(BaseModel):
    strategy: str
    symbol: str
    status: str
    uptime: str


class WalletDetail(WalletSummary):
    strategies: list[WalletStrategyAssignment] = []
    positions: list[WalletPosition] = []
    open_orders: list[WalletOpenOrder] = []
    executions: list[WalletExecution] = []
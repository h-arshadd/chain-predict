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
    symbol: str
    side: str
    size: float
    entry: float
    mark: float
    pnl: float


class WalletOpenOrder(BaseModel):
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
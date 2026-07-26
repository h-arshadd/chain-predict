"""
schemas/backtests.py
-----------------------
Response/request models for /api/backtests.

BacktestRequestIn -- what the "new backtest" form (Backtest Requests
page) submits: a strategy_id plus any config.yaml fields it wants to
override (everything else falls back to backtest/config.yaml's
defaults, see backtests_repo._default_backtest_config).

BacktestSummary -- one row in the Backtest Requests list. status is one
of pending/running/completed/failed (metadata.backtest.status, added by
this feature -- see metadata_utils.create_backtest_table).

BacktestDetail -- full Backtest Details page: request config, strategy
config (real entry/exit rules), trade list, equity curve, and the full
stats/plots bundle -- populated only once status is 'completed'.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class EquityCurvePoint(BaseModel):
    timestamp: datetime
    balance: float


class BacktestTrade(BaseModel):
    """One row in the trade ledger -- exact columns run_backtest() produces."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    exit_reason: str
    quantity: float
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    balance_after_trade: float
    cumulative_pnl: float


class BacktestWinLoss(BaseModel):
    wins: int
    losses: int


class BacktestRequestIn(BaseModel):
    strategy_id: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_balance: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    position_size: Optional[dict] = None
    take_profit: Optional[dict] = None
    stop_loss: Optional[dict] = None
    allow_long: Optional[bool] = None
    allow_short: Optional[bool] = None
    max_open_positions: Optional[int] = None


class BacktestSummary(BaseModel):
    backtest_id: int
    strategy_id: Optional[int] = None
    strategy_name: str
    status: str
    error: Optional[str] = None
    backtest_config: dict
    result_summary: Optional[dict] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime


class BacktestDetail(BaseModel):
    backtest_id: int
    strategy_id: Optional[int] = None
    strategy_name: str
    exchange: Optional[str] = None
    coin: Optional[str] = None
    time_horizon: Optional[str] = None
    status: str
    error: Optional[str] = None
    backtest_config: dict
    strategy_config: dict
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime

    final_balance: Optional[float] = None
    total_net_profit: Optional[float] = None
    total_trades: int = 0
    win_loss: Optional[BacktestWinLoss] = None
    trades: list[BacktestTrade] = []
    equity_curve: list[EquityCurvePoint] = []
    stats: Optional[dict] = None
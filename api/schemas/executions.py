"""
schemas/executions.py
----------------------
Request/response models for /api/executions -- backs BOTH frontend
pages that read the same underlying data:
  - Strategy Deployment (PDF section 4): the list, ExecutionSummary rows
  - Execution Details (PDF section 5): one row's drill-down, ExecutionDetail

Maps to execution.config (which pairs are set up to trade, and which
wallet places their orders), execution.positions (current state: balance,
open position, last_processed), and execution.{symbol}_{strategy}_trades
(the permanent trade ledger) -- see crypto_pipeline/utils/db_utils.py's
execution.* functions, all reused as-is rather than re-implemented here.

No "Last Signal" column exists anywhere in the DB as a standalone log --
today's closest available signal is the open position's `leaning` (only
meaningful while a position is open) or, if flat, the pair's status/
last_processed time. ExecutionSummary.last_signal is best-effort from
those, not a real signal history; a proper signal log is a future
addition, not invented here.

ExecutionDetail additionally carries:
  - risk config (commission/slippage/allow_long/allow_short) straight
    from execution.config -- the PDF's "Risk Statistics" panel.
  - strategy_config: real entry/exit rule text + TP/SL, derived from
    metadata.strategy.strategy_config (see executions_repo._describe_side).
  - live_position: Bybit's own current position + native TP/SL for this
    pair's wallet, fetched live (nothing here is stored in Postgres --
    same reasoning as wallets_live.py's balance fetch).
  - stats: the same {"metrics", "trade_summary", "plots", ...} shape
    crypto_pipeline.stats.calculator.compute_stats() already returns for
    backtests/simulator runs, computed here from execution's own equity
    curve -- covers Equity Curve, Drawdown, Rolling Sharpe/Volatility,
    Monthly/Yearly Returns, and Return Distribution in one field, left as
    a loose dict (not modeled field-by-field) since its shape is exactly
    whatever stats/config.yaml's `plots:` list currently produces.
"""

from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel


class ExecutionPosition(BaseModel):
    """Current open position, if any -- mirrors get_execution_state()'s "position" dict."""
    direction: str
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    quantity: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    leaning: Optional[str] = None
    status: str


class ExecutionSummary(BaseModel):
    """One row in the Strategy Deployment table."""
    exchange: str
    symbol: str
    strategy_name: str
    account_name: Optional[str] = None       # wallet assigned to this pair, if any
    wallet_enabled: Optional[bool] = None    # None if no wallet assigned yet
    status: str                              # "running" | "paused" | "unassigned" | "never_run"
    position: Optional[ExecutionPosition] = None
    balance: Optional[float] = None
    cumulative_pnl: Optional[float] = None
    daily_return_pct: Optional[float] = None
    last_signal: Optional[str] = None
    last_processed: Optional[datetime] = None


class ExecutionTrade(BaseModel):
    """One row in the Trade Ledger -- Filled Orders / Trade History."""
    entry_date_time: datetime
    direction: str
    entry_price: float
    quantity: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    exit_date_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    gross_pnl: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    net_pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    balance: Optional[float] = None
    status: str


class ExecutionWinLoss(BaseModel):
    wins: int
    losses: int
    win_rate: float


class EquityCurvePoint(BaseModel):
    timestamp: datetime
    balance: float


class StrategyConfigDetail(BaseModel):
    """Real entry/exit logic + TP/SL, derived from metadata.strategy.strategy_config."""
    indicators: list[str] = []
    entry_logic_long: Optional[str] = None
    entry_logic_short: Optional[str] = None
    take_profit_type: Optional[str] = None
    take_profit_value: Optional[float] = None
    stop_loss_type: Optional[str] = None
    stop_loss_value: Optional[float] = None


class LivePosition(BaseModel):
    """Bybit's own current position for this pair -- see bybit_client.get_open_position()."""
    side: str
    size: float
    avg_price: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    created_time: Optional[datetime] = None


class ExecutionDetail(ExecutionSummary):
    """Full detail for one (exchange, symbol, strategy) execution combo."""
    time_horizon: Optional[str] = None
    initial_balance: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    allow_long: Optional[bool] = None
    allow_short: Optional[bool] = None
    total_net_profit: Optional[float] = None
    total_trades: int = 0
    win_loss: Optional[ExecutionWinLoss] = None
    equity_curve: list[EquityCurvePoint] = []
    trades: list[ExecutionTrade] = []
    strategy_config: Optional[StrategyConfigDetail] = None
    live_position: Optional[LivePosition] = None
    # Loose dict, not modeled field-by-field -- shape mirrors
    # compute_stats()'s output {"metrics", "trade_summary", "plots",
    # "resample_freq_used", "insufficient_data", "returns_count"}, which
    # is driven by stats/config.yaml's plot list, not something we want
    # to hardcode into a schema and have to update in two places.
    stats: Optional[dict[str, Any]] = None
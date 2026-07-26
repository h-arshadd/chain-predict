"""
schemas/simulator.py
----------------------
Response models for /api/simulator -- backs the Dashboard's Simulation
Details page (see api/repos/simulator_repo.py).

Deliberately its own module rather than reusing schemas/executions.py:
simulator has no wallet/live-exchange concept (paper trading, no Bybit
call) and no execution_enabled exclusivity (several strategies can run
per pair at once) -- the shapes are similar but not identical, and
conflating them would misrepresent what's actually real for each.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class SimulatorPosition(BaseModel):
    """Current simulated open position, if any -- mirrors get_simulator_state()'s "position" dict."""
    direction: str
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    quantity: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    leaning: Optional[str] = None
    status: str


class SimulationSummary(BaseModel):
    """One row of simulator activity for a (exchange, symbol, strategy_name) combo."""
    exchange: str
    symbol: str
    strategy_id: int
    strategy_name: str
    time_horizon: str
    simulator_enabled: bool
    status: str  # "running" | "flat" | "never_run"
    position: Optional[SimulatorPosition] = None
    balance: Optional[float] = None
    cumulative_pnl: Optional[float] = None
    daily_return_pct: Optional[float] = None
    last_processed: Optional[datetime] = None


class SimulatorTrade(BaseModel):
    """One row in the simulator's Trade Ledger."""
    entry_date_time: datetime
    direction: str
    entry_price: float
    quantity: float
    exit_date_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    gross_pnl: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    net_pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    balance: Optional[float] = None


class SimulatorStrategyConfig(BaseModel):
    indicators: list[str] = []
    entry_logic_long: Optional[str] = None
    entry_logic_short: Optional[str] = None
    take_profit_type: Optional[str] = None
    take_profit_value: Optional[float] = None
    stop_loss_type: Optional[str] = None
    stop_loss_value: Optional[float] = None


class WinLoss(BaseModel):
    wins: int = 0
    losses: int = 0
    win_rate: Optional[float] = None


class EquityPoint(BaseModel):
    timestamp: str
    balance: float


class SimulationDetail(SimulationSummary):
    initial_balance: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[float] = None
    allow_long: Optional[bool] = None
    allow_short: Optional[bool] = None
    total_net_profit: Optional[float] = None
    total_trades: int = 0
    win_loss: Optional[WinLoss] = None
    strategy_config: SimulatorStrategyConfig = SimulatorStrategyConfig()
    equity_curve: list[EquityPoint] = []
    trades: list[SimulatorTrade] = []
    # Full compute_stats() bundle (metrics + plots) off the real
    # simulator equity curve. None if too little history yet.
    stats: Optional[dict] = None
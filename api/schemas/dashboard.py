"""
schemas/dashboard.py
----------------------
Response models for /api/dashboard.

DashboardSummary -- the top stat-card strip. Exactly the 10 widgets the
spec asks for: Total Strategies, Active Strategies, Running Executions,
Running Simulations, Connected Accounts, Trained ML Models, Total
Backtests, Today's PnL, Overall Portfolio Value, Total Return.

DashboardStrategyRow -- the Dashboard's own strategy table shape.
Deliberately separate from schemas/strategies.py's StrategySummary: the
Dashboard table is SIMULATOR data (this pipeline runs the simulator
continuously across every registered pair) while StrategySummary is
execution-only, used by the Strategies page -- per instruction, the two
pages are meant to show different things, not the same numbers twice.
No execution_enabled/pair_status here -- those are execution-exclusivity
concepts (see strategies_repo._pair_status) that don't apply to
simulator, which has no such exclusivity rule.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class DashboardSummary(BaseModel):
    total_strategies: int
    active_strategies: int
    running_executions: int
    running_simulations: int
    connected_accounts: int
    trained_ml_models: int
    # Total requests submitted, any status. See backtests_repo.py.
    total_backtests: Optional[int] = None

    # Real closed-trade PnL for live trades that exited today (UTC).
    # None only if execution has never traded at all yet.
    today_pnl: Optional[float] = None

    # Execution + simulator balances combined -- None if neither side
    # has any real data yet.
    overall_portfolio_value: Optional[float] = None
    total_return_pct: Optional[float] = None


class PnlSeriesPoint(BaseModel):
    """One point in a strategy row's dashboard sparkline -- v is % return vs initial_balance, not raw balance."""
    t: str
    v: float


class DashboardStrategyRow(BaseModel):
    """One row in the Dashboard's strategy table -- real simulator performance."""
    strategy_id: int
    strategy_name: str
    exchange: str
    coin: str
    time_horizon: str
    simulator_enabled: bool
    # Real return / win rate from simulator.positions + the simulator's
    # own Trade Ledger (see dashboard_repo.list_strategies). None
    # (rendered as "—") if this strategy hasn't produced a simulator
    # trade yet -- never borrowed from execution.
    latest_return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    win_rate_pct: Optional[float] = None
    pnl_series: Optional[list[PnlSeriesPoint]] = None
    created_at: Optional[datetime] = None
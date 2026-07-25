"""
schemas/dashboard.py
----------------------
Response models for /api/dashboard.

Per instruction, the Dashboard is simulation-only end to end -- no
execution/live-account fields anywhere here (that's Wallets'/
Deployment's territory). Two response shapes, matching
dashboard_repo.py's two functions:

DashboardSummary -- the top stat-card strip. total_backtests is
Optional and None until a real Backtests module/DB exists.

DashboardStrategyRow -- one row in the Dashboard's strategy table,
which is 100% simulator-sourced (see dashboard_repo.list_simulator_strategies
docstring) -- distinct from schemas/strategies.py's StrategySummary,
which uses the execution-first-fallback-to-simulator performance model
for the Strategies page.
"""

from typing import Optional
from pydantic import BaseModel


class DashboardSummary(BaseModel):
    total_strategies: int
    active_strategies: int
    running_simulations: int
    connected_accounts: int
    trained_ml_models: int
    # None until a real Backtests module/DB exists -- never fabricated.
    total_backtests: Optional[int] = None

    # Simulator (paper) portfolio rollup, from simulator.config/positions
    # across every registered pair/strategy.
    simulator_balance: Optional[float] = None
    simulator_net_profit: Optional[float] = None
    simulator_return_pct: Optional[float] = None


class DashboardStrategyRow(BaseModel):
    strategy_id: int
    strategy_name: str
    exchange: str
    coin: str
    time_horizon: str
    simulator_enabled: bool
    has_run: bool
    total_trades: int
    latest_return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    win_rate_pct: Optional[float] = None
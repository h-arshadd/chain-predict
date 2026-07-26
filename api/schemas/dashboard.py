"""
schemas/dashboard.py
----------------------
Response models for /api/dashboard.

DashboardSummary -- the top stat-card strip. Exactly the 10 widgets the
spec asks for: Total Strategies, Active Strategies, Running Executions,
Running Simulations, Connected Accounts, Trained ML Models, Total
Backtests, Today's PnL, Overall Portfolio Value, Total Return.

The strategy table reuses schemas/strategies.py's StrategySummary
directly (see dashboard_repo.list_strategies(), which calls
strategies_repo.list_strategies() as-is) -- one shape, one source of
truth, so the Dashboard and Strategies pages can never show different
numbers for the same strategy.
"""

from typing import Optional
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
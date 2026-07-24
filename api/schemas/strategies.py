"""
schemas/strategies.py
----------------------
Request/response models for /api/strategies.

Maps to metadata.strategy (see crypto_pipeline/utils/metadata_utils.py --
create_strategy_table()'s docstring is the source of truth for what each
column means). One row per (strategy_name, exchange, coin) combination.

Two real on/off flags live on this row: simulator_enabled and
execution_enabled. This module only exposes execution_enabled for now
(per current instructions) -- simulator_enabled is read back on the
detail response so the frontend can display it, but there is no PATCH
for it yet. That endpoint/toggle lands when the Simulator module is
built, since simulator_enabled has no exclusivity rule and deserves its
own considered UI rather than being bolted on here.

No "Stopped"/"Paused" 3-state status anywhere -- per the real data, a
strategy row only has two real states: execution_enabled True or False.
Whether it's actually the one live for its pair (vs. sitting disabled,
vs. its pair having zero enabled strategies at all) is a derived
property surfaced as `is_live_for_pair` / `pair_status`, not stored.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class StrategyEnabledUpdate(BaseModel):
    execution_enabled: bool


class StrategySummary(BaseModel):
    """One row in the Strategies table."""
    strategy_id: int
    strategy_name: str
    exchange: str
    coin: str
    time_horizon: str
    execution_enabled: bool
    simulator_enabled: bool
    # True only if this row is THE currently-enabled strategy for its
    # pair AND no other row for that same pair is also enabled (i.e. the
    # pair isn't misconfigured). A strategy can have execution_enabled
    # True yet still show is_live_for_pair False if a sibling row for
    # the same (exchange, coin) is also True -- surfaces the exact
    # "2+ enabled, pair skipped" misconfiguration execution/main.py
    # itself warns about, instead of hiding it.
    is_live_for_pair: bool
    # "live" | "disabled" | "conflicted" | "unassigned" -- see
    # strategies_repo._pair_status() for exactly what each means.
    # "unassigned" only ever appears when a DIFFERENT strategy on the
    # same pair holds that state, not on the row that's actually
    # disabled -- disabled rows show "disabled".
    pair_status: str
    # Latest real return / Sharpe / win rate, sourced from whichever of
    # execution or simulator has actually produced trades for this row
    # (see strategies_repo._performance_for_strategy) -- None (rendered
    # as "—" by the frontend) if this strategy has never run in either.
    latest_return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    win_rate_pct: Optional[float] = None
    created_at: Optional[datetime] = None


class StrategyConfigDetail(BaseModel):
    """Real entry/exit logic + risk fields, parsed from strategy_config JSON."""
    indicators: list[str] = []
    entry_logic_long: Optional[str] = None
    entry_logic_short: Optional[str] = None
    take_profit_type: Optional[str] = None
    take_profit_value: Optional[float] = None
    stop_loss_type: Optional[str] = None
    stop_loss_value: Optional[float] = None


class TradeStats(BaseModel):
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate_pct: Optional[float] = None


class EquityPoint(BaseModel):
    timestamp: str
    balance: float


class RecentTrade(BaseModel):
    entry_date_time: Optional[datetime] = None
    direction: Optional[str] = None
    entry_price: Optional[float] = None
    exit_date_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    net_pnl: Optional[float] = None
    exit_reason: Optional[str] = None


class StrategyDetail(StrategySummary):
    strategy_config: StrategyConfigDetail
    # Which module the performance numbers above / charts below actually
    # came from -- "execution", "simulator", or None if this strategy has
    # never produced a single trade in either. The frontend uses this to
    # label the Performance/Trade sections honestly instead of implying
    # a specific source.
    data_source: Optional[str] = None
    trade_stats: TradeStats = TradeStats()
    recent_trades: list[RecentTrade] = []
    # Real equity curve array (timestamp/balance pairs) -- same shape
    # ExecutionDetails already exposes as `equity_curve`, built from the
    # same execution/simulator ledger, so the frontend chart code stays
    # identical between the two pages.
    equity_curve: list[EquityPoint] = []
    # Full compute_stats() bundle (metrics + plots: equity curve,
    # drawdown, rolling sharpe/vol, monthly heatmap, yearly returns,
    # distribution) off the real equity curve for whichever module ran
    # this strategy. None if it's never traded yet (too little/no
    # history to compute anything from).
    stats: Optional[dict] = None
    # Real backtest data doesn't exist yet in this codebase (no backtest
    # reader in db_utils.py, no Backtests module built) -- left explicit
    # and None rather than fabricated, so the frontend can render an
    # honest "not available yet" placeholder for this PDF-spec section.
    backtest_summary: Optional[dict] = None
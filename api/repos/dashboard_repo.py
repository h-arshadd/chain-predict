"""
repos/dashboard_repo.py
------------------------
DB access for the Dashboard (landing page).

Total Strategies, Active Strategies, Running Executions, Running
Simulations, Connected Accounts, Trained ML Models, Total Backtests,
Today's PnL, Total Return -- plus the strategies table. Nothing here
is invented -- every widget reads a real number from
execution/simulator/backtest data.

Two pieces:

1. get_summary() -- the top stat-card strip. Running Executions/
   Simulations are read from execution.positions/simulator.positions
   directly (open position = running). Today's PnL and Total Return
   are read from SIMULATOR data only (simulator.positions +
   simulator.{...}_trades), not execution.

2. list_strategies() -- the Dashboard's strategy table. This table is
   SIMULATOR data, deliberately separate from strategies_repo's
   execution-only model used on the Strategies page (per instruction --
   the two pages are meant to differ, one is the live/execution view,
   this one is the continuously-running simulator view). Every
   simulator_enabled metadata.strategy row, with real performance read
   from simulator.positions / simulator.{...}_trades. No pair_status/
   Live-Disabled-Conflicted concept here -- that's an execution-only
   exclusivity rule (see strategies_repo._pair_status) that doesn't apply
   to simulator, which runs multiple strategies per pair with no
   exclusivity at all.
"""

import datetime as _dt
import re

from psycopg2 import sql

from crypto_pipeline.utils.metadata_utils import get_strategies, get_backtests
from crypto_pipeline.utils.db_utils import (
    get_simulator_universe,
    get_simulator_state,
    get_simulator_config,
    get_simulator_summary,
    get_simulator_stats,
    build_equity_curve_from_ledger,
    get_execution_universe,
    get_execution_state,
)
from crypto_pipeline.accounts.accounts_utils import list_accounts
from api.repos import ml_repo


# ----------------------------------------------------------------------
# Summary strip
# ----------------------------------------------------------------------

def _simulator_strategy_lookup(conn):
    def lookup(exchange, symbol):
        rows = get_strategies(conn, exchange=exchange, coin=symbol)
        # No exclusivity for simulator -- every simulator_enabled row on
        # this pair is a real, independently-running strategy.
        return [r for r in rows if r.get("simulator_enabled", True)]
    return lookup


def _running_simulations(conn) -> int:
    """
    Count of (exchange, symbol) pairs with at least one simulator
    strategy currently holding an open position (simulator.positions,
    via get_simulator_state) -- real running-simulation count, not
    derived from simulator_enabled alone (a strategy can be enabled but
    flat between trades).
    """
    running = 0
    for exchange, symbol in get_simulator_universe(conn):
        for strategy_row in _simulator_strategy_lookup(conn)(exchange, symbol):
            state = get_simulator_state(conn, exchange, symbol, strategy_row["strategy_name"])
            if state is not None and state.get("position") is not None:
                running += 1
                break  # one open position is enough to count this pair as running
    return running


def _current_execution_strategy(conn, exchange, symbol):
    """
    Same rule executions_repo._current_strategy_for_pair() and
    execution/main.py's own loop use: the one metadata.strategy row for
    this pair with execution_enabled=True. None if zero or more than one
    are enabled (unassigned/conflicted -- never guessed).
    """
    rows = get_strategies(conn, exchange=exchange, coin=symbol)
    enabled = [r for r in rows if r.get("execution_enabled", True)]
    if len(enabled) != 1:
        return None
    return enabled[0]


def _running_executions(conn) -> int:
    """
    Count of (exchange, symbol) pairs with an open LIVE position
    (execution.positions, via get_execution_state) for that pair's
    single execution-enabled strategy -- mirrors _running_simulations()
    but on the execution side.
    """
    running = 0
    for exchange, symbol in get_execution_universe(conn):
        strategy_row = _current_execution_strategy(conn, exchange, symbol)
        if strategy_row is None:
            continue
        state = get_execution_state(conn, exchange, symbol, strategy_row["strategy_name"])
        if state is not None and state.get("position") is not None:
            running += 1
    return running


def _simulator_portfolio(conn) -> dict:
    """
    Real balance/PnL rollup across every simulator pair/strategy
    currently registered -- summed from simulator.positions (via
    get_simulator_config/get_simulator_state), pair by pair, strategy by
    strategy.
    """
    pairs = get_simulator_universe(conn)
    total_balance = 0.0
    total_initial = 0.0
    any_data = False

    for exchange, symbol in pairs:
        config = get_simulator_config(conn, exchange, symbol)
        if config is None:
            continue
        strategy_rows = [
            r for r in get_strategies(conn, exchange=exchange, coin=symbol)
            if r.get("simulator_enabled", True)
        ]
        for strategy_row in strategy_rows:
            state = get_simulator_state(conn, exchange, symbol, strategy_row["strategy_name"])
            if state is None:
                continue
            any_data = True
            total_balance += state.get("balance") or 0.0
            total_initial += config.get("initial_balance") or 0.0

    if not any_data:
        return {"balance": None, "net_profit": None, "return_pct": None}

    net_profit = total_balance - total_initial
    return_pct = (net_profit / total_initial * 100.0) if total_initial else None
    return {"balance": total_balance, "net_profit": net_profit, "return_pct": return_pct}


def _simulator_trades_table(symbol, strategy_name, time_horizon):
    """
    Same table-name builder get_simulator_summary() uses internally for
    reading the Trade Ledger: simulator.{symbol}_{strategy}_{time_horizon}_trades.
    """
    safe_strategy_name = re.sub(r"[^0-9a-zA-Z_]", "_", strategy_name)
    safe_time_horizon = re.sub(r"[^0-9a-zA-Z_]", "_", time_horizon)
    return f"{symbol}_{safe_strategy_name}_{safe_time_horizon}_trades"


def _today_simulator_pnl(conn) -> float | None:
    """
    Sum of net_pnl for every simulator trade that closed today (UTC),
    across every (exchange, symbol) pair and every simulator_enabled
    strategy's trades table (simulator.{symbol}_{strategy}_{time_horizon}_trades).
    Real closed-trade PnL, not an estimate -- returns None if no pair
    has ever traded (table doesn't exist for any of them yet), 0.0 if
    tables exist but nothing closed today.

    Table lookup goes through sql.Identifier + as_string() for
    to_regclass, same quoting fix get_simulator_summary() uses, so
    strategy names with uppercase letters resolve correctly.

    NOTE: unlike execution's trades table, the simulator's Trade Ledger
    (see append_simulator_trades) has no `status` column -- its columns
    are whatever's in the closed-trade DataFrame (direction,
    entry_date_time, exit_date_time, entry_price, exit_price, quantity,
    gross_pnl, commission, slippage, net_pnl, exit_reason, balance).
    Every row in this table is already a closed trade (only closed
    trades ever get appended), so there's nothing to filter on besides
    exit_date_time.
    """
    today = _dt.datetime.now(_dt.timezone.utc).date()
    total = 0.0
    any_table = False

    cursor = conn.cursor()
    for exchange, symbol in get_simulator_universe(conn):
        strategy_rows = [
            r for r in get_strategies(conn, exchange=exchange, coin=symbol)
            if r.get("simulator_enabled", True)
        ]
        for row in strategy_rows:
            time_horizon = row.get("time_horizon") or "1h"
            table_name = _simulator_trades_table(symbol, row["strategy_name"], time_horizon)
            qualified_name = sql.SQL(".").join(
                [sql.Identifier("simulator"), sql.Identifier(table_name)]
            ).as_string(conn)
            cursor.execute(sql.SQL("SELECT to_regclass(%s)"), (qualified_name,))
            if cursor.fetchone()[0] is None:
                continue

            any_table = True
            cursor.execute(sql.SQL("""
                SELECT COALESCE(SUM(net_pnl), 0) FROM {schema}.{table}
                WHERE exit_date_time::date = %s
            """).format(
                schema=sql.Identifier("simulator"),
                table=sql.Identifier(table_name),
            ), (today,))
            total += cursor.fetchone()[0] or 0.0

    cursor.close()
    return total if any_table else None


def get_summary(conn) -> dict:
    all_strategies = get_strategies(conn)
    total_strategies = len(all_strategies)
    # "Active" = simulator_enabled OR execution_enabled -- a strategy
    # actively running in either mode counts, matching the PDF's
    # system-wide "Active Strategies" widget (not scoped to one mode).
    active_strategies = len([
        s for s in all_strategies
        if s.get("simulator_enabled", True) or s.get("execution_enabled", True)
    ])

    connected_accounts = len(list_accounts(conn))
    ml_model_count = len(ml_repo.list_runs(conn))

    # Today's PnL and Total Return are read entirely from simulator
    # data (simulator.positions + simulator.{...}_trades) -- no
    # execution money involved, no Overall Portfolio Value widget.
    sim_portfolio = _simulator_portfolio(conn)
    today_sim_pnl = _today_simulator_pnl(conn)

    return {
        "total_strategies": total_strategies,
        "active_strategies": active_strategies,
        "running_executions": _running_executions(conn),
        "running_simulations": _running_simulations(conn),
        "connected_accounts": connected_accounts,
        "trained_ml_models": ml_model_count,
        # Real count now that the Backtests module exists (see
        # backtests_repo.py / metadata_utils.get_backtests) -- every
        # request ever submitted, any status.
        "total_backtests": len(get_backtests(conn)),
        # Real closed-trade PnL for simulator trades that exited today
        # (UTC). None if the simulator has never traded at all yet.
        "today_pnl": today_sim_pnl,
        # Simulator balance vs. simulator initial balance across every
        # registered pair/strategy. None if the simulator has no real
        # data yet.
        "total_return_pct": sim_portfolio["return_pct"],
    }


# ----------------------------------------------------------------------
# Strategy table -- SIMULATOR data. Deliberately separate from
# strategies_repo.list_strategies() (execution-only, used by the
# Strategies page) -- the two pages show different things on purpose.
# ----------------------------------------------------------------------

_PNL_SPARKLINE_POINTS = 30


def _pnl_series_from_equity(equity, initial_balance) -> list[dict] | None:
    """
    Downsample a pandas equity Series into <= _PNL_SPARKLINE_POINTS
    {"t", "v"} points (v = % return vs initial_balance) for the table's
    per-row sparkline. Same approach strategies_repo.py uses for its own
    (execution) sparkline, applied here to simulator's equity curve
    builder instead.
    """
    if equity is None or len(equity) == 0 or not initial_balance:
        return None

    if len(equity) > _PNL_SPARKLINE_POINTS:
        step = len(equity) // _PNL_SPARKLINE_POINTS
        equity = equity.iloc[::step]

    return [
        {"t": ts.isoformat() if hasattr(ts, "isoformat") else str(ts), "v": (float(val) - initial_balance) / initial_balance * 100.0}
        for ts, val in equity.items()
    ]


def list_strategies(conn) -> list[dict]:
    """
    Every simulator_enabled metadata.strategy row, with real performance
    read from simulator.positions (get_simulator_state), the simulator's
    own Trade Ledger (get_simulator_summary), and simulator.stats
    (get_simulator_stats, for sharpe_ratio -- precomputed by whatever job
    runs compute_stats() for the simulator), for every pair in
    simulator.config (get_simulator_universe) -- this pipeline is
    continuously running the simulator across all registered pairs, so
    this table reflects that real, ongoing activity. No pair_status
    (Live/Disabled/Conflicted) here -- that's an execution-only
    exclusivity concept that doesn't apply to simulator, which can run
    several strategies per pair at once with no exclusivity rule.

    A strategy with no simulator trades yet still shows in the table
    (simulator_enabled=True is real state) but with null performance
    fields -- never fabricated, never borrowed from execution.
    """
    results = []
    for exchange, symbol in get_simulator_universe(conn):
        config = get_simulator_config(conn, exchange, symbol)
        initial_balance = (config or {}).get("initial_balance")

        strategy_rows = [
            r for r in get_strategies(conn, exchange=exchange, coin=symbol)
            if r.get("simulator_enabled", True)
        ]
        for row in strategy_rows:
            strategy_name = row["strategy_name"]
            time_horizon = row.get("time_horizon") or "1h"

            summary = get_simulator_summary(conn, exchange, symbol, strategy_name, time_horizon)
            latest_return_pct = None
            win_rate_pct = None
            pnl_series = None

            if summary is not None and summary.get("total_trades", 0) > 0 and initial_balance:
                total_net_profit = summary.get("total_net_profit")
                if total_net_profit is not None:
                    latest_return_pct = (total_net_profit / initial_balance) * 100.0
                win_loss = summary.get("win_loss") or {}
                if win_loss.get("win_rate") is not None:
                    win_rate_pct = win_loss["win_rate"] * 100.0

                equity = build_equity_curve_from_ledger(
                    conn, exchange, symbol, strategy_name, time_horizon, initial_balance
                )
                pnl_series = _pnl_series_from_equity(equity, initial_balance)

            results.append({
                "strategy_id": row["strategy_id"],
                "strategy_name": strategy_name,
                "exchange": exchange,
                "coin": symbol,
                "time_horizon": time_horizon,
                "simulator_enabled": row.get("simulator_enabled", True),
                "latest_return_pct": latest_return_pct,
                # Precomputed by whatever job runs compute_stats() for the
                # simulator (see save_simulator_stats) -- read straight
                # from simulator.stats instead of recomputing here.
                "sharpe_ratio": (get_simulator_stats(conn, exchange, symbol, strategy_name) or {}).get("sharpe"),
                "win_rate_pct": win_rate_pct,
                "pnl_series": pnl_series,
                "created_at": row.get("created_at"),
            })
    return results
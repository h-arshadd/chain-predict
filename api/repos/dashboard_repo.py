"""
repos/dashboard_repo.py
------------------------
DB access for the Dashboard (landing page).

Exactly the 10 widgets the spec asks for -- Total Strategies, Active
Strategies, Running Executions, Running Simulations, Connected
Accounts, Trained ML Models, Total Backtests, Today's PnL, Overall
Portfolio Value, Total Return -- plus the strategies table. Nothing
extra. Nothing here is invented -- every widget reads a real number
from execution/simulator/backtest data.

Two pieces:

1. get_summary() -- the top stat-card strip. Running Executions/
   Simulations are read from execution.positions/simulator.positions
   directly (open position = running). Today's PnL, Overall Portfolio
   Value, and Total Return combine BOTH execution and simulator money
   (no separate execution/simulator line items are exposed here -- the
   spec asks for one Overall Portfolio Value / one Total Return, not a
   breakdown).

2. list_strategies() -- the Dashboard's strategy table. Reuses
   strategies_repo.list_strategies() as-is (the real execution-first-
   fallback-to-simulator performance model, same pair-status logic used
   on the Strategies page) rather than a second, simulator-only
   implementation -- so this table and the Strategies page never
   disagree about which numbers are "real" for a given strategy.
"""

import datetime as _dt

from psycopg2 import sql

from crypto_pipeline.utils.metadata_utils import get_strategies, get_backtests
from crypto_pipeline.utils.db_utils import (
    get_simulator_universe,
    get_simulator_state,
    get_simulator_config,
    get_execution_universe,
    get_execution_state,
    get_execution_config,
    _execution_trades_table,
)
from crypto_pipeline.accounts.accounts_utils import list_accounts
from api.repos import ml_repo
from api.repos.strategies_repo import list_strategies as _list_strategies


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


def _execution_portfolio(conn) -> dict:
    """
    Real balance/PnL rollup across every (exchange, symbol) pair's
    single execution-enabled strategy -- summed from execution.positions
    (via get_execution_config/get_execution_state). Exact mirror of
    _simulator_portfolio(), execution side, one strategy per pair
    (execution's real exclusivity rule) instead of simulator's "every
    enabled row".
    """
    pairs = get_execution_universe(conn)
    total_balance = 0.0
    total_initial = 0.0
    any_data = False

    for exchange, symbol in pairs:
        config = get_execution_config(conn, exchange, symbol)
        if config is None:
            continue
        strategy_row = _current_execution_strategy(conn, exchange, symbol)
        if strategy_row is None:
            continue
        state = get_execution_state(conn, exchange, symbol, strategy_row["strategy_name"])
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


def _today_execution_pnl(conn) -> float | None:
    """
    Sum of net_pnl for every LIVE trade that closed today (UTC), across
    every (exchange, symbol) pair's execution-enabled strategy trades
    table (execution.{symbol}_{strategy}_trades, see
    _execution_trades_table). Real closed-trade PnL, not an estimate --
    returns None if no pair has ever traded (table doesn't exist for
    any of them yet), 0.0 if tables exist but nothing closed today.
    """
    today = _dt.datetime.now(_dt.timezone.utc).date()
    total = 0.0
    any_table = False

    cursor = conn.cursor()
    for exchange, symbol in get_execution_universe(conn):
        strategy_row = _current_execution_strategy(conn, exchange, symbol)
        if strategy_row is None:
            continue
        table_name = _execution_trades_table(exchange, symbol, strategy_row["strategy_name"])
        qualified_name = sql.SQL(".").join(
            [sql.Identifier("execution"), sql.Identifier(table_name)]
        ).as_string(conn)
        cursor.execute(sql.SQL("SELECT to_regclass(%s)"), (qualified_name,))
        if cursor.fetchone()[0] is None:
            continue

        any_table = True
        cursor.execute(sql.SQL("""
            SELECT COALESCE(SUM(net_pnl), 0) FROM {schema}.{table}
            WHERE status = 'closed' AND exit_date_time::date = %s
        """).format(
            schema=sql.Identifier("execution"),
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

    sim_portfolio = _simulator_portfolio(conn)
    exec_portfolio = _execution_portfolio(conn)
    today_exec_pnl = _today_execution_pnl(conn)

    # Overall Portfolio Value / Total Return combine execution + sim
    # balances only where at least one has real data -- never silently
    # treats a missing side as zero if BOTH are missing.
    balances = [b for b in (sim_portfolio["balance"], exec_portfolio["balance"]) if b is not None]
    initials = []
    if sim_portfolio["balance"] is not None and sim_portfolio["net_profit"] is not None:
        initials.append(sim_portfolio["balance"] - sim_portfolio["net_profit"])
    if exec_portfolio["balance"] is not None and exec_portfolio["net_profit"] is not None:
        initials.append(exec_portfolio["balance"] - exec_portfolio["net_profit"])

    overall_portfolio_value = sum(balances) if balances else None
    if balances and initials and sum(initials):
        total_return_pct = (sum(balances) - sum(initials)) / sum(initials) * 100.0
    else:
        total_return_pct = None

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
        # Real closed-trade PnL for live trades that exited today (UTC).
        # None if execution has never traded at all yet.
        "today_pnl": today_exec_pnl,
        # Execution + simulator balances combined. None if neither side
        # has any real data yet.
        "overall_portfolio_value": overall_portfolio_value,
        "total_return_pct": total_return_pct,
    }


# ----------------------------------------------------------------------
# Strategy table -- reuses strategies_repo's real execution-first-
# fallback-to-simulator model, same numbers as the Strategies page.
# ----------------------------------------------------------------------

def list_strategies(conn) -> list[dict]:
    """
    Every metadata.strategy row, enriched exactly the way the Strategies
    page is (see strategies_repo.list_strategies): real pair status
    (live/conflicted/disabled) and performance pulled from execution
    first, falling back to simulator, never fabricated.
    """
    return _list_strategies(conn)
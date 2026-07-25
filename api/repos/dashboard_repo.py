"""
repos/dashboard_repo.py
------------------------
DB access for the Dashboard (landing page).

Per instruction, the Dashboard is simulation-only end to end -- no
execution/live-account data anywhere on this page (that's Wallets'/
Deployment's territory). Two pieces:

1. get_summary() -- the top stat-card strip (Total/Active Strategies,
   Running Simulations, Connected Accounts, Trained ML Models, Total
   Backtests, Simulator Balance/Net Profit/Return). Running Simulations
   is read from simulator.config/positions directly -- no
   execution-side lookup happens anywhere in this module.

2. list_simulator_strategies() -- the Dashboard's strategy table. Per
   instruction, this table is 100% simulator-sourced (not the
   execution-first-fallback-to-simulator _performance_for_strategy()
   pattern strategies_repo.py uses for the Strategies page). It also
   does NOT filter by simulator_enabled -- every metadata.strategy row
   is shown (a strategy can have 15 rows per coin, per
   load_strategies_from_yaml()'s own docstring), with real simulator
   stats if it has ever run and zero/empty stats if it hasn't. This is
   deliberately a new, separate function/file rather than a change to
   strategies_repo.list_strategies(), which stays exactly as-is for the
   Strategies page.

Total Backtests has no real reader -- no Backtests module/DB exists yet
(see PROJECT_SUMMARY.md) -- so it's returned as None here, rendered by
the frontend as an honest "not available yet", never fabricated.
"""

from crypto_pipeline.utils.metadata_utils import get_strategies
from crypto_pipeline.utils.db_utils import (
    get_simulator_universe,
    get_simulator_state,
    get_simulator_config,
    get_simulator_summary,
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


def get_summary(conn) -> dict:
    all_strategies = get_strategies(conn)
    total_strategies = len(all_strategies)
    # "Active" here means simulator_enabled, matching this page's
    # simulation-only scope (Strategies page's own "active" concept is
    # execution_enabled, a different real toggle for a different page).
    active_strategies = len([s for s in all_strategies if s.get("simulator_enabled", True)])

    connected_accounts = len(list_accounts(conn))
    ml_model_count = len(ml_repo.list_runs(conn))
    sim_portfolio = _simulator_portfolio(conn)

    return {
        "total_strategies": total_strategies,
        "active_strategies": active_strategies,
        "running_simulations": _running_simulations(conn),
        "connected_accounts": connected_accounts,
        "trained_ml_models": ml_model_count,
        # No backtest reader/module exists yet -- honest None, not a
        # fabricated count. See PROJECT_SUMMARY.md section 3.
        "total_backtests": None,
        "simulator_balance": sim_portfolio["balance"],
        "simulator_net_profit": sim_portfolio["net_profit"],
        "simulator_return_pct": sim_portfolio["return_pct"],
    }


# ----------------------------------------------------------------------
# Strategy table -- simulator-only, no simulator_enabled filter
# ----------------------------------------------------------------------

def list_simulator_strategies(conn) -> list[dict]:
    """
    Every metadata.strategy row (ALL of them -- no simulator_enabled
    filter, per instruction: the simulator runs every registered
    strategy), enriched purely with real simulator data via
    get_simulator_summary(). A strategy that has never actually run in
    the simulator yet gets zero/empty stats (total_trades=0, PnL/win
    rate/sharpe all None) rather than being hidden or marked
    "unassigned" -- that concept doesn't apply here, this table isn't
    about which one strategy is live for a pair (that's execution's
    exclusivity rule), it's "how is every registered strategy doing in
    the simulator".
    """
    all_rows = get_strategies(conn)
    results = []

    for row in all_rows:
        exchange = row["exchange"]
        coin = row["coin"]
        strategy_name = row["strategy_name"]
        time_horizon = row.get("time_horizon") or "1h"

        latest_return_pct = None
        win_rate_pct = None
        total_trades = 0
        has_run = False

        sim_config = get_simulator_config(conn, exchange, coin)
        if sim_config is not None:
            sim_summary = get_simulator_summary(conn, exchange, coin, strategy_name, time_horizon)
            if sim_summary is not None:
                total_trades = sim_summary.get("total_trades", 0)
                if total_trades > 0:
                    has_run = True
                    initial_balance = sim_config.get("initial_balance")
                    total_net_profit = sim_summary.get("total_net_profit")
                    if initial_balance:
                        latest_return_pct = (total_net_profit / initial_balance) * 100.0
                    win_loss = sim_summary.get("win_loss") or {}
                    win_rate = win_loss.get("win_rate")
                    if win_rate is not None:
                        win_rate_pct = win_rate * 100.0

        results.append({
            "strategy_id": row["strategy_id"],
            "strategy_name": strategy_name,
            "exchange": exchange,
            "coin": coin,
            "time_horizon": time_horizon,
            "simulator_enabled": row.get("simulator_enabled", True),
            "has_run": has_run,
            "total_trades": total_trades,
            "latest_return_pct": latest_return_pct,
            # Sharpe isn't part of get_simulator_summary()'s lightweight
            # roll-up (same reasoning as strategies_repo._perf_from_summary)
            # -- left None on this list view rather than computing full
            # quantstats per row, which would be expensive at up to ~15
            # strategies x N coins.
            "sharpe_ratio": None,
            "win_rate_pct": win_rate_pct,
        })

    return results
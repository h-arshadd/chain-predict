"""
repos/simulator_repo.py
--------------------------
DB access for the Dashboard's Simulation Details page -- the simulator
counterpart to executions_repo.py's Execution Details. This pipeline
runs the simulator continuously across every registered pair/strategy
(see simulator/main.py, run_simulator.bat), so this reads that real,
ongoing state -- nothing here is invented or borrowed from execution.

The universe is simulator.config (one row per (exchange, symbol) pair
set up to simulate) x every simulator_enabled metadata.strategy row for
that pair -- unlike execution (at most one enabled strategy per pair),
simulator has NO exclusivity rule, so a pair can have several strategies
running side by side. A detail page is therefore keyed on
(exchange, symbol, strategy_name), not just (exchange, symbol).

No live-exchange call here (simulator is paper trading -- there is no
live Bybit position to fetch), and no wallet concept (simulator doesn't
use accounts.api_keys) -- both are real differences from
executions_repo.py, not omissions.
"""

import re
import yaml
from pathlib import Path

from psycopg2 import sql

from crypto_pipeline.utils.db_utils import (
    get_simulator_universe,
    get_simulator_config,
    get_simulator_state,
    get_simulator_summary,
    build_equity_curve_from_ledger,
)
from crypto_pipeline.utils.metadata_utils import get_strategies
from crypto_pipeline.stats.calculator import compute_stats

_STATS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "crypto_pipeline" / "stats" / "config.yaml"
_stats_config_cache = None


def _stats_config() -> dict:
    global _stats_config_cache
    if _stats_config_cache is None:
        with open(_STATS_CONFIG_PATH) as f:
            _stats_config_cache = yaml.safe_load(f)
    return _stats_config_cache


# ----------------------------------------------------------------------
# Entry/exit logic text -- same rendering approach executions_repo.py /
# strategies_repo.py use, reused here rather than duplicated differently.
# ----------------------------------------------------------------------

_OPERATOR_TEXT = {
    "cross_above": "crosses above",
    "cross_below": "crosses below",
    "greater_than": ">",
    "less_than": "<",
    "greater_equal": ">=",
    "less_equal": "<=",
    "equal": "==",
}


def _describe_condition(cond: dict) -> str:
    left = cond.get("left", "?")
    right = cond.get("right", "?")
    op = _OPERATOR_TEXT.get(cond.get("operator"), cond.get("operator", "?"))
    text = f"{left} {op} {right}"
    if cond.get("persist_bars"):
        text += f" (held {cond['persist_bars']} bars)"
    return text


def _describe_side(strategy_config: dict, side: str) -> str | None:
    if not strategy_config:
        return None
    side_block = (strategy_config.get("strategy") or {}).get(side)
    if not side_block:
        return None
    conditions = side_block.get("conditions") or []
    if not conditions:
        return None
    rule = (side_block.get("rule") or "AND").upper()
    parts = [_describe_condition(c) for c in conditions]
    return f" {rule} ".join(parts)


def _strategy_config_detail(strategy_row: dict | None) -> dict:
    if strategy_row is None:
        return {
            "indicators": [],
            "entry_logic_long": None,
            "entry_logic_short": None,
            "take_profit_type": None,
            "take_profit_value": None,
            "stop_loss_type": None,
            "stop_loss_value": None,
        }
    config = strategy_row.get("strategy_config") or {}
    indicator_keys = [k for k in config.keys() if k != "strategy"]
    return {
        "indicators": indicator_keys,
        "entry_logic_long": _describe_side(config, "long"),
        "entry_logic_short": _describe_side(config, "short"),
        "take_profit_type": strategy_row.get("take_profit_type"),
        "take_profit_value": strategy_row.get("take_profit_value"),
        "stop_loss_type": strategy_row.get("stop_loss_type"),
        "stop_loss_value": strategy_row.get("stop_loss_value"),
    }


def _find_strategy_row(conn, exchange, symbol, strategy_name):
    rows = get_strategies(conn, exchange=exchange, coin=symbol)
    for r in rows:
        if r["strategy_name"] == strategy_name:
            return r
    return None


# ----------------------------------------------------------------------
# List -- every (pair, simulator_enabled strategy) combination
# ----------------------------------------------------------------------

def list_simulations(conn) -> list[dict]:
    """
    One row per (exchange, symbol, strategy_name) actually running in
    the simulator (simulator.config pair x that pair's
    simulator_enabled metadata.strategy rows) -- same universe
    dashboard_repo.list_strategies() already iterates, exposed here as
    its own listing for completeness/potential reuse.
    """
    rows = []
    for exchange, symbol in get_simulator_universe(conn):
        strategy_rows = [
            r for r in get_strategies(conn, exchange=exchange, coin=symbol)
            if r.get("simulator_enabled", True)
        ]
        for strategy_row in strategy_rows:
            rows.append(_build_summary(conn, exchange, symbol, strategy_row))
    return rows


def _build_summary(conn, exchange, symbol, strategy_row) -> dict:
    strategy_name = strategy_row["strategy_name"]
    time_horizon = strategy_row.get("time_horizon") or "1h"
    config = get_simulator_config(conn, exchange, symbol)
    state = get_simulator_state(conn, exchange, symbol, strategy_name)

    if state is None:
        return {
            "exchange": exchange,
            "symbol": symbol,
            "strategy_id": strategy_row["strategy_id"],
            "strategy_name": strategy_name,
            "time_horizon": time_horizon,
            "simulator_enabled": strategy_row.get("simulator_enabled", True),
            "status": "never_run",
            "position": None,
            "balance": (config or {}).get("initial_balance"),
            "cumulative_pnl": None,
            "daily_return_pct": None,
            "last_processed": None,
        }

    position = state["position"]
    balance = state["balance"]
    initial_balance = (config or {}).get("initial_balance") or balance
    daily_return_pct = None
    if initial_balance:
        daily_return_pct = ((balance - initial_balance) / initial_balance) * 100.0

    status = "running" if position is not None else "flat"

    return {
        "exchange": exchange,
        "symbol": symbol,
        "strategy_id": strategy_row["strategy_id"],
        "strategy_name": strategy_name,
        "time_horizon": time_horizon,
        "simulator_enabled": strategy_row.get("simulator_enabled", True),
        "status": status,
        "position": position,
        "balance": balance,
        "cumulative_pnl": state["cumulative_pnl"],
        "daily_return_pct": daily_return_pct,
        "last_processed": state["last_processed"],
    }


# ----------------------------------------------------------------------
# Detail
# ----------------------------------------------------------------------

def get_simulation_detail(conn, exchange, symbol, strategy_name):
    """
    Full detail for one (exchange, symbol, strategy_name) simulator run:
    summary fields, risk config, strategy config (real entry/exit logic
    + TP/SL), current simulated position, trades, and the full
    stats/plots bundle computed off the real simulator equity curve.

    Returns None if this pair was never set up in simulator.config, or
    if no metadata.strategy row with this exact strategy_name exists for
    the pair (never guesses/fabricates a strategy that isn't real).
    """
    config = get_simulator_config(conn, exchange, symbol)
    if config is None:
        return None

    strategy_row = _find_strategy_row(conn, exchange, symbol, strategy_name)
    if strategy_row is None:
        return None

    summary = _build_summary(conn, exchange, symbol, strategy_row)

    detail = dict(summary)
    detail["initial_balance"] = config.get("initial_balance")
    detail["commission"] = config.get("commission")
    detail["slippage"] = config.get("slippage")
    detail["allow_long"] = config.get("allow_long")
    detail["allow_short"] = config.get("allow_short")
    detail["total_net_profit"] = None
    detail["total_trades"] = 0
    detail["win_loss"] = None
    detail["equity_curve"] = []
    detail["trades"] = []
    detail["strategy_config"] = _strategy_config_detail(strategy_row)
    detail["stats"] = None

    time_horizon = strategy_row.get("time_horizon") or "1h"

    sim_summary = get_simulator_summary(conn, exchange, symbol, strategy_name, time_horizon)
    if sim_summary is not None:
        detail["total_net_profit"] = sim_summary["total_net_profit"]
        detail["total_trades"] = sim_summary["total_trades"]
        detail["win_loss"] = sim_summary["win_loss"]

    equity = build_equity_curve_from_ledger(
        conn, exchange, symbol, strategy_name, time_horizon, config.get("initial_balance") or 0.0
    )
    if equity is not None:
        detail["equity_curve"] = [
            {"timestamp": str(ts), "balance": float(val)} for ts, val in equity.items()
        ]
        try:
            detail["stats"] = compute_stats({"equity_curve": equity}, _stats_config())
        except Exception:
            # Too little history / a quantstats edge case shouldn't break
            # the whole page -- trades/equity curve above still render.
            detail["stats"] = None

    detail["trades"] = _list_trades(conn, exchange, symbol, strategy_name, time_horizon)

    return detail


def _list_trades(conn, exchange, symbol, strategy_name, time_horizon, limit: int = 200) -> list[dict]:
    """
    Most recent simulator trades (closed -- simulator's trade ledger only
    ever has closed rows, see append_simulator_trades) for this exact
    (symbol, strategy_name, time_horizon), newest entry first.
    """
    safe_strategy_name = re.sub(r"[^0-9a-zA-Z_]", "_", strategy_name)
    safe_time_horizon = re.sub(r"[^0-9a-zA-Z_]", "_", time_horizon)
    table_name = f"{symbol}_{safe_strategy_name}_{safe_time_horizon}_trades"

    cursor = conn.cursor()
    qualified_name = sql.SQL(".").join(
        [sql.Identifier("simulator"), sql.Identifier(table_name)]
    ).as_string(conn)
    cursor.execute(sql.SQL("SELECT to_regclass(%s)"), (qualified_name,))
    table_exists = cursor.fetchone()[0] is not None
    if not table_exists:
        cursor.close()
        return []

    cursor.execute(sql.SQL("""
        SELECT entry_date_time, direction, entry_price, quantity,
               exit_date_time, exit_price, gross_pnl, commission, slippage, net_pnl,
               exit_reason, balance
        FROM {schema}.{table}
        ORDER BY entry_date_time DESC
        LIMIT %s
    """).format(schema=sql.Identifier("simulator"), table=sql.Identifier(table_name)), (limit,))

    columns = [
        "entry_date_time", "direction", "entry_price", "quantity",
        "exit_date_time", "exit_price", "gross_pnl", "commission", "slippage", "net_pnl",
        "exit_reason", "balance",
    ]
    rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
    cursor.close()
    return rows
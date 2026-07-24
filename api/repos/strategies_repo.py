"""
repos/strategies_repo.py
--------------------------
DB access for the Strategies (list) + Strategy Details pages. Builds on
metadata_utils.get_strategies()/get_strategy()/set_strategy_enabled() --
nothing here talks to Postgres directly except the two small read-only
trade-ledger queries at the bottom, added here for the same reason
executions_repo.py added _list_trades(): db_utils.py/simulator ledger
tables only have writers, no "give me recent trades" reader.

Real status model (see metadata_utils.create_strategy_table's docstring
and execution/main.py's main loop, ~line 775-803):

  - metadata.strategy has TWO independent per-row toggles:
    simulator_enabled (no exclusivity -- simulator is allowed to try
    several strategies per pair at once) and execution_enabled (execution
    allows AT MOST ONE True per (exchange, coin) pair -- enforced only at
    READ time by execution/main.py and by this API, never by the DB or by
    metadata_utils.set_strategy_enabled() itself).
  - This module only exposes execution_enabled via PATCH for now, per
    current instructions -- simulator_enabled is read-only here (shown on
    the detail response) until the Simulator module defines its own
    toggle UI.
  - There is no separate "Active/Paused/Stopped" status column anywhere.
    The real, only pause mechanism IS execution_enabled itself -- see
    _pair_status() below for the four derived states shown to the user.

Critical constraint this repo enforces (the DB/pipeline don't):
setting execution_enabled=True on a strategy for a pair that already has
a different strategy execution_enabled=True must atomically disable the
previous one in the same request -- otherwise the UI creates exactly the
"2+ enabled, pair skipped" misconfiguration execution/main.py already
warns about (and executions_repo._current_strategy_for_pair() already
treats as "unassigned"). See set_execution_enabled() below.
"""

import re
import yaml
from pathlib import Path

from psycopg2 import sql

from crypto_pipeline.utils.metadata_utils import (
    get_strategies,
    get_strategy,
    set_strategy_enabled,
)
from crypto_pipeline.utils.db_utils import (
    get_execution_config,
    get_execution_summary,
    build_execution_equity_curve_from_ledger,
    _execution_trades_table,
    get_simulator_config,
    get_simulator_summary,
    build_equity_curve_from_ledger as build_simulator_equity_curve_from_ledger,
)
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
# Entry/exit logic text -- same rendering approach executions_repo.py
# uses for Execution Details, reused here rather than duplicated
# differently. (Not imported from there to avoid a repo-to-repo import;
# it's ~15 lines and both are read-only pure functions over the same
# strategy_config JSON shape.)
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


def _strategy_config_detail(strategy_row: dict) -> dict:
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


# ----------------------------------------------------------------------
# Pair status / exclusivity
# ----------------------------------------------------------------------

def _pair_status(strategy_row: dict, siblings: list[dict]) -> tuple[bool, str]:
    """
    Derive (is_live_for_pair, pair_status) for one strategy row, given
    every other metadata.strategy row for the same (exchange, coin) pair
    (siblings does NOT include strategy_row itself).

    - This row disabled                              -> (False, "disabled")
    - This row enabled, zero siblings also enabled    -> (True,  "live")
    - This row enabled, 1+ siblings ALSO enabled       -> (False, "conflicted")
      (matches execution/main.py's real behavior: 2+ enabled rows means
      execution/main.py skips the pair entirely -- nobody is actually
      live, even though this row's own flag says True)
    """
    if not strategy_row.get("execution_enabled", True):
        return False, "disabled"

    other_enabled = [s for s in siblings if s.get("execution_enabled", True)]
    if other_enabled:
        return False, "conflicted"
    return True, "live"


# ----------------------------------------------------------------------
# Performance (real data from whichever of execution/simulator this
# strategy has actually run in -- never fabricated)
# ----------------------------------------------------------------------

def _performance_for_strategy(conn, strategy_row: dict) -> dict:
    """
    Returns {"latest_return_pct", "sharpe_ratio", "win_rate_pct",
    "data_source"} for one strategy row. Tries execution first (a
    strategy row that's execution_enabled and has actually traded live is
    the more meaningful number), falls back to simulator, and returns all
    None / data_source=None if neither has ever run this exact
    (strategy_name, exchange, coin).
    """
    exchange = strategy_row["exchange"]
    coin = strategy_row["coin"]
    strategy_name = strategy_row["strategy_name"]
    time_horizon = strategy_row.get("time_horizon") or "1h"

    exec_config = get_execution_config(conn, exchange, coin)
    if exec_config is not None:
        exec_summary = get_execution_summary(conn, exchange, coin, strategy_name)
        if exec_summary is not None and exec_summary.get("total_trades", 0) > 0:
            perf = _perf_from_summary(exec_summary, exec_config.get("initial_balance"))
            if perf is not None:
                return {**perf, "data_source": "execution"}

    sim_config = get_simulator_config(conn, exchange, coin)
    if sim_config is not None:
        sim_summary = get_simulator_summary(conn, exchange, coin, strategy_name, time_horizon)
        if sim_summary is not None and sim_summary.get("total_trades", 0) > 0:
            perf = _perf_from_summary(sim_summary, sim_config.get("initial_balance"))
            if perf is not None:
                return {**perf, "data_source": "simulator"}

    return {"latest_return_pct": None, "sharpe_ratio": None, "win_rate_pct": None, "data_source": None}


def _perf_from_summary(summary: dict, initial_balance) -> dict | None:
    if not initial_balance:
        return None
    total_net_profit = summary.get("total_net_profit")
    latest_return_pct = None
    if total_net_profit is not None:
        latest_return_pct = (total_net_profit / initial_balance) * 100.0

    win_loss = summary.get("win_loss") or {}
    win_rate_pct = win_loss.get("win_rate")
    if win_rate_pct is not None:
        win_rate_pct = win_rate_pct * 100.0

    # Sharpe isn't part of get_execution_summary/get_simulator_summary's
    # lightweight roll-up (that's compute_stats()'s job, run separately
    # for the detail page since it's more expensive) -- left None on the
    # list view rather than computed twice; StrategyDetail's `stats`
    # bundle has the real quantstats sharpe.
    return {
        "latest_return_pct": latest_return_pct,
        "sharpe_ratio": None,
        "win_rate_pct": win_rate_pct,
    }


# ----------------------------------------------------------------------
# List
# ----------------------------------------------------------------------

def list_strategies(conn) -> list[dict]:
    """
    Every metadata.strategy row, newest first (same order
    get_strategies() already returns), enriched with derived pair status
    and real performance numbers.
    """
    all_rows = get_strategies(conn)

    # Group by (exchange, coin) once so _pair_status doesn't re-query
    # get_strategies() per row (O(n) DB round trips avoided).
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for row in all_rows:
        by_pair.setdefault((row["exchange"], row["coin"]), []).append(row)

    results = []
    for row in all_rows:
        pair_key = (row["exchange"], row["coin"])
        siblings = [r for r in by_pair[pair_key] if r["strategy_id"] != row["strategy_id"]]
        is_live, pair_status = _pair_status(row, siblings)
        perf = _performance_for_strategy(conn, row)

        results.append({
            "strategy_id": row["strategy_id"],
            "strategy_name": row["strategy_name"],
            "exchange": row["exchange"],
            "coin": row["coin"],
            "time_horizon": row["time_horizon"],
            "execution_enabled": row["execution_enabled"],
            "simulator_enabled": row["simulator_enabled"],
            "is_live_for_pair": is_live,
            "pair_status": pair_status,
            "latest_return_pct": perf["latest_return_pct"],
            "sharpe_ratio": perf["sharpe_ratio"],
            "win_rate_pct": perf["win_rate_pct"],
            "created_at": row.get("created_at"),
        })
    return results


# ----------------------------------------------------------------------
# Detail
# ----------------------------------------------------------------------

def get_strategy_detail(conn, strategy_id: int) -> dict | None:
    row = get_strategy(conn, strategy_id)
    if row is None:
        return None

    siblings = [
        r for r in get_strategies(conn, exchange=row["exchange"], coin=row["coin"])
        if r["strategy_id"] != row["strategy_id"]
    ]
    is_live, pair_status = _pair_status(row, siblings)
    perf = _performance_for_strategy(conn, row)

    detail = {
        "strategy_id": row["strategy_id"],
        "strategy_name": row["strategy_name"],
        "exchange": row["exchange"],
        "coin": row["coin"],
        "time_horizon": row["time_horizon"],
        "execution_enabled": row["execution_enabled"],
        "simulator_enabled": row["simulator_enabled"],
        "is_live_for_pair": is_live,
        "pair_status": pair_status,
        "latest_return_pct": perf["latest_return_pct"],
        "sharpe_ratio": perf["sharpe_ratio"],
        "win_rate_pct": perf["win_rate_pct"],
        "created_at": row.get("created_at"),
        "strategy_config": _strategy_config_detail(row),
        "data_source": perf["data_source"],
        "trade_stats": {"total_trades": 0, "wins": 0, "losses": 0, "win_rate_pct": None},
        "recent_trades": [],
        "stats": None,
        # No backtest reader exists in db_utils.py yet and no Backtests
        # module is built -- left explicitly None (not fabricated) so
        # the frontend renders an honest "not available yet" placeholder
        # for this PDF-spec section instead of inventing numbers.
        "backtest_summary": None,
    }

    if perf["data_source"] is None:
        return detail

    exchange, coin, strategy_name = row["exchange"], row["coin"], row["strategy_name"]
    time_horizon = row.get("time_horizon") or "1h"

    if perf["data_source"] == "execution":
        exec_config = get_execution_config(conn, exchange, coin)
        exec_summary = get_execution_summary(conn, exchange, coin, strategy_name)
        if exec_summary is not None:
            win_loss = exec_summary.get("win_loss") or {}
            detail["trade_stats"] = {
                "total_trades": exec_summary.get("total_trades", 0),
                "wins": win_loss.get("wins", 0),
                "losses": win_loss.get("losses", 0),
                "win_rate_pct": (win_loss.get("win_rate") or 0.0) * 100.0,
            }
        equity = build_execution_equity_curve_from_ledger(
            conn, exchange, coin, strategy_name, (exec_config or {}).get("initial_balance") or 0.0
        )
        detail["recent_trades"] = _list_execution_trades(conn, exchange, coin, strategy_name)
    else:  # "simulator"
        sim_config = get_simulator_config(conn, exchange, coin)
        sim_summary = get_simulator_summary(conn, exchange, coin, strategy_name, time_horizon)
        if sim_summary is not None:
            win_loss = sim_summary.get("win_loss") or {}
            detail["trade_stats"] = {
                "total_trades": sim_summary.get("total_trades", 0),
                "wins": win_loss.get("wins", 0),
                "losses": win_loss.get("losses", 0),
                "win_rate_pct": (win_loss.get("win_rate") or 0.0) * 100.0,
            }
        equity = build_simulator_equity_curve_from_ledger(
            conn, exchange, coin, strategy_name, time_horizon, (sim_config or {}).get("initial_balance") or 0.0
        )
        detail["recent_trades"] = _list_simulator_trades(conn, exchange, coin, strategy_name, time_horizon)

    if equity is not None:
        # Same real array shape ExecutionDetails' `equity_curve` field
        # already uses (executions_repo.get_execution_detail) -- built
        # directly from the pandas Series both equity-curve builders
        # return, not from compute_stats()'s plots (which only has
        # returns/cumulative_returns, no raw balance series).
        detail["equity_curve"] = [
            {"timestamp": str(ts), "balance": float(val)} for ts, val in equity.items()
        ]
        try:
            detail["stats"] = compute_stats({"equity_curve": equity}, _stats_config())
        except Exception:
            # Too little history / a quantstats edge case shouldn't break
            # the whole page -- trade stats/recent trades above still render.
            detail["stats"] = None

    return detail


# ----------------------------------------------------------------------
# Exclusivity-enforcing PATCH
# ----------------------------------------------------------------------

def set_execution_enabled(conn, strategy_id: int, execution_enabled: bool) -> dict | None:
    """
    Toggle execution_enabled for one strategy row.

    Turning OFF is always safe (no exclusivity concern). Turning ON
    atomically disables any other strategy row on the same (exchange,
    coin) pair that's currently execution_enabled=True, in the same
    request -- this is the constraint the DB/metadata_utils don't
    enforce (set_strategy_enabled() is a bare flag flip, no exclusivity
    check, per its own docstring). Without this, the frontend could
    silently create the exact "2+ enabled, pair skipped" misconfiguration
    execution/main.py's main loop already warns about.

    Returns the updated row (as get_strategy() would return it), or None
    if strategy_id doesn't exist.
    """
    row = get_strategy(conn, strategy_id)
    if row is None:
        return None

    if execution_enabled:
        siblings = get_strategies(conn, exchange=row["exchange"], coin=row["coin"])
        for sibling in siblings:
            if sibling["strategy_id"] != strategy_id and sibling.get("execution_enabled", True):
                set_strategy_enabled(conn, sibling["strategy_id"], execution_enabled=False)

    set_strategy_enabled(conn, strategy_id, execution_enabled=execution_enabled)
    return get_strategy(conn, strategy_id)


# ----------------------------------------------------------------------
# Trade readers (read-only, mirrors executions_repo._list_trades /
# no equivalent existed for simulator either)
# ----------------------------------------------------------------------

def _list_execution_trades(conn, exchange, symbol, strategy_name, limit: int = 20) -> list[dict]:
    table_name = _execution_trades_table(exchange, symbol, strategy_name)
    return _list_trades_from_table(conn, "execution", table_name, limit)


def _list_simulator_trades(conn, exchange, symbol, strategy_name, time_horizon, limit: int = 20) -> list[dict]:
    safe_strategy_name = re.sub(r"[^0-9a-zA-Z_]", "_", strategy_name)
    safe_time_horizon = re.sub(r"[^0-9a-zA-Z_]", "_", time_horizon)
    table_name = f"{symbol}_{safe_strategy_name}_{safe_time_horizon}_trades"
    return _list_trades_from_table(conn, "simulator", table_name, limit)


def _list_trades_from_table(conn, schema_name: str, table_name: str, limit: int) -> list[dict]:
    cursor = conn.cursor()
    qualified_name = sql.SQL(".").join(
        [sql.Identifier(schema_name), sql.Identifier(table_name)]
    ).as_string(conn)
    cursor.execute(sql.SQL("SELECT to_regclass(%s)"), (qualified_name,))
    table_exists = cursor.fetchone()[0] is not None
    if not table_exists:
        cursor.close()
        return []

    cursor.execute(sql.SQL("""
        SELECT entry_date_time, direction, entry_price,
               exit_date_time, exit_price, net_pnl, exit_reason
        FROM {schema}.{table}
        ORDER BY entry_date_time DESC
        LIMIT %s
    """).format(schema=sql.Identifier(schema_name), table=sql.Identifier(table_name)), (limit,))

    columns = ["entry_date_time", "direction", "entry_price", "exit_date_time", "exit_price", "net_pnl", "exit_reason"]
    rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
    cursor.close()
    return rows
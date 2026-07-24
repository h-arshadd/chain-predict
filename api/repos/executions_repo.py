"""
repos/executions_repo.py
--------------------------
DB access for the Strategy Deployment + Execution Details pages. Builds
entirely on existing crypto_pipeline functions -- nothing here talks to
Postgres directly except one small paginated trade-list query, added
here because db_utils.py didn't have a "give me N trades, newest first"
reader (open/close/append are all write-side).

The universe of "deployed strategies" is defined by execution.config
(one row per (exchange, symbol) pair that's been set up to trade) --
see db_utils.get_execution_universe/get_execution_config. For each pair,
the currently execution_enabled strategy comes from metadata.strategy
(same lookup execution/main.py itself does), and current state/PnL from
execution.positions + the trade ledger (get_execution_summary).

A pair can be present in execution.config but have never actually
traded yet (no execution.positions row) -- these still show up in the
Deployment list (status "never_run"), same as the PDF's "Display all
deployed strategies" -- "deployed" means configured, not necessarily
already filled.

Execution Details additionally pulls in (all from data that already
exists, nothing invented):
  - strategy_config JSON (metadata.strategy.strategy_config) -- rendered
    into human-readable entry/exit rule text via _describe_side(),
    instead of a hand-written English sentence.
  - live open position + native TP/SL straight from Bybit
    (execution.bybit_client.get_open_position) -- this execution module
    never stores open-order/position data in Postgres itself, it only
    lives on the exchange, so there is nothing to "join" here, only a
    live call, same pattern as wallets_live.py's balance fetch.
  - full stats (metrics + chart data) via
    crypto_pipeline.stats.calculator.compute_stats(), fed the same
    equity curve build_execution_equity_curve_from_ledger() already
    builds -- reuses the exact plot set (returns/drawdown/rolling
    sharpe/rolling volatility/monthly heatmap/yearly returns/
    distribution) stats/config.yaml already defines, no new plot code.
"""

import yaml
from pathlib import Path

from psycopg2 import sql

from crypto_pipeline.utils.db_utils import (
    get_execution_universe,
    get_execution_config,
    get_execution_state,
    get_execution_summary,
    build_execution_equity_curve_from_ledger,
    _execution_trades_table,
)
from crypto_pipeline.utils.metadata_utils import get_strategies
from crypto_pipeline.accounts.accounts_utils import get_account_api_key
from crypto_pipeline.execution.bybit_client import get_client, get_open_position
from crypto_pipeline.stats.calculator import compute_stats

_STATS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "crypto_pipeline" / "stats" / "config.yaml"
_stats_config_cache = None


def _stats_config() -> dict:
    """Loaded once per process -- same config every backtest/simulator run uses."""
    global _stats_config_cache
    if _stats_config_cache is None:
        with open(_STATS_CONFIG_PATH) as f:
            _stats_config_cache = yaml.safe_load(f)
    return _stats_config_cache


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
    """
    Render metadata.strategy.strategy_config["strategy"]["long"|"short"]
    (rule + conditions list, same shape as signals/strategies/*.yaml) into
    one readable line, e.g. "ind_sma_20 crosses above close" or
    "A AND B" for multiple conditions. Returns None if this strategy has
    no rule for that side (long-only/short-only strategies are common).
    """
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
    """
    Real Entry Logic / Exit Logic / indicators-used text, built from the
    actual stored strategy_config JSON -- not a hardcoded description.
    "Exit logic" here means the short-side (flip/close) rule if the
    strategy has one; take_profit/stop_loss are surfaced separately since
    they're already their own columns.
    """
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


def _live_bybit_position(account_name: str | None, conn, symbol: str) -> dict | None:
    """
    Live open position straight from Bybit (side, size, avg_price, native
    take_profit/stop_loss, created_time) for the wallet assigned to this
    pair -- this is the real "current position / native TP-SL" source,
    since execution/main.py never persists Bybit's own position rows in
    Postgres (see bybit_client.get_open_position's docstring). Returns
    None if no wallet is assigned, the wallet's credentials are missing,
    or Bybit reports flat/errors -- callers should treat None as "no live
    data available", not as an error.
    """
    if not account_name:
        return None
    wallet = get_account_api_key(conn, account_name)
    if wallet is None:
        return None
    try:
        client = get_client(api_key=wallet["api_key"], api_secret=wallet["api_secret"], demo=wallet["demo"])
        return get_open_position(client, symbol)
    except Exception:
        return None


def _current_strategy_for_pair(conn, exchange, symbol):
    """
    Same lookup execution/main.py's __main__ loop does: the one
    metadata.strategy row for this pair with execution_enabled=True.
    Returns None if zero or more-than-one are enabled (misconfigured or
    not yet assigned) -- callers show "unassigned" rather than guessing.
    """
    strategy_rows = get_strategies(conn, exchange=exchange, coin=symbol)
    enabled_rows = [s for s in strategy_rows if s.get("execution_enabled", True)]
    if len(enabled_rows) != 1:
        return None
    return enabled_rows[0]


def list_executions(conn) -> list[dict]:
    """
    One row per (exchange, symbol) pair registered in execution.config,
    enriched with its current strategy, wallet, state, and PnL.

    Takes a single connection -- execution.config/positions/*_trades,
    metadata.strategy, and accounts.api_keys all live in the same
    Postgres database (see crypto_pipeline.utils.db_utils/metadata_utils/
    accounts_utils, which each open their own connection with identical
    env vars purely for module-boundary clarity, not because they're
    different databases).
    """
    pairs = get_execution_universe(conn)
    rows = []

    for exchange, symbol in pairs:
        config = get_execution_config(conn, exchange, symbol)
        if config is None:
            continue

        strategy_row = _current_strategy_for_pair(conn, exchange, symbol)
        rows.append(_build_summary(conn, exchange, symbol, config, strategy_row))

    return rows


def get_execution_detail(conn, exchange, symbol):
    """
    Full detail for one pair: summary fields, risk config, strategy
    config (real entry/exit logic + TP/SL), live Bybit position, trades,
    and the full stats/plots bundle computed off the real trade ledger.
    """
    config = get_execution_config(conn, exchange, symbol)
    if config is None:
        return None

    strategy_row = _current_strategy_for_pair(conn, exchange, symbol)
    summary = _build_summary(conn, exchange, symbol, config, strategy_row)

    detail = dict(summary)
    detail["time_horizon"] = None
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
    detail["live_position"] = _live_bybit_position(summary.get("account_name"), conn, symbol)
    detail["stats"] = None

    if strategy_row is None:
        return detail

    strategy_name = strategy_row["strategy_name"]
    detail["time_horizon"] = strategy_row.get("time_horizon")

    exec_summary = get_execution_summary(conn, exchange, symbol, strategy_name)
    if exec_summary is not None:
        detail["total_net_profit"] = exec_summary["total_net_profit"]
        detail["total_trades"] = exec_summary["total_trades"]
        detail["win_loss"] = exec_summary["win_loss"]

    equity = build_execution_equity_curve_from_ledger(
        conn, exchange, symbol, strategy_name, config.get("initial_balance") or 0.0
    )
    if equity is not None:
        detail["equity_curve"] = [
            {"timestamp": ts, "balance": float(val)} for ts, val in equity.items()
        ]
        # Same compute_stats() the backtest/simulator paths use, fed this
        # execution's own real equity curve -- gives metrics (55+
        # quantstats stats) plus every configured plot's numeric data
        # (returns/cumulative returns, drawdown series + periods, rolling
        # sharpe, rolling volatility, monthly heatmap, yearly returns,
        # return distribution) for free, no separate chart code needed.
        try:
            detail["stats"] = compute_stats({"equity_curve": equity}, _stats_config())
        except Exception:
            # Too little history / a quantstats edge case shouldn't break
            # the whole page -- trades/equity curve above still render.
            detail["stats"] = None

    detail["trades"] = _list_trades(conn, exchange, symbol, strategy_name)

    return detail


def _build_summary(conn, exchange, symbol, config, strategy_row) -> dict:
    account_name = config.get("account_name")

    wallet_enabled = None
    if account_name:
        wallet = get_account_api_key(conn, account_name)
        wallet_enabled = wallet["enabled"] if wallet else None

    if strategy_row is None:
        return {
            "exchange": exchange,
            "symbol": symbol,
            "strategy_name": "—",
            "account_name": account_name,
            "wallet_enabled": wallet_enabled,
            "status": "unassigned",
            "position": None,
            "balance": config.get("initial_balance"),
            "cumulative_pnl": None,
            "daily_return_pct": None,
            "last_signal": None,
            "last_processed": None,
        }

    strategy_name = strategy_row["strategy_name"]
    state = get_execution_state(conn, exchange, symbol, strategy_name)

    if state is None:
        return {
            "exchange": exchange,
            "symbol": symbol,
            "strategy_name": strategy_name,
            "account_name": account_name,
            "wallet_enabled": wallet_enabled,
            "status": "never_run",
            "position": None,
            "balance": config.get("initial_balance"),
            "cumulative_pnl": None,
            "daily_return_pct": None,
            "last_signal": None,
            "last_processed": None,
        }

    position = state["position"]
    balance = state["balance"]
    initial_balance = config.get("initial_balance") or balance
    daily_return_pct = None
    if initial_balance:
        daily_return_pct = ((balance - initial_balance) / initial_balance) * 100.0

    if account_name and wallet_enabled is False:
        status = "paused"
    elif position is not None:
        status = "running"
    else:
        status = "flat"

    last_signal = None
    if position is not None:
        last_signal = f"{position['direction']} ({position.get('leaning') or 'open'})"

    return {
        "exchange": exchange,
        "symbol": symbol,
        "strategy_name": strategy_name,
        "account_name": account_name,
        "wallet_enabled": wallet_enabled,
        "status": status,
        "position": position,
        "balance": balance,
        "cumulative_pnl": state["cumulative_pnl"],
        "daily_return_pct": daily_return_pct,
        "last_signal": last_signal,
        "last_processed": state["last_processed"],
    }


def _list_trades(conn, exchange, symbol, strategy_name, limit: int = 200) -> list[dict]:
    """
    Most recent trades (open or closed) for this pair, newest entry
    first. No reader for this existed in db_utils.py (only writers) --
    added here rather than there since it's read-only/API-specific and
    doesn't change any write-path behavior.
    """
    table_name = _execution_trades_table(exchange, symbol, strategy_name)

    cursor = conn.cursor()
    qualified_name = sql.SQL(".").join(
        [sql.Identifier("execution"), sql.Identifier(table_name)]
    ).as_string(conn)
    cursor.execute(sql.SQL("SELECT to_regclass(%s)"), (qualified_name,))
    table_exists = cursor.fetchone()[0] is not None
    if not table_exists:
        cursor.close()
        return []

    cursor.execute(sql.SQL("""
        SELECT entry_date_time, direction, entry_price, quantity, take_profit, stop_loss,
               exit_date_time, exit_price, gross_pnl, commission, slippage, net_pnl,
               exit_reason, balance, status
        FROM {schema}.{table}
        ORDER BY entry_date_time DESC
        LIMIT %s
    """).format(schema=sql.Identifier("execution"), table=sql.Identifier(table_name)), (limit,))

    columns = [
        "entry_date_time", "direction", "entry_price", "quantity", "take_profit", "stop_loss",
        "exit_date_time", "exit_price", "gross_pnl", "commission", "slippage", "net_pnl",
        "exit_reason", "balance", "status",
    ]
    rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
    cursor.close()
    return rows
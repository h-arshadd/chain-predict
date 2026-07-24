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
"""

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


def get_execution_detail(conn, exchange, symbol) -> dict | None:
    """Full detail for one pair: same summary fields plus trades + equity curve."""
    config = get_execution_config(conn, exchange, symbol)
    if config is None:
        return None

    strategy_row = _current_strategy_for_pair(conn, exchange, symbol)
    summary = _build_summary(conn, exchange, symbol, config, strategy_row)

    detail = dict(summary)
    detail["time_horizon"] = None
    detail["initial_balance"] = config.get("initial_balance")
    detail["total_net_profit"] = None
    detail["total_trades"] = 0
    detail["win_loss"] = None
    detail["equity_curve"] = []
    detail["trades"] = []

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
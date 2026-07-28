"""
repos/backtests_repo.py
--------------------------
DB access + job runner for the Backtest Requests / Backtest Details
pages. Builds entirely on existing crypto_pipeline functions -- the
same get_data -> generate_signals -> run_backtest pipeline
backtest/main.py's __main__ block already runs for all 8 coins, just
scoped here to one specific metadata.strategy row (by strategy_id) and
triggered on demand instead of by a scheduled script.

Two real gaps this file had to close (see metadata_utils.py /
db_utils.py for the actual additions, not duplicated here):

  1. metadata.backtest existed but insert_backtest() was never called
     anywhere in the codebase, and had no status/lifecycle columns --
     every row would've looked "completed" the instant it was created.
     Added strategy_id + status/error/started_at/finished_at/
     result_summary columns (self-healing ALTER, see
     metadata_utils.create_backtest_table), and start_backtest/
     complete_backtest/fail_backtest to actually drive that lifecycle.

  2. backtest.{exchange}_{symbol} (insert_trades) is DROPPED and
     rebuilt on every run of that pair -- only the latest run per pair
     ever survives, with no backtest_id link at all. That's fine for
     the old script's "just re-run and look at the latest" use, but
     wrong for a "Backtest Requests" list where every request is
     supposed to keep its own results. Added insert_backtest_trades()/
     get_backtest_trades() (backtest.run_{backtest_id}, permanent, one
     table per run) and save_backtest_equity_curve()/
     get_backtest_equity_curve() alongside it.

run_backtest_job() is the actual work, meant to run inside a FastAPI
BackgroundTask (see routers/backtests.py) -- opens its own DB
connection (background tasks run after the request's own `conn`
dependency has already been closed), same pattern get_data()/
get_1m_data() use for the same reason.
"""

import yaml
from pathlib import Path
from datetime import datetime

import pandas as pd

from crypto_pipeline.utils.db_utils import (
    get_db_connection,
    get_candles_from_db,
    insert_backtest_trades,
    save_backtest_equity_curve,
    get_backtest_trades,
    get_backtest_equity_curve,
)
from crypto_pipeline.utils.metadata_utils import (
    get_strategy,
    create_backtest_table,
    insert_backtest,
    start_backtest,
    complete_backtest,
    fail_backtest,
    get_backtest,
    get_backtests,
)
from crypto_pipeline.data.data_downloader import get_data
from crypto_pipeline.signals.main import generate_signals
from crypto_pipeline.backtest.backtest import run_backtest
from crypto_pipeline.stats.calculator import compute_stats

from api.repos.executions_repo import _strategy_config_detail, _stats_config

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "crypto_pipeline" / "backtest" / "config.yaml"
_default_config_cache = None


def _default_backtest_config() -> dict:
    """
    backtest/config.yaml's own defaults (commission, slippage, TP/SL,
    position sizing, max_open_positions, entry/exit price convention) --
    used to fill in anything the request form doesn't override, same
    file backtest/main.py's script itself loads via load_config().
    """
    global _default_config_cache
    if _default_config_cache is None:
        with open(_DEFAULT_CONFIG_PATH) as f:
            _default_config_cache = yaml.safe_load(f)
    return _default_config_cache


def _parse_date(value):
    """
    Accepts a date-only ("2026-04-01") or full timestamp
    ("2026-04-01 00:00:00") string, or an already-parsed datetime --
    same two formats backtest/main.py's parse_backtest_dates() accepts.
    """
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {value!r}")


def _validate_backtest_dates(config: dict) -> None:
    """
    Reject nonsensical date ranges before a backtest is even queued,
    rather than letting run_backtest_job() silently truncate an
    end_date that's in the future down to whatever data happens to
    exist right now (get_data() falls back to a live exchange fetch
    for any gap between the DB and end_date, so a future end_date
    doesn't error -- it just quietly returns less data than requested
    and the run still reports "completed").

    Raises ValueError, which the router turns into a 400.
    """
    start_date = _parse_date(config["start_date"]) if config.get("start_date") not in (None, "now") else None
    end_date = _parse_date(config["end_date"]) if config.get("end_date") not in (None, "now") else None
    now = datetime.utcnow()

    if end_date is not None and end_date > now:
        raise ValueError(
            f"end_date {end_date:%Y-%m-%d} is in the future -- a backtest can only run "
            f"over historical data up to now ({now:%Y-%m-%d})."
        )
    if start_date is not None and end_date is not None and start_date >= end_date:
        raise ValueError("start_date must be before end_date.")


def create_backtest_request(conn, strategy_id: int, overrides: dict) -> dict:
    """
    Register a new backtest request (metadata.backtest, status
    'pending') and merge the request's overrides on top of backtest/
    config.yaml's defaults. Does NOT run the backtest itself -- the
    router hands the returned backtest_id to run_backtest_job() as a
    FastAPI BackgroundTask right after this returns, same "insert the
    row first, do the work after" order the frontend's Backtest
    Requests page needs to show a pending row immediately.

    overrides: whatever subset of backtest/config.yaml's keys the
    request form actually sent (start_date, end_date, initial_balance,
    commission, slippage, take_profit, stop_loss, position_size,
    allow_long, allow_short, max_open_positions) -- anything omitted
    falls back to the config.yaml default, exactly like running the
    script with no config changes would.

    Raises ValueError if strategy_id doesn't exist -- the router turns
    this into a 404.
    """
    strategy_row = get_strategy(conn, strategy_id)
    if strategy_row is None:
        raise ValueError(f"strategy_id {strategy_id} not found")

    # Idempotent -- CREATE TABLE IF NOT EXISTS + self-healing ALTER for
    # the strategy_id/status/error/started_at/finished_at/result_summary
    # columns this file added. Safe to call on every request; only ever
    # does real work the first time it runs against a given DB.
    create_backtest_table(conn)

    config = dict(_default_backtest_config())
    config.update({k: v for k, v in overrides.items() if v is not None})

    _validate_backtest_dates(config)

    backtest_id = insert_backtest(
        conn,
        strategy_name=strategy_row["strategy_name"],
        backtest_config=config,
        strategy_id=strategy_id,
        status="pending",
    )
    return get_backtest(conn, backtest_id)


def run_backtest_job(backtest_id: int):
    """
    The actual backtest run -- meant to be scheduled as a FastAPI
    BackgroundTask right after create_backtest_request() returns, not
    called directly inside a request handler (a real backtest can pull
    weeks of 1-minute candles and run quantstats over them; blocking a
    request on that would time out a browser tab for no reason).

    Opens its own DB connection since BackgroundTasks run after the
    triggering request (and its `conn` dependency) has already closed --
    same reason get_data()/get_1m_data() each open/close their own
    connection internally.

    On success: trade ledger -> backtest.run_{backtest_id}, full equity
    curve -> backtest.run_{backtest_id}_equity, metadata.backtest gets
    status='completed' + a small result_summary (final_balance,
    total_net_profit, total_trades, win_loss) so the Backtest Requests
    list has a headline number without joining the ledger table.

    On any failure (bad date range, no data in the DB for that window,
    an unexpected exception from the pipeline) metadata.backtest gets
    status='failed' + the real error message -- never silently drops a
    request or leaves it stuck at 'running' forever.
    """
    conn = get_db_connection()
    try:
        backtest_row = get_backtest(conn, backtest_id)
        if backtest_row is None:
            return  # request was deleted/never existed -- nothing to run

        start_backtest(conn, backtest_id)

        strategy_row = get_strategy(conn, backtest_row["strategy_id"]) if backtest_row["strategy_id"] else None
        if strategy_row is None:
            fail_backtest(conn, backtest_id, "Strategy no longer exists.")
            return

        config = dict(backtest_row["backtest_config"])
        config["start_date"] = _parse_date(config["start_date"])
        config["end_date"] = _parse_date(config["end_date"])

        exchange = strategy_row["exchange"]
        symbol = strategy_row["coin"]
        timeframe = strategy_row.get("time_horizon") or "1h"

        hourly_result = get_data(
            exchange=exchange,
            symbol=symbol,
            start_date=config["start_date"],
            end_date=config["end_date"],
            timeframe=timeframe,
        )
        ohlcv_resampled = hourly_result["resampled"]
        if ohlcv_resampled.empty:
            fail_backtest(conn, backtest_id, f"No {timeframe} data available for {exchange}/{symbol} in that date range.")
            return

        # Same shape build_signals() in backtest/main.py produces, just
        # fed this strategy's own DB-stored config instead of a yaml
        # file (see generate_signals()'s config_dict parameter).
        #
        # A strategy saved via the Strategy Builder (POST /api/strategies/
        # build) has a different strategy_config shape -- {"builder":
        # {"components": [...], "combine_rule": ...}} -- see
        # crypto_pipeline.strategy_builder.assemble.assemble_strategy_config.
        # generate_signals() doesn't understand that shape (it's not a
        # single strategy's indicator/condition config), so combined
        # strategies are routed through build_combined_signal() instead,
        # which re-resolves each component live against this run's own
        # OHLCV window (per STRATEGY_BUILDER_SPEC.md decision 5: "recompute
        # live, no new signal-cache table" -- the only durable/cached piece
        # is an ML model's own signal series, read via get_model_signals).
        raw_config = strategy_row["strategy_config"]
        if isinstance(raw_config, dict) and "builder" in raw_config:
            from crypto_pipeline.strategy_builder.assemble import build_combined_signal
            from crypto_pipeline.utils.db_utils import get_model_signals
            from crypto_pipeline.utils.metadata_utils import get_playbook

            builder = raw_config["builder"]

            # assemble_strategy_config() deliberately stores only
            # playbook_id/strategy_name per playbook component (a lean,
            # inspectable reference -- see its own docstring), NOT the
            # full strategy_config JSON, to avoid duplicating the whole
            # config into every saved combined strategy. That means
            # resolve_component_signal() (which needs the actual
            # strategy_config to call generate_signals()) can't work off
            # the saved builder dict as-is -- each playbook component has
            # to be re-hydrated here, by looking its playbook_id back up
            # in metadata.playbook, right before combining. If a playbook
            # entry was deleted after this strategy was saved, fail loudly
            # with a clear message rather than a bare KeyError.
            hydrated_components = []
            for c in builder["components"]:
                if c["kind"] == "playbook":
                    playbook_row = get_playbook(conn, c["playbook_id"])
                    if playbook_row is None:
                        fail_backtest(
                            conn, backtest_id,
                            f"Playbook entry {c['playbook_id']} ({c.get('strategy_name')!r}) "
                            "no longer exists -- this strategy can't be re-backtested."
                        )
                        return
                    hydrated_components.append({
                        **c,
                        "strategy_config": playbook_row["strategy_config"],
                    })
                else:
                    hydrated_components.append(c)

            ohlcv_indexed = ohlcv_resampled.set_index("datetime")

            def _get_model_signals(run_id):
                return get_model_signals(conn, run_id, start_date=config["start_date"], end_date=config["end_date"])

            signal_series = build_combined_signal(
                hydrated_components, ohlcv_indexed, builder["combine_rule"],
                weights=builder.get("weights"), threshold=builder.get("threshold"),
                get_model_signals=_get_model_signals,
            )
            combined = ohlcv_resampled.copy().set_index("datetime")
            combined["signal"] = signal_series
            combined = combined.dropna(subset=["signal"]).reset_index()
            signals = combined[["datetime", "signal"]]
        else:
            indicator_df, condition_df, signal_series = generate_signals(
                ohlcv_resampled, config_dict=raw_config
            )
            combined = pd.concat([indicator_df, condition_df], axis=1)
            combined["signal"] = signal_series
            combined = combined.dropna().reset_index(drop=True)
            signals = combined[["datetime", "signal"]]

        ohlcv_1m = get_candles_from_db(conn, exchange, symbol, config["start_date"], config["end_date"])
        if ohlcv_1m.empty:
            fail_backtest(conn, backtest_id, f"No 1-minute data available for {exchange}/{symbol} in that date range.")
            return

        # get_data()/get_candles_from_db() don't error on a partially
        # covered range -- they just return whatever data exists, which
        # could be a small slice of what was actually requested (e.g. a
        # start_date before the DB's earliest stored candle, or an
        # end_date past what's been ingested/lives in the future). Check
        # actual coverage against the request so a partial window fails
        # loudly instead of quietly running on less data than asked for.
        actual_start = ohlcv_1m["datetime"].min()
        actual_end = ohlcv_1m["datetime"].max()
        missing_start = actual_start > config["start_date"] + pd.Timedelta(days=1)
        missing_end = actual_end < config["end_date"] - pd.Timedelta(days=1)
        if missing_start or missing_end:
            fail_backtest(
                conn, backtest_id,
                f"Requested {config['start_date']:%Y-%m-%d} \u2192 {config['end_date']:%Y-%m-%d}, "
                f"but {exchange}/{symbol} only has data from {actual_start:%Y-%m-%d} to {actual_end:%Y-%m-%d}."
            )
            return

        result = run_backtest(ohlcv_1m, signals, config)

        insert_backtest_trades(conn, backtest_id, result["trade_ledger"])
        save_backtest_equity_curve(conn, backtest_id, result["equity_curve"])

        complete_backtest(conn, backtest_id, {
            "final_balance": result["final_balance"],
            "total_net_profit": result["total_net_profit"],
            "total_trades": result["total_trades"],
            "win_loss": result["win_loss"],
        })

    except Exception as exc:
        fail_backtest(conn, backtest_id, str(exc))
    finally:
        conn.close()


def list_backtests(conn) -> list[dict]:
    """
    Every backtest request, newest first -- pending/running/completed/
    failed all included, same rows the Backtest Requests page splits
    into its four buckets by `status`.
    """
    return get_backtests(conn)


def get_backtest_detail(conn, backtest_id: int) -> dict | None:
    """
    Full detail for one backtest run: request config, lifecycle status,
    strategy config (real entry/exit rules, same _strategy_config_detail
    executions_repo already builds), trade list, and the full
    stats/plots bundle (compute_stats fed this run's own saved equity
    curve) -- only populated once status is 'completed'.
    """
    backtest_row = get_backtest(conn, backtest_id)
    if backtest_row is None:
        return None

    strategy_row = get_strategy(conn, backtest_row["strategy_id"]) if backtest_row["strategy_id"] else None

    detail = {
        "backtest_id": backtest_row["backtest_id"],
        "strategy_id": backtest_row["strategy_id"],
        "strategy_name": backtest_row["strategy_name"],
        "exchange": strategy_row["exchange"] if strategy_row else None,
        "coin": strategy_row["coin"] if strategy_row else None,
        "time_horizon": strategy_row.get("time_horizon") if strategy_row else None,
        "status": backtest_row["status"],
        "error": backtest_row["error"],
        "backtest_config": backtest_row["backtest_config"],
        "strategy_config": _strategy_config_detail(strategy_row),
        "started_at": backtest_row["started_at"],
        "finished_at": backtest_row["finished_at"],
        "created_at": backtest_row["created_at"],
        "final_balance": None,
        "total_net_profit": None,
        "total_trades": 0,
        "win_loss": None,
        "trades": [],
        "equity_curve": [],
        "stats": None,
    }

    if backtest_row["status"] != "completed":
        return detail

    result_summary = backtest_row.get("result_summary") or {}
    detail["final_balance"] = result_summary.get("final_balance")
    detail["total_net_profit"] = result_summary.get("total_net_profit")
    detail["total_trades"] = result_summary.get("total_trades", 0)
    detail["win_loss"] = result_summary.get("win_loss")

    trades_df = get_backtest_trades(conn, backtest_id)
    if not trades_df.empty:
        detail["trades"] = trades_df.to_dict(orient="records")

    equity = get_backtest_equity_curve(conn, backtest_id)
    if equity is not None:
        # Downsample for the chart -- long date ranges at 1-minute resolution
        # can produce a point per minute (e.g. a 2-year range is 1M+ rows),
        # which is far more than a line chart needs and is enough to freeze
        # the browser trying to render it. Cap at MAX_EQUITY_POINTS evenly
        # spaced points; stats below still run on the full, un-downsampled
        # series so Sharpe/drawdown/etc. stay accurate.
        MAX_EQUITY_POINTS = 2000
        if len(equity) > MAX_EQUITY_POINTS:
            step = len(equity) // MAX_EQUITY_POINTS
            equity_for_chart = equity.iloc[::step]
        else:
            equity_for_chart = equity

        detail["equity_curve"] = [
            {"timestamp": ts, "balance": float(val)} for ts, val in equity_for_chart.items()
        ]
        try:
            detail["stats"] = compute_stats({"equity_curve": equity}, _stats_config())
        except Exception:
            # Too little history / a quantstats edge case shouldn't break
            # the whole page -- trades/equity curve above still render.
            detail["stats"] = None

    return detail
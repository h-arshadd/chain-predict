"""
bybit_client.py
----------------
Thin wrapper around Bybit's REST API for the execution folder. Jobs:

  1. get_live_ohlcv() -- pull recent 1-minute candles directly from
     Bybit (not the DB -- execution trades off live exchange data, not
     whatever's been backfilled into binance.*/bybit.* tables).
  2. place_market_order() -- send a real market order when a position
     opens or closes, then look up what it actually filled at
     (avgPrice / cumExecQty / cumExecFee) and return that -- a market
     order's ack response does NOT include fill data, only orderId, so
     the fill has to be fetched back separately right after placing.

Uses pybit (Bybit's official Python SDK) since it already handles request
signing. Install with: pip install pybit

Symbol format: Bybit expects e.g. "BTCUSDT", not "btc" -- to_bybit_symbol()
does that conversion so the rest of execution/ can keep using the same
lowercase "btc" convention as simulator/backtest/signals.
"""

import os
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()

# Cache of qtyStep per Bybit symbol, so we only hit get_instruments_info()
# once per symbol instead of on every single order.
_QTY_STEP_CACHE: dict[str, Decimal] = {}


def get_qty_step(client: HTTP, bybit_symbol: str) -> Decimal:
    """
    Bybit rejects any order qty that isn't an exact multiple of the
    instrument's qtyStep (ErrCode 10001 "Qty invalid") -- e.g. BTCUSDT
    linear has qtyStep "0.001", so 0.0149 is rejected but 0.014 or 0.015
    is accepted. Fetched live (not hardcoded) so this stays correct if a
    symbol's lot size ever changes or a new symbol is added.
    """
    if bybit_symbol not in _QTY_STEP_CACHE:
        info = client.get_instruments_info(category="linear", symbol=bybit_symbol)
        lot_filter = info["result"]["list"][0]["lotSizeFilter"]
        _QTY_STEP_CACHE[bybit_symbol] = Decimal(lot_filter["qtyStep"])
    return _QTY_STEP_CACHE[bybit_symbol]


def round_to_qty_step(quantity: float, qty_step: Decimal) -> str:
    """
    Round DOWN to the instrument's qtyStep, as Decimal to avoid float
    binary-representation drift (e.g. 0.0149 rendering as
    0.014900000000000001). Rounding down (not to nearest) guarantees we
    never send a qty larger than what was computed -- never risk
    over-ordering relative to the sized position.
    """
    step = Decimal(str(quantity)).quantize(qty_step, rounding=ROUND_DOWN)
    return str(step)


def to_bybit_symbol(symbol: str) -> str:
    """'btc' -> 'BTCUSDT'. Execution is USDT-perpetual only for now."""
    return f"{symbol.upper()}USDT"


def get_client(api_key: str, api_secret: str, demo: bool = False) -> HTTP:
    """
    Build a pybit HTTP session. Called once in main.py and reused.

    demo=True routes to Bybit's isolated Demo Trading sub-account
    (simulated balance, real production domain under the hood).
    demo=False is real production trading with real funds. Testnet
    support was removed entirely per team decision -- there is no
    testnet option here anymore, intentionally.
    """
    return HTTP(api_key=api_key, api_secret=api_secret, demo=demo)


def get_client_from_env() -> HTTP:
    """
    Build a pybit HTTP session from BYBIT_API_KEY / BYBIT_API_SECRET /
    BYBIT_DEMO in .env -- same pattern db_utils.get_db_connection() and
    metadata_utils.get_db_connection() already use for DB credentials.
    Secrets never go in execution/config.yaml or the DB -- .env only.

    Testnet has been removed entirely (per team decision) -- there are
    now only two environments, and each requires its OWN API key
    generated while your account is switched into that specific mode; a
    key from one will not work against the other:
      - demo       (BYBIT_DEMO=true):  api.bybit.com under the hood but
                     routed to an isolated demo sub-account with
                     simulated balance -- generate this key from the
                     Bybit website while your account is switched into
                     "Demo Trading" mode (hover the profile icon ->
                     Demo Trading), NOT from your regular API management
                     page. Note demo trading doesn't support every
                     endpoint production does (it's meant for practice,
                     not full parity).
      - production (BYBIT_DEMO=false or unset): api.bybit.com, REAL
                     funds, REAL orders.

    BYBIT_DEMO: "true"/"false" (case-insensitive), defaults to "false".
    If left unset, this connects to real production Bybit with real
    money -- there is no "fails safe" default anymore now that testnet
    is gone; make sure .env explicitly sets BYBIT_DEMO=true until you're
    actually ready for production.
    """
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    demo = os.getenv("BYBIT_DEMO", "false").strip().lower() == "true"

    if not api_key or not api_secret:
        raise RuntimeError(
            "BYBIT_API_KEY / BYBIT_API_SECRET not set in .env -- add them before running execution/main.py."
        )

    return get_client(api_key=api_key, api_secret=api_secret, demo=demo)


def get_live_ohlcv(client: HTTP, symbol: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch the most recent `limit` 1-minute candles for `symbol` from
    Bybit directly (category="linear" -- USDT perpetuals).

    Returns a DataFrame with columns: datetime, open, high, low, close,
    volume -- sorted oldest to newest, same shape/column names get_data()
    returns elsewhere in the pipeline (see data_downloader.get_data()),
    so build_resampled_signals() works unchanged.
    """
    bybit_symbol = to_bybit_symbol(symbol)
    response = client.get_kline(
        category="linear",
        symbol=bybit_symbol,
        interval="1",
        limit=limit,
    )
    rows = response["result"]["list"]

    df = pd.DataFrame(
        rows,
        columns=["start", "open", "high", "low", "close", "volume", "turnover"],
    )
    df["datetime"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True).dt.tz_localize(None)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def _extract_fill(row: dict) -> dict:
    return {
        "avg_price": float(row["avgPrice"]),
        "filled_qty": float(row["cumExecQty"]),
        "fee": float(row["cumExecFee"]),
        "order_status": row["orderStatus"],
    }


def get_order_fill(client: HTTP, bybit_symbol: str, order_id: str,
                    max_attempts: int = 10, poll_seconds: float = 1.0) -> dict:
    """
    Look up what an order actually filled at. place_order()'s own response
    only echoes back orderId (no avgPrice/cumExecQty) -- the real fill has
    to be read back separately. Market orders fill almost immediately,
    but propagation to the history endpoints can lag by a second or two
    (more on demo trading, which doesn't have full endpoint parity with
    production) -- so this polls, and checks TWO endpoints each attempt:

      1. get_open_orders(openOnly=0) -- the "realtime" order endpoint.
         Despite the name, passing openOnly=0 also returns recently
         closed orders, and this endpoint updates faster than order
         history right after a fill.
      2. get_order_history() -- the historical endpoint, meant for
         orders that already fully settled. Slower to update, but used
         as a fallback each attempt in case the order already aged out
         of the realtime endpoint's short window.

    Returns dict: avg_price, filled_qty, fee (all float), order_status.
    Raises RuntimeError if the order can't be found or never gets filled
    within max_attempts.
    """
    bybit_symbol = to_bybit_symbol(bybit_symbol) if not bybit_symbol.endswith("USDT") else bybit_symbol

    for attempt in range(max_attempts):
        response = client.get_open_orders(
            category="linear",
            symbol=bybit_symbol,
            orderId=order_id,
            openOnly=0,
            limit=1,
        )
        rows = response["result"]["list"]

        if not rows:
            response = client.get_order_history(
                category="linear",
                symbol=bybit_symbol,
                orderId=order_id,
                limit=1,
            )
            rows = response["result"]["list"]

        if rows:
            row = rows[0]
            status = row["orderStatus"]
            if status == "Filled":
                return _extract_fill(row)
            if status in ("Cancelled", "Rejected"):
                raise RuntimeError(
                    f"Order {order_id} for {bybit_symbol} ended in status {status!r} -- not filled."
                )

        if attempt < max_attempts - 1:
            time.sleep(poll_seconds)

    raise RuntimeError(
        f"Order {order_id} for {bybit_symbol} did not reach 'Filled' status after "
        f"{max_attempts} attempt(s) (~{max_attempts * poll_seconds:.0f}s) -- check Bybit manually, "
        f"DB state was not updated. If this keeps happening on demo trading, confirm the order "
        f"actually filled on the Bybit website (Demo Trading -> Order History) -- demo accounts "
        f"don't support every endpoint with full parity to production."
    )


def place_market_order(client: HTTP, symbol: str, direction: str, quantity: float) -> dict:
    """
    Send a real market order on Bybit, then read back its actual fill.

    direction : "long" or "short" -- "long" opens/closes toward Buy,
        "short" opens/closes toward Sell. main.py calls this once to
        open a position and once to close it -- the side passed in is
        whichever direction the trade needs at that moment.
    quantity  : float -- desired order size in base currency (e.g. BTC
        amount), from the sizing math in simulator.py.

    take_profit/stop_loss are NOT sent here. They used to be attached
    to this same market order, computed off the pre-fill candle-open
    estimate -- but a market order's real fill price can move away from
    that estimate by the time Bybit processes it, and if price moves
    enough, the pre-computed SL/TP can end up on the wrong side of the
    real fill (Bybit then rejects the whole order with e.g. "StopLoss
    should be lower than base_price" for a Buy). TP/SL are now attached
    AFTER the fill, priced off the real fill price, via
    set_trading_stop() -- see main.py's _open_live_position().

    Returns dict: order_id, side, avg_price, filled_qty, fee -- the REAL
    numbers Bybit filled at, not the requested quantity or any candle
    price. main.py must build the ledger from this dict, not from
    step_candle()'s paper-fill numbers, since a market order can fill at
    a different price/size than requested (slippage, partial fill).

    Raises if the API call itself errors (network/auth) or if the fill
    can't be confirmed -- main.py should not swallow that silently,
    since a failed/uncertain order means the DB state and the real
    exchange position may have gone out of sync.
    """
    bybit_symbol = to_bybit_symbol(symbol)
    side = "Buy" if direction == "long" else "Sell"

    qty_step = get_qty_step(client, bybit_symbol)
    qty_str = round_to_qty_step(quantity, qty_step)

    if Decimal(qty_str) <= 0:
        # position_size in execution.config produced a quantity smaller
        # than one qtyStep (e.g. BTCUSDT's 0.001) -- rounding down to
        # Bybit's lot size collapsed it to zero. Fail loudly: this means
        # execution.config's position_size is too small for current
        # price/leverage and needs to be raised.
        raise ValueError(
            f"Computed order qty {quantity} rounds down to {qty_str} for {bybit_symbol}, "
            f"below its qtyStep ({qty_step}). Increase position_size in execution.config."
        )

    order_kwargs = dict(
        category="linear",
        symbol=bybit_symbol,
        side=side,
        orderType="Market",
        qty=qty_str,
    )

    order_response = client.place_order(**order_kwargs)

    order_id = order_response["result"]["orderId"]
    fill = get_order_fill(client, bybit_symbol, order_id)

    return {
        "order_id": order_id,
        "side": side,
        "avg_price": fill["avg_price"],
        "filled_qty": fill["filled_qty"],
        "fee": fill["fee"],
    }


def set_trading_stop(client: HTTP, symbol: str, take_profit: float = None,
                      stop_loss: float = None) -> dict:
    """
    Attach native exchange-side TP/SL to a position that's already open,
    via Bybit's POST /v5/position/trading-stop endpoint (pybit:
    client.set_trading_stop(...)).

    Called right after place_market_order()'s fill is confirmed, with
    TP/SL computed off the REAL fill price (position["entry_price"]),
    not the pre-fill candle-open estimate that used to be embedded in
    the opening order itself -- that stale price could cross the live
    market by fill time and get the whole order rejected (e.g. "StopLoss
    should be lower than base_price" for a Buy, when the market had
    moved between candle-open and actual fill).

    take_profit/stop_loss : float, optional -- absolute price levels.
    Passing both is normal for an opening position; pass only one to
    update just that side. Bybit registers these as native exchange-
    side TP/SL on the position (same as the "+ Add" button under TP/SL
    on the Positions tab) -- Bybit's own engine then watches price and
    auto-closes on hit, independent of whether this script is running.

    Raises if the API call itself errors -- a position opened without
    its protective TP/SL attached is a real risk, so this should not be
    swallowed silently by the caller.
    """
    bybit_symbol = to_bybit_symbol(symbol)

    trading_stop_kwargs = dict(
        category="linear",
        symbol=bybit_symbol,
        positionIdx=0,
    )
    if take_profit is not None:
        trading_stop_kwargs["takeProfit"] = str(take_profit)
    if stop_loss is not None:
        trading_stop_kwargs["stopLoss"] = str(stop_loss)

    return client.set_trading_stop(**trading_stop_kwargs)


def get_open_position(client: HTTP, symbol: str) -> dict:
    """
    Ask Bybit directly whether a position is currently open for this
    symbol -- used by run_execution()'s reconciliation step to detect a
    Bybit-side auto-close (native TP/SL hit) that happened between runs,
    which this script's own candle walk would otherwise have no way of
    knowing about. Also used to ADOPT a live Bybit position into
    execution.positions when our own DB state has drifted out of sync
    with Bybit (e.g. a previous run placed a real order and then crashed
    before saving state) -- see run_execution()'s DB-vs-Bybit check.

    Returns None if flat (size == 0 or no row), else a dict: side
    ("Buy"/"Sell"), size (float), avg_price (float), mark_price (float
    or None), take_profit (float or None), stop_loss (float or None),
    created_time (naive UTC datetime or None) -- Bybit's own view of the
    currently open position, not this script's DB state. take_profit/
    stop_loss/created_time come straight off the same position row
    (Bybit returns them whenever they were set natively at order time,
    same as place_market_order does). mark_price is Bybit's live mark
    (used for unrealized PnL, not just the entry fill) -- same response,
    no extra call, it was previously read off this row and discarded.
    """
    bybit_symbol = to_bybit_symbol(symbol)
    response = client.get_positions(category="linear", symbol=bybit_symbol)
    rows = response["result"]["list"]

    if not rows or float(rows[0]["size"]) == 0:
        return None

    row = rows[0]

    def _f(key):
        val = row.get(key)
        return float(val) if val not in (None, "") else None

    created_time_ms = row.get("createdTime")
    created_time = utcnow_from_ms(int(created_time_ms)) if created_time_ms not in (None, "", "0") else None

    return {
        "side": row["side"],
        "size": float(row["size"]),
        "avg_price": float(row["avgPrice"]),
        "mark_price": _f("markPrice"),
        "take_profit": _f("takeProfit"),
        "stop_loss": _f("stopLoss"),
        "created_time": created_time,
    }


def get_last_closed_pnl(client: HTTP, symbol: str) -> dict:
    """
    Look up the most recent closed-PnL record for this symbol -- used
    when run_execution() detects Bybit auto-closed a position (native
    TP/SL hit) that this script didn't itself close, so the trade ledger
    can be built from Bybit's REAL exit fill/fee, not a guess.

    Returns dict: exit_price (avgExitPrice), exit_time (updatedTime, as
    a naive UTC datetime), closed_pnl (float), exit_type ("TakeProfit"/
    "StopLoss"/etc, from closedPnl's own reason). Raises RuntimeError if
    Bybit has no closed-PnL record yet for this symbol (e.g. called
    before the close has propagated) -- caller should not silently
    fabricate a fill in that case, same reasoning as get_order_fill().
    """
    bybit_symbol = to_bybit_symbol(symbol)
    response = client.get_closed_pnl(category="linear", symbol=bybit_symbol, limit=1)
    rows = response["result"]["list"]

    if not rows:
        raise RuntimeError(
            f"No closed-PnL record found for {bybit_symbol} -- Bybit shows the position "
            f"closed but hasn't surfaced the fill yet. Check Bybit manually before retrying."
        )

    row = rows[0]
    return {
        "exit_price": float(row["avgExitPrice"]),
        "exit_time": utcnow_from_ms(int(row["updatedTime"])),
        "closed_pnl": float(row["closedPnl"]),
        "exit_type": row.get("execType", "unknown"),
    }


def utcnow_from_ms(ms: int) -> datetime:
    """Convert a Bybit millisecond timestamp string/int into a naive UTC datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(tzinfo=None)


def utcnow() -> datetime:
    """Plain UTC now, naive (matches how datetime columns are stored elsewhere)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
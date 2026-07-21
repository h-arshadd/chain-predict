"""
bybit_client.py
----------------
Thin wrapper around Bybit's REST API for the execution folder. Two jobs
only:

  1. get_live_ohlcv() -- pull recent 1-minute candles directly from
     Bybit (not the DB -- execution trades off live exchange data, not
     whatever's been backfilled into binance.*/bybit.* tables).
  2. place_market_order() -- send a real market order when a position
     opens or closes.

Uses pybit (Bybit's official Python SDK) since it already handles request
signing. Install with: pip install pybit

Symbol format: Bybit expects e.g. "BTCUSDT", not "btc" -- to_bybit_symbol()
does that conversion so the rest of execution/ can keep using the same
lowercase "btc" convention as simulator/backtest/signals.
"""

import os
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
    is accepted. round(quantity, 6) in place_market_order() is NOT the
    same thing as rounding to qtyStep, which is why that error happens.
    Fetched live (not hardcoded) so this stays correct if a symbol's
    lot size ever changes or a new symbol is added.
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
    never send a qty larger than step_candle() computed -- never risk
    over-ordering relative to the sized position.
    """
    step = Decimal(str(quantity)).quantize(qty_step, rounding=ROUND_DOWN)
    return str(step)


def to_bybit_symbol(symbol: str) -> str:
    """'btc' -> 'BTCUSDT'. Execution is USDT-perpetual only for now."""
    return f"{symbol.upper()}USDT"


def get_client(api_key: str, api_secret: str, testnet: bool) -> HTTP:
    """Build a pybit HTTP session. Called once in main.py and reused."""
    return HTTP(api_key=api_key, api_secret=api_secret, testnet=testnet)


def get_client_from_env() -> HTTP:
    """
    Build a pybit HTTP session from BYBIT_API_KEY / BYBIT_API_SECRET /
    BYBIT_TESTNET in .env -- same pattern db_utils.get_db_connection() and
    metadata_utils.get_db_connection() already use for DB credentials.
    Secrets never go in execution/config.yaml or the DB -- .env only.

    BYBIT_TESTNET: "true"/"false" (case-insensitive), defaults to "true"
    so a missing/misconfigured .env fails safe onto testnet rather than
    accidentally trading live.
    """
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    testnet = os.getenv("BYBIT_TESTNET", "true").strip().lower() != "false"

    if not api_key or not api_secret:
        raise RuntimeError(
            "BYBIT_API_KEY / BYBIT_API_SECRET not set in .env -- add them before running execution/main.py."
        )

    return get_client(api_key=api_key, api_secret=api_secret, testnet=testnet)


def get_live_ohlcv(client: HTTP, symbol: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch the most recent `limit` 1-minute candles for `symbol` from
    Bybit directly (category="linear" -- USDT perpetuals).

    Returns a DataFrame with columns: datetime, open, high, low, close,
    volume -- sorted oldest to newest, same shape/column names get_data()
    returns elsewhere in the pipeline (see data_downloader.get_data()),
    so build_resampled_signals() / step_candle() work unchanged.
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


def place_market_order(client: HTTP, symbol: str, direction: str, quantity: float) -> dict:
    """
    Send a real market order on Bybit.

    direction : "long" or "short" -- "long" opens/closes toward Buy,
        "short" opens/closes toward Sell. main.py calls this once to
        open a position and once to close it, same as it calls
        simulator.step_candle() for both -- the side passed in is
        whichever direction the trade needs at that moment (see main.py).
    quantity  : float -- order size in base currency (e.g. BTC amount),
        same "quantity" step_candle()/simulator.py already computes.

    Returns Bybit's raw order response dict. Raises if the API call
    itself errors (network/auth) -- main.py should not swallow that
    silently, since a failed order means the DB state and the real
    exchange position have gone out of sync.
    """
    bybit_symbol = to_bybit_symbol(symbol)
    side = "Buy" if direction == "long" else "Sell"

    qty_step = get_qty_step(client, bybit_symbol)
    qty_str = round_to_qty_step(quantity, qty_step)

    if Decimal(qty_str) <= 0:
        # position_size in execution.config produced a quantity smaller
        # than one qtyStep (e.g. BTCUSDT's 0.001) -- rounding down to
        # Bybit's lot size collapsed it to zero. Sending qty="0.000"
        # would just trade a different flavor of "Qty invalid" than the
        # one this fix addresses, so fail loudly instead: this means
        # execution.config's position_size is too small for current
        # price/leverage and needs to be raised.
        raise ValueError(
            f"Computed order qty {quantity} rounds down to {qty_str} for {bybit_symbol}, "
            f"below its qtyStep ({qty_step}). Increase position_size in execution.config."
        )

    return client.place_order(
        category="linear",
        symbol=bybit_symbol,
        side=side,
        orderType="Market",
        qty=qty_str,
    )


def utcnow() -> datetime:
    """Plain UTC now, naive (matches how datetime columns are stored elsewhere)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
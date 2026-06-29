"""
talib_indicators.py
--------------------
TA-Lib based technical indicators for crypto OHLCV data.

Each function:
  - Takes a DataFrame with columns: open, high, low, close, volume
  - Returns a pd.Series (or dict of pd.Series for multi-output indicators)
  - Applies .shift(1) to every output to prevent look-ahead bias
    (row N only sees data from candles 0..N-1)

Column usage per indicator group:
  - Most overlap studies use `close`
  - Price-range indicators (MIDPRICE, SAR, SAREXT) use `high` and `low`
  - BBANDS, DEMA, EMA, etc. use `close`

Installation
------------
TA-Lib requires the underlying C library before the Python wrapper.

Linux / WSL:
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib && ./configure --prefix=/usr && make && sudo make install
    pip install ta-lib

Windows:
    Download the .whl from https://github.com/cgohlke/talib-build/releases
    pip install <downloaded_file>.whl

After the C library is installed:
    pip install ta-lib pandas numpy
"""

import pandas as pd
import talib

# ---------------------------------------------------------------------------
# Overlap Studies
# ---------------------------------------------------------------------------
def overlap_bbands(df, period=20, nbdev_up=2.0, nbdev_dn=2.0, matype=0):
    """
    Bollinger Bands — upper / middle / lower bands around a moving average.

    Multi-output: returns a dict with keys 'upper', 'middle', 'lower'.
    Standard quant convention: shift all three bands by 1 so the signal
    for bar N is built from bars 0..N-1 only.

    Args:
        period:    lookback window (default 20)
        nbdev_up:  std dev multiplier for upper band (default 2.0)
        nbdev_dn:  std dev multiplier for lower band (default 2.0)
        matype:    moving average type (0=SMA, 1=EMA, ...)
    """
    close = df["close"].values
    upper, middle, lower = talib.BBANDS(
        close, timeperiod=period, nbdevup=nbdev_up, nbdevdn=nbdev_dn, matype=matype
    )
    idx = df.index
    return {
        "upper":  pd.Series(upper,  index=idx).shift(1),
        "middle": pd.Series(middle, index=idx).shift(1),
        "lower":  pd.Series(lower,  index=idx).shift(1),
    }
    
def overlap_dema(df, period=30):
    """
    Double Exponential Moving Average — reduces EMA lag by doubling the EMA
    and subtracting an EMA of the EMA: DEMA = 2*EMA(n) - EMA(EMA(n)).

    Args:
        period: lookback window (default 30)
    """
    close = df["close"].values
    result = talib.DEMA(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def overlap_ema(df, period=20):
    """
    Exponential Moving Average — weights recent prices more heavily than SMA.

    Args:
        period: lookback window (default 20)
    """
    close = df["close"].values
    result = talib.EMA(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def overlap_ht_trendline(df):
    """
    Hilbert Transform – Instantaneous Trendline.
    Adaptive cycle-based trendline; no period parameter needed.
    Applied to close.
    """
    close = df["close"].values
    result = talib.HT_TRENDLINE(close)
    return pd.Series(result, index=df.index).shift(1)


def overlap_kama(df, period=30):
    """
    Kaufman Adaptive Moving Average — adapts its speed to market noise.
    Slow in choppy markets, fast in trending ones.

    Args:
        period: lookback window (default 30)
    """
    close = df["close"].values
    result = talib.KAMA(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def overlap_ma(df, period=30, matype=0):
    """
    Generic Moving Average — wraps all TA-Lib MA types via matype parameter.

    matype values:
        0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3

    Args:
        period: lookback window (default 30)
        matype: MA type integer (default 0 = SMA)
    """
    close = df["close"].values
    result = talib.MA(close, timeperiod=period, matype=matype)
    return pd.Series(result, index=df.index).shift(1)


def overlap_mama(df, fastlimit=0.5, slowlimit=0.05):
    """
    MESA Adaptive Moving Average — dual output: MAMA and FAMA (following MA).

    Multi-output: returns a dict with keys 'mama' and 'fama'.
    Both are shifted by 1 to prevent look-ahead.

    Args:
        fastlimit: upper alpha limit (default 0.5)
        slowlimit: lower alpha limit (default 0.05)
    """
    close = df["close"].values
    mama, fama = talib.MAMA(close, fastlimit=fastlimit, slowlimit=slowlimit)
    idx = df.index
    return {
        "mama": pd.Series(mama, index=idx).shift(1),
        "fama": pd.Series(fama, index=idx).shift(1),
    }


def overlap_mavp(df, periods_col, minperiod=2, maxperiod=30, matype=0):
    """
    Moving Average with Variable Period — each row can use a different window.
    Useful when the period itself is a computed signal (e.g. ATR-based).

    Args:
        periods_col: name of the DataFrame column holding per-row period values
        minperiod:   floor for period values (default 2)
        maxperiod:   ceiling for period values (default 30)
        matype:      MA type (default 0 = SMA)
    """
    close = df["close"].values
    periods = df[periods_col].astype(float).values
    result = talib.MAVP(
        close, periods, minperiod=minperiod, maxperiod=maxperiod, matype=matype
    )
    return pd.Series(result, index=df.index).shift(1)


def overlap_midpoint(df, period=14):
    """
    MidPoint over Period — (highest_close + lowest_close) / 2 over N bars.
    Applied to close.

    Args:
        period: lookback window (default 14)
    """
    close = df["close"].values
    result = talib.MIDPOINT(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def overlap_midprice(df, period=14):
    """
    Midpoint Price over Period — (highest_high + lowest_low) / 2 over N bars.
    Uses high and low columns (price range, not just close).

    Args:
        period: lookback window (default 14)
    """
    high, low = df["high"].values, df["low"].values
    result = talib.MIDPRICE(high, low, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def overlap_sar(df, acceleration=0.02, maximum=0.2):
    """
    Parabolic SAR — trailing stop/reversal signal that flips above/below price.
    Uses high and low; no close needed.

    Args:
        acceleration: AF starting value (default 0.02)
        maximum:      AF ceiling (default 0.2)
    """
    high, low = df["high"].values, df["low"].values
    result = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
    return pd.Series(result, index=df.index).shift(1)


def overlap_sarext(
    df,
    startvalue=0.0,
    offsetonreverse=0.0,
    accelerationinitlong=0.02,
    accelerationlong=0.02,
    accelerationmaxlong=0.2,
    accelerationinitshort=0.02,
    accelerationshort=0.02,
    accelerationmaxshort=0.2,
):
    """
    Parabolic SAR Extended — full control over long/short AF parameters
    separately. Single output (same shape as SAR), just with more knobs.

    Uses high and low columns.
    """
    high, low = df["high"].values, df["low"].values
    result = talib.SAREXT(
        high,
        low,
        startvalue=startvalue,
        offsetonreverse=offsetonreverse,
        accelerationinitlong=accelerationinitlong,
        accelerationlong=accelerationlong,
        accelerationmaxlong=accelerationmaxlong,
        accelerationinitshort=accelerationinitshort,
        accelerationshort=accelerationshort,
        accelerationmaxshort=accelerationmaxshort,
    )
    return pd.Series(result, index=df.index).shift(1)


def overlap_sma(df, period=20):
    """
    Simple Moving Average — equal-weight average of the last N closes.

    Args:
        period: lookback window (default 20)
    """
    close = df["close"].values
    result = talib.SMA(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def overlap_t3(df, period=5, vfactor=0.7):
    """
    Triple Exponential Moving Average (T3) — smoother than TEMA, controlled
    by vfactor (volume factor). Higher vfactor = more smoothing.

    Args:
        period:  lookback window (default 5)
        vfactor: smoothing aggressiveness, 0..1 (default 0.7)
    """
    close = df["close"].values
    result = talib.T3(close, timeperiod=period, vfactor=vfactor)
    return pd.Series(result, index=df.index).shift(1)


def overlap_tema(df, period=30):
    """
    Triple Exponential Moving Average — 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA)).
    Faster response than DEMA with less overshoot than raw EMA subtraction.

    Args:
        period: lookback window (default 30)
    """
    close = df["close"].values
    result = talib.TEMA(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def overlap_trima(df, period=30):
    """
    Triangular Moving Average — double-smoothed SMA, weights middle of the
    window more heavily. Smoother than SMA but slower to react.

    Args:
        period: lookback window (default 30)
    """
    close = df["close"].values
    result = talib.TRIMA(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def overlap_wma(df, period=30):
    """
    Weighted Moving Average — linearly increasing weights, most recent bar
    gets weight N, oldest gets weight 1.

    Args:
        period: lookback window (default 30)
    """
    close = df["close"].values
    result = talib.WMA(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


# ---------------------------------------------------------------------------
# Momentum Indicators
# ---------------------------------------------------------------------------

def momentum_adx(df, period=14):
    """
    Average Directional Movement Index — measures trend strength (not direction).
    Values above 25 generally indicate a strong trend.
    Uses high, low, close.

    Args:
        period: lookback window (default 14)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.ADX(high, low, close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_adxr(df, period=14):
    """
    Average Directional Movement Index Rating — average of current ADX and
    ADX from `period` bars ago. Smoother but more lagged than ADX.
    Uses high, low, close.

    Args:
        period: lookback window (default 14)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.ADXR(high, low, close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_apo(df, fastperiod=12, slowperiod=26, matype=0):
    """
    Absolute Price Oscillator — difference between two MAs of close.
    Like MACD but without the signal line.

    Args:
        fastperiod: fast MA window (default 12)
        slowperiod: slow MA window (default 26)
        matype:     MA type (default 0 = SMA)
    """
    close = df["close"].values
    result = talib.APO(close, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
    return pd.Series(result, index=df.index).shift(1)


def momentum_aroon(df, period=14):
    """
    Aroon — dual output: aroon_down and aroon_up, both in range [0, 100].
    Measures how recently the highest high / lowest low occurred.

    Multi-output: returns dict with keys 'aroon_down' and 'aroon_up'.
    Uses high and low.

    Args:
        period: lookback window (default 14)
    """
    high, low = df["high"].values, df["low"].values
    aroon_down, aroon_up = talib.AROON(high, low, timeperiod=period)
    idx = df.index
    return {
        "aroon_down": pd.Series(aroon_down, index=idx).shift(1),
        "aroon_up":   pd.Series(aroon_up,   index=idx).shift(1),
    }


def momentum_aroonosc(df, period=14):
    """
    Aroon Oscillator — aroon_up minus aroon_down. Single value in [-100, 100].
    Positive = bullish momentum, negative = bearish.
    Uses high and low.

    Args:
        period: lookback window (default 14)
    """
    high, low = df["high"].values, df["low"].values
    result = talib.AROONOSC(high, low, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_bop(df):
    """
    Balance of Power — (close - open) / (high - low).
    Measures buying vs selling pressure. Range roughly [-1, 1].
    Uses open, high, low, close.
    """
    open_, high, low, close = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    result = talib.BOP(open_, high, low, close)
    return pd.Series(result, index=df.index).shift(1)


def momentum_cci(df, period=14):
    """
    Commodity Channel Index — measures deviation of price from its average.
    Values beyond ±100 suggest overbought/oversold conditions.
    Uses high, low, close.

    Args:
        period: lookback window (default 14)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.CCI(high, low, close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_cmo(df, period=14):
    """
    Chande Momentum Oscillator — ratio of sum of up-days vs down-days.
    Range [-100, 100]; similar to RSI but unbounded symmetry around 0.

    Args:
        period: lookback window (default 14)
    """
    close = df["close"].values
    result = talib.CMO(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_dx(df, period=14):
    """
    Directional Movement Index — raw input to ADX before smoothing.
    High values indicate strong directional movement.
    Uses high, low, close.

    Args:
        period: lookback window (default 14)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.DX(high, low, close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_macd(df, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    MACD — Moving Average Convergence/Divergence.
    Classic trend-following momentum indicator.

    Multi-output: returns dict with keys 'macd', 'signal', 'hist'.
      macd   = EMA(fast) - EMA(slow)
      signal = EMA(macd, signalperiod)
      hist   = macd - signal

    Args:
        fastperiod:   fast EMA window (default 12)
        slowperiod:   slow EMA window (default 26)
        signalperiod: signal line EMA window (default 9)
    """
    close = df["close"].values
    macd, signal, hist = talib.MACD(
        close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
    )
    idx = df.index
    return {
        "macd":   pd.Series(macd,   index=idx).shift(1),
        "signal": pd.Series(signal, index=idx).shift(1),
        "hist":   pd.Series(hist,   index=idx).shift(1),
    }


def momentum_macdext(
    df,
    fastperiod=12, fastmatype=0,
    slowperiod=26, slowmatype=0,
    signalperiod=9, signalmatype=0,
):
    """
    MACD with controllable MA type — same three outputs as MACD but each
    of the three MAs (fast, slow, signal) can be a different MA type.

    Multi-output: returns dict with keys 'macd', 'signal', 'hist'.

    Args:
        fastperiod/fastmatype:     fast MA window and type
        slowperiod/slowmatype:     slow MA window and type
        signalperiod/signalmatype: signal MA window and type
        matype values: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, ...
    """
    close = df["close"].values
    macd, signal, hist = talib.MACDEXT(
        close,
        fastperiod=fastperiod, fastmatype=fastmatype,
        slowperiod=slowperiod, slowmatype=slowmatype,
        signalperiod=signalperiod, signalmatype=signalmatype,
    )
    idx = df.index
    return {
        "macd":   pd.Series(macd,   index=idx).shift(1),
        "signal": pd.Series(signal, index=idx).shift(1),
        "hist":   pd.Series(hist,   index=idx).shift(1),
    }


def momentum_macdfix(df, signalperiod=9):
    """
    MACD Fix 12/26 — MACD with fast/slow periods locked at 12 and 26.
    Only the signal period is configurable.

    Multi-output: returns dict with keys 'macd', 'signal', 'hist'.

    Args:
        signalperiod: signal line EMA window (default 9)
    """
    close = df["close"].values
    macd, signal, hist = talib.MACDFIX(close, signalperiod=signalperiod)
    idx = df.index
    return {
        "macd":   pd.Series(macd,   index=idx).shift(1),
        "signal": pd.Series(signal, index=idx).shift(1),
        "hist":   pd.Series(hist,   index=idx).shift(1),
    }


def momentum_mfi(df, period=14):
    """
    Money Flow Index — volume-weighted RSI. Uses price AND volume.
    Range [0, 100]; above 80 = overbought, below 20 = oversold.
    Uses high, low, close, volume.

    Args:
        period: lookback window (default 14)
    """
    high, low, close, volume = df["high"].values, df["low"].values, df["close"].values, df["volume"].values
    result = talib.MFI(high, low, close, volume, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_minus_di(df, period=14):
    """
    Minus Directional Indicator (-DI) — measures downward price movement strength.
    Part of the Directional Movement System alongside +DI and ADX.
    Uses high, low, close.

    Args:
        period: lookback window (default 14)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.MINUS_DI(high, low, close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_minus_dm(df, period=14):
    """
    Minus Directional Movement — raw -DM values before the DI smoothing step.
    Uses high and low only.

    Args:
        period: lookback window (default 14)
    """
    high, low = df["high"].values, df["low"].values
    result = talib.MINUS_DM(high, low, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_mom(df, period=10):
    """
    Momentum — close[i] - close[i - period]. Raw price change over N bars.
    No normalization; scale depends on asset price.

    Args:
        period: lookback window (default 10)
    """
    close = df["close"].values
    result = talib.MOM(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_plus_di(df, period=14):
    """
    Plus Directional Indicator (+DI) — measures upward price movement strength.
    When +DI > -DI, trend is bullish.
    Uses high, low, close.

    Args:
        period: lookback window (default 14)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.PLUS_DI(high, low, close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_plus_dm(df, period=14):
    """
    Plus Directional Movement — raw +DM values before the DI smoothing step.
    Uses high and low only.

    Args:
        period: lookback window (default 14)
    """
    high, low = df["high"].values, df["low"].values
    result = talib.PLUS_DM(high, low, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_ppo(df, fastperiod=12, slowperiod=26, matype=0):
    """
    Percentage Price Oscillator — like APO but expressed as a percentage,
    so it's comparable across different-priced assets.
    ((fast_MA - slow_MA) / slow_MA) * 100

    Args:
        fastperiod: fast MA window (default 12)
        slowperiod: slow MA window (default 26)
        matype:     MA type (default 0 = SMA)
    """
    close = df["close"].values
    result = talib.PPO(close, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
    return pd.Series(result, index=df.index).shift(1)


def momentum_roc(df, period=10):
    """
    Rate of Change — ((close / close[N periods ago]) - 1) * 100.
    Expressed as a percentage.

    Args:
        period: lookback window (default 10)
    """
    close = df["close"].values
    result = talib.ROC(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_rocp(df, period=10):
    """
    Rate of Change Percentage — (close - close[N]) / close[N].
    Same as ROC but as a decimal ratio, not multiplied by 100.

    Args:
        period: lookback window (default 10)
    """
    close = df["close"].values
    result = talib.ROCP(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_rocr(df, period=10):
    """
    Rate of Change Ratio — close / close[N periods ago].
    Value of 1.0 means no change; >1 = up, <1 = down.

    Args:
        period: lookback window (default 10)
    """
    close = df["close"].values
    result = talib.ROCR(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_rocr100(df, period=10):
    """
    Rate of Change Ratio 100 Scale — (close / close[N]) * 100.
    Value of 100 means no change; >100 = up, <100 = down.

    Args:
        period: lookback window (default 10)
    """
    close = df["close"].values
    result = talib.ROCR100(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_rsi(df, period=14):
    """
    Relative Strength Index — ratio of average gains to average losses.
    Range [0, 100]; above 70 = overbought, below 30 = oversold.

    Args:
        period: lookback window (default 14)
    """
    close = df["close"].values
    result = talib.RSI(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_stoch(
    df,
    fastk_period=5,
    slowk_period=3, slowk_matype=0,
    slowd_period=3, slowd_matype=0,
):
    """
    Stochastic — classic slow stochastic oscillator.
    %K measures where close sits within the recent high-low range.
    %D is a smoothed signal line of %K.

    Multi-output: returns dict with keys 'slowk' and 'slowd'.
    Uses high, low, close.

    Args:
        fastk_period:  raw %K lookback (default 5)
        slowk_period:  %K smoothing window (default 3)
        slowk_matype:  %K smoothing MA type (default 0 = SMA)
        slowd_period:  %D smoothing window (default 3)
        slowd_matype:  %D smoothing MA type (default 0 = SMA)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    slowk, slowd = talib.STOCH(
        high, low, close,
        fastk_period=fastk_period,
        slowk_period=slowk_period, slowk_matype=slowk_matype,
        slowd_period=slowd_period, slowd_matype=slowd_matype,
    )
    idx = df.index
    return {
        "slowk": pd.Series(slowk, index=idx).shift(1),
        "slowd": pd.Series(slowd, index=idx).shift(1),
    }


def momentum_stochf(df, fastk_period=5, fastd_period=3, fastd_matype=0):
    """
    Stochastic Fast — unsmoothed %K with a smoothed %D signal.
    More reactive than slow Stochastic, noisier in choppy markets.

    Multi-output: returns dict with keys 'fastk' and 'fastd'.
    Uses high, low, close.

    Args:
        fastk_period: raw %K lookback (default 5)
        fastd_period: %D smoothing window (default 3)
        fastd_matype: %D smoothing MA type (default 0 = SMA)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    fastk, fastd = talib.STOCHF(
        high, low, close,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype,
    )
    idx = df.index
    return {
        "fastk": pd.Series(fastk, index=idx).shift(1),
        "fastd": pd.Series(fastd, index=idx).shift(1),
    }


def momentum_stochrsi(df, period=14, fastk_period=5, fastd_period=3, fastd_matype=0):
    """
    Stochastic RSI — applies the Stochastic formula to RSI values instead
    of price. More sensitive than plain RSI; useful for overbought/oversold
    signals in strong trends.

    Multi-output: returns dict with keys 'fastk' and 'fastd'.

    Args:
        period:       RSI lookback window (default 14)
        fastk_period: Stochastic %K window applied to RSI (default 5)
        fastd_period: %D smoothing window (default 3)
        fastd_matype: %D smoothing MA type (default 0 = SMA)
    """
    close = df["close"].values
    fastk, fastd = talib.STOCHRSI(
        close,
        timeperiod=period,
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype,
    )
    idx = df.index
    return {
        "fastk": pd.Series(fastk, index=idx).shift(1),
        "fastd": pd.Series(fastd, index=idx).shift(1),
    }


def momentum_trix(df, period=30):
    """
    TRIX — 1-day ROC of a triple-smoothed EMA. Filters out short cycles;
    crossovers of the zero line are used as buy/sell signals.

    Args:
        period: EMA smoothing window (default 30)
    """
    close = df["close"].values
    result = talib.TRIX(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def momentum_ultosc(df, period1=7, period2=14, period3=28):
    """
    Ultimate Oscillator — weighted average of three stochastic oscillators
    over three different timeframes, reducing false divergence signals.
    Range [0, 100].
    Uses high, low, close.

    Args:
        period1: short window (default 7)
        period2: medium window (default 14)
        period3: long window (default 28)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.ULTOSC(high, low, close, timeperiod1=period1, timeperiod2=period2, timeperiod3=period3)
    return pd.Series(result, index=df.index).shift(1)


def momentum_willr(df, period=14):
    """
    Williams' %R — inverse of the Stochastic %K. Range [-100, 0].
    Above -20 = overbought, below -80 = oversold.
    Uses high, low, close.

    Args:
        period: lookback window (default 14)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.WILLR(high, low, close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


# ---------------------------------------------------------------------------
# Volume Indicators
# ---------------------------------------------------------------------------

def volume_ad(df):
    """
    Chaikin A/D Line — cumulative sum of the Money Flow Volume.
    Measures the cumulative flow of money into/out of an asset.
    Rising A/D with falling price = bullish divergence (and vice versa).
    Uses high, low, close, volume.
    """
    high, low, close, volume = df["high"].values, df["low"].values, df["close"].values, df["volume"].values
    result = talib.AD(high, low, close, volume)
    return pd.Series(result, index=df.index).shift(1)


def volume_adosc(df, fastperiod=3, slowperiod=10):
    """
    Chaikin A/D Oscillator — difference between fast and slow EMAs of the
    A/D Line. Crosses above zero = bullish, below = bearish.
    Uses high, low, close, volume.

    Args:
        fastperiod: fast EMA window applied to A/D Line (default 3)
        slowperiod: slow EMA window applied to A/D Line (default 10)
    """
    high, low, close, volume = df["high"].values, df["low"].values, df["close"].values, df["volume"].values
    result = talib.ADOSC(high, low, close, volume, fastperiod=fastperiod, slowperiod=slowperiod)
    return pd.Series(result, index=df.index).shift(1)


def volume_obv(df):
    """
    On Balance Volume — running total that adds volume on up days and
    subtracts on down days. Tracks whether volume is flowing in or out.
    Uses close and volume.
    """
    close, volume = df["close"].values, df["volume"].values
    result = talib.OBV(close, volume)
    return pd.Series(result, index=df.index).shift(1)


# ---------------------------------------------------------------------------
# Cycle Indicators
# ---------------------------------------------------------------------------

def cycle_ht_dcperiod(df):
    """
    Hilbert Transform — Dominant Cycle Period.
    Estimates the period (in bars) of the dominant market cycle.
    Applied to close. No period parameter; the HT algorithm is self-adaptive.
    """
    close = df["close"].values
    result = talib.HT_DCPERIOD(close)
    return pd.Series(result, index=df.index).shift(1)


def cycle_ht_dcphase(df):
    """
    Hilbert Transform — Dominant Cycle Phase.
    Returns the current phase angle (0–360°) within the dominant cycle.
    Applied to close.
    """
    close = df["close"].values
    result = talib.HT_DCPHASE(close)
    return pd.Series(result, index=df.index).shift(1)


def cycle_ht_phasor(df):
    """
    Hilbert Transform — Phasor Components.
    Decomposes price into two quadrature components of the dominant cycle.

    Multi-output: returns dict with keys 'inphase' and 'quadrature'.
    Applied to close.
    """
    close = df["close"].values
    inphase, quadrature = talib.HT_PHASOR(close)
    idx = df.index
    return {
        "inphase":    pd.Series(inphase,    index=idx).shift(1),
        "quadrature": pd.Series(quadrature, index=idx).shift(1),
    }


def cycle_ht_sine(df):
    """
    Hilbert Transform — SineWave.
    Projects the dominant cycle onto a sine and lead-sine wave.
    Crossovers between sine and leadsine signal cycle turning points.

    Multi-output: returns dict with keys 'sine' and 'leadsine'.
    Applied to close.
    """
    close = df["close"].values
    sine, leadsine = talib.HT_SINE(close)
    idx = df.index
    return {
        "sine":     pd.Series(sine,     index=idx).shift(1),
        "leadsine": pd.Series(leadsine, index=idx).shift(1),
    }


def cycle_ht_trendmode(df):
    """
    Hilbert Transform — Trend vs Cycle Mode.
    Returns 1 when price is trending, 0 when it is cycling.
    Useful for switching between trend-following and mean-reversion strategies.
    Applied to close.
    """
    close = df["close"].values
    result = talib.HT_TRENDMODE(close)
    return pd.Series(result, index=df.index).shift(1)


# ---------------------------------------------------------------------------
# Price Transform
# ---------------------------------------------------------------------------

def price_avgprice(df):
    """
    Average Price — (open + high + low + close) / 4.
    Simple four-way average of the candle's OHLC values.
    Uses open, high, low, close.
    """
    open_, high, low, close = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    result = talib.AVGPRICE(open_, high, low, close)
    return pd.Series(result, index=df.index).shift(1)


def price_medprice(df):
    """
    Median Price — (high + low) / 2.
    Midpoint of the candle's price range; ignores open and close.
    Uses high and low.
    """
    high, low = df["high"].values, df["low"].values
    result = talib.MEDPRICE(high, low)
    return pd.Series(result, index=df.index).shift(1)


def price_typprice(df):
    """
    Typical Price — (high + low + close) / 3.
    Standard base input for indicators like CCI and MFI.
    Uses high, low, close.
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.TYPPRICE(high, low, close)
    return pd.Series(result, index=df.index).shift(1)


def price_wclprice(df):
    """
    Weighted Close Price — (high + low + close * 2) / 4.
    Like Typical Price but gives double weight to the close.
    Uses high, low, close.
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.WCLPRICE(high, low, close)
    return pd.Series(result, index=df.index).shift(1)


# ---------------------------------------------------------------------------
# Volatility Indicators
# ---------------------------------------------------------------------------

def volatility_atr(df, period=14):
    """
    Average True Range — smoothed average of True Range over N bars.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|).
    Measures volatility in price units; larger = more volatile.
    Uses high, low, close.

    Args:
        period: lookback window (default 14)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.ATR(high, low, close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def volatility_natr(df, period=14):
    """
    Normalized Average True Range — ATR expressed as a percentage of close.
    (ATR / close) * 100. Makes volatility comparable across different assets
    regardless of price level.
    Uses high, low, close.

    Args:
        period: lookback window (default 14)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.NATR(high, low, close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def volatility_trange(df):
    """
    True Range — single-bar volatility measure, unsmoothed.
    max(high-low, |high-prev_close|, |low-prev_close|).
    Raw input to ATR before smoothing.
    Uses high, low, close.
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.TRANGE(high, low, close)
    return pd.Series(result, index=df.index).shift(1)

# ---------------------------------------------------------------------------
# Pattern Recognition
# ---------------------------------------------------------------------------
# All candlestick pattern functions:
#   - Use open, high, low, close
#   - Return integer Series: +100 (bullish), -100 (bearish), 0 (no pattern)
#   - All shifted by 1 to prevent look-ahead bias
# ---------------------------------------------------------------------------

def _ohlc(df):
    """Extract open, high, low, close as numpy arrays."""
    return df["open"].values, df["high"].values, df["low"].values, df["close"].values


def pattern_cdl2crows(df):
    """Two Crows — bearish reversal. Two black candles gap above then close into prior white candle."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDL2CROWS(o, h, l, c), index=df.index).shift(1)


def pattern_cdl3blackcrows(df):
    """Three Black Crows — three consecutive long bearish candles, each closing lower."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDL3BLACKCROWS(o, h, l, c), index=df.index).shift(1)


def pattern_cdl3inside(df):
    """Three Inside Up/Down — harami confirmed by a third candle closing beyond the first."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDL3INSIDE(o, h, l, c), index=df.index).shift(1)


def pattern_cdl3linestrike(df):
    """Three-Line Strike — three candles in one direction, fourth reverses and engulfs all three."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDL3LINESTRIKE(o, h, l, c), index=df.index).shift(1)


def pattern_cdl3outside(df):
    """Three Outside Up/Down — engulfing pattern confirmed by a third candle closing further."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDL3OUTSIDE(o, h, l, c), index=df.index).shift(1)


def pattern_cdl3starsinsouth(df):
    """Three Stars In The South — three progressively smaller bearish candles signaling bullish reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDL3STARSINSOUTH(o, h, l, c), index=df.index).shift(1)


def pattern_cdl3whitesoldiers(df):
    """Three Advancing White Soldiers — three consecutive long bullish candles, each closing higher."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDL3WHITESOLDIERS(o, h, l, c), index=df.index).shift(1)


def pattern_cdlabandonedbaby(df, penetration=0.3):
    """
    Abandoned Baby — doji gaps away from prior trend then gaps back; strong reversal signal.

    Args:
        penetration: gap penetration factor (default 0.3)
    """
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLABANDONEDBABY(o, h, l, c, penetration=penetration), index=df.index).shift(1)


def pattern_cdladvanceblock(df):
    """Advance Block — three white candles with progressively weakening bodies; bearish warning."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLADVANCEBLOCK(o, h, l, c), index=df.index).shift(1)


def pattern_cdlbelthold(df):
    """Belt-hold — long candle opening at its extreme (no shadow on one side); trend signal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLBELTHOLD(o, h, l, c), index=df.index).shift(1)


def pattern_cdlbreakaway(df):
    """Breakaway — five-candle pattern with a gap and reversal close back into the gap area."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLBREAKAWAY(o, h, l, c), index=df.index).shift(1)


def pattern_cdlclosingmarubozu(df):
    """Closing Marubozu — candle closes at its high (bullish) or low (bearish) with no closing shadow."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLCLOSINGMARUBOZU(o, h, l, c), index=df.index).shift(1)


def pattern_cdlconcealbabyswall(df):
    """Concealing Baby Swallow — four-candle bullish reversal in a downtrend with marubozu bodies."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLCONCEALBABYSWALL(o, h, l, c), index=df.index).shift(1)


def pattern_cdlcounterattack(df):
    """Counterattack — two candles of opposite color closing at the same level; reversal signal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLCOUNTERATTACK(o, h, l, c), index=df.index).shift(1)


def pattern_cdldarkcloudcover(df, penetration=0.5):
    """
    Dark Cloud Cover — bearish reversal; black candle opens above prior close and closes into its body.

    Args:
        penetration: how far the second candle must close into the first body (default 0.5)
    """
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLDARKCLOUDCOVER(o, h, l, c, penetration=penetration), index=df.index).shift(1)


def pattern_cdldoji(df):
    """Doji — open and close are nearly equal; signals indecision."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLDOJI(o, h, l, c), index=df.index).shift(1)


def pattern_cdldojistar(df):
    """Doji Star — doji that gaps away from the prior candle; potential reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLDOJISTAR(o, h, l, c), index=df.index).shift(1)


def pattern_cdldragonflydoji(df):
    """Dragonfly Doji — open/high/close at top, long lower shadow; bullish reversal at bottoms."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLDRAGONFLYDOJI(o, h, l, c), index=df.index).shift(1)


def pattern_cdlengulfing(df):
    """Engulfing Pattern — second candle's body completely engulfs the first; strong reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLENGULFING(o, h, l, c), index=df.index).shift(1)


def pattern_cdleveningdojistar(df, penetration=0.3):
    """
    Evening Doji Star — three-candle bearish reversal: white candle, doji gap up, black candle.

    Args:
        penetration: third candle penetration into first candle body (default 0.3)
    """
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLEVENINGDOJISTAR(o, h, l, c, penetration=penetration), index=df.index).shift(1)


def pattern_cdleveningstar(df, penetration=0.3):
    """
    Evening Star — three-candle bearish reversal: white candle, small body gap up, black candle.

    Args:
        penetration: third candle penetration into first candle body (default 0.3)
    """
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLEVENINGSTAR(o, h, l, c, penetration=penetration), index=df.index).shift(1)


def pattern_cdlgapsidesidewhite(df):
    """Up/Down-gap side-by-side white lines — two white candles of similar size after a gap; continuation."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLGAPSIDESIDEWHITE(o, h, l, c), index=df.index).shift(1)


def pattern_cdlgravestonedoji(df):
    """Gravestone Doji — open/low/close at bottom, long upper shadow; bearish reversal at tops."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLGRAVESTONEDOJI(o, h, l, c), index=df.index).shift(1)


def pattern_cdlhammer(df):
    """Hammer — small body at top, long lower shadow, little/no upper shadow; bullish reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLHAMMER(o, h, l, c), index=df.index).shift(1)


def pattern_cdlhangingman(df):
    """Hanging Man — same shape as Hammer but appears in an uptrend; bearish warning."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLHANGINGMAN(o, h, l, c), index=df.index).shift(1)


def pattern_cdlharami(df):
    """Harami Pattern — small second candle contained within the first candle's body; reversal warning."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLHARAMI(o, h, l, c), index=df.index).shift(1)


def pattern_cdlharamicross(df):
    """Harami Cross — harami where the second candle is a doji; stronger reversal signal than plain harami."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLHARAMICROSS(o, h, l, c), index=df.index).shift(1)


def pattern_cdlhighwave(df):
    """High-Wave Candle — very long upper and lower shadows with a small body; extreme indecision."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLHIGHWAVE(o, h, l, c), index=df.index).shift(1)


def pattern_cdlhikkake(df):
    """Hikkake Pattern — inside bar breakout that fails and reverses; trap pattern."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLHIKKAKE(o, h, l, c), index=df.index).shift(1)


def pattern_cdlhikkakemod(df):
    """Modified Hikkake Pattern — stricter version of Hikkake with additional confirmation bar."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLHIKKAKEMOD(o, h, l, c), index=df.index).shift(1)


def pattern_cdlhomingpigeon(df):
    """Homing Pigeon — two black candles where the second is smaller and contained; bullish reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLHOMINGPIGEON(o, h, l, c), index=df.index).shift(1)


def pattern_cdlidentical3crows(df):
    """Identical Three Crows — three black candles opening at prior close; strong bearish reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLIDENTICAL3CROWS(o, h, l, c), index=df.index).shift(1)


def pattern_cdlinneck(df):
    """In-Neck Pattern — black candle followed by white candle closing near prior low; bearish continuation."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLINNECK(o, h, l, c), index=df.index).shift(1)


def pattern_cdlinvertedhammer(df):
    """Inverted Hammer — small body at bottom, long upper shadow; potential bullish reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLINVERTEDHAMMER(o, h, l, c), index=df.index).shift(1)


def pattern_cdlkicking(df):
    """Kicking — two marubozu candles of opposite color with a gap; very strong reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLKICKING(o, h, l, c), index=df.index).shift(1)


def pattern_cdlkickingbylength(df):
    """Kicking by Length — kicking pattern where direction is set by the longer marubozu."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLKICKINGBYLENGTH(o, h, l, c), index=df.index).shift(1)


def pattern_cdlladderbottom(df):
    """Ladder Bottom — three black candles followed by an inverted hammer and white candle; bullish reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLLADDERBOTTOM(o, h, l, c), index=df.index).shift(1)


def pattern_cdllongleggeddoji(df):
    """Long Legged Doji — doji with long upper and lower shadows; strong indecision."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLLONGLEGGEDDOJI(o, h, l, c), index=df.index).shift(1)


def pattern_cdllongline(df):
    """Long Line Candle — candle with a long body relative to recent range; momentum signal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLLONGLINE(o, h, l, c), index=df.index).shift(1)


def pattern_cdlmarubozu(df):
    """Marubozu — candle with no shadows at all; full-body momentum candle."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLMARUBOZU(o, h, l, c), index=df.index).shift(1)


def pattern_cdlmatchinglow(df):
    """Matching Low — two black candles closing at the same low; support level signal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLMATCHINGLOW(o, h, l, c), index=df.index).shift(1)


def pattern_cdlmathold(df, penetration=0.5):
    """
    Mat Hold — bullish continuation: white candle, three small pullback candles, then another white.

    Args:
        penetration: pullback penetration factor (default 0.5)
    """
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLMATHOLD(o, h, l, c, penetration=penetration), index=df.index).shift(1)


def pattern_cdlmorningdojistar(df, penetration=0.3):
    """
    Morning Doji Star — three-candle bullish reversal: black candle, doji gap down, white candle.

    Args:
        penetration: third candle penetration into first candle body (default 0.3)
    """
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLMORNINGDOJISTAR(o, h, l, c, penetration=penetration), index=df.index).shift(1)


def pattern_cdlmorningstar(df, penetration=0.3):
    """
    Morning Star — three-candle bullish reversal: black candle, small body gap down, white candle.

    Args:
        penetration: third candle penetration into first candle body (default 0.3)
    """
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLMORNINGSTAR(o, h, l, c, penetration=penetration), index=df.index).shift(1)


def pattern_cdlonneck(df):
    """On-Neck Pattern — black candle followed by white candle closing at prior low; bearish continuation."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLONNECK(o, h, l, c), index=df.index).shift(1)


def pattern_cdlpiercing(df):
    """Piercing Pattern — black candle followed by white candle closing above midpoint; bullish reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLPIERCING(o, h, l, c), index=df.index).shift(1)


def pattern_cdlrickshawman(df):
    """Rickshaw Man — long-legged doji with body near the middle of the range; indecision."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLRICKSHAWMAN(o, h, l, c), index=df.index).shift(1)


def pattern_cdlrisefall3methods(df):
    """Rising/Falling Three Methods — continuation: long candle, three small retracements, resuming candle."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLRISEFALL3METHODS(o, h, l, c), index=df.index).shift(1)


def pattern_cdlseparatinglines(df):
    """Separating Lines — two candles of opposite color opening at the same price; continuation signal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLSEPARATINGLINES(o, h, l, c), index=df.index).shift(1)


def pattern_cdlshootingstar(df):
    """Shooting Star — small body at bottom, long upper shadow; bearish reversal at tops."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLSHOOTINGSTAR(o, h, l, c), index=df.index).shift(1)


def pattern_cdlshortline(df):
    """Short Line Candle — candle with a short body relative to recent range; low conviction bar."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLSHORTLINE(o, h, l, c), index=df.index).shift(1)


def pattern_cdlspinningtop(df):
    """Spinning Top — small body with upper and lower shadows of similar length; indecision."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLSPINNINGTOP(o, h, l, c), index=df.index).shift(1)


def pattern_cdlstalledpattern(df):
    """Stalled Pattern — three white candles where momentum visibly weakens on the third; bearish warning."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLSTALLEDPATTERN(o, h, l, c), index=df.index).shift(1)


def pattern_cdlsticksandwich(df):
    """Stick Sandwich — two black candles with same close sandwiching a white candle; bullish reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLSTICKSANDWICH(o, h, l, c), index=df.index).shift(1)


def pattern_cdltakuri(df):
    """Takuri — Dragonfly Doji with an exceptionally long lower shadow; strong bullish reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLTAKURI(o, h, l, c), index=df.index).shift(1)


def pattern_cdltasukigap(df):
    """Tasuki Gap — gap followed by two candles where the second partially fills the gap; continuation."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLTASUKIGAP(o, h, l, c), index=df.index).shift(1)


def pattern_cdlthrusting(df):
    """Thrusting Pattern — black candle followed by white closing below midpoint; weak bearish continuation."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLTHRUSTING(o, h, l, c), index=df.index).shift(1)


def pattern_cdltristar(df):
    """Tristar Pattern — three consecutive dojis; rare but strong reversal signal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLTRISTAR(o, h, l, c), index=df.index).shift(1)


def pattern_cdlunique3river(df):
    """Unique 3 River — three-candle bullish reversal with a hammering third candle at a new low."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLUNIQUE3RIVER(o, h, l, c), index=df.index).shift(1)


def pattern_cdlupsidegap2crows(df):
    """Upside Gap Two Crows — white candle gaps up, two black candles fill the gap; bearish reversal."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLUPSIDEGAP2CROWS(o, h, l, c), index=df.index).shift(1)


def pattern_cdlxsidegap3methods(df):
    """Upside/Downside Gap Three Methods — gap with two candles then a third that closes the gap; continuation."""
    o, h, l, c = _ohlc(df)
    return pd.Series(talib.CDLXSIDEGAP3METHODS(o, h, l, c), index=df.index).shift(1)


# ---------------------------------------------------------------------------
# Statistic Functions
# ---------------------------------------------------------------------------

def stats_beta(df, period=5):
    """
    Beta — slope of close relative to high/low midpoint over N bars.
    Measures how much `close` moves relative to the high-low range.
    Uses high and low as the "market" series, close as the "asset" series.

    Args:
        period: lookback window (default 5)
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    result = talib.BETA(high, low, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def stats_correl(df, period=30):
    """
    Pearson's Correlation Coefficient — correlation between high and low
    over N bars. Range [-1, 1]; 1 = perfect positive, -1 = perfect negative.
    Uses high and low.

    Args:
        period: lookback window (default 30)
    """
    high, low = df["high"].values, df["low"].values
    result = talib.CORREL(high, low, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def stats_linearreg(df, period=14):
    """
    Linear Regression — endpoint of the least-squares regression line fit
    to close over N bars. Smoother than a moving average.

    Args:
        period: lookback window (default 14)
    """
    close = df["close"].values
    result = talib.LINEARREG(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def stats_linearreg_angle(df, period=14):
    """
    Linear Regression Angle — slope of the regression line expressed
    in degrees. Positive = uptrend, negative = downtrend.

    Args:
        period: lookback window (default 14)
    """
    close = df["close"].values
    result = talib.LINEARREG_ANGLE(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def stats_linearreg_intercept(df, period=14):
    """
    Linear Regression Intercept — y-intercept of the regression line.
    Together with slope, fully describes the fitted line.

    Args:
        period: lookback window (default 14)
    """
    close = df["close"].values
    result = talib.LINEARREG_INTERCEPT(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def stats_linearreg_slope(df, period=14):
    """
    Linear Regression Slope — rate of change of the regression line per bar.
    Positive = rising trend, negative = falling trend.

    Args:
        period: lookback window (default 14)
    """
    close = df["close"].values
    result = talib.LINEARREG_SLOPE(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def stats_stddev(df, period=5, nbdev=1.0):
    """
    Standard Deviation — rolling std of close over N bars.
    Used directly as a volatility measure or as input to BBANDS.

    Args:
        period: lookback window (default 5)
        nbdev:  number of deviations to scale output by (default 1.0)
    """
    close = df["close"].values
    result = talib.STDDEV(close, timeperiod=period, nbdev=nbdev)
    return pd.Series(result, index=df.index).shift(1)


def stats_tsf(df, period=14):
    """
    Time Series Forecast — projects the regression line one bar forward.
    Equivalent to LINEARREG_SLOPE * period + LINEARREG_INTERCEPT.
    More responsive than a plain moving average.

    Args:
        period: lookback window (default 14)
    """
    close = df["close"].values
    result = talib.TSF(close, timeperiod=period)
    return pd.Series(result, index=df.index).shift(1)


def stats_var(df, period=5, nbdev=1.0):
    """
    Variance — rolling variance of close over N bars.
    Square of standard deviation; used in position sizing and risk models.

    Args:
        period: lookback window (default 5)
        nbdev:  scaling factor (default 1.0)
    """
    close = df["close"].values
    result = talib.VAR(close, timeperiod=period, nbdev=nbdev)
    return pd.Series(result, index=df.index).shift(1)
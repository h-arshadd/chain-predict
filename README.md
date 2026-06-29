# chain-predict

A production-grade data pipeline for fetching, cleaning, and storing historical OHLCV (Open, High, Low, Close, Volume) candlestick data from Binance and Bybit exchanges into a PostgreSQL database. Includes a full TA-Lib indicator library and visualization tooling. Built as the data foundation for training quantitative ML models on crypto markets.

---

## Features

- Fetches 1-minute OHLCV data for 8 coins: DOGE, SOL, BTC, ETH, ADA, LTC, MINA, SUI
- Supports both Binance (spot) and Bybit (linear perpetuals)
- Smart resume — tracks last stored timestamp and fetches only new data on each run
- Data completeness validation with linear interpolation or forward fill for missing candles
- Zero-volume candle correction via forward fill
- On-demand resampling to any pandas-compatible timeframe (e.g. `5min`, `1h`, `1D`)
- 134 TA-Lib indicators across all categories: Overlap Studies, Momentum, Volume, Cycle, Price Transform, Volatility, Pattern Recognition, and Statistics
- Look-ahead bias prevention — every indicator output is shifted by 1 bar
- Interactive Plotly charts with candlestick, overlays, and indicator subplots
- Configurable via YAML — no hardcoded values
- Full logging to console and file with configurable retry logic

---

## Project Structure

```
chain-predict/
├── crypto_pipeline/
│   ├── data/
│   │   ├── binance/
│   │   │   ├── config_binance.yml      # Binance pipeline config
│   │   │   ├── exchange_binance.py     # Binance API fetcher
│   │   │   └── main.py                # Binance pipeline entry point
│   │   ├── bybit/
│   │   │   ├── config_bybit.yml        # Bybit pipeline config
│   │   │   ├── exchange_bybit.py       # Bybit API fetcher
│   │   │   └── main.py                 # Bybit pipeline entry point
│   │   └── data_downloader.py          # Core orchestrator: parse, clean, resample, store
│   ├── indicators/
│   │   └── talib_indicators.py         # 134 TA-Lib indicators, all look-ahead safe
│   └── utils/
│       ├── db_utils.py                 # PostgreSQL connection, schema, insert, query
│       └── pipeline_utils.py           # Logging setup, config loading, date parsing
├── .env                                # DB credentials (not committed)
├── requirements.txt
└── setup.py
```

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/h-arshadd/chain-predict.git
cd chain-predict
```

**2. Create and activate virtual environment**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
pip install -e .
```

**4. Install TA-Lib**

TA-Lib requires a prebuilt wheel on Windows. Download the `.whl` matching your Python version from [github.com/cgohlke/talib-build/releases](https://github.com/cgohlke/talib-build/releases), then:

```bash
pip install TA_Lib-0.6.4-cp310-cp310-win_amd64.whl
```

On Linux/WSL:

```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib && ./configure --prefix=/usr && make && sudo make install
pip install ta-lib
```

**5. Configure environment variables**

Create a `.env` file in the root directory:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chain_predict
DB_USER=postgres
DB_PASSWORD=your_password_here
```

**6. Set up PostgreSQL**

Create the database:

```sql
CREATE DATABASE chain_predict;
```

Schemas and tables are created automatically on first run.

---

## Configuration

Edit the YAML config files to customize the pipeline:

`crypto_pipeline/data/binance/config_binance.yml`
`crypto_pipeline/data/bybit/config_bybit.yml`

Key parameters:

```yaml
symbols: ["doge", "sol", "btc", "eth", "ada", "ltc", "mina", "sui"]
start_date: "2023-01-01"
end_date: "now"
filling_missing_method: "interpolation"
fill_zero_volume: "ffill"
retries: 5
retry_delay: 10
```

---

## Usage

**Run Binance pipeline:**

```bash
python -m crypto_pipeline.data.binance.main
```

**Run Bybit pipeline:**

```bash
python -m crypto_pipeline.data.bybit.main
```

**Fetch and resample data at runtime (no DB write):**

```python
from datetime import datetime
from crypto_pipeline.utils.db_utils import get_db_connection
from crypto_pipeline.data.binance.exchange_binance import BinanceExchange
from crypto_pipeline.data.data_downloader import DataDownloader

conn     = get_db_connection()
exchange = BinanceExchange()
dl       = DataDownloader(config={}, exchange_fetcher=exchange, conn=conn)

result = dl.get_data(
    exchange="binance",
    symbol="btc",
    start_date=datetime(2026, 6, 29, 0, 0),
    end_date="now",
    df_1m=True,
)

df_1m        = result["one_min"]      # 1-minute candles
df_resampled = result["resampled"]    # 5-minute resampled
```

**Compute indicators:**

```python
from crypto_pipeline.indicators.talib_indicators import (
    overlap_bbands, overlap_ema, momentum_rsi, momentum_macd
)

bbands = overlap_bbands(df_1m, period=20)   # returns dict: upper, middle, lower
ema20  = overlap_ema(df_1m, period=20)      # returns pd.Series
rsi    = momentum_rsi(df_1m, period=14)     # returns pd.Series
macd   = momentum_macd(df_1m)               # returns dict: macd, signal, hist
```

All indicator outputs are shifted by 1 bar — row N only uses data from bars 0..N-1.

**Plot indicators:**

```bash
python plot_indicators.py
```

Opens an interactive Plotly chart in the browser with candlestick, Bollinger Bands, SMA, EMA, SAR, RSI, MACD, and volume panels.

---

## Indicators

All 134 indicators from TA-Lib are implemented in `crypto_pipeline/indicators/talib_indicators.py`, organized by category:

| Category | Count | Examples |
|---|---|---|
| Overlap Studies | 17 | BBANDS, EMA, SMA, MAMA, SAR, KAMA |
| Momentum | 30 | RSI, MACD, STOCH, ADX, CCI, MFI, WILLR |
| Volume | 3 | AD, ADOSC, OBV |
| Cycle | 5 | HT_DCPERIOD, HT_SINE, HT_TRENDMODE |
| Price Transform | 4 | AVGPRICE, TYPPRICE, WCLPRICE |
| Volatility | 3 | ATR, NATR, TRANGE |
| Pattern Recognition | 61 | CDLENGULFING, CDLHAMMER, CDLMARUBOZU |
| Statistics | 9 | LINEARREG, STDDEV, CORREL, BETA |

Multi-output indicators (BBANDS, MAMA, MACD, AROON, STOCH, etc.) return a `dict` of named `pd.Series`.

---

## Database Schema

Each exchange has its own PostgreSQL schema. Tables follow the naming convention `{symbol}_{timeframe}`:

| Column   | Type             | Description                 |
|----------|------------------|-----------------------------|
| datetime | TIMESTAMP (PK)   | Candle open timestamp (UTC) |
| open     | DOUBLE PRECISION | Opening price               |
| high     | DOUBLE PRECISION | Highest price in the candle |
| low      | DOUBLE PRECISION | Lowest price in the candle  |
| close    | DOUBLE PRECISION | Closing price               |
| volume   | DOUBLE PRECISION | Trading volume              |

Example tables: `binance.btc_1m`, `bybit.doge_1m`

---

## Tech Stack

- **Python 3.10**
- **PostgreSQL** — data storage
- **TA-Lib 0.6.4** — technical indicators
- **Plotly** — interactive charting
- **python-binance** — Binance API client
- **pybit** — Bybit API client
- **psycopg2** — PostgreSQL connector
- **pandas / numpy** — data processing and resampling
- **pyyaml** — config management
- **python-dotenv** — environment variable management
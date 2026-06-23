# chain-predict

A production-grade data pipeline for fetching, cleaning, and storing historical OHLCV (Open, High, Low, Close, Volume) candlestick data from Binance and Bybit exchanges into a PostgreSQL database. Built as the data foundation for training quantitative ML models on crypto markets.

---

## Features

- Fetches 1-minute OHLCV data for 8 coins: DOGE, SOL, BTC, ETH, ADA, LTC, MINA, SUI
- Supports both Binance (linear futures) and Bybit (linear perpetuals)
- Smart resume — tracks last stored timestamp and fetches only new data on each run
- Data completeness validation with linear interpolation for missing candles
- Zero-volume candle correction via forward fill
- Configurable via YAML — no hardcoded values
- OOP architecture with `DatabaseManager` and `DataDownloader` classes
- Full logging to console and file with configurable retry logic

---

## Project Structure

```
chain-predict/
├── crypto_pipeline/
│   ├── data/
│   │   ├── binance/
│   │   │   ├── config_binance.yml
│   │   │   ├── exchange_binance.py
│   │   │   └── main_binance.py
│   │   ├── bybit/
│   │   │   ├── config_bybit.yml
│   │   │   ├── exchange_bybit.py
│   │   │   └── main_bybit.py
│   │   └── data_downloader.py
│   └── utils/
│       └── db_utils.py
├── .env
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

**4. Configure environment variables**

Create a `.env` file in the root directory:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chain_predict
DB_USER=postgres
DB_PASSWORD=your_password_here
```

**5. Set up PostgreSQL**

Create the database:

```sql
CREATE DATABASE chain_predict;
```

Schemas are created automatically on first run.

---

## Configuration

Edit the YAML config files to customize the pipeline:

`crypto_pipeline/data/binance/config_binance.yml`
`crypto_pipeline/data/bybit/config_bybit.yml`

Key parameters:

```yaml
symbols: ["doge", "sol", "btc", "eth", "ada", "ltc", "mina", "sui"]
time_horizons: ["1m"]
start_date: "2023-01-01"
end_date: "now"
filling_missing_method: "interpolation"
fill_zero_volume: "ffill"
retries: 5
retry_delay: 10
```

---

## Usage

Run Binance pipeline:

```bash
python -m crypto_pipeline.data.binance.main_binance
```

Run Bybit pipeline:

```bash
python -m crypto_pipeline.data.bybit.main_bybit
```

---

## Database Schema

Each exchange has its own PostgreSQL schema. Tables follow the naming convention `{symbol}_{timeframe}`:

| Column    | Type             | Description                   |
|-----------|------------------|-------------------------------|
| date_time | TIMESTAMP (PK)   | Candle open timestamp (UTC)   |
| open      | DOUBLE PRECISION | Opening price                 |
| high      | DOUBLE PRECISION | Highest price in the candle   |
| low       | DOUBLE PRECISION | Lowest price in the candle    |
| close     | DOUBLE PRECISION | Closing price                 |
| volume    | DOUBLE PRECISION | Trading volume                |

Example tables: `binance.btc_1m`, `bybit.doge_1m`

---

## Tech Stack

- **Python 3.10**
- **PostgreSQL 18** — data storage
- **python-binance** — Binance API client
- **pybit** — Bybit API client
- **psycopg2** — PostgreSQL connector
- **pandas / numpy** — data processing and interpolation
- **pyyaml** — config management
- **python-dotenv** — environment variable management
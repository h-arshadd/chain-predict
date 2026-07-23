# chain-predict

A production-grade quantitative trading pipeline combining **market data**, **social sentiment analysis**, **machine learning**, and **live/paper trade execution on Bybit**.

## Core Modules

### 1. **crypto_pipeline** — Data, Signals, ML, Backtesting & Live Trading
Started as an OHLCV + technical-indicator foundation and has grown into the full stack: fetches and stores historical candlestick data from Binance and Bybit, generates strategy signals, trains ML models, backtests and paper-trades strategies, and places real orders on Bybit. See [crypto_pipeline in depth](#crypto_pipeline-in-depth) below for the full module breakdown.

### 2. **sentiment_pipeline** — Reddit Sentiment Analysis  
Fetches Reddit posts, analyzes sentiment (Bullish/Bearish/Neutral) using CryptoBERT, classifies which coin they mention, extracts tickers and entities, and aggregates engagement-weighted sentiment scores per coin. Stores raw and cleaned data in separate PostgreSQL schemas for downstream analysis and portfolio signals.

### 3. **frontend** — Dashboard (early stage)
A React + Vite dashboard scaffolded with Tailwind, shadcn/ui, Ant Design, and Recharts for visualizing pipeline output — pages for Dashboard, Backtests, Models, Strategies, Wallets, Sentiment, Deployment, and Execution are stubbed in `frontend/src/pages/`. Not yet wired up to a backend API.

---

## Features

### crypto_pipeline — data & indicators
- Fetches 1-minute OHLCV data for 8 coins: DOGE, SOL, BTC, ETH, ADA, LTC, MINA, SUI
- Supports both Binance (spot) and Bybit (linear perpetuals)
- Smart resume — tracks last stored timestamp and fetches only new data on each run
- Data completeness validation with linear interpolation or forward fill for missing candles
- Zero-volume candle correction via forward fill
- On-demand resampling to any pandas-compatible timeframe (e.g. `5min`, `1h`, `1D`)
- 134 TA-Lib indicators across all categories with look-ahead bias prevention
- Interactive Plotly charts with candlestick, overlays, and indicator subplots
- Configurable via YAML — no hardcoded values
- Full logging to console and file with configurable retry logic

### crypto_pipeline — signals, backtesting & trading
- Strategy signal generation from YAML files or DB-backed rows, shared across backtest/simulator/execution so every stage runs off the same strategy definition
- 15 example strategies included: RSI reversal, golden cross, MACD crossover, EMA trend cross, doji/engulfing reversal, weighted multi-indicator confluence, majority vote, and more
- Vectorized backtesting engine with configurable position sizing, commission, and slippage — trade ledger stored per symbol in Postgres
- Paper-trading simulator that runs every strategy against every active pair, walking forward candle by candle with persisted state, designed to run repeatedly via Task Scheduler
- Live execution on Bybit: real market orders, native TP/SL registered with Bybit's own engine, ledger built from actual fill data (not candle prices), auto-reconciles any Bybit-side auto-close before making new decisions
- Bybit account-level history and stats: pulls fill history live from Bybit and computes ~85 pooled account stats (realized PnL via FIFO, win rate, profit factor, streaks, drawdown, per-symbol breakdown, etc.)

### crypto_pipeline — ML & research
- End-to-end ML pipeline (data → features → sentiment merge → target → feature selection → split → preprocessing → training → signals → evaluation) driven by one config file
- Regression, classification, and timeseries model types, each with several sklearn/boosting/deep-learning algorithms plus `darts`-based timeseries models (N-BEATS, TCN, StatsForecast)
- Preprocessing experimentation lab for comparing scaling/differencing/stationarity transforms against a live dataset
- Target validation module that backtests ML-generated targets to confirm they're profitable before trusting them as training labels
- Standalone statistics module (Sharpe, Sortino, Calmar, max drawdown, CAGR, profit factor, win rate, recovery factor, risk of ruin) that works off any backtest result dict, independent of the ML module

### sentiment_pipeline
- Fetches Reddit posts from configurable subreddits per coin
- NLP processing: text cleaning, tokenization, stopword removal, lemmatization, stemming, POS tagging, NER
- **CryptoBERT** fine-tuned sentiment model (outputs: Bullish/Bearish/Neutral with confidence scores)
- Zero-shot coin classification (auto-detects which coin a post is actually about)
- Ticker extraction and entity recognition
- Engagement-weighted sentiment aggregation (upvotes/comments influence final mean)
- Separate PostgreSQL schemas for raw fetched data and cleaned/analyzed data
- Stores sentiment per post, compute daily/weekly/yearly rolling averages

---

## Project Structure

```
chain-predict/
├── crypto_pipeline/
│   ├── data/
│   │   ├── binance/
│   │   │   ├── config_binance.yml      # Binance pipeline config
│   │   │   ├── exchange_binance.py     # Binance API fetcher
│   │   │   └── main.py                 # Binance pipeline entry point
│   │   ├── bybit/
│   │   │   ├── config_bybit.yml        # Bybit pipeline config
│   │   │   ├── exchange_bybit.py       # Bybit API fetcher
│   │   │   └── main.py                 # Bybit pipeline entry point
│   │   └── data_downloader.py          # Core orchestrator: parse, clean, resample, store
│   │
│   ├── indicators/
│   │   └── talib_indicators.py         # 134 TA-Lib indicators, all look-ahead safe
│   │
│   ├── signals/
│   │   ├── config.yaml                 # Default indicator config
│   │   ├── main.py                     # generate_signals() entry point
│   │   ├── signal_helpers.py           # Indicator calculation for signals
│   │   ├── conditions.py               # Condition evaluation
│   │   ├── rules.py                    # Rule application → final signal
│   │   └── strategies/                 # 15 example strategy YAML files
│   │
│   ├── backtest/
│   │   ├── config.yaml                 # Date range, balance, sizing, commission, slippage
│   │   ├── backtest.py                 # Vectorized backtest engine
│   │   └── main.py                     # Entry point: pulls data, signals, runs backtest, stores ledger
│   │
│   ├── simulator/
│   │   ├── simulator.py                # step_candle() paper-fill logic, TP/SL checks
│   │   └── main.py                     # Entry point (Task Scheduler): walks every active pair/strategy
│   │
│   ├── execution/
│   │   ├── bybit_client.py             # Bybit API client, order placement, symbol helpers
│   │   └── main.py                     # Entry point (Task Scheduler): live trading on Bybit
│   │
│   ├── accounts/
│   │   ├── accounts_utils.py           # accounts.api_keys / .history / .stats DB logic
│   │   ├── ledger_stats.py             # get_ledger_stats() — ~85-stat FIFO-based ledger stats
│   │   └── run_accounts.py             # Entry point (Task Scheduler): refresh account history/stats
│   │
│   ├── ml/
│   │   ├── config.yaml                 # Single config for the whole ML module
│   │   ├── main.py                     # Top-level entry point, routes by model_type
│   │   ├── data_prep/                  # Dataset/feature/sentiment/target pipelines
│   │   ├── preprocessing/              # Feature selection, scaling, stationarity
│   │   ├── pipeline/                   # Regression/classification/timeseries pipeline runners
│   │   ├── regressors/                 # Ridge, Lasso, ElasticNet, SVR, RF, XGBoost, LightGBM, CatBoost...
│   │   ├── classifiers/                # Logistic Regression, KNN, Naive Bayes, SVM, RF, boosting...
│   │   ├── deep_learning/              # MLP, LSTM, GRU (shared trainer/callbacks/losses)
│   │   ├── timeseries/                 # N-BEATS, TCN, StatsForecast, sklearn-classifier wrapper
│   │   ├── signals/                    # Convert model predictions into trading signals
│   │   ├── evaluation/                 # Regression/classification metrics, evaluator
│   │   ├── persistence/                # Model/artifact saving and loading
│   │   ├── inference/                  # Inference sanity checks
│   │   └── utils/                      # ML-specific logging
│   │
│   ├── preprocessing_lab/
│   │   ├── config.yaml                 # Which methods + target types to run
│   │   ├── main.py                     # run_experiment.py — applies registry methods, saves CSVs
│   │   ├── registry.py                 # PREPROCESSING_REGISTRY
│   │   ├── scalers.py / stationarity.py / distribution.py
│   │   ├── analysis/                   # Stationarity + trend-preservation analysis
│   │   └── model_evaluation/           # Backtests transformed features
│   │
│   ├── validation/
│   │   ├── config.yaml
│   │   ├── validate_targets.py         # Backtests ML targets to confirm they're profitable
│   │   └── threshold_analysis.py
│   │
│   ├── stats/
│   │   ├── config.yaml
│   │   ├── calculator.py               # compute_stats()
│   │   ├── metrics.py                  # Sharpe, Sortino, Calmar, drawdown, CAGR, etc.
│   │   ├── plots.py
│   │   ├── stats_runner.py             # Batch driver: run() over backtest results
│   │   └── utils.py
│   │
│   └── utils/
│       ├── db_utils.py                 # PostgreSQL connection, schema, insert, query
│       ├── metadata_utils.py           # metadata.strategy reads/writes
│       └── pipeline_utils.py           # Logging setup, config loading, date parsing
│
├── sentiment_pipeline/
│   ├── config.py                       # Coin config, model names, token limits
│   ├── database.py                     # PostgreSQL schemas (raw/clean), insert/fetch logic
│   ├── reddit_fetcher.py               # PRAW Reddit API wrapper
│   ├── text_cleaner.py                 # Text normalization (URLs, HTML, contractions)
│   ├── text_features.py                # spaCy NLP: tokenization, POS, NER, lemmatization, stemming
│   ├── sentiment_model.py              # CryptoBERT inference with chunking for long posts
│   ├── topic_classifier.py             # Zero-shot coin classification
│   ├── weighting.py                    # Log-scaled engagement weighting
│   ├── structured_output.py            # JSON packaging of results
│   ├── chunking.py                     # Token-aware text splitting (reusable)
│   ├── main.py                         # Full orchestration: fetch → clean → analyze → store
│   ├── .env                            # DB + Reddit API credentials (not committed)
│   └── requirements.txt
│
├── frontend/                           # React + Vite dashboard (early stage, not yet wired to a backend)
│   └── src/
│       ├── pages/                      # Dashboard, Backtests, Models, Strategies, Wallets, Sentiment, Deployment, Execution
│       ├── components/
│       └── layouts/
│
├── run_pipeline.bat                    # Task Scheduler: Binance + Bybit data pipelines
├── run_simulator.bat                   # Task Scheduler: paper-trading simulator
├── run_execution.bat                   # Task Scheduler: live Bybit execution
├── run_accounts.bat                    # Task Scheduler: Bybit account history/stats refresh
├── plot_indicators.py                  # Standalone Plotly indicator chart script
├── setup.py                            # Makes crypto_pipeline pip-installable (pip install -e .)
├── .env                                # Shared DB credentials (not committed)
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 12+
- Reddit API credentials (https://www.reddit.com/prefs/apps — create a "script" app)

### 1. Clone the repo

```bash
git clone https://github.com/h-arshadd/chain-predict.git
cd chain-predict
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

For **crypto_pipeline only:**
```bash
pip install -r requirements.txt
```

For **sentiment_pipeline**, go to its directory and install:
```bash
cd sentiment_pipeline
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl
cd ..
```

### 4. Install TA-Lib (crypto_pipeline only)

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

### 5. Configure environment variables

Create a `.env` file in the root directory:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chain_predict
DB_USER=postgres
DB_PASSWORD=your_password_here
```

**For sentiment_pipeline**, also create `sentiment_pipeline/.env`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chain_predict
DB_USER=postgres
DB_PASSWORD=your_password_here

REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=sentiment-pipeline/0.1
```

### 6. Set up PostgreSQL

Create the database:

```sql
CREATE DATABASE chain_predict;
```

Schemas and tables are created automatically on first run of each pipeline.

---

## Usage

### crypto_pipeline

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

### sentiment_pipeline

**Run the full sentiment pipeline:**

```bash
cd sentiment_pipeline
python main.py
```

**What it does in one run:**
1. Fetches new Reddit posts from configured subreddits (per coin in `config.py`)
2. Stores raw posts in PostgreSQL `raw.btc_posts` / `raw.eth_posts` / etc.
3. For each unprocessed post:
   - Cleans text (lowercase, remove URLs/HTML, normalize tickers, expand contractions)
   - Runs NLP: tokenization, stopword removal, lemmatization, stemming, POS tagging, NER
   - Analyzes sentiment with CryptoBERT (outputs: Bullish/Bearish/Neutral + confidence)
   - Classifies which coin the post is about (zero-shot classification)
   - Extracts mentioned tickers and entities
   - Computes engagement weight (log-scaled upvotes + comments)
4. Stores cleaned analysis in PostgreSQL `clean.btc_posts` / `clean.eth_posts` / etc.
5. Prints plain and **engagement-weighted mean sentiment** for each coin

**Example output:**
```
--- BTC ---
Fetched & stored 150 raw posts for BTC
47 posts to analyze for BTC
BTC mean sentiment: 0.32 | weighted mean: 0.38
--- ETH ---
Fetched & stored 120 raw posts for ETH
33 posts to analyze for ETH
ETH mean sentiment: 0.18 | weighted mean: 0.21
```

**Add a new coin:**

Edit `sentiment_pipeline/config.py`:

```python
COINS = {
    "BTC": {...},
    "ETH": {...},
    "SOL": {                           # New coin
        "subreddits": ["solana", "CryptoCurrency"],
        "search_query": "SOL OR Solana",
    },
}
```

The pipeline auto-detects the new coin. No code changes needed.

**Query results from PostgreSQL:**

```python
from sentiment_pipeline.database import get_db_connection, get_mean_score, get_weighted_mean_score

conn = get_db_connection()
btc_mean = get_mean_score(conn, "BTC")              # Plain average
btc_weighted = get_weighted_mean_score(conn, "BTC") # Engagement-weighted

# With time windows:
btc_1day = get_weighted_mean_score(conn, "BTC", days=1)
btc_7day = get_weighted_mean_score(conn, "BTC", days=7)
btc_1year = get_weighted_mean_score(conn, "BTC", days=365)
```

---

## Database Schema

### crypto_pipeline

Each exchange has its own schema. Tables follow the naming convention `{symbol}_{timeframe}`:

| Column   | Type             | Description                 |
|----------|------------------|-----------------------------|
| datetime | TIMESTAMP (PK)   | Candle open timestamp (UTC) |
| open     | DOUBLE PRECISION | Opening price               |
| high     | DOUBLE PRECISION | Highest price in the candle |
| low      | DOUBLE PRECISION | Lowest price in the candle  |
| close    | DOUBLE PRECISION | Closing price               |
| volume   | DOUBLE PRECISION | Trading volume              |

Example tables: `binance.btc_1m`, `bybit.doge_1m`

### sentiment_pipeline

Two schemas: `raw` (fetched posts) and `clean` (analyzed posts).

**`raw.btc_posts` / `raw.eth_posts` / etc.:**

| Column      | Type      | Description                 |
|-------------|-----------|----------------------------|
| post_id     | TEXT (PK) | Unique Reddit post ID       |
| subreddit   | TEXT      | Source subreddit            |
| title       | TEXT      | Post title                  |
| body        | TEXT      | Post body text              |
| created_utc | TIMESTAMP | Post creation time (UTC)    |
| score       | INTEGER   | Net upvotes (used for weight) |
| num_comments| INTEGER   | Comment count (used for weight) |
| upvote_ratio| FLOAT     | Ratio of upvotes           |
| fetched_at  | TIMESTAMP | When this post was fetched  |

**`clean.btc_posts` / `clean.eth_posts` / etc.:**

| Column          | Type              | Description                            |
|-----------------|-------------------|----------------------------------------|
| post_id         | TEXT (PK/FK)      | Links to raw.btc_posts                |
| clean_text      | TEXT              | Cleaned text (for storage/analysis)    |
| sentiment_label | TEXT              | Bullish / Bearish / Neutral            |
| sentiment_score | FLOAT (-1 to +1)  | Signed score (used for averaging)      |
| confidence      | FLOAT (0 to 1)    | Model confidence in sentiment           |
| topic           | TEXT              | Detected coin (BTC, ETH, etc.)         |
| topic_confidence| FLOAT (0 to 1)    | Confidence in coin detection           |
| tickers         | TEXT[]            | Array of mentioned tickers             |
| weight          | FLOAT             | Log-scaled engagement weight           |
| processed_at    | TIMESTAMP         | When this post was analyzed            |

---

## Models & NLP Pipeline

### sentiment_pipeline NLP Steps

Follows the classical NLP pipeline with modern transformer sentiment analysis:

1. **Text Collection** → PRAW Reddit API
2. **Text Cleaning** → lowercase, remove URLs/HTML, normalize tickers, expand contractions
3. **Tokenization** → spaCy word tokenizer
4. **Stop Word Removal** → spaCy (filters common words)
5. **Lemmatization** → spaCy (running → run, studies → study)
6. **Stemming** → NLTK PorterStemmer
7. **POS Tagging** → spaCy (mark NOUN, VERB, ADJ, etc.)
8. **Named Entity Recognition (NER)** → spaCy (extract named entities)
9. **Ticker Extraction** → Regex + config (find $BTC, ETH mentions)
10. **Sentiment Analysis** → **CryptoBERT** (Bullish/Bearish/Neutral)
11. **Topic Classification** → Zero-shot BART (which coin is post about?)
12. **Structured Output** → JSON packaging with all results

### Models

| Step | Model | Training Data | Purpose |
|------|-------|---------------|---------|
| Sentiment | ElKulako/cryptobert | Reddit/Twitter/StockTwits crypto posts | Crypto-specific sentiment (Bullish/Bearish/Neutral) |
| Topic Classification | facebook/bart-large-mnli | MNLI dataset | Zero-shot coin classification |
| NER/POS/Lemmatization | spacy en_core_web_sm | English web text | Multi-task NLP (tokenization, POS, NER, lemmas) |
| Stemming | NLTK PorterStemmer | — | Root word reduction |

### Handling Long Posts

BERT-family models max out at **512 tokens**. For longer posts:
- Text is split into 510-token chunks
- Each chunk is scored separately
- Probabilities are averaged to get final sentiment score

This prevents truncation and ensures consistent sentiment across long-form Reddit discussions.

---

## Configuration

### crypto_pipeline

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

### sentiment_pipeline

Edit `sentiment_pipeline/config.py`:

```python
COINS = {
    "BTC": {
        "subreddits": ["Bitcoin", "BitcoinMarkets", "CryptoCurrency"],
        "search_query": "BTC OR Bitcoin",
    },
    "ETH": {
        "subreddits": ["ethereum", "ethtrader", "CryptoCurrency"],
        "search_query": "ETH OR Ethereum",
    },
}

REDDIT_POST_LIMIT = 100  # Posts per subreddit per run
SENTIMENT_MODEL_NAME = "ElKulako/cryptobert"
MAX_TOKENS = 512  # BERT token limit
```

---

## Indicators (crypto_pipeline)

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

## Tech Stack

### crypto_pipeline
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

### sentiment_pipeline
- **Python 3.10**
- **PostgreSQL** — raw and cleaned data storage
- **PRAW 7.7.0** — Reddit API client
- **transformers 4.37.2** — HuggingFace models (CryptoBERT, BART)
- **torch 2.2.1** — deep learning framework
- **spacy 3.7.2** — NLP toolkit (tokenization, POS, NER, lemmatization)
- **nltk 3.8.1** — stemming (PorterStemmer)
- **psycopg2-binary** — PostgreSQL connector
- **emoji, contractions** — text cleaning
- **python-dotenv** — environment variable management

---

## Roadmap

- [ ] Combine crypto_pipeline OHLCV signals with sentiment_pipeline sentiment for ML feature engineering
- [ ] Daily/weekly/yearly rolling sentiment averages in sentiment_pipeline
- [ ] Real-time streaming sentiment updates (WebSocket)
- [ ] Backtesting module combining price action + sentiment signals
- [ ] REST API to query mean sentiment scores per coin
- [ ] Multi-asset support (stocks, forex, commodities) for sentiment_pipeline
- [ ] Fine-tune sentiment model on Neurog.ai internal trading forum data
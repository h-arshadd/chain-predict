"""
insert_strategy_and_sentiment.py
----------------------------------
One-off script: inserts the current strategy (from signals/config.yaml)
and sentiment coins (from sentiment_pipeline/config.yaml) into
metadata.strategy and metadata.sentiment.

Requires metadata.data to already be populated (run discover_data.py
first) -- the strategy needs a real data_id to attach to.

Run from the same folder as create_metadata_tables.py:

    python insert_strategy_and_sentiment.py
"""

from crypto_pipeline.utils.metadata_utils import (
    get_db_connection, get_data_rows, insert_strategy, insert_sentiment,
)

conn = get_db_connection()

# ------------------------------------------------------------
# Strategy: RSI 30/70 (the currently ACTIVE strategy in
# signals/config.yaml -- long when RSI < 30, short when RSI > 70)
# ------------------------------------------------------------
# Attach it to one (exchange, symbol) pair to start -- change these to
# whichever pair you actually want it tied to. If you want the SAME
# strategy attached to multiple pairs, call insert_strategy() again below
# with a different data_id (each call makes a new strategy row, since
# strategy isn't upserted -- see metadata_utils.py's insert_strategy docstring).
EXCHANGE = "binance"
SYMBOL = "btc"

data_rows = get_data_rows(conn, exchange=EXCHANGE, symbol=SYMBOL)
if not data_rows:
    raise RuntimeError(
        f"No metadata.data row for {EXCHANGE}/{SYMBOL} -- run discover_data.py first."
    )
data_id = data_rows[0]["data_id"]

strategy_config = {
    "RSI": [
        {
            "inputs": ["close"],
            "parameters": {"period": 14},
            "aliases": {"rsi": "ind_RSI_14"},
        }
    ],
    "strategy": {
        "long": {
            "rule": "AND",
            "conditions": [
                {"left": "ind_RSI_14", "operator": "<", "right": 30, "persist_bars": 0}
            ],
        },
        "short": {
            "rule": "AND",
            "conditions": [
                {"left": "ind_RSI_14", "operator": ">", "right": 70, "persist_bars": 0}
            ],
        },
    },
}

strategy_id = insert_strategy(conn, data_id, strategy_config, timeframe="1h")
print(f"Inserted strategy_id={strategy_id} for {EXCHANGE}/{SYMBOL} (data_id={data_id})")

# ------------------------------------------------------------
# Sentiment: BTC and ETH (from sentiment_pipeline/config.yaml)
# ------------------------------------------------------------
sentiment_id_btc = insert_sentiment(
    conn,
    coin="BTC",
    subreddits=["Bitcoin", "BitcoinMarkets", "CryptoCurrency"],
    search_query="BTC OR Bitcoin",
    post_limit=5,
)
print(f"Inserted sentiment_id={sentiment_id_btc} for BTC")

sentiment_id_eth = insert_sentiment(
    conn,
    coin="ETH",
    subreddits=["ethereum", "ethtrader", "CryptoCurrency"],
    search_query="ETH OR Ethereum",
    post_limit=5,
)
print(f"Inserted sentiment_id={sentiment_id_eth} for ETH")

conn.close()
print("Done.")
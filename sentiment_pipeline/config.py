"""
config.py
---------
Add a new coin here and the whole pipeline (fetch -> clean -> sentiment -> DB)
will pick it up automatically. Nothing else needs to change.
"""

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

# how many posts to pull per subreddit, per run
REDDIT_POST_LIMIT = 100

# HuggingFace model — trained on crypto social media text (Reddit/Twitter/StockTwits),
# outputs Bullish / Bearish / Neutral directly. Tokenizer is bundled with the model.
SENTIMENT_MODEL_NAME = "ElKulako/cryptobert"

# the model's own tokenizer limit (BERT-family = 512). Text longer than this gets
# split into chunks in sentiment_model.py, scored separately, then averaged.
MAX_TOKENS = 512
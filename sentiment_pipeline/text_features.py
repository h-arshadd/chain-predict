"""
text_features.py
----------------
Step 9: Extract crypto tickers from text.

NOTE: Steps 3-8 (tokenization, stopword removal, lemmatization, stemming, 
POS tagging, NER) are NOT used in this pipeline because CryptoBERT and BART 
(transformers) handle all that internally. These steps would only be useful 
if we were doing classic NLP analysis, which we're not.

We keep only ticker extraction because it's a simple regex operation 
that doesn't conflict with transformer models.
"""

import re
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
KNOWN_SYMBOLS = set(config["coins"].keys())


def extract_tickers(raw_text: str, known_symbols=None) -> list:
    """
    Extract uppercase 2-5 letter words that match known coins.
    
    Step 9: Ticker Extraction
    
    Args:
        raw_text: The original text (before cleaning, so tickers like $BTC are still present)
        known_symbols: Set of coin symbols to match against (defaults to config coins)
    
    Returns:
        Sorted list of detected tickers (e.g., ["BTC", "ETH"])
    
    Example:
        extract_tickers("I bought $BTC and ETH yesterday")
        → ["BTC", "ETH"]
    """
    if known_symbols is None:
        known_symbols = KNOWN_SYMBOLS
    
    # Find all uppercase 2-5 letter words
    candidates = set(TICKER_RE.findall(raw_text))
    
    # Filter to only known coins from config
    if known_symbols:
        return sorted(candidates & set(known_symbols))
    return sorted(candidates)
"""
text_cleaner.py
----------------
Step 2: Clean the Text

Removes noise from raw Reddit posts:
- URLs (https://..., www....)
- HTML tags (<br>, <p>, etc.)
- Emojis (🚀 → "rocket")
- Contractions (don't → do not)
- Dollar signs before tickers ($BTC → BTC)
- Extra whitespace

Keeps punctuation because transformers (CryptoBERT, BART) were trained
on text with punctuation and use it as a signal (e.g., !!! = excitement).

Duplicate posts aren't handled here — they're handled in database.py via 
post_id primary key + ON CONFLICT DO NOTHING.
"""

import re
import emoji
import contractions

URL_RE = re.compile(r"http\S+|www\.\S+")          # Remove URLs
HTML_RE = re.compile(r"<.*?>")                    # Remove HTML tags
TICKER_RE = re.compile(r"\$(\w+)")                # Normalize tickers ($BTC → BTC)
MULTI_SPACE_RE = re.compile(r"\s+")               # Clean extra spaces


def _base_clean(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub("", text)
    text = HTML_RE.sub("", text)
    text = emoji.demojize(text, delimiters=(" ", " "))    # emoji -> text
    text = contractions.fix(text)                          # can't -> cannot
    text = TICKER_RE.sub(r"\1", text)                      # $TSLA -> TSLA
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def clean_text_for_model(text: str) -> str:
    """
    Clean text for transformer models (CryptoBERT, BART).
    
    Returns cleaned text with:
    - Lowercase
    - URLs removed
    - HTML removed
    - Emojis converted to text
    - Contractions expanded (don't → do not)
    - Tickers normalized ($BTC → BTC)
    - Extra spaces cleaned
    - PUNCTUATION KEPT (transformers need it)
    
    Example:
        Input: "I don't like $BTC!!! 🚀 https://example.com"
        Output: "i do not like btc!!!  rocket "
    """
    return _base_clean(text)
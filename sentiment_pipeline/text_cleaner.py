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

URL_RE = re.compile(r"http\S+|www\.\S+")          # Remove bare URLs
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](url) -> text
MENTION_RE = re.compile(r"\b[ur]/[\w-]+")          # Remove u/username, r/subreddit mentions (names can have hyphens)
HTML_RE = re.compile(r"<.*?>")                    # Remove HTML tags
MD_RE = re.compile(r"\*+")                        # Remove markdown bold/italic (** ** or * *)
HASHTAG_RE = re.compile(r"#(\w+)")                # Normalize hashtags (#Bitcoin → Bitcoin)
TICKER_RE = re.compile(r"\$(\w+)")                # Normalize tickers ($BTC → BTC)
MULTI_SPACE_RE = re.compile(r"\s+")               # Clean extra spaces


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
    text = text.lower()
    text = MD_LINK_RE.sub(r"\1", text)                     # [text](url) -> text
    text = URL_RE.sub("", text)                            # any remaining bare urls
    text = HTML_RE.sub("", text)
    text = MD_RE.sub("", text)                             # ** ** / * * -> gone
    text = MENTION_RE.sub("", text)                        # u/name, r/name -> gone (no sentiment signal)
    text = emoji.demojize(text, delimiters=(" ", " "))    # emoji -> text
    text = contractions.fix(text)                          # can't -> cannot
    text = TICKER_RE.sub(r"\1", text)                      # $TSLA -> TSLA
    text = HASHTAG_RE.sub(r"\1", text)                     # #Bitcoin -> Bitcoin
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text
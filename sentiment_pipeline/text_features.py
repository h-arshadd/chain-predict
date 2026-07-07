"""
text_features.py
----------------
Steps 3-9: tokenization, stopword removal, lemmatization, stemming, POS, NER, ticker extraction.
"""

import re
import yaml
import spacy
from nltk.stem import PorterStemmer

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

_nlp = spacy.load("en_core_web_sm")
_stemmer = PorterStemmer()

TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
KNOWN_SYMBOLS = set(config["coins"].keys())


def analyze(text: str) -> dict:
    """One pass: tokenization, stopword removal, lemmatization, POS tagging, NER."""
    doc = _nlp(text)
    return {
        "tokens": [t.text for t in doc],
        "tokens_no_stopwords": [t.text for t in doc if not t.is_stop and not t.is_punct],
        "lemmas": [t.lemma_ for t in doc if not t.is_stop and not t.is_punct],
        "pos_tags": [(t.text, t.pos_) for t in doc],
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
    }


def stem_tokens(tokens: list) -> list:
    """e.g. ['running', 'studies'] -> ['run', 'studi']"""
    return [_stemmer.stem(t) for t in tokens]


def extract_tickers(raw_text: str, known_symbols=None) -> list:
    """Extract uppercase 2-5 letter words that match known coins."""
    if known_symbols is None:
        known_symbols = KNOWN_SYMBOLS
    candidates = set(TICKER_RE.findall(raw_text))
    if known_symbols:
        return sorted(candidates & set(known_symbols))
    return sorted(candidates)
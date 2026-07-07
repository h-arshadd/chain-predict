"""
nlp_extras.py
--------------
Steps 3-9 from the PDF: tokenization, stopword removal, lemmatization,
stemming, POS tagging, NER, ticker extraction.

These are "classic NLP" steps - good for exploring/debugging your data
later (e.g. "which entities/tickers get mentioned most with each coin"),
but they are NOT fed into the sentiment model. See sentiment_model.py's
docstring for why - short version: the transformer needs raw sentences,
not a bag of lemmas with stopwords stripped out.

Run these on text from text_cleaner.clean_text_traditional().

Uses spaCy's pretrained "en_core_web_sm" model - one pretrained model
covers tokenization, stopwords, lemmatization, POS tagging, AND NER, so
one parse of the text gets you 5 of these steps at once. NLTK's
PorterStemmer covers stemming separately, since spaCy doesn't do stemming
(it considers lemmatization the better modern replacement - but the PDF
wants both, so both are here).
"""

import re
import spacy
from nltk.stem import PorterStemmer
from config import COINS

_nlp = spacy.load("en_core_web_sm")
_stemmer = PorterStemmer()

TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
KNOWN_SYMBOLS = set(COINS.keys())  # pulled from config.py - add a coin there, it's known here too


def analyze(text: str) -> dict:
    """
    One pass over the text -> tokenization, stopword removal,
    lemmatization, POS tagging, and NER together.
    """
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


def extract_tickers(raw_text: str, known_symbols=KNOWN_SYMBOLS) -> list:
    """
    Finds uppercase 2-5 letter words. IMPORTANT: pass the ORIGINAL
    (uncleaned, not-lowercased) text here - ticker detection depends on
    case, and clean_text_for_model()/clean_text_traditional() both
    lowercase everything.

    Defaults to only matching coins listed in config.py, so random
    uppercase noise like "PSA" or "CEO" doesn't get picked up as a
    ticker. Pass known_symbols=None to get every uppercase match instead.
    """
    candidates = set(TICKER_RE.findall(raw_text))
    if known_symbols:
        return sorted(candidates & set(known_symbols))
    return sorted(candidates)
"""
text_cleaner.py
----------------
Step 2 from the PDF: cleaning.

Why two functions instead of one:
  clean_text_for_model()  -> KEEPS punctuation/casing. This is what goes
                             into the sentiment transformer. Transformers
                             actually use punctuation and capitalization as
                             signal, so stripping it would make the
                             sentiment model slightly dumber, not cleaner.

  clean_text_traditional() -> strips punctuation too. This is what the
                             classic NLP steps (stopword removal,
                             lemmatization, stemming, POS tagging, NER)
                             will run on in the next file, since those
                             techniques were built around bag-of-words
                             style text, not full sentences.

Duplicate posts aren't handled here — they're handled in db.py via the
post_id primary key + ON CONFLICT DO NOTHING, so the same post never gets
cleaned/scored twice even if fetched again.
"""

import re
import emoji
import contractions

URL_RE = re.compile(r"http\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
TICKER_RE = re.compile(r"\$(\w+)")          # $TSLA -> TSLA
PUNCT_RE = re.compile(r"[^\w\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


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
    """Feed this version to the sentiment model."""
    return _base_clean(text)


def clean_text_traditional(text: str) -> str:
    """Feed this version to tokenization/stopwords/lemmatization/POS/NER."""
    text = _base_clean(text)
    text = PUNCT_RE.sub("", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text
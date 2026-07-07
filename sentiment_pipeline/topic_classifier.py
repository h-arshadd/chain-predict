"""
topic_classifier.py
--------------------
Step 11 from the PDF, adjusted per your lead: classify which COIN a post
is mainly about (BTC, ETH, ...) instead of a generic finance topic
(earnings/inflation/mergers/etc).

Why this matters: posts get fetched from mixed subreddits like
r/CryptoCurrency, where a BTC search can still pull in a post that's
mostly about ETH. This tells you what the post is really about, on top
of which coin's search query happened to catch it.

Candidate labels come straight from config.COINS - add a coin there and
this file automatically starts recognizing it, no changes needed here.
That's the "interconnected" part: one source of truth for coin names.

Model: zero-shot classification (facebook/bart-large-mnli). "Zero-shot"
means it can sort text into any label list you hand it at request time -
no training/fine-tuning needed, which is exactly what we want since the
label list (coins) grows over time via config.py.
"""

from transformers import pipeline
from config import COINS
from chunking import split_into_token_chunks

_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
MAX_TOKENS = 1000  # bart-large-mnli caps at 1024 - leaving a little headroom


def classify_topic(text: str) -> dict:
    """
    Returns: {"topic": "BTC", "confidence": 0.87}
    Re-reads config.COINS on every call so a newly added coin is picked
    up immediately, without restarting anything mid-session.

    Long posts get split into chunks first (same idea as
    sentiment_model.py - every model has its own token limit and needs
    this check). Each chunk is classified separately, per-coin scores
    are averaged across chunks, then the highest-average coin wins.
    """
    candidate_labels = list(COINS.keys())
    chunks = split_into_token_chunks(_classifier.tokenizer, text, MAX_TOKENS)

    totals = {label: 0.0 for label in candidate_labels}
    for chunk in chunks:
        result = _classifier(chunk, candidate_labels=candidate_labels)
        for label, score in zip(result["labels"], result["scores"]):
            totals[label] += score

    avg_scores = {label: total / len(chunks) for label, total in totals.items()}
    top_label = max(avg_scores, key=avg_scores.get)
    return {"topic": top_label, "confidence": avg_scores[top_label]}
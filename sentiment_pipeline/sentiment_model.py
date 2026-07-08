"""
sentiment_model.py
-------------------
Sentiment analysis using CryptoBERT.
"""

import math
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHUNK_SIZE = 510
CHUNK_OVERLAP = 50  # tokens re-shown at the start of the next chunk

_tokenizer = AutoTokenizer.from_pretrained(config["model"]["sentiment_model"])
_model = AutoModelForSequenceClassification.from_pretrained(config["model"]["sentiment_model"])
_model.eval()

_id2label = _model.config.id2label
_label2id = {v.lower(): k for k, v in _id2label.items()}


def split_into_token_chunks(tokenizer, text: str, max_tokens: int, overlap: int = 0) -> list:
    """
    Returns a list of decoded text chunks, each within max_tokens according
    to the given tokenizer. A short post just comes back as a single-item
    list, so callers can always loop over the result the same way whether
    chunking actually happened or not.

    overlap: how many tokens from the end of one chunk are repeated at the
    start of the next. This means a sentence cut in half at a chunk boundary
    still shows up whole in the following chunk, instead of being lost.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return [""]

    step = max_tokens - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than max_tokens")

    token_chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), step)]
    return [tokenizer.decode(chunk) for chunk in token_chunks]


def _score_chunk(text: str) -> torch.Tensor:
    """Score a single text chunk, return probability distribution."""
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = _model(**inputs).logits
    return F.softmax(logits, dim=1)[0]


def _chunk_confidence(probs: torch.Tensor) -> float:
    """
    How informative/confident a chunk's prediction is, 0-1.
    A near-uniform distribution (model shrugging, e.g. no real signal in
    that chunk) scores near 0. A peaked distribution (model is sure)
    scores near 1. Based on normalized entropy, so it works for any
    number of classes.
    """
    eps = 1e-9
    entropy = -(probs * (probs + eps).log()).sum()
    max_entropy = math.log(probs.shape[0])
    return float(1 - entropy / max_entropy)


def get_sentiment(text: str) -> dict:
    """
    Analyze sentiment of text using CryptoBERT.

    Args:
        text: Text to analyze

    Returns:
        Dict with keys:
        - label: "Bullish", "Bearish", or "Neutral"
        - confidence: 0-1 confidence score
        - score: -1 to 1 (bullish_prob - bearish_prob)
    """
    chunks = split_into_token_chunks(_tokenizer, text, CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    probs_per_chunk = [_score_chunk(chunk) for chunk in chunks]

    weights = torch.tensor([_chunk_confidence(p) for p in probs_per_chunk])

    if weights.sum() < 1e-6:
        # every chunk was basically a shrug -- fall back to a plain average
        # instead of dividing by ~0
        avg_probs = torch.stack(probs_per_chunk).mean(dim=0)
    else:
        weights = weights / weights.sum()
        avg_probs = (torch.stack(probs_per_chunk) * weights.unsqueeze(1)).sum(dim=0)

    label_idx = int(torch.argmax(avg_probs))
    label = _id2label[label_idx]
    confidence = float(avg_probs[label_idx])

    bullish_idx = _label2id["bullish"]
    bearish_idx = _label2id["bearish"]
    score = float(avg_probs[bullish_idx] - avg_probs[bearish_idx])

    return {"label": label, "confidence": confidence, "score": score}
"""
sentiment_model.py
-------------------
Sentiment analysis using CryptoBERT.
"""

import yaml
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHUNK_SIZE = 510

_tokenizer = AutoTokenizer.from_pretrained(config["model"]["sentiment_model"])
_model = AutoModelForSequenceClassification.from_pretrained(config["model"]["sentiment_model"])
_model.eval()

_id2label = _model.config.id2label
_label2id = {v.lower(): k for k, v in _id2label.items()}


def split_into_token_chunks(tokenizer, text: str, max_tokens: int) -> list:
    """
    Returns a list of decoded text chunks, each within max_tokens according
    to the given tokenizer. A short post just comes back as a single-item
    list, so callers can always loop over the result the same way whether
    chunking actually happened or not.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return [""]

    token_chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in token_chunks]

def _score_chunk(text: str) -> torch.Tensor:
    """Score a single text chunk, return probability distribution."""
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = _model(**inputs).logits
    return F.softmax(logits, dim=1)[0]


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
    chunks = split_into_token_chunks(_tokenizer, text, CHUNK_SIZE)
    probs_per_chunk = [_score_chunk(chunk) for chunk in chunks]
    avg_probs = torch.stack(probs_per_chunk).mean(dim=0)

    label_idx = int(torch.argmax(avg_probs))
    label = _id2label[label_idx]
    confidence = float(avg_probs[label_idx])

    bullish_idx = _label2id["bullish"]
    bearish_idx = _label2id["bearish"]
    score = float(avg_probs[bullish_idx] - avg_probs[bearish_idx])

    return {"label": label, "confidence": confidence, "score": score}
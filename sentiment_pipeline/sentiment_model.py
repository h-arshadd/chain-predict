"""
sentiment_model.py
-------------------
Step 10: sentiment analysis using CryptoBERT.
"""

import yaml
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from chunking import split_into_token_chunks

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHUNK_SIZE = 510

_tokenizer = AutoTokenizer.from_pretrained(config["model"]["sentiment_model"])
_model = AutoModelForSequenceClassification.from_pretrained(config["model"]["sentiment_model"])
_model.eval()

_id2label = _model.config.id2label
_label2id = {v.lower(): k for k, v in _id2label.items()}


def _score_chunk(text: str) -> torch.Tensor:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = _model(**inputs).logits
    return F.softmax(logits, dim=1)[0]


def get_sentiment(text: str) -> dict:
    """Returns: {"label": "Bullish"/"Bearish"/"Neutral", "confidence": 0-1, "score": -1 to 1}"""
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
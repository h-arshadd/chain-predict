"""
sentiment_model.py
-------------------
Step 10 from the PDF: sentiment analysis.

Model: ElKulako/cryptobert (HuggingFace) — a BERT model fine-tuned on
crypto social media posts (Reddit/Twitter/StockTwits), outputs
Bearish / Neutral / Bullish directly. It comes with its own tokenizer,
so tokenization (step 3) for this branch of the pipeline is just
"whatever this model's tokenizer does" — no separate step needed.

Chunking: BERT-family models cap out around 512 tokens. If a post is
longer, we split its tokens into 510-token pieces (leaving room for the
model's own [CLS]/[SEP] markers), score each piece separately, then
average the probabilities across pieces to get one final score.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from chunking import split_into_token_chunks

MODEL_NAME = "ElKulako/cryptobert"
CHUNK_SIZE = 510  # leaves room for the model's own [CLS]/[SEP] markers

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
_model.eval()

# read straight from the model config instead of hardcoding label order,
# e.g. {0: "Bearish", 1: "Neutral", 2: "Bullish"}
_id2label = _model.config.id2label
_label2id = {v.lower(): k for k, v in _id2label.items()}


def _score_chunk(text: str) -> torch.Tensor:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = _model(**inputs).logits
    return F.softmax(logits, dim=1)[0]


def get_sentiment(text: str) -> dict:
    """
    Returns: {"label": "Bullish"/"Bearish"/"Neutral", "confidence": 0-1, "score": -1 to 1}
    score is signed: positive = bullish, negative = bearish, ~0 = neutral.
    That's the number you'll average later for the daily/weekly/yearly mean.
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
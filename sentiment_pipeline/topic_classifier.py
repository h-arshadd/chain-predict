"""
topic_classifier.py
--------------------
Step 11: Zero-shot coin classification.
"""

import yaml
from transformers import pipeline
from chunking import split_into_token_chunks

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

_classifier = pipeline("zero-shot-classification", model=config["model"]["topic_model"])


def classify_topic(text: str) -> dict:
    """Returns: {"topic": "BTC", "confidence": 0.87}"""
    candidate_labels = list(config["coins"].keys())
    chunks = split_into_token_chunks(_classifier.tokenizer, text, config["model"]["topic_max_tokens"])

    totals = {label: 0.0 for label in candidate_labels}
    for chunk in chunks:
        result = _classifier(chunk, candidate_labels=candidate_labels)
        for label, score in zip(result["labels"], result["scores"]):
            totals[label] += score

    avg_scores = {label: total / len(chunks) for label, total in totals.items()}
    top_label = max(avg_scores, key=avg_scores.get)
    return {"topic": top_label, "confidence": avg_scores[top_label]}
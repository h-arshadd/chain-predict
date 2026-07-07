"""
structured_output.py
---------------------
Step 14 from the PDF: structured JSON output.

No new model here - by this point every piece (sentiment, topic, tickers,
weight) has already been computed by the earlier steps and is sitting in
clean.<coin>_posts. This file's only job is to package it into the clean
JSON shape the PDF asked for, either fresh (right after analysis, in
main.py) or later by pulling a row back out of the DB.

PDF's example shape was:
    {"ticker": "TSLA", "sentiment": "Bullish", "confidence": 0.94,
     "topic": "Earnings", "summary": "..."}

Since summarization got dropped (whole post gets passed through instead),
"summary" here is just the cleaned post text itself - renamed to make
that clear rather than pretending it's a generated summary.
"""

from database import get_db_connection
from psycopg2 import sql


def build_output(coin, post_id, clean_text, sentiment, topic, weight):
    """Assemble the JSON object right after analysis, no DB round-trip needed."""
    return {
        "post_id": post_id,
        "coin": coin,
        "ticker": topic["topic"],
        "sentiment": sentiment["label"],
        "confidence": round(sentiment["confidence"], 4),
        "topic": topic["topic"],
        "topic_confidence": round(topic["confidence"], 4),
        "weight": round(weight, 4),
        "text": clean_text,
    }


def get_structured_output(conn, coin, post_id):
    """Rebuild the same JSON shape later, by pulling an already-analyzed
    post back out of the DB (e.g. for an API endpoint or a report)."""
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()
    cur.execute(sql.SQL("""
        SELECT post_id, clean_text, sentiment_label, sentiment_score,
               confidence, topic, topic_confidence, weight
        FROM clean.{table}
        WHERE post_id = %s
    """).format(table=sql.Identifier(table)), (post_id,))
    row = cur.fetchone()
    cur.close()

    if row is None:
        return None

    post_id, clean_text, label, score, confidence, topic, topic_confidence, weight = row
    return {
        "post_id": post_id,
        "coin": coin,
        "ticker": topic,
        "sentiment": label,
        "confidence": round(confidence, 4),
        "topic": topic,
        "topic_confidence": round(topic_confidence, 4),
        "weight": round(weight, 4),
        "text": clean_text,
    }
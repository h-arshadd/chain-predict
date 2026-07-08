"""
structured_output.py
---------------------
Structured JSON output for sentiment results.

By this point sentiment has already been computed by the earlier steps
and is sitting in sentiment_clean.<coin>_posts. This file's only job is
to package it into a clean JSON shape, either fresh (right after
analysis, in main.py) or later by pulling a row back out of the DB.

Topic/ticker classification has been removed from the pipeline, so those
fields are no longer produced or stored. There's no weighting logic
either — the pipeline reports plain (unweighted) sentiment only.
"""

from database import get_db_connection
from psycopg2 import sql


def build_output(coin, post_id, clean_text, sentiment):
    """Assemble the JSON object right after analysis, no DB round-trip needed."""
    return {
        "post_id": post_id,
        "coin": coin,
        "sentiment": sentiment["label"],
        "confidence": round(sentiment["confidence"], 4),
        "text": clean_text,
    }


def get_structured_output(conn, coin, post_id):
    """Rebuild the same JSON shape later, by pulling an already-analyzed
    post back out of the DB (e.g. for an API endpoint or a report)."""
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()
    cur.execute(sql.SQL("""
        SELECT post_id, clean_text, sentiment_label, sentiment_score, confidence
        FROM sentiment_clean.{table}
        WHERE post_id = %s
    """).format(table=sql.Identifier(table)), (post_id,))
    row = cur.fetchone()
    cur.close()

    if row is None:
        return None

    post_id, clean_text, label, score, confidence = row
    return {
        "post_id": post_id,
        "coin": coin,
        "sentiment": label,
        "confidence": round(confidence, 4),
        "text": clean_text,
    }
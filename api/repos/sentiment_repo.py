"""
repos/sentiment_repo.py
------------------------
DB access for the NLP & Sentiment page (spec section 10).

Real data source: sentiment_clean.{coin}_posts (one table per coin,
created by sentiment_pipeline/database.py::create_tables, populated by
sentiment_pipeline/main.py -- CryptoBERT-scored Reddit posts, see
sentiment_pipeline/sentiment_model.py). Each row has sentiment_label
("Bullish"/"Neutral"/"Bearish"), sentiment_score (-1..+1,
bullish_prob - bearish_prob), confidence (0-1), plus the post's own
score/upvote_ratio/created_utc/subreddit/title/body.

Two things NOT real anywhere in this codebase, handled honestly rather
than invented:
  - "News Sentiment" (PDF spec wording) -- there is no news source, only
    Reddit. This module surfaces real Reddit posts (top ones by score),
    labeled as such, not as "news" from a source that doesn't exist.
  - Fear & Greed Index -- no external index (e.g. alternative.me) is
    fetched anywhere. What IS real: a derived 0-100 score computed from
    the actual stored bullish/neutral/bearish distribution, same shape
    as the real F&G index but sourced from this pipeline's own data --
    clearly labeled "derived" so it's never confused with the real
    external index.

Coin list is NOT hardcoded to BTC/ETH (the only two sentiment_pipeline/
config.yaml currently configures) -- discovered live from
information_schema, same pattern as
metadata_utils.find_existing_candle_tables/discover_data_pairs, so new
coins show up automatically as their sentiment_clean.{coin}_posts table
is created and populated, no code change needed here.
"""

from psycopg2 import sql
from psycopg2.extras import RealDictCursor


def discover_sentiment_coins(conn) -> list[str]:
    """
    Every coin with a real sentiment_clean.{coin}_posts table right now,
    uppercased (e.g. ["BTC", "ETH"]). Read-only introspection, same
    pattern as metadata_utils.find_existing_candle_tables -- returns
    whatever actually exists, not a hardcoded/config-driven list.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'sentiment_clean'
          AND table_name LIKE %s
        ORDER BY table_name
    """), ("%\\_posts",))
    rows = cursor.fetchall()
    cursor.close()

    coins = []
    for (table_name,) in rows:
        coin = table_name[: -len("_posts")]
        coins.append(coin.upper())
    return coins


def _table_exists(conn, coin: str) -> bool:
    cursor = conn.cursor()
    qualified_name = f"sentiment_clean.{coin.lower()}_posts"
    cursor.execute(sql.SQL("SELECT to_regclass(%s)"), (qualified_name,))
    exists = cursor.fetchone()[0] is not None
    cursor.close()
    return exists


def _label_counts(conn, coin: str) -> dict:
    """
    Raw bullish/neutral/bearish post counts for this coin, all-time --
    the input both the Overall Sentiment breakdown and the derived
    Fear & Greed score are built from. Empty dict (all zeros) if the
    coin's table doesn't exist or has no rows yet.
    """
    table = f"{coin.lower()}_posts"
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        SELECT LOWER(sentiment_label), COUNT(*)
        FROM sentiment_clean.{table}
        GROUP BY LOWER(sentiment_label)
    """).format(table=sql.Identifier(table)))
    rows = dict(cursor.fetchall())
    cursor.close()
    return {
        "bullish": rows.get("bullish", 0),
        "neutral": rows.get("neutral", 0),
        "bearish": rows.get("bearish", 0),
    }


def _mean_score(conn, coin: str):
    """Plain mean of sentiment_score (-1..+1) across all posts for this coin. None if no rows."""
    table = f"{coin.lower()}_posts"
    cursor = conn.cursor()
    cursor.execute(sql.SQL("SELECT AVG(sentiment_score) FROM sentiment_clean.{table}").format(
        table=sql.Identifier(table)
    ))
    result = cursor.fetchone()[0]
    cursor.close()
    return float(result) if result is not None else None


def get_overall_sentiment(conn, coin: str) -> dict:
    """
    Overall Market Sentiment widget: mean score (-1..+1), a Bullish/
    Neutral/Bearish label off that mean, and bullish/neutral/bearish
    post percentages. All real, all from sentiment_clean.{coin}_posts.
    Every field is None/0 (not fabricated) if the coin has no table or
    no posts yet.
    """
    if not _table_exists(conn, coin):
        return {
            "coin": coin, "score": None, "label": None,
            "bullish_pct": 0.0, "neutral_pct": 0.0, "bearish_pct": 0.0,
            "post_count": 0,
        }

    counts = _label_counts(conn, coin)
    total = sum(counts.values())
    score = _mean_score(conn, coin)

    if score is None:
        label = None
    elif score > 0.15:
        label = "Bullish"
    elif score < -0.15:
        label = "Bearish"
    else:
        label = "Neutral"

    def pct(n):
        return round((n / total) * 100.0, 1) if total else 0.0

    return {
        "coin": coin,
        "score": round(score, 4) if score is not None else None,
        "label": label,
        "bullish_pct": pct(counts["bullish"]),
        "neutral_pct": pct(counts["neutral"]),
        "bearish_pct": pct(counts["bearish"]),
        "post_count": total,
    }


def get_fear_greed(conn, coin: str) -> dict:
    """
    Derived 0-100 Fear & Greed-style score, computed from this coin's
    real bullish/neutral/bearish post distribution -- NOT the real
    external Fear & Greed Index (that would require fetching
    alternative.me or similar, not built here). Formula: start at 50
    (neutral), shift by the bullish-minus-bearish share of all labeled
    posts, scaled to fill the 0-100 range. E.g. 100% bullish -> 100,
    100% bearish -> 0, even split -> 50.

    yesterday/last_week/last_month are the same score computed from
    posts up to that point back in time, using created_utc, so the
    caller can show real movement over time -- not fabricated deltas.
    None fields mean "not enough posts in that window yet", not zero.
    """
    from datetime import datetime, timedelta, timezone

    if not _table_exists(conn, coin):
        return {"coin": coin, "score": None, "label": None,
                "yesterday": None, "last_week": None, "last_month": None}

    def score_as_of(cutoff) -> float | None:
        table = f"{coin.lower()}_posts"
        cursor = conn.cursor()
        if cutoff is None:
            cursor.execute(sql.SQL("""
                SELECT LOWER(sentiment_label), COUNT(*) FROM sentiment_clean.{table}
                GROUP BY LOWER(sentiment_label)
            """).format(table=sql.Identifier(table)))
        else:
            cursor.execute(sql.SQL("""
                SELECT LOWER(sentiment_label), COUNT(*) FROM sentiment_clean.{table}
                WHERE created_utc <= %s
                GROUP BY LOWER(sentiment_label)
            """).format(table=sql.Identifier(table)), (cutoff,))
        rows = dict(cursor.fetchall())
        cursor.close()

        bullish = rows.get("bullish", 0)
        bearish = rows.get("bearish", 0)
        neutral = rows.get("neutral", 0)
        total = bullish + bearish + neutral
        if total == 0:
            return None
        net = (bullish - bearish) / total  # -1..+1
        return round(50 + net * 50, 1)      # 0..100

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    def fg_label(score):
        if score is None:
            return None
        if score < 20:
            return "Extreme Fear"
        if score < 40:
            return "Fear"
        if score < 60:
            return "Neutral"
        if score < 80:
            return "Greed"
        return "Extreme Greed"

    current = score_as_of(None)
    return {
        "coin": coin,
        "score": current,
        "label": fg_label(current),
        "yesterday": score_as_of(now - timedelta(days=1)),
        "last_week": score_as_of(now - timedelta(days=7)),
        "last_month": score_as_of(now - timedelta(days=30)),
    }


def get_sentiment_timeline(conn, coin: str, days: int = 30) -> list[dict]:
    """
    Daily mean sentiment_score over the last `days` days -- real numeric
    series for the Sentiment Timeline chart, one point per day that has
    at least one post (no fabricated interpolation for empty days).
    """
    if not _table_exists(conn, coin):
        return []

    table = f"{coin.lower()}_posts"
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("""
        SELECT DATE_TRUNC('day', created_utc) AS day,
               AVG(sentiment_score) AS avg_score,
               COUNT(*) AS post_count
        FROM sentiment_clean.{table}
        WHERE created_utc >= now() - (%s || ' days')::interval
        GROUP BY DATE_TRUNC('day', created_utc)
        ORDER BY day
    """).format(table=sql.Identifier(table)), (days,))
    rows = [dict(r) for r in cursor.fetchall()]
    cursor.close()
    return [
        {"date": r["day"], "score": round(float(r["avg_score"]), 4), "post_count": r["post_count"]}
        for r in rows
    ]


def get_fear_greed_timeline(conn, coin: str, days: int = 30) -> list[dict]:
    """
    Daily derived Fear & Greed score (see get_fear_greed's formula) for
    each of the last `days` days that has at least one post -- built
    from the same daily bullish/neutral/bearish counts, not a separate
    data source.
    """
    if not _table_exists(conn, coin):
        return []

    table = f"{coin.lower()}_posts"
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("""
        SELECT DATE_TRUNC('day', created_utc) AS day,
               COUNT(*) FILTER (WHERE LOWER(sentiment_label) = 'bullish') AS bullish,
               COUNT(*) FILTER (WHERE LOWER(sentiment_label) = 'bearish') AS bearish,
               COUNT(*) AS total
        FROM sentiment_clean.{table}
        WHERE created_utc >= now() - (%s || ' days')::interval
        GROUP BY DATE_TRUNC('day', created_utc)
        ORDER BY day
    """).format(table=sql.Identifier(table)), (days,))
    rows = [dict(r) for r in cursor.fetchall()]
    cursor.close()

    out = []
    for r in rows:
        net = (r["bullish"] - r["bearish"]) / r["total"] if r["total"] else 0.0
        out.append({"date": r["day"], "score": round(50 + net * 50, 1), "post_count": r["total"]})
    return out


def get_post_volume(conn, coin: str, days: int = 14) -> list[dict]:
    """
    Daily post counts split by label -- real numeric series for the
    (renamed) Post Volume chart. bullish/neutral/bearish counts per day,
    last `days` days, days with zero posts simply don't appear (no
    fabricated zero-fill).
    """
    if not _table_exists(conn, coin):
        return []

    table = f"{coin.lower()}_posts"
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("""
        SELECT DATE_TRUNC('day', created_utc) AS day,
               COUNT(*) FILTER (WHERE LOWER(sentiment_label) = 'bullish') AS bullish,
               COUNT(*) FILTER (WHERE LOWER(sentiment_label) = 'neutral') AS neutral,
               COUNT(*) FILTER (WHERE LOWER(sentiment_label) = 'bearish') AS bearish
        FROM sentiment_clean.{table}
        WHERE created_utc >= now() - (%s || ' days')::interval
        GROUP BY DATE_TRUNC('day', created_utc)
        ORDER BY day
    """).format(table=sql.Identifier(table)), (days,))
    rows = [dict(r) for r in cursor.fetchall()]
    cursor.close()
    return rows


def get_top_posts(conn, coin: str, limit: int = 20) -> list[dict]:
    """
    Real Reddit posts for this coin, most-upvoted first (post `score`,
    the same Reddit score reddit_fetcher.py stored, not a sentiment
    score) -- this is what the frontend shows in place of the PDF's
    "News Sentiment" table, since there is no news source in this
    codebase, only Reddit. title/subreddit/sentiment_label/
    sentiment_score/created_utc are all real stored columns; there is no
    stored permalink/URL (reddit_fetcher.py never captured one), so
    posts are not clickable -- shown as subreddit + title only.
    """
    if not _table_exists(conn, coin):
        return []

    table = f"{coin.lower()}_posts"
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("""
        SELECT post_id, subreddit, title, sentiment_label, sentiment_score,
               confidence, score, upvote_ratio, created_utc
        FROM sentiment_clean.{table}
        ORDER BY score DESC NULLS LAST, created_utc DESC
        LIMIT %s
    """).format(table=sql.Identifier(table)), (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    cursor.close()
    return rows
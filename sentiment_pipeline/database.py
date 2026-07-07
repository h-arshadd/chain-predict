"""
db.py
-----
DB utilities for the sentiment pipeline. Separate module, separate concern
from the chain-predict OHLCV db_utils.py — only similarity is style/pattern.

Two schemas:
    raw   - raw fetched Reddit posts, one table per coin (e.g. raw.btc_posts)
    clean - cleaned text + sentiment score, one table per coin (e.g. clean.btc_posts)

A row only exists in `clean` once it's been scored, so "posts in raw with no
matching row in clean" = the work queue for the sentiment model.
"""

import os
import logging
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    logger.info("Database connection established successfully.")
    return conn


def create_tables(conn, coin):
    """Create raw/clean schemas (if missing) and this coin's tables."""
    coin = coin.lower()
    table = f"{coin}_posts"
    cur = conn.cursor()

    cur.execute("CREATE SCHEMA IF NOT EXISTS raw")
    cur.execute("CREATE SCHEMA IF NOT EXISTS clean")

    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS raw.{table} (
            post_id       TEXT PRIMARY KEY,
            subreddit     TEXT,
            title         TEXT,
            body          TEXT,
            created_utc   TIMESTAMP,
            score         INTEGER,
            num_comments  INTEGER,
            upvote_ratio  DOUBLE PRECISION,
            fetched_at    TIMESTAMP DEFAULT NOW()
        )
    """).format(table=sql.Identifier(table)))

    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS clean.{table} (
            post_id           TEXT PRIMARY KEY REFERENCES raw.{table}(post_id),
            clean_text        TEXT,
            sentiment_label   TEXT,
            sentiment_score   DOUBLE PRECISION,
            confidence        DOUBLE PRECISION,
            topic             TEXT,
            topic_confidence  DOUBLE PRECISION,
            tickers           TEXT[],
            weight            DOUBLE PRECISION,
            processed_at      TIMESTAMP DEFAULT NOW()
        )
    """).format(table=sql.Identifier(table)))

    conn.commit()
    cur.close()
    logger.info(f"Tables ensured: raw.{table}, clean.{table}")


def insert_raw_posts(conn, coin, posts):
    """posts: list of dicts with post_id, subreddit, title, body, created_utc.
    Duplicate post_ids (re-fetching the same post) are just ignored."""
    if not posts:
        return
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()

    query = sql.SQL("""
        INSERT INTO raw.{table} (post_id, subreddit, title, body, created_utc, score, num_comments, upvote_ratio)
        VALUES %s
        ON CONFLICT (post_id) DO NOTHING
    """).format(table=sql.Identifier(table)).as_string(conn)

    rows = [(
        p["post_id"], p["subreddit"], p["title"], p["body"], p["created_utc"],
        p["score"], p["num_comments"], p["upvote_ratio"],
    ) for p in posts]
    execute_values(cur, query, rows)

    conn.commit()
    cur.close()


def get_unprocessed_posts(conn, coin):
    """Posts sitting in raw that don't have a sentiment row in clean yet.
    Returns score/num_comments too, since main.py needs them to compute weight."""
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()
    cur.execute(sql.SQL("""
        SELECT r.post_id, r.title, r.body, r.score, r.num_comments
        FROM raw.{table} r
        LEFT JOIN clean.{table} c ON r.post_id = c.post_id
        WHERE c.post_id IS NULL
    """).format(table=sql.Identifier(table)))
    rows = cur.fetchall()
    cur.close()
    return rows  # [(post_id, title, body, score, num_comments), ...]


def insert_analysis(conn, coin, post_id, clean_text, sentiment, topic, tickers, weight):
    """
    sentiment: {"label", "score", "confidence"} from sentiment_model.get_sentiment()
    topic: {"topic", "confidence"} from topic_classifier.classify_topic()
    tickers: list of strings from text_features.extract_tickers()
    weight: float from weighting.compute_weight()
    """
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()
    cur.execute(sql.SQL("""
        INSERT INTO clean.{table}
            (post_id, clean_text, sentiment_label, sentiment_score, confidence,
             topic, topic_confidence, tickers, weight)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (post_id) DO NOTHING
    """).format(table=sql.Identifier(table)), (
        post_id, clean_text,
        sentiment["label"], sentiment["score"], sentiment["confidence"],
        topic["topic"], topic["confidence"], tickers, weight,
    ))
    conn.commit()
    cur.close()


def get_mean_score(conn, coin, days=None):
    """
    Plain (unweighted) mean sentiment score for a coin. Pass days=1 / days=3 /
    days=365 for the 1-day / 3-day / yearly windows your lead mentioned -
    that's the only change needed when you build those filters later.
    """
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()

    if days is None:
        cur.execute(sql.SQL("SELECT AVG(sentiment_score) FROM clean.{table}").format(
            table=sql.Identifier(table)
        ))
    else:
        cur.execute(sql.SQL("""
            SELECT AVG(sentiment_score) FROM clean.{table}
            WHERE processed_at > NOW() - INTERVAL %s
        """).format(table=sql.Identifier(table)), (f"{days} days",))

    result = cur.fetchone()[0]
    cur.close()
    return result


def get_weighted_mean_score(conn, coin, days=None):
    """
    Engagement-weighted mean: posts with higher weight (more upvotes/comments)
    count for more. Standard weighted-average formula:
    sum(score * weight) / sum(weight).
    """
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()

    base_query = """
        SELECT SUM(sentiment_score * weight) / NULLIF(SUM(weight), 0)
        FROM clean.{table}
    """
    if days is None:
        cur.execute(sql.SQL(base_query).format(table=sql.Identifier(table)))
    else:
        cur.execute(sql.SQL(base_query + " WHERE processed_at > NOW() - INTERVAL %s").format(
            table=sql.Identifier(table)
        ), (f"{days} days",))

    result = cur.fetchone()[0]
    cur.close()
    return result
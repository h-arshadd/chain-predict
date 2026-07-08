"""
database.py
-----
DB utilities for sentiment pipeline.

Two schemas:
    sentiment_raw   - raw fetched Reddit posts, one table per coin.
                       Top comments stored as JSONB in the `comments` column
                       (list of {comment_id, body, score, created_utc}).
    sentiment_clean - cleaned title/body/comments (comments joined into one
                       text column) + sentiment score, one table per coin.
"""

import os
import logging
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values, Json
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
    """Create sentiment_raw/sentiment_clean schemas (if missing) and this coin's tables."""
    coin = coin.lower()
    table = f"{coin}_posts"
    cur = conn.cursor()

    cur.execute("CREATE SCHEMA IF NOT EXISTS sentiment_raw")
    cur.execute("CREATE SCHEMA IF NOT EXISTS sentiment_clean")

    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS sentiment_raw.{table} (
            post_id       TEXT PRIMARY KEY,
            subreddit     TEXT,
            title         TEXT,
            body          TEXT,
            comments      JSONB,
            created_utc   TIMESTAMP,
            score         INTEGER,
            upvote_ratio  DOUBLE PRECISION
        )
    """).format(table=sql.Identifier(table)))

    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS sentiment_clean.{table} (
            post_id           TEXT PRIMARY KEY REFERENCES sentiment_raw.{table}(post_id),
            title             TEXT,
            body              TEXT,
            comments          TEXT,
            sentiment_label   TEXT,
            sentiment_score   DOUBLE PRECISION,
            confidence        DOUBLE PRECISION
        )
    """).format(table=sql.Identifier(table)))

    conn.commit()
    cur.close()
    logger.info(f"Tables ensured: sentiment_raw.{table}, sentiment_clean.{table}")


def insert_raw_posts(conn, coin, posts):
    """posts: list of dicts with post_id, subreddit, title, body, created_utc."""
    if not posts:
        return
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()

    query = sql.SQL("""
        INSERT INTO sentiment_raw.{table} (post_id, subreddit, title, body, created_utc, score, upvote_ratio)
        VALUES %s
        ON CONFLICT (post_id) DO NOTHING
    """).format(table=sql.Identifier(table)).as_string(conn)

    rows = [(
        p["post_id"], p["subreddit"], p["title"], p["body"], p["created_utc"],
        p["score"], p["upvote_ratio"],
    ) for p in posts]
    execute_values(cur, query, rows)

    conn.commit()
    cur.close()


def insert_post_comments(conn, coin, post_comments):
    """post_comments: dict of {post_id: [ {comment_id, body, score, created_utc}, ... ]}.
    Stored as {"comment_1": "text", "comment_2": "text", ...} — just the comment text,
    ordered by Reddit's own top-comment ranking (score was only used to pick/sort them).
    Batches all updates into a single round trip instead of one UPDATE per post."""
    if not post_comments:
        return
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()

    rows = []
    for post_id, comments in post_comments.items():
        payload = {
            f"comment_{i}": c["body"]
            for i, c in enumerate(comments, start=1)
        }
        rows.append((post_id, Json(payload)))

    query = sql.SQL("""
        UPDATE sentiment_raw.{table} AS t
        SET comments = v.comments
        FROM (VALUES %s) AS v(post_id, comments)
        WHERE t.post_id = v.post_id
    """).format(table=sql.Identifier(table)).as_string(conn)

    execute_values(cur, query, rows, template="(%s, %s::jsonb)")

    conn.commit()
    cur.close()


def get_unprocessed_posts(conn, coin):
    """Posts in sentiment_raw that don't have sentiment analysis in sentiment_clean yet."""
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()
    cur.execute(sql.SQL("""
        SELECT r.post_id, r.title, r.body, r.comments
        FROM sentiment_raw.{table} r
        LEFT JOIN sentiment_clean.{table} c ON r.post_id = c.post_id
        WHERE c.post_id IS NULL
    """).format(table=sql.Identifier(table)))
    rows = cur.fetchall()
    cur.close()
    return rows


def insert_analysis(conn, coin, post_id, clean_title, clean_body, clean_comments, sentiment):
    """
    clean_comments: single string, cleaned comments joined together.
    sentiment: {"label", "score", "confidence"}
    """
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()
    cur.execute(sql.SQL("""
        INSERT INTO sentiment_clean.{table}
            (post_id, title, body, comments, sentiment_label, sentiment_score, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (post_id) DO NOTHING
    """).format(table=sql.Identifier(table)), (
        post_id, clean_title, clean_body, clean_comments,
        sentiment["label"], sentiment["score"], sentiment["confidence"],
    ))
    conn.commit()
    cur.close()


def get_mean_score(conn, coin):
    """Plain (unweighted) mean sentiment score."""
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()
    cur.execute(sql.SQL("SELECT AVG(sentiment_score) FROM sentiment_clean.{table}").format(
        table=sql.Identifier(table)
    ))
    result = cur.fetchone()[0]
    cur.close()
    return result
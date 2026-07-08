"""
database.py
-----
DB utilities for sentiment pipeline.

Two schemas:
    sentiment_raw   - raw fetched Reddit posts, one table per coin.
                       Each row (post + its comments) is inserted together in one
                       write, so a row never sits with NULL comments waiting on a
                       second pass. Comments stored as JSONB: {"comment_1": "...", ...}.
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


def insert_raw_post(conn, coin, post, comments):
    """Insert a single post row, fully populated including its comments, in one write.
    post: dict with post_id, subreddit, title, body, created_utc, score, upvote_ratio.
    comments: list of dicts with comment_id, body, score, created_utc (top N for this post).
    Stored as {"comment_1": "text", "comment_2": "text", ...} — just the comment text."""
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()

    payload = {
        f"comment_{i}": c["body"]
        for i, c in enumerate(comments, start=1)
    }

    cur.execute(sql.SQL("""
        INSERT INTO sentiment_raw.{table}
            (post_id, subreddit, title, body, comments, created_utc, score, upvote_ratio)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (post_id) DO NOTHING
    """).format(table=sql.Identifier(table)), (
        post["post_id"], post["subreddit"], post["title"], post["body"], Json(payload),
        post["created_utc"], post["score"], post["upvote_ratio"],
    ))

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
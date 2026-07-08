"""
database.py
-----
DB utilities for sentiment pipeline.

Two schemas:
    sentiment_raw   - raw fetched Reddit posts + top comments, one table per coin
    sentiment_clean - cleaned title/body/comments + sentiment score, one table per coin
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
    """Create sentiment_raw/sentiment_clean schemas (if missing) and this coin's tables."""
    coin = coin.lower()
    table = f"{coin}_posts"
    comments_table = f"{coin}_comments"
    cur = conn.cursor()

    cur.execute("CREATE SCHEMA IF NOT EXISTS sentiment_raw")
    cur.execute("CREATE SCHEMA IF NOT EXISTS sentiment_clean")

    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS sentiment_raw.{table} (
            post_id       TEXT PRIMARY KEY,
            subreddit     TEXT,
            title         TEXT,
            body          TEXT,
            created_utc   TIMESTAMP,
            score         INTEGER,
            upvote_ratio  DOUBLE PRECISION
        )
    """).format(table=sql.Identifier(table)))

    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS sentiment_raw.{comments_table} (
            comment_id    TEXT PRIMARY KEY,
            post_id       TEXT REFERENCES sentiment_raw.{table}(post_id),
            body          TEXT,
            score         INTEGER,
            created_utc   TIMESTAMP
        )
    """).format(comments_table=sql.Identifier(comments_table), table=sql.Identifier(table)))

    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS sentiment_clean.{table} (
            post_id           TEXT PRIMARY KEY REFERENCES sentiment_raw.{table}(post_id),
            title             TEXT,
            body              TEXT,
            sentiment_label   TEXT,
            sentiment_score   DOUBLE PRECISION,
            confidence        DOUBLE PRECISION
        )
    """).format(table=sql.Identifier(table)))

    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS sentiment_clean.{comments_table} (
            comment_id    TEXT PRIMARY KEY,
            post_id       TEXT REFERENCES sentiment_clean.{table}(post_id),
            body          TEXT
        )
    """).format(comments_table=sql.Identifier(comments_table), table=sql.Identifier(table)))

    conn.commit()
    cur.close()
    logger.info(
        f"Tables ensured: sentiment_raw.{table}, sentiment_raw.{comments_table}, "
        f"sentiment_clean.{table}, sentiment_clean.{comments_table}"
    )


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


def insert_top_comments(conn, coin, post_id, comments):
    """comments: list of dicts with comment_id, body, score, created_utc (top 10 per post)."""
    if not comments:
        return
    comments_table = f"{coin.lower()}_comments"
    cur = conn.cursor()

    query = sql.SQL("""
        INSERT INTO sentiment_raw.{comments_table} (comment_id, post_id, body, score, created_utc)
        VALUES %s
        ON CONFLICT (comment_id) DO NOTHING
    """).format(comments_table=sql.Identifier(comments_table)).as_string(conn)

    rows = [(
        c["comment_id"], post_id, c["body"], c["score"], c["created_utc"],
    ) for c in comments]
    execute_values(cur, query, rows)

    conn.commit()
    cur.close()


def get_unprocessed_posts(conn, coin):
    """Posts in sentiment_raw that don't have sentiment analysis in sentiment_clean yet."""
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()
    cur.execute(sql.SQL("""
        SELECT r.post_id, r.title, r.body
        FROM sentiment_raw.{table} r
        LEFT JOIN sentiment_clean.{table} c ON r.post_id = c.post_id
        WHERE c.post_id IS NULL
    """).format(table=sql.Identifier(table)))
    rows = cur.fetchall()
    cur.close()
    return rows


def get_raw_comments(conn, coin, post_id):
    """Raw (uncleaned) top comments for a post, from sentiment_raw."""
    comments_table = f"{coin.lower()}_comments"
    cur = conn.cursor()
    cur.execute(sql.SQL("""
        SELECT comment_id, body
        FROM sentiment_raw.{comments_table}
        WHERE post_id = %s
    """).format(comments_table=sql.Identifier(comments_table)), (post_id,))
    rows = cur.fetchall()
    cur.close()
    return rows


def insert_analysis(conn, coin, post_id, clean_title, clean_body, sentiment):
    """
    sentiment: {"label", "score", "confidence"}
    """
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()
    cur.execute(sql.SQL("""
        INSERT INTO sentiment_clean.{table}
            (post_id, title, body, sentiment_label, sentiment_score, confidence)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (post_id) DO NOTHING
    """).format(table=sql.Identifier(table)), (
        post_id, clean_title, clean_body,
        sentiment["label"], sentiment["score"], sentiment["confidence"],
    ))
    conn.commit()
    cur.close()


def insert_clean_comments(conn, coin, post_id, clean_comments):
    """clean_comments: list of dicts with comment_id, body (cleaned text)."""
    if not clean_comments:
        return
    comments_table = f"{coin.lower()}_comments"
    cur = conn.cursor()

    query = sql.SQL("""
        INSERT INTO sentiment_clean.{comments_table} (comment_id, post_id, body)
        VALUES %s
        ON CONFLICT (comment_id) DO NOTHING
    """).format(comments_table=sql.Identifier(comments_table)).as_string(conn)

    rows = [(c["comment_id"], post_id, c["body"]) for c in clean_comments]
    execute_values(cur, query, rows)

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
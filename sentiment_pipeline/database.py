"""
database.py
-----
DB utilities for sentiment pipeline.
"""

import os
import logging
import psycopg2
from psycopg2 import sql
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
    """Create sentiment_clean schema (if missing) and this coin's table."""
    coin = coin.lower()
    table = f"{coin}_posts"
    cur = conn.cursor()

    cur.execute("CREATE SCHEMA IF NOT EXISTS sentiment_clean")

    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS sentiment_clean.{table} (
            post_id           TEXT PRIMARY KEY,
            created_utc       TIMESTAMP,
            subreddit         TEXT,
            title             TEXT,
            body              TEXT,
            comments          TEXT,
            sentiment_label   TEXT,
            sentiment_score   DOUBLE PRECISION,
            confidence        DOUBLE PRECISION,
            score             INTEGER,
            upvote_ratio      DOUBLE PRECISION
        )
    """).format(table=sql.Identifier(table)))

    conn.commit()
    cur.close()
    logger.info(f"Table ensured: sentiment_clean.{table}")


def insert_analysis(conn, coin, post_id, post, subreddit, sentiment, created_utc, score, upvote_ratio):
    """
    Insert one row: the post itself + its metadata/analysis.

    post: dict with just the actual post content -> {"title", "body", "comments"}
    Everything else (subreddit, created_utc, score, upvote_ratio, sentiment)
    is passed straight through as-is.
    sentiment: {"label", "score", "confidence"}
    """
    table = f"{coin.lower()}_posts"
    cur = conn.cursor()
    cur.execute(sql.SQL("""
        INSERT INTO sentiment_clean.{table}
            (post_id, created_utc, subreddit, title, body, comments, sentiment_label, sentiment_score, confidence, score, upvote_ratio)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (post_id) DO NOTHING
    """).format(table=sql.Identifier(table)), (
        post_id, created_utc, subreddit, post["title"], post["body"], post["comments"],
        sentiment["label"], sentiment["score"], sentiment["confidence"],
        score, upvote_ratio,
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
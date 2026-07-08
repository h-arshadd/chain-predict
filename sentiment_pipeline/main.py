"""
main.py
-------
Full sentiment pipeline orchestration.
"""

import yaml
import logging

from database import (
    get_db_connection, create_tables, insert_raw_posts, insert_top_comments,
    get_unprocessed_posts, insert_analysis, get_mean_score,
)
from reddit_fetcher import get_reddit_client, fetch_posts, fetch_top_comments
from text_cleaner import clean_text_for_model
from sentiment_model import get_sentiment
from structured_output import build_output

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    conn = get_db_connection()
    reddit = get_reddit_client()

    for coin, cfg in config["coins"].items():
        logger.info(f"--- {coin} ---")
        create_tables(conn, coin)

        posts = fetch_posts(reddit, cfg["subreddits"], cfg["search_query"], 
                          limit=config["reddit"]["post_limit"])
        insert_raw_posts(conn, coin, posts)
        logger.info(f"Fetched & stored {len(posts)} raw posts for {coin}")

        for post in posts:
            top_comments = fetch_top_comments(reddit, post["post_id"], limit=10)
            insert_top_comments(conn, coin, post["post_id"], top_comments)
        logger.info(f"Fetched & stored top comments for {len(posts)} posts for {coin}")

        unprocessed = get_unprocessed_posts(conn, coin)
        logger.info(f"{len(unprocessed)} posts to analyze for {coin}")

        for post_id, title, body in unprocessed:
            raw_text = f"{title} {body}"
            clean_text = clean_text_for_model(raw_text)
            if not clean_text:
                continue

            sentiment = get_sentiment(clean_text)

            insert_analysis(conn, coin, post_id, clean_text, sentiment)

            output = build_output(coin, post_id, clean_text, sentiment)
            logger.info(output)

        plain_mean = get_mean_score(conn, coin)
        logger.info(f"{coin} mean sentiment: {plain_mean}")

    conn.close()


if __name__ == "__main__":
    run()
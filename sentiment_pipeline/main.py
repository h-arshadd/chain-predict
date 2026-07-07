"""
main.py
-------
Full sentiment pipeline orchestration.
"""

import yaml
import logging

from database import (
    get_db_connection, create_tables, insert_raw_posts,
    get_unprocessed_posts, insert_analysis, get_mean_score, get_weighted_mean_score,
)
from reddit_fetcher import get_reddit_client, fetch_posts
from text_cleaner import clean_text_for_model
from sentiment_model import get_sentiment
from topic_classifier import classify_topic
from weighting import compute_weight
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

        unprocessed = get_unprocessed_posts(conn, coin)
        logger.info(f"{len(unprocessed)} posts to analyze for {coin}")

        for post_id, title, body, score, num_comments in unprocessed:
            raw_text = f"{title} {body}"
            clean_text = clean_text_for_model(raw_text)
            if not clean_text:
                continue

            sentiment = get_sentiment(clean_text)
            topic = classify_topic(clean_text)
            weight = compute_weight(score, num_comments)

            insert_analysis(conn, coin, post_id, clean_text, sentiment, topic, weight)

            output = build_output(coin, post_id, clean_text, sentiment, topic, weight)
            logger.info(output)

        plain_mean = get_mean_score(conn, coin)
        weighted_mean = get_weighted_mean_score(conn, coin)
        logger.info(f"{coin} mean sentiment: {plain_mean} | weighted mean: {weighted_mean}")

    conn.close()


if __name__ == "__main__":
    run()
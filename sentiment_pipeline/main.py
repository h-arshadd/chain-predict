"""
main.py
-------
Full sentiment pipeline orchestration.
"""

import yaml
import logging

from database import (
    get_db_connection, create_tables, insert_raw_post,
    get_unprocessed_posts, insert_analysis,
    get_mean_score,
)
from reddit_fetcher import get_reddit_client, fetch_posts, fetch_top_comments
from text_cleaner import clean_text_for_model
from sentiment_model import get_sentiment
from topic_classifier import classify_topic
from structured_output import build_output

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run():
    conn = get_db_connection()
    reddit = get_reddit_client()
    
    coins = config["coins"]
    subreddits = config["reddit"]["subreddits"]
    post_limit = config["reddit"]["post_limit"]
    
    # Create tables for all coins upfront
    for coin in coins:
        create_tables(conn, coin)
        logger.info(f"Tables ready for {coin}")

    # Fetch posts once from all subreddits (no coin filtering)
    logger.info(f"Fetching from {len(subreddits)} subreddits...")
    posts = fetch_posts(reddit, subreddits, limit=post_limit)
    logger.info(f"Fetched {len(posts)} total posts")

    # Process each post: classify coin, fetch comments, store in raw
    for post in posts:
        top_comments = fetch_top_comments(reddit, post["post_id"], limit=10)
        
        # Classify which coin this post is about
        combined_text = f"{post['title']} {post['body']}".strip()
        topic_result = classify_topic(combined_text)
        coin = topic_result["topic"]
        topic_confidence = topic_result["confidence"]
        
        # Only store if confidence is reasonable (e.g., > 0.1)
        if topic_confidence > 0.1:
            insert_raw_post(conn, coin, post, top_comments)
            logger.info(f"Stored {post['post_id']} → {coin} (confidence: {topic_confidence:.3f})")
        else:
            logger.info(f"Skipped {post['post_id']} (low confidence: {topic_confidence:.3f})")

    # Now analyze sentiment per coin
    for coin in coins:
        logger.info(f"\n--- Analyzing {coin} ---")
        unprocessed = get_unprocessed_posts(conn, coin)
        logger.info(f"{len(unprocessed)} posts to analyze for {coin}")

        for post_id, title, body, comments in unprocessed:
            clean_title = clean_text_for_model(title)
            clean_body = clean_text_for_model(body)

            comments = comments or {}
            clean_comment_list = [
                cleaned for text in comments.values()
                if (cleaned := clean_text_for_model(text))
            ]
            clean_comments = " | ".join(clean_comment_list)

            if not clean_title and not clean_body:
                continue

            # Sentiment for the whole post
            combined_text = f"{clean_title} {clean_body}".strip()
            sentiment = get_sentiment(combined_text)

            insert_analysis(conn, coin, post_id, clean_title, clean_body, clean_comments, sentiment)

            output = build_output(coin, post_id, clean_title, clean_body, clean_comments, sentiment)
            logger.info(
                f"{coin} {post_id}: {sentiment['label']} "
                f"(score={sentiment['score']:.3f}, confidence={sentiment['confidence']:.3f})"
            )

        plain_mean = get_mean_score(conn, coin)
        logger.info(f"{coin} mean sentiment: {plain_mean}")

    conn.close()


if __name__ == "__main__":
    run()
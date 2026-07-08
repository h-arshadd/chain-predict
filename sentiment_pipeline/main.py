"""
main.py
-------
Full sentiment pipeline orchestration.
"""

import yaml
import logging

from database import (
    get_db_connection, create_tables, insert_raw_post,
)
from reddit_fetcher import get_reddit_client, fetch_posts, fetch_top_comments
from text_cleaner import clean_text_for_model
from sentiment_model import get_sentiment

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

        posts = fetch_posts(reddit, cfg["subreddits"], cfg["search_query"])

        for post in posts:
            top_comments = fetch_top_comments(reddit, post["post_id"], limit=10)
            
            # Clean text
            clean_title = clean_text_for_model(post["title"])
            clean_body = clean_text_for_model(post["body"])
            
            clean_comment_list = [
                cleaned for text in top_comments
                if (cleaned := clean_text_for_model(text["body"]))
            ]
            clean_comments = " | ".join(clean_comment_list)
            
            if not clean_title and not clean_body:
                continue
            
            # Analyze sentiment
            combined_text = f"{clean_title} {clean_body}".strip()
            sentiment = get_sentiment(combined_text)
            
            # Insert cleaned + sentiment + metadata in one go
            insert_raw_post(
                conn, coin, 
                post["post_id"], post["subreddit"],
                clean_title, clean_body, clean_comments, 
                sentiment,
                post["created_utc"], post["score"], post["upvote_ratio"]
            )
            
            logger.info(
                f"{coin} {post['post_id']}: {sentiment['label']} "
                f"(score={sentiment['score']:.3f}, confidence={sentiment['confidence']:.3f})"
            )
        
        logger.info(f"Processed {len(posts)} posts for {coin}")

    conn.close()


if __name__ == "__main__":
    run()
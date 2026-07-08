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
            post["comments"] = top_comments  # Add comments to post dict
            
            # Combine all text into single string
            all_text = f"{post['title']} {post['body']}"
            for comment in post["comments"]:
                all_text += f" {comment['body']}"
            
            # Clean combined text
            clean_text = clean_text_for_model(all_text)
            
            if not clean_text:
                continue
            
            # Analyze sentiment on combined text
            sentiment = get_sentiment(clean_text)
            
            # Insert into DB
            insert_raw_post(
                conn, coin, 
                post["post_id"], post["subreddit"],
                post["title"], post["body"], " | ".join([c["body"] for c in post["comments"]]),
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
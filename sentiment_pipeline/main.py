"""
main.py
-------
Full sentiment pipeline orchestration.
"""

import yaml
import logging

from database import (
    get_db_connection, create_tables, insert_analysis,
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
            
            # Clean each piece of text individually
            clean_title = clean_text_for_model(post["title"])
            clean_body = clean_text_for_model(post["body"])
            clean_comment_bodies = [clean_text_for_model(c["body"]) for c in post["comments"]]

            # Combine cleaned text for the model
            all_text = f"{clean_title} {clean_body} " + " ".join(clean_comment_bodies)
            all_text = all_text.strip()

            if not all_text:
                continue

            # Analyze sentiment on cleaned combined text
            sentiment = get_sentiment(all_text)

            # The post itself: just title, body, comments -- all cleaned
            post_data = {
                "title": clean_title,
                "body": clean_body,
                "comments": " | ".join(clean_comment_bodies),
            }

            # Insert into DB
            insert_analysis(
                conn, coin,
                post["post_id"], post_data,
                post["subreddit"], sentiment,
                post["created_utc"], post["score"], post["upvote_ratio"],
            )
            
            logger.info(
                f"{coin} {post['post_id']}: {sentiment['label']} "
                f"(score={sentiment['score']:.3f}, confidence={sentiment['confidence']:.3f})"
            )
        
        logger.info(f"Processed {len(posts)} posts for {coin}")

    conn.close()


if __name__ == "__main__":
    run()
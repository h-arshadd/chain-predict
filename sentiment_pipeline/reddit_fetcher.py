"""
reddit_fetcher.py
-----------------
Thin wrapper around PRAW (Python Reddit API Wrapper).
"""

import os
import logging
from datetime import datetime, timezone
import praw
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def get_reddit_client():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=config["reddit"]["user_agent"],
    )


def fetch_posts(reddit, subreddits, search_query, limit=100):
    """Search each subreddit for the coin's query."""
    posts = []
    for sub_name in subreddits:
        subreddit = reddit.subreddit(sub_name)
        for post in subreddit.search(search_query, limit=limit, sort="new"):
            posts.append({
                "post_id": post.id,
                "subreddit": sub_name,
                "title": post.title or "",
                "body": post.selftext or "",
                "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
            })
        logger.info(f"Fetched from r/{sub_name}")
    return posts


def fetch_top_comments(reddit, post_id, limit=10):
    """Fetch the top `limit` comments (by score) for a given post."""
    submission = reddit.submission(id=post_id)
    submission.comment_sort = "top"
    submission.comments.replace_more(limit=0)  # drop "load more comments" stubs

    comments = []
    for comment in submission.comments[:limit]:
        comments.append({
            "comment_id": comment.id,
            "body": comment.body or "",
            "score": comment.score,
            "created_utc": datetime.fromtimestamp(comment.created_utc, tz=timezone.utc),
        })
    return comments
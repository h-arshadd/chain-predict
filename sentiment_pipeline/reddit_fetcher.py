"""
reddit_fetcher.py
-----------------
Thin wrapper around PRAW (Python Reddit API Wrapper) — the standard,
pretrained-nothing-needed client for Reddit's API. You just need an
app registered at https://www.reddit.com/prefs/apps (type: "script"),
which gives you REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET for the .env.
"""

import os
import logging
from datetime import datetime, timezone
import praw
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_reddit_client():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "sentiment-pipeline/0.1"),
    )


def fetch_posts(reddit, subreddits, search_query, limit=100):
    """
    Search each subreddit for the coin's query, return a flat list of dicts
    ready for db.insert_raw_posts(). Duplicate handling happens at the DB
    layer (ON CONFLICT DO NOTHING), so no dedup logic needed here.
    """
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
                "score": post.score,                  # net upvotes, used for weighting later
                "num_comments": post.num_comments,
                "upvote_ratio": post.upvote_ratio,
            })
        logger.info(f"Fetched from r/{sub_name}")
    return posts
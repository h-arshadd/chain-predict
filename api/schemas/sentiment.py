"""
schemas/sentiment.py
---------------------
Response models for /api/sentiment. See repos/sentiment_repo.py's module
docstring for what's real vs. derived here -- summary:
  - Overall sentiment, timelines, post volume, top posts: real, straight
    from sentiment_clean.{coin}_posts.
  - Fear & Greed: a derived 0-100 score computed from this same real
    data (bullish/neutral/bearish distribution), NOT the real external
    Fear & Greed Index -- no such index is fetched anywhere in this
    codebase. Labeled "derived" throughout so it's never confused with
    the real thing.
  - "Top Posts" stands in for the PDF's "News Sentiment" -- there is no
    news source in this codebase, only Reddit.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class SentimentCoins(BaseModel):
    """Every coin with a real sentiment_clean.{coin}_posts table right now."""
    coins: list[str] = []


class OverallSentiment(BaseModel):
    coin: str
    score: Optional[float] = None       # mean sentiment_score, -1..+1
    label: Optional[str] = None         # "Bullish" / "Neutral" / "Bearish"
    bullish_pct: float = 0.0
    neutral_pct: float = 0.0
    bearish_pct: float = 0.0
    post_count: int = 0


class FearGreed(BaseModel):
    """Derived score -- see this file's module docstring. Not the real external index."""
    coin: str
    score: Optional[float] = None       # 0-100
    label: Optional[str] = None         # "Extreme Fear" .. "Extreme Greed"
    yesterday: Optional[float] = None
    last_week: Optional[float] = None
    last_month: Optional[float] = None


class SentimentTimelinePoint(BaseModel):
    date: datetime
    score: float                        # mean sentiment_score that day, -1..+1
    post_count: int


class FearGreedTimelinePoint(BaseModel):
    date: datetime
    score: float                        # derived 0-100 that day
    post_count: int


class PostVolumePoint(BaseModel):
    day: datetime
    bullish: int
    neutral: int
    bearish: int


class TopPost(BaseModel):
    post_id: str
    subreddit: str
    title: str
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    confidence: Optional[float] = None
    score: Optional[int] = None          # Reddit's own upvote score, not sentiment
    upvote_ratio: Optional[float] = None
    created_utc: Optional[datetime] = None


class SentimentOverview(BaseModel):
    """Everything the Sentiment page needs for one coin, in one call."""
    coin: str
    overall: OverallSentiment
    fear_greed: FearGreed
    sentiment_timeline: list[SentimentTimelinePoint] = []
    fear_greed_timeline: list[FearGreedTimelinePoint] = []
    post_volume: list[PostVolumePoint] = []
    top_posts: list[TopPost] = []
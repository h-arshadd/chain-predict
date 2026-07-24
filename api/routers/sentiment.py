"""
routers/sentiment.py
---------------------
/api/sentiment -- NLP & Sentiment page (spec section 10).

Coin list is discovered live from information_schema (whichever
sentiment_clean.{coin}_posts tables actually exist), not hardcoded --
see repos/sentiment_repo.discover_sentiment_coins. Right now that's just
BTC/ETH (the only two sentiment_pipeline/config.yaml tracks), but this
endpoint needs no code change as more coins are added and their tables
get created/populated by the pipeline.

No /api/sentiment/{coin}/refresh or similar trigger-the-pipeline
endpoint here -- sentiment_pipeline/main.py is a separate offline
process (same relationship the API has to backtest/simulator/execution
scripts elsewhere), this module only reads what it already wrote.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from api.core.db import get_conn
from api.core.responses import item
from api.schemas.sentiment import (
    SentimentCoins, SentimentOverview, OverallSentiment, FearGreed,
    SentimentTimelinePoint, FearGreedTimelinePoint, PostVolumePoint, TopPost,
)
from api.repos import sentiment_repo

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])


@router.get("/coins")
def list_coins(conn=Depends(get_conn)):
    """Every coin with real sentiment data right now -- drives the frontend's coin filter."""
    coins = sentiment_repo.discover_sentiment_coins(conn)
    return item(SentimentCoins(coins=coins).model_dump())


@router.get("/{coin}")
def get_sentiment_overview(
    coin: str,
    timeline_days: int = Query(default=30, ge=1, le=180),
    volume_days: int = Query(default=14, ge=1, le=90),
    top_posts_limit: int = Query(default=20, ge=1, le=100),
    conn=Depends(get_conn),
):
    """
    Everything the Sentiment page needs for one coin: overall breakdown,
    derived Fear & Greed, sentiment timeline, Fear & Greed timeline,
    daily post volume, and top Reddit posts (standing in for the PDF's
    "News Sentiment" -- see repos/sentiment_repo.py's module docstring).

    404 if this coin has no sentiment_clean.{coin}_posts table at all
    (not yet tracked), rather than silently returning an all-empty
    overview that looks identical to "tracked but no posts yet".
    """
    coin = coin.upper()
    known_coins = sentiment_repo.discover_sentiment_coins(conn)
    if coin not in known_coins:
        raise HTTPException(
            status_code=404,
            detail=f"No sentiment data tracked for '{coin}'. Tracked coins: {known_coins or 'none yet'}",
        )

    overall = OverallSentiment(**sentiment_repo.get_overall_sentiment(conn, coin))
    fear_greed = FearGreed(**sentiment_repo.get_fear_greed(conn, coin))
    sentiment_timeline = [
        SentimentTimelinePoint(**p) for p in sentiment_repo.get_sentiment_timeline(conn, coin, timeline_days)
    ]
    fear_greed_timeline = [
        FearGreedTimelinePoint(**p) for p in sentiment_repo.get_fear_greed_timeline(conn, coin, timeline_days)
    ]
    post_volume = [PostVolumePoint(**p) for p in sentiment_repo.get_post_volume(conn, coin, volume_days)]
    top_posts = [TopPost(**p) for p in sentiment_repo.get_top_posts(conn, coin, top_posts_limit)]

    overview = SentimentOverview(
        coin=coin,
        overall=overall,
        fear_greed=fear_greed,
        sentiment_timeline=sentiment_timeline,
        fear_greed_timeline=fear_greed_timeline,
        post_volume=post_volume,
        top_posts=top_posts,
    )
    return item(overview.model_dump())
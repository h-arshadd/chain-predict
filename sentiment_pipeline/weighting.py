"""
weighting.py
------------
Per your lead: a post with a lot of upvotes/comments likely reflects a
real, shared opinion more than a random 1-upvote post, so it should count
for more when we average sentiment later.

Log-scaled on purpose: without the log, one viral post (5,000 upvotes)
would completely dominate the average and drown out everything else.
With it, a viral post still counts for more, just not absurdly more.
"""

import math


def compute_weight(score: int, num_comments: int) -> float:
    """
    score = Reddit's net upvotes (upvotes - downvotes) for the post.
    Negative/downvoted posts are floored at 0 - a downvoted post
    shouldn't count as "less than nothing" against the average.
    """
    score = max(score, 0)
    return 1 + math.log10(score + 1) + math.log10(num_comments + 1)
"""
responses.py
------------
Shared response envelope so every module's routes return JSON in the same
shape (per spec: "Return consistent JSON response formats").

Two shapes, used consistently everywhere:

    Single item / action result:
        {"data": {...}}

    List endpoint:
        {"data": [...], "meta": {"total": N, "limit": L, "offset": O}}

Errors are NOT wrapped here -- FastAPI's HTTPException already produces a
consistent {"detail": "..."} body with the right status code, which is
its own consistent format. We don't re-wrap that.
"""

from typing import Any, Sequence


def item(data: Any) -> dict:
    return {"data": data}


def list_response(data: Sequence[Any], total: int, limit: int, offset: int) -> dict:
    return {
        "data": data,
        "meta": {"total": total, "limit": limit, "offset": offset},
    }
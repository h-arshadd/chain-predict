"""
routers/playbook.py
---------------------
/api/playbook -- Strategy Builder's Playbook panel (Strategy_Builder_
Module.pdf Step 1: "Load Playbook"). Plain CRUD, no lifecycle/enabled
flags -- a playbook entry is a reusable template, not a running strategy
(see metadata_utils.create_playbook_table's docstring for why).
"""

from fastapi import APIRouter, Depends, HTTPException

from api.core.db import get_conn
from api.core.responses import item, list_response
from api.schemas.playbook import PlaybookCreate, PlaybookSummary, PlaybookDetail
from api.repos import playbook_repo

router = APIRouter(prefix="/api/playbook", tags=["playbook"])


@router.get("")
def list_playbooks(limit: int = 50, offset: int = 0, conn=Depends(get_conn)):
    rows = playbook_repo.list_playbooks(conn)
    total = len(rows)
    page = rows[offset: offset + limit]
    summaries = [PlaybookSummary(**r).model_dump() for r in page]
    return list_response(summaries, total, limit, offset)


@router.get("/{playbook_id}")
def get_playbook(playbook_id: int, conn=Depends(get_conn)):
    row = playbook_repo.get_playbook_detail(conn, playbook_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Playbook entry {playbook_id} not found")
    return item(PlaybookDetail(**row).model_dump())


@router.post("")
def create_playbook(body: PlaybookCreate, conn=Depends(get_conn)):
    row = playbook_repo.create_playbook(conn, body.strategy_name, body.strategy_config)
    return item(PlaybookDetail(**row).model_dump())


@router.delete("/{playbook_id}")
def delete_playbook(playbook_id: int, conn=Depends(get_conn)):
    deleted = playbook_repo.remove_playbook(conn, playbook_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Playbook entry {playbook_id} not found")
    return item({"deleted": True, "playbook_id": playbook_id})
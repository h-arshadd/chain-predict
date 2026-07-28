"""
schemas/playbook.py
--------------------
Request/response models for /api/playbook.

Maps to metadata.playbook (see crypto_pipeline/utils/metadata_utils.py --
create_playbook_table()'s docstring is the source of truth for what each
column means). Deliberately three fields only, same as the DB table --
no take_profit/stop_loss/position_size/exchange/coin/time_horizon here.
Those only exist once a playbook entry is combined into a real
metadata.strategy row via /api/strategies/build (see routers/strategies.py).
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class PlaybookCreate(BaseModel):
    strategy_name: str
    strategy_config: dict


class PlaybookSummary(BaseModel):
    """One row in the Strategy Builder's Playbook picker list."""
    playbook_id: int
    strategy_name: str
    created_at: Optional[datetime] = None


class PlaybookDetail(PlaybookSummary):
    strategy_config: dict

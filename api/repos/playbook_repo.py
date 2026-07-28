"""
repos/playbook_repo.py
------------------------
Thin wrapper over metadata_utils' playbook CRUD -- same pattern as every
other *_repo.py in this folder (strategies_repo, wallets_repo, etc.): the
router stays a plain HTTP layer, the repo is where DB calls happen.
"""

from crypto_pipeline.utils.metadata_utils import (
    create_playbook_table,
    insert_playbook,
    get_playbook,
    get_playbooks,
    delete_playbook,
)


def list_playbooks(conn) -> list[dict]:
    """Every playbook entry, newest first."""
    create_playbook_table(conn)
    return get_playbooks(conn)


def get_playbook_detail(conn, playbook_id: int) -> dict | None:
    create_playbook_table(conn)
    return get_playbook(conn, playbook_id)


def create_playbook(conn, strategy_name: str, strategy_config: dict) -> dict:
    """
    Save a new playbook entry, or overwrite an existing one with the same
    strategy_name (insert_playbook() upserts on the UNIQUE constraint).
    """
    create_playbook_table(conn)
    playbook_id = insert_playbook(conn, strategy_name, strategy_config)
    return get_playbook(conn, playbook_id)


def remove_playbook(conn, playbook_id: int) -> bool:
    create_playbook_table(conn)
    return delete_playbook(conn, playbook_id)
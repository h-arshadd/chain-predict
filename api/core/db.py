"""
db.py
-----
FastAPI dependency for a per-request Postgres connection.

Deliberately reuses crypto_pipeline.utils.db_utils.get_db_connection()
instead of opening a second, separate connection path with its own env
var handling -- same .env, same psycopg2 connection, one source of truth
for DB credentials across the pipeline and the API.

Usage in a route:

    from api.core.db import get_conn

    @router.get("/wallets")
    def list_wallets(conn = Depends(get_conn)):
        ...
"""

from crypto_pipeline.utils.db_utils import get_db_connection


def get_conn():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()
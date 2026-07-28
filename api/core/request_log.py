"""
core/request_log.py
--------------------
Backs the public.request_log table -- one row per HTTP request the API
receives, written by RequestLogMiddleware in main.py.

Schema: public (not its own schema) since this is cross-cutting API
infrastructure, not pipeline/domain data -- same reasoning execution/
simulator/backtest each get their own schema because they ARE domain
data, this explicitly isn't.

Deliberately does NOT log request or response bodies. Several POST/PUT
bodies in this app carry real secrets (WalletCreate.api_key/api_secret,
WalletUpdate.api_key/api_secret) -- persisting those verbatim into a log
table would put live exchange credentials at rest in a second place,
outside accounts.api_keys, with no encryption and no access control of
its own. Path, method, query string, status code, and timing are enough
to answer "what hit the API and when" without that risk. If per-request
body auditing is ever genuinely needed, it should redact secret fields
explicitly rather than log the raw body.

No user/actor column -- this app has no auth layer (see main.py's own
docstring: "No /api/users, /api/auth -- intentionally absent per
instructions, this is a single-operator tool with no login"), so there
is no identity to attribute a request to beyond client_ip.
"""

from psycopg2 import sql

from crypto_pipeline.utils.db_utils import get_db_connection


def _ensure_table(conn):
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS public.request_log (
            id            BIGSERIAL PRIMARY KEY,
            requested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            method        TEXT NOT NULL,
            path          TEXT NOT NULL,
            query_string  TEXT,
            status_code   INTEGER,
            duration_ms   DOUBLE PRECISION,
            client_ip     TEXT
        )
    """))
    conn.commit()
    cursor.close()


def log_request(method: str, path: str, query_string: str, status_code: int,
                 duration_ms: float, client_ip: str | None) -> None:
    """
    Insert one row for a completed request. Called from
    RequestLogMiddleware after the response is ready, so status_code and
    duration_ms are always real, not guessed. Swallows its own DB errors
    (logs to stdout instead of raising) so a logging failure never turns
    into a 500 for the actual request being served.
    """
    conn = None
    try:
        conn = get_db_connection()
        _ensure_table(conn)
        cursor = conn.cursor()
        cursor.execute(sql.SQL("""
            INSERT INTO public.request_log
                (method, path, query_string, status_code, duration_ms, client_ip)
            VALUES (%s, %s, %s, %s, %s, %s)
        """), (method, path, query_string or None, status_code, duration_ms, client_ip))
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"request_log: failed to write row -- {e}")
    finally:
        if conn is not None:
            conn.close()
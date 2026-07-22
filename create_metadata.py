"""
apply_strategy_enabled_columns.py
----------------------------------
One-off script: adds the simulator_enabled/execution_enabled columns to
metadata.strategy (self-heals existing tables via the ALTER TABLE block
inside create_strategy_table()). Safe to re-run -- it's all
IF NOT EXISTS / idempotent.

Usage:
    python apply_strategy_enabled_columns.py
"""
from crypto_pipeline.utils.metadata_utils import get_db_connection, create_strategy_table

conn = get_db_connection()
try:
    create_strategy_table(conn)
    print("metadata.strategy schema updated: simulator_enabled/execution_enabled columns ensured.")
finally:
    conn.close()
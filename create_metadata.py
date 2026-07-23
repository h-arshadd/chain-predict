"""
seed_execution_config_multi.py
-------------------------------
Registers execution.config rows for multiple bybit pairs at once, so
execution/main.py's universe (get_execution_universe -- every row in
execution.config) includes all of them. Safe to re-run (upsert on
exchange+symbol).

Before running: make sure each symbol below has exactly one
execution_enabled=True row in metadata.strategy, or execution/main.py
will skip that pair.

Usage:
    python seed_execution_config_multi.py
"""
from crypto_pipeline.utils.db_utils import get_db_connection, save_execution_config

# One entry per (exchange, symbol) you want execution/main.py to trade.
# Edit this list to match the coins you're ready to go live on.
PAIRS = [
    "btc",
    "eth",
    "sol",
    "doge",
    "ada",
    "ltc",
    "mina",
    "sui",
]

conn = get_db_connection()
try:
    for symbol in PAIRS:
        save_execution_config(
            conn,
            exchange="bybit",
            symbol=symbol,
            initial_balance=10000,
            position_size={"type": "fixed_percentage", "value": 10},
            commission=0.05,
            slippage=0.02,
            allow_long=True,
            allow_short=True,
            max_open_positions=1,
        )
        print(f"Saved execution.config: bybit/{symbol}")
finally:
    conn.close()
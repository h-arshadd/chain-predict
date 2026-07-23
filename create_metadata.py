"""
seed_simulator_config.py
--------------------------
Registers simulator.config rows for bybit pairs so simulator/main.py's
universe (get_simulator_universe -- every row in simulator.config)
includes them. Safe to re-run (upsert on exchange+symbol).

Usage:
    python seed_simulator_config.py
"""
from crypto_pipeline.utils.db_utils import get_db_connection, save_simulator_config

# One entry per (exchange, symbol) you want simulator/main.py to run.
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
        save_simulator_config(
            conn,
            exchange="bybit",
            symbol=symbol,
            # Starting balance the simulator's own bookkeeping tracks
            # P&L against (simulator.positions/*_trades) -- not a real
            # wallet balance.
            initial_balance=10000,
            # Percentage of current tracked balance risked per trade.
            position_size={"type": "fixed_percentage", "value": 10},
            commission=0.05,   # % of trade value
            slippage=0.02,     # % of trade value
            allow_long=True,
            allow_short=True,
            max_open_positions=1,
        )
        print(f"Saved simulator.config: bybit/{symbol}")
finally:
    conn.close()

print("simulator.config seeded.")
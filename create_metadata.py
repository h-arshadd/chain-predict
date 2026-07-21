from crypto_pipeline.utils.db_utils import get_db_connection, save_simulator_config

conn = get_db_connection()
save_simulator_config(
    conn, exchange="bybit", symbol="btc",
    start_date="2026-07-20",
    initial_balance=10000,
    position_size={"type": "fixed_percentage", "value": 10},
    commission=0.05,
    slippage=0.02,
    allow_long=True,
    allow_short=True,
    max_open_positions=1,
    enabled=True,
)
conn.close()
print("simulator.config seeded.")
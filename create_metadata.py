"""
seed_execution_config.py
-------------------------
One-off script: registers the first execution.config row so
execution/main.py has something to run. Run this once (or again anytime
you want to change these settings -- it's an upsert, safe to re-run).
Usage:
    python seed_execution_config.py
"""
from crypto_pipeline.utils.db_utils import get_db_connection, save_execution_config
conn = get_db_connection()
try:
    save_execution_config(
        conn,
        exchange="bybit",
        symbol="btc",
        # Starting account balance execution tracks P&L against. This is
        # NOT your real Bybit wallet balance -- it's the baseline this
        # module's own bookkeeping (execution.positions/*_trades) uses to
        # compute position size and P&L, same as simulator does.
        initial_balance=10000,
        # Percentage of current tracked balance risked per trade. Stored
        # as two flat columns (position_size_type, position_size_value)
        # in execution.config now, not JSONB -- still passed in here as
        # the same dict shape, save_execution_config splits it internally.
        position_size={"type": "fixed_percentage", "value": 10},
        commission=0.05,   # % of trade value
        slippage=0.02,     # % of trade value
        allow_long=True,
        allow_short=True,
        max_open_positions=1,
    )
    print("Saved execution.config: bybit/btc, strategy=RSI_14_reversal")
finally:
    conn.close()
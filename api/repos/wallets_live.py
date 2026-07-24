"""
repos/wallet_live.py
---------------------
Live balance/unrealized-PnL lookup straight from Bybit for one wallet.
Nothing here is stored in Postgres -- accounts.api_keys has no balance
column because balance only exists on the exchange, it's not derived
from anything in our own tables.

"Total PnL" is different: that one DOES come from our DB
(accounts.stats.realized_pnl_total, built by
accounts_utils.refresh_account_stats() from accounts.history). This
module only covers the two live fields, balance and unrealized_pnl.

Kept isolated (own try/except) so one wallet's API hiccup doesn't break
the whole /api/wallets list -- see fetch_balance()'s return shape.
"""

from crypto_pipeline.execution.bybit_client import get_client


def fetch_balance(api_key: str, api_secret: str, demo: bool) -> dict:
    """
    Returns {"balance": float, "unrealized_pnl": float} on success, or
    {"balance": None, "unrealized_pnl": None, "error": "..."} if the
    Bybit call fails (bad/rotated key, network issue, etc) -- callers
    should show the row with an error badge, not drop it or 500.
    """
    try:
        client = get_client(api_key=api_key, api_secret=api_secret, demo=demo)
        resp = client.get_wallet_balance(accountType="UNIFIED")
        result_list = resp.get("result", {}).get("list", [])
        if not result_list:
            return {"balance": 0.0, "unrealized_pnl": 0.0, "error": None}

        account = result_list[0]
        balance = float(account.get("totalEquity") or 0)
        unrealized = float(account.get("totalPerpUPL") or 0)
        return {"balance": balance, "unrealized_pnl": unrealized, "error": None}
    except Exception as exc:
        return {"balance": None, "unrealized_pnl": None, "error": str(exc)}
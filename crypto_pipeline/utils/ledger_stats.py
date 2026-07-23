"""
ledger_stats.py
----------------
One function, `get_ledger_stats(ledger)`, that takes a Bybit-style execution
ledger (a pandas DataFrame, a CSV path, or a CSV string) and returns a flat
dict of ~85 plain, readable stats: PnL, win rate, fees, volume, streaks,
drawdown, time-based breakdowns, per-symbol stats, long/short, holding time,
etc.

Realized PnL is derived with FIFO matching per symbol, since raw fills don't
carry a "profit" column directly (this mirrors Bybit's get_executions
fill-level data, not get_closed_pnl).

Usage:
    from ledger_stats import get_ledger_stats
    stats = get_ledger_stats(df)          # df = accounts.history slice
    # or
    stats = get_ledger_stats("history-1.csv")
"""

import pandas as pd
import numpy as np
from collections import defaultdict


def _to_native(obj):
    """
    Recursively convert numpy scalar types (np.float64, np.int64, np.bool_,
    etc.) and pandas NaT/Timestamp-ish leftovers into plain Python types, so
    downstream consumers (e.g. psycopg2) never see a numpy repr like
    'np.float64(1.23)' where a plain float/int/str/None is expected.
    Dicts and lists are walked recursively; everything else is returned
    through pd.isna-checked np.generic/np.ndarray handling.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        val = obj.item()
        return None if pd.isna(val) else val
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, float) and pd.isna(obj):
        return None
    return obj


def get_ledger_stats(ledger):
    # ---------- 1. Load ----------
    if isinstance(ledger, pd.DataFrame):
        df = ledger.copy()
    else:
        df = pd.read_csv(ledger)

    stats = {}
    if df.empty:
        stats["row_count"] = 0
        return stats

    df["trade_time"] = pd.to_datetime(df["trade_time"], format="mixed")
    df = df.sort_values("trade_time").reset_index(drop=True)

    trades = df[df["exec_type"] == "Trade"].copy()
    funding = df[df["exec_type"] == "Funding"].copy()

    # ---------- 2. FIFO realized PnL per symbol ----------
    fifo_lots = defaultdict(list)   # symbol -> list of [qty_remaining, price, side, time]
    closed_trades = []              # symbol, qty, entry, exit, pnl, open_time, close_time, direction

    for _, row in trades.iterrows():
        sym = row["symbol"]
        side = row["side"]
        qty = row["exec_qty"]
        price = row["exec_price"]
        t = row["trade_time"]
        lots = fifo_lots[sym]

        opposite = "Sell" if side == "Buy" else "Buy"
        remaining = qty

        while remaining > 1e-12 and lots and lots[0][2] == opposite:
            lot_qty, lot_price, lot_side, lot_time = lots[0]
            matched = min(lot_qty, remaining)
            if lot_side == "Buy":  # closing a long with a sell
                pnl = (price - lot_price) * matched
            else:  # closing a short with a buy
                pnl = (lot_price - price) * matched
            closed_trades.append({
                "symbol": sym, "qty": matched, "entry_price": lot_price,
                "exit_price": price, "pnl": pnl,
                "open_time": lot_time, "close_time": t,
                "direction": "long" if lot_side == "Buy" else "short",
            })
            lot_qty -= matched
            remaining -= matched
            if lot_qty <= 1e-12:
                lots.pop(0)
            else:
                lots[0][0] = lot_qty

        if remaining > 1e-12:
            lots.append([remaining, price, side, t])

    closed_df = pd.DataFrame(closed_trades)

    # ---------- 3. Basic counts ----------
    stats["row_count"] = len(df)
    stats["trade_row_count"] = len(trades)
    stats["funding_row_count"] = len(funding)
    stats["closed_trade_count"] = len(closed_df)
    stats["unique_symbols"] = trades["symbol"].nunique()
    stats["symbols_traded"] = sorted(trades["symbol"].unique().tolist())
    stats["unique_accounts"] = df["account_name"].nunique()
    stats["unique_exchanges"] = df["exchange"].nunique()
    stats["unique_order_ids"] = trades["order_id"].nunique()

    # ---------- 4. Side / order-type breakdown ----------
    stats["buy_count"] = int((trades["side"] == "Buy").sum())
    stats["sell_count"] = int((trades["side"] == "Sell").sum())
    stats["maker_count"] = int(trades["is_maker"].sum())
    stats["taker_count"] = int((~trades["is_maker"]).sum())
    stats["maker_ratio"] = round(stats["maker_count"] / max(len(trades), 1), 4)
    stats["market_order_count"] = int((trades["order_type"] == "Market").sum())

    # ---------- 5. Volume / value ----------
    stats["total_qty_traded"] = round(trades["exec_qty"].sum(), 8)
    stats["total_exec_value"] = round(trades["exec_value"].sum(), 8)
    stats["avg_exec_value"] = round(trades["exec_value"].mean(), 8)
    stats["median_exec_value"] = round(trades["exec_value"].median(), 8)
    stats["max_exec_value"] = round(trades["exec_value"].max(), 8)
    stats["min_exec_value"] = round(trades["exec_value"].min(), 8)
    stats["avg_exec_qty"] = round(trades["exec_qty"].mean(), 8)
    stats["total_buy_value"] = round(trades.loc[trades.side == "Buy", "exec_value"].sum(), 8)
    stats["total_sell_value"] = round(trades.loc[trades.side == "Sell", "exec_value"].sum(), 8)
    stats["total_buy_qty"] = round(trades.loc[trades.side == "Buy", "exec_qty"].sum(), 8)
    stats["total_sell_qty"] = round(trades.loc[trades.side == "Sell", "exec_qty"].sum(), 8)

    # ---------- 6. Price stats ----------
    stats["avg_exec_price"] = round(trades["exec_price"].mean(), 8)
    stats["median_exec_price"] = round(trades["exec_price"].median(), 8)
    stats["max_exec_price"] = round(trades["exec_price"].max(), 8)
    stats["min_exec_price"] = round(trades["exec_price"].min(), 8)
    stats["first_price"] = round(trades["exec_price"].iloc[0], 8)
    stats["last_price"] = round(trades["exec_price"].iloc[-1], 8)
    stats["price_change"] = round(stats["last_price"] - stats["first_price"], 8)
    stats["price_change_pct"] = round(
        (stats["last_price"] / stats["first_price"] - 1) * 100, 4
    ) if stats["first_price"] else None

    # ---------- 7. Fees ----------
    stats["total_fees"] = round(df["exec_fee"].sum(), 8)
    stats["total_trade_fees"] = round(trades["exec_fee"].sum(), 8)
    stats["total_funding_fees"] = round(funding["exec_fee"].sum(), 8)
    stats["avg_fee_per_trade"] = round(trades["exec_fee"].mean(), 8)
    stats["max_fee"] = round(trades["exec_fee"].max(), 8)
    stats["min_fee"] = round(trades["exec_fee"].min(), 8)
    stats["maker_fees_paid"] = round(trades.loc[trades.is_maker, "exec_fee"].sum(), 8)
    stats["taker_fees_paid"] = round(trades.loc[~trades.is_maker, "exec_fee"].sum(), 8)

    # ---------- 8. Realized PnL (FIFO) ----------
    if not closed_df.empty:
        gross_pnl = closed_df["pnl"].sum()
        wins = closed_df[closed_df["pnl"] > 0]
        losses = closed_df[closed_df["pnl"] < 0]

        stats["gross_realized_pnl"] = round(gross_pnl, 8)
        stats["net_realized_pnl"] = round(gross_pnl - stats["total_trade_fees"], 8)
        stats["win_count"] = len(wins)
        stats["loss_count"] = len(losses)
        stats["win_rate_pct"] = round(len(wins) / len(closed_df) * 100, 4)
        stats["loss_rate_pct"] = round(len(losses) / len(closed_df) * 100, 4)
        stats["avg_win"] = round(wins["pnl"].mean(), 8) if len(wins) else 0
        stats["avg_loss"] = round(losses["pnl"].mean(), 8) if len(losses) else 0
        stats["largest_win"] = round(wins["pnl"].max(), 8) if len(wins) else 0
        stats["largest_loss"] = round(losses["pnl"].min(), 8) if len(losses) else 0
        stats["total_win_amount"] = round(wins["pnl"].sum(), 8) if len(wins) else 0
        stats["total_loss_amount"] = round(losses["pnl"].sum(), 8) if len(losses) else 0
        stats["profit_factor"] = round(
            abs(stats["total_win_amount"] / stats["total_loss_amount"]), 4
        ) if stats["total_loss_amount"] else None
        stats["expectancy"] = round(closed_df["pnl"].mean(), 8)
        stats["avg_win_loss_ratio"] = round(
            abs(stats["avg_win"] / stats["avg_loss"]), 4
        ) if stats["avg_loss"] else None

        closed_df = closed_df.sort_values("close_time")
        cum_pnl = closed_df["pnl"].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        stats["max_drawdown"] = round(drawdown.min(), 8)
        stats["ending_cum_pnl"] = round(cum_pnl.iloc[-1], 8)

        # streaks
        cur_streak, cur_sign, max_win_streak, max_loss_streak = 0, 0, 0, 0
        for pnl in closed_df["pnl"]:
            sign = 1 if pnl > 0 else (-1 if pnl < 0 else 0)
            if sign == cur_sign and sign != 0:
                cur_streak += 1
            else:
                cur_streak = 1
                cur_sign = sign
            if sign == 1:
                max_win_streak = max(max_win_streak, cur_streak)
            elif sign == -1:
                max_loss_streak = max(max_loss_streak, cur_streak)
        stats["max_win_streak"] = max_win_streak
        stats["max_loss_streak"] = max_loss_streak
        stats["current_streak_len"] = cur_streak
        stats["current_streak_type"] = (
            "win" if cur_sign == 1 else "loss" if cur_sign == -1 else "none"
        )

        # long vs short
        long_df = closed_df[closed_df["direction"] == "long"]
        short_df = closed_df[closed_df["direction"] == "short"]
        stats["long_trade_count"] = len(long_df)
        stats["short_trade_count"] = len(short_df)
        stats["long_pnl"] = round(long_df["pnl"].sum(), 8)
        stats["short_pnl"] = round(short_df["pnl"].sum(), 8)
        stats["long_win_rate_pct"] = round(
            (long_df["pnl"] > 0).mean() * 100, 4
        ) if len(long_df) else None
        stats["short_win_rate_pct"] = round(
            (short_df["pnl"] > 0).mean() * 100, 4
        ) if len(short_df) else None

        # holding time
        holding = (closed_df["close_time"] - closed_df["open_time"]).dt.total_seconds()
        stats["avg_holding_time_sec"] = round(holding.mean(), 2)
        stats["median_holding_time_sec"] = round(holding.median(), 2)
        stats["max_holding_time_sec"] = round(holding.max(), 2)
        stats["min_holding_time_sec"] = round(holding.min(), 2)
    else:
        stats["gross_realized_pnl"] = 0
        stats["net_realized_pnl"] = -stats["total_trade_fees"]
        stats["win_count"] = 0
        stats["loss_count"] = 0
        stats["win_rate_pct"] = None
        stats["loss_rate_pct"] = None
        stats["avg_win"] = 0
        stats["avg_loss"] = 0
        stats["largest_win"] = 0
        stats["largest_loss"] = 0
        stats["total_win_amount"] = 0
        stats["total_loss_amount"] = 0
        stats["profit_factor"] = None
        stats["expectancy"] = None
        stats["avg_win_loss_ratio"] = None
        stats["max_drawdown"] = 0
        stats["ending_cum_pnl"] = 0
        stats["max_win_streak"] = 0
        stats["max_loss_streak"] = 0
        stats["current_streak_len"] = 0
        stats["current_streak_type"] = "none"
        stats["long_trade_count"] = 0
        stats["short_trade_count"] = 0
        stats["long_pnl"] = 0
        stats["short_pnl"] = 0
        stats["long_win_rate_pct"] = None
        stats["short_win_rate_pct"] = None
        stats["avg_holding_time_sec"] = None
        stats["median_holding_time_sec"] = None
        stats["max_holding_time_sec"] = None
        stats["min_holding_time_sec"] = None

    # ---------- 9. Per-symbol breakdown ----------
    per_symbol = {}
    for sym, g in trades.groupby("symbol"):
        sym_closed = closed_df[closed_df["symbol"] == sym] if not closed_df.empty else pd.DataFrame()
        per_symbol[sym] = {
            "trade_count": len(g),
            "total_qty": round(g["exec_qty"].sum(), 8),
            "total_value": round(g["exec_value"].sum(), 8),
            "total_fees": round(g["exec_fee"].sum(), 8),
            "avg_price": round(g["exec_price"].mean(), 8),
            "realized_pnl": round(sym_closed["pnl"].sum(), 8) if not sym_closed.empty else 0,
            "win_rate_pct": round((sym_closed["pnl"] > 0).mean() * 100, 4) if len(sym_closed) else None,
        }
    stats["per_symbol"] = per_symbol

    # ---------- 10. Time-based ----------
    stats["first_trade_time"] = str(df["trade_time"].min())
    stats["last_trade_time"] = str(df["trade_time"].max())
    span = df["trade_time"].max() - df["trade_time"].min()
    stats["trading_span_days"] = round(span.total_seconds() / 86400, 4)
    stats["trades_per_day_avg"] = round(
        len(trades) / max(stats["trading_span_days"], 1e-9), 4
    )
    stats["trades_by_hour_of_day"] = trades["trade_time"].dt.hour.value_counts().sort_index().to_dict()
    stats["trades_by_day_of_week"] = trades["trade_time"].dt.day_name().value_counts().to_dict()
    stats["trades_by_date"] = trades["trade_time"].dt.date.astype(str).value_counts().sort_index().to_dict()
    stats["busiest_hour"] = int(trades["trade_time"].dt.hour.value_counts().idxmax()) if len(trades) else None
    stats["busiest_day_of_week"] = trades["trade_time"].dt.day_name().value_counts().idxmax() if len(trades) else None

    # ---------- 11. Position / closing behavior ----------
    stats["fully_closing_trades"] = int((trades["closed_size"] > 0).sum())
    stats["position_opening_trades"] = int((trades["closed_size"] == 0).sum())

    return _to_native(stats)


if __name__ == "__main__":
    import json
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "history-1.csv"
    result = get_ledger_stats(path)
    print(json.dumps(result, indent=2, default=str))
    print(f"\nTotal top-level stats: {len(result)}")
"""Walk-forward backtest.

Uses ONLY out-of-sample predictions (each day is traded by a model that had
never seen that day) and runs them through the same decision code and the
same paper broker as live trading:

  * a decision made when a bar closes is filled at the NEXT bar's open,
    paying the spread/slippage model;
  * stops are checked against each bar's high/low (stop first if a bar
    touches both stop and target);
  * everything is flattened before the close.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .broker import PaperBroker
from .config import Config
from .features import FeatureSet
from .market_calendar import OPEN_MIN
from .panel import Panel
from .strategy import make_signals, plan, session_gates


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame
    stats: Dict[str, Any]


def performance_stats(daily: pd.DataFrame, trades: pd.DataFrame, start_capital: float,
                      equity: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Headline numbers from a daily P&L table and a trade list."""
    st: Dict[str, Any] = {"days": int(len(daily)), "trades": int(len(trades)), "start_capital": float(start_capital)}
    if len(daily) == 0:
        st.update({"end_equity": float(start_capital), "total_return_pct": 0.0})
        return st
    end_eq = float(daily["end_equity"].iloc[-1])
    rets = daily["ret_pct"].astype(float).to_numpy() / 100.0
    st["end_equity"] = end_eq
    st["total_pnl"] = end_eq - float(start_capital)
    st["total_return_pct"] = (end_eq / start_capital - 1.0) * 100.0
    st["avg_daily_return_pct"] = float(rets.mean() * 100.0)
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    st["daily_vol_pct"] = sd * 100.0
    st["sharpe"] = float(rets.mean() / sd * np.sqrt(252.0)) if sd > 0 else 0.0
    st["t_stat"] = float(rets.mean() / sd * np.sqrt(len(rets))) if sd > 0 else 0.0
    st["win_days"] = int((rets > 0).sum())
    st["lose_days"] = int((rets < 0).sum())
    st["best_day_pct"] = float(rets.max() * 100.0)
    st["worst_day_pct"] = float(rets.min() * 100.0)
    curve = equity.astype(float).to_numpy() if equity is not None and len(equity) else daily["end_equity"].astype(float).to_numpy()
    curve = np.concatenate([[float(start_capital)], curve])
    peak = np.maximum.accumulate(curve)
    st["max_drawdown_pct"] = float(((curve - peak) / peak).min() * 100.0)
    if len(trades):
        pnl = trades["pnl"].astype(float).to_numpy()
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        st["win_rate_pct"] = float(len(wins) / len(pnl) * 100.0)
        st["profit_factor"] = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
        st["avg_trade_bps"] = float(trades["ret_bps"].astype(float).mean())
        st["avg_trade_pnl"] = float(pnl.mean())
        st["trades_per_day"] = float(len(pnl) / max(len(daily), 1))
        st["avg_bars_held"] = float(trades["bars_held"].astype(float).mean())
        if "predicted_bps" in trades:
            signed_pred = np.abs(trades["predicted_bps"].astype(float))
            st["avg_predicted_bps"] = float(signed_pred.mean())
        st["exit_reasons"] = {str(k): int(v) for k, v in trades["reason"].value_counts().items()}
    return st


def run_backtest(panel: Panel, feats: FeatureSet, oos: pd.DataFrame, cfg: Config,
                 capital: Optional[float] = None) -> BacktestResult:
    T, S = panel.T, panel.S
    exp_z = np.full((T, S), np.nan)
    exp_z[oos["t"].to_numpy(), oos["s"].to_numpy()] = oos["exp_z"].to_numpy()
    days = np.unique(oos["day"].to_numpy())
    offsets = np.flatnonzero(np.r_[True, panel.day[1:] != panel.day[:-1], True])
    sym_idx = {s: i for i, s in enumerate(panel.symbols)}
    iv = panel.interval_min
    broker = PaperBroker(cfg, None, capital)
    O, Hi, Lo, C = panel.O, panel.H, panel.L, panel.C
    daily_rows: List[dict] = []
    eq_t: List[pd.Timestamp] = []
    eq_v: List[float] = []

    for d in days:
        a, b = int(offsets[d]), int(offsets[d + 1])
        close_min = OPEN_MIN + int(panel.nbar[a]) * iv
        g_start, g_end, g_flat = session_gates(cfg, close_min)
        broker.start_day(str(panel.dates[d]))
        pend_exits: list = []
        pend_orders: list = []
        flat = False
        for t in range(a, b):
            ts = panel.ts[t]
            stamp = ts.strftime("%Y-%m-%d %H:%M")
            bar_end = OPEN_MIN + (int(panel.k[t]) + 1) * iv
            o, h, l, c = O[t], Hi[t], Lo[t], C[t]
            # 1) fill what was decided at the previous close, at this bar's open
            for sym, reason in pend_exits:
                broker.close(sym, stamp, reason, price=float(o[sym_idx[sym]]))
            for od in pend_orders:
                px = float(o[sym_idx[od.symbol]])
                if np.isfinite(px) and px > 0:
                    broker.open(od, stamp, price=px)
            pend_exits, pend_orders = [], []
            # 2) stops / targets inside the bar
            for sym, pos in list(broker.positions.items()):
                i = sym_idx[sym]
                hit = broker.stop_or_target_hit(pos, float(l[i]), float(h[i]), open_px=float(o[i]))
                if hit:
                    broker.close(sym, stamp, hit[0], price=hit[1])
            # 3) mark to the close
            broker.mark({sym: float(c[sym_idx[sym]]) for sym in broker.positions})
            broker.end_bar()
            # 4) end-of-day flatten and the daily loss limit
            if not flat and bar_end >= g_flat:
                for sym in list(broker.positions):
                    broker.close(sym, stamp, "eod", price=float(c[sym_idx[sym]]))
                flat = True
            elif not broker.halted and broker.daily_loss_breached():
                for sym in list(broker.positions):
                    broker.close(sym, stamp, "loss-limit", price=float(c[sym_idx[sym]]))
                broker.halted = True
            # 5) new decisions (filled next bar)
            if not flat and t + 1 < b and np.isfinite(exp_z[t]).any():
                sig = make_signals(panel.symbols, c, exp_z[t], feats.sigH[t], feats.usable[t], cfg,
                                   beta=feats.beta(t))
                can_enter = (g_start <= bar_end <= g_end) and not broker.halted
                pend_exits, pend_orders = plan(sig, broker, cfg, can_enter)
            eq_t.append(ts + pd.Timedelta(minutes=iv))
            eq_v.append(broker.equity())
        for sym in list(broker.positions):   # safety net: never carry overnight
            broker.close(sym, str(panel.dates[d]), "eod", price=float(C[b - 1][sym_idx[sym]]))
        daily_rows.append(broker.day_summary())

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(broker.trades)
    equity = pd.DataFrame({"time": eq_t, "equity": eq_v})
    stats = performance_stats(daily, trades, broker.start_capital, equity["equity"] if len(equity) else None)
    return BacktestResult(daily=daily, trades=trades, equity=equity, stats=stats)


def format_stats(st: Dict[str, Any], title: str = "Performance") -> str:
    money = lambda v: "${:,.2f}".format(v)
    lines = ["  %s" % title]
    if st.get("days", 0) == 0:
        lines.append("    no completed sessions yet")
        return "\n".join(lines)
    lines.append("    %-22s %s -> %s  (%+.2f%%, %s)" % (
        "equity", money(st["start_capital"]), money(st["end_equity"]), st["total_return_pct"],
        ("+" if st["total_pnl"] >= 0 else "-") + money(abs(st["total_pnl"]))))
    lines.append("    %-22s %d  (%d up, %d down; best %+.2f%%, worst %+.2f%%)" % (
        "sessions", st["days"], st["win_days"], st["lose_days"], st["best_day_pct"], st["worst_day_pct"]))
    lines.append("    %-22s %+.3f%% avg, %.3f%% vol  ->  Sharpe %.2f (t-stat %.2f)%s" % (
        "daily return", st["avg_daily_return_pct"], st["daily_vol_pct"], st["sharpe"], st["t_stat"],
        "  [too few sessions to mean much]" if st["days"] < 20 else ""))
    lines.append("    %-22s %.2f%%" % ("max drawdown", st["max_drawdown_pct"]))
    if st.get("trades"):
        pf = st["profit_factor"]
        lines.append("    %-22s %d  (%.1f/day, %.0f%% winners, profit factor %s)" % (
            "trades", st["trades"], st["trades_per_day"], st["win_rate_pct"],
            "inf" if pf == float("inf") else "%.2f" % pf))
        lines.append("    %-22s %+.1f bp realized after costs vs %.1f bp predicted before costs" % (
            "average trade", st["avg_trade_bps"], st.get("avg_predicted_bps", float("nan"))))
        lines.append("    %-22s %s" % ("exits", ", ".join("%s %d" % kv for kv in st["exit_reasons"].items())))
    else:
        lines.append("    %-22s 0  (no signal cleared the cost hurdle)" % "trades")
    return "\n".join(lines)

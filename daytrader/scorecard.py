"""End-of-day scorecard: how did today's predictions actually turn out?

Every bar the live engine writes the prediction for every stock to
state/predictions/<day>.csv. After the close this grades them against what
happened next, measured exactly as the model's training label is:

  enter at the open of the bar after the decision, exit at the open
  `horizon` bars later (or at the close if the session ends first);
  "relative" = that move minus the average move of all graded stocks.

It answers three questions for the day:
  1. Did the ranking work?   rank IC (as in training) and the direction hit rate.
  2. Were the strongest calls right, and would they have paid after costs?
  3. What did the near misses (ideas that almost cleared the cost hurdle) do?

Days without a predictions file (older runs) are graded from signals.csv,
which only holds the top few ideas each bar, so the rank IC there is omitted.
A one-day scorecard is noise; the running totals in state/scorecard.csv are
what to judge the model by.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .market_calendar import ET, UTC, MarketCalendar

PRED_FIELDS = ["time", "symbol", "price", "expected_bps", "sigH_bps", "cost_bps"]
HISTORY_FIELDS = ["day", "source", "bars", "predictions", "rank_ic", "hit_rate", "top_signed_rel_bps",
                  "top_signed_raw_bps", "top_cost_bps", "top_net_bps", "trades", "pnl"]
SIZE_BUCKETS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, np.inf)]


def predictions_path(state_dir: Path, day) -> Path:
    return Path(state_dir) / "predictions" / ("%s.csv" % day)


def write_predictions(state_dir: Path, stamp: str, symbols, price, exp_bps, sigH_bps, cost_bps) -> None:
    day = stamp.split(" ")[0]
    path = predictions_path(state_dir, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(PRED_FIELDS)
        for i, s in enumerate(symbols):
            e = exp_bps[i]
            if not np.isfinite(e):
                continue
            w.writerow([stamp, s, "%.4f" % price[i], "%.3f" % e, "%.2f" % sigH_bps[i], "%.2f" % cost_bps[i]])


def _load_predictions(state_dir: Path, day: date):
    p = predictions_path(state_dir, day)
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p), "predictions"
    s = Path(state_dir) / "signals.csv"
    if s.exists() and s.stat().st_size > 0:
        df = pd.read_csv(s)
        df = df[df["time"].astype(str).str.startswith(str(day))]
        if len(df):
            return df.rename(columns={})[["time", "symbol", "price", "expected_bps", "cost_bps"]], "signals"
    return None, ""


def _forward_moves(store, symbols: List[str], day: date, cal: MarketCalendar, cfg: Config):
    """Per symbol, arrays of bar opens and the session's last close, indexed by bar number."""
    sess = cal.session(day)
    iv = cfg.data.interval_min
    bars = store.load(symbols, start=sess.open.astimezone(UTC), end=sess.close.astimezone(UTC))
    n = int((sess.close - sess.open).total_seconds() // 60 // iv)
    out: Dict[str, tuple] = {}
    if len(bars) == 0:
        return out, n
    bars = bars.copy()
    bars["k"] = ((bars["ts"].dt.tz_convert(ET) - pd.Timestamp(sess.open)).dt.total_seconds() // 60 // iv).astype(int)
    for sym, g in bars.groupby("symbol"):
        opens = np.full(n, np.nan)
        g = g[(g["k"] >= 0) & (g["k"] < n)]
        opens[g["k"].to_numpy()] = g["open"].to_numpy(dtype=float)
        last = g.sort_values("k")
        out[sym] = (opens, float(last["close"].iloc[-1]) if len(last) else np.nan, int(last["k"].max()) if len(last) else -1)
    return out, n


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return np.nan
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def grade(df: pd.DataFrame, moves: Dict[str, tuple], n_bars: int, day: date, cal: MarketCalendar,
          cfg: Config) -> pd.DataFrame:
    """Attach realised forward moves (bp) to each prediction row."""
    sess = cal.session(day)
    iv, H = cfg.data.interval_min, cfg.model.horizon_bars
    t = pd.to_datetime(df["time"].astype(str).str.replace(" ET", "", regex=False))
    k_end = ((t - pd.Timestamp(sess.open.replace(tzinfo=None))).dt.total_seconds() // 60 // iv).astype(int)
    # forward move of every stock from every bar: R[s, ke] (bp); the market is their average
    names = sorted(moves)
    pos = {s: i for i, s in enumerate(names)}
    R = np.full((len(names), n_bars), np.nan)
    for i, sym in enumerate(names):
        opens, last_close, last_k = moves[sym]
        exit_px = np.full(n_bars, np.nan)
        exit_px[: max(n_bars - H, 0)] = opens[H:]
        if last_k == n_bars - 1:
            exit_px[max(n_bars - H, 0):] = last_close
        with np.errstate(divide="ignore", invalid="ignore"):
            R[i] = np.log(exit_px / opens) * 1e4
    R[~np.isfinite(R)] = np.nan
    with np.errstate(invalid="ignore"):
        mkt = np.nanmean(R, axis=0) if len(names) else np.full(n_bars, np.nan)
    ke = k_end.to_numpy()
    si = np.array([pos.get(s, -1) for s in df["symbol"].to_numpy()])
    okk = (si >= 0) & (ke >= 0) & (ke < n_bars)
    raw = np.full(len(df), np.nan)
    rel = np.full(len(df), np.nan)
    raw[okk] = R[si[okk], ke[okk]]
    rel[okk] = raw[okk] - mkt[ke[okk]]
    out = df.copy()
    out["k"] = ke
    out["raw_bps"] = raw
    out["rel_bps"] = rel
    ok = np.isfinite(raw) & np.isfinite(rel)
    return out[ok].reset_index(drop=True)


def build_scorecard(day: date, state_dir: Path, store, symbols: List[str], cfg: Config,
                    cal: MarketCalendar, top_k: int = 5) -> Optional[dict]:
    df, source = _load_predictions(state_dir, day)
    if df is None or cal.session(day) is None:
        return None
    moves, n_bars = _forward_moves(store, symbols, day, cal, cfg)
    g = grade(df, moves, n_bars, day, cal, cfg)
    if len(g) == 0:
        return {"day": str(day), "source": source, "graded": 0}
    e = g["expected_bps"].to_numpy(dtype=float)
    side = np.sign(e)
    g["signed_rel"] = side * g["rel_bps"].to_numpy()
    g["signed_raw"] = side * g["raw_bps"].to_numpy()
    g["abs_exp"] = np.abs(e)
    card: dict = {"day": str(day), "source": source, "graded": int(len(g)),
                  "bars": int(g["time"].nunique()), "symbols": int(g["symbol"].nunique()),
                  "horizon_min": cfg.model.horizon_bars * cfg.data.interval_min}

    # 1) ranking quality
    if source == "predictions":
        ics = [(_spearman(x["expected_bps"].to_numpy(float), x["rel_bps"].to_numpy(float)))
               for _, x in g.groupby("time")]
        ics = np.array([v for v in ics if np.isfinite(v)])
        card["rank_ic"] = float(ics.mean()) if len(ics) else np.nan
        card["ic_bars_positive"] = int((ics > 0).sum())
        card["ic_bars"] = int(len(ics))
    else:
        card["rank_ic"] = np.nan
    nz = g["abs_exp"] > 0
    card["hit_rate"] = float((g.loc[nz, "signed_rel"] > 0).mean()) if nz.any() else np.nan

    # 2) the strongest calls of each bar
    top = g.sort_values("abs_exp", ascending=False).groupby("time").head(top_k)
    card["top_k"] = top_k
    card["top_n"] = int(len(top))
    card["top_expected_bps"] = float(top["abs_exp"].mean())
    card["top_signed_rel_bps"] = float(top["signed_rel"].mean())
    card["top_signed_raw_bps"] = float(top["signed_raw"].mean())
    card["top_hit_rate"] = float((top["signed_rel"] > 0).mean())
    card["top_cost_bps"] = float(top["cost_bps"].astype(float).mean())
    card["top_net_bps"] = card["top_signed_raw_bps"] - card["top_cost_bps"]

    # calibration by predicted size
    rows = []
    for lo, hi in SIZE_BUCKETS:
        m = (g["abs_exp"] >= lo) & (g["abs_exp"] < hi)
        if m.sum() == 0:
            continue
        x = g[m]
        rows.append({"bucket": ("%.1f+" % lo) if not np.isfinite(hi) else "%.1f-%.1f" % (lo, hi),
                     "n": int(m.sum()), "expected": float(x["abs_exp"].mean()),
                     "realised_rel": float(x["signed_rel"].mean()), "realised_raw": float(x["signed_raw"].mean()),
                     "hit": float((x["signed_rel"] > 0).mean())})
    card["by_size"] = rows

    # 3) near misses: within 1.5 bp of clearing the hurdle
    need = g["cost_bps"].astype(float) + cfg.strategy.min_edge_bps
    gap = need - g["abs_exp"]
    near = g[(gap > 0) & (gap <= 1.5)].copy()
    near["gap"] = gap[near.index]
    card["near_n"] = int(len(near))
    card["near_signed_raw_bps"] = float(near["signed_raw"].mean()) if len(near) else np.nan
    card["near_net_bps"] = float((near["signed_raw"] - near["cost_bps"].astype(float)).mean()) if len(near) else np.nan
    best = g.sort_values("abs_exp", ascending=False).drop_duplicates("symbol").head(8)
    card["best"] = [{"time": r["time"], "symbol": r["symbol"], "expected": float(r["expected_bps"]),
                     "realised_raw": float(r["raw_bps"]), "realised_rel": float(r["rel_bps"]),
                     "cost": float(r["cost_bps"])} for _, r in best.iterrows()]
    card["cost_median_bps"] = float(g["cost_bps"].astype(float).median())
    return card


def _trades_today(state_dir: Path, day: date) -> tuple:
    p = Path(state_dir) / "trades.csv"
    if not p.exists() or p.stat().st_size == 0:
        return 0, 0.0
    t = pd.read_csv(p)
    col = "exit_time" if "exit_time" in t else t.columns[0]
    t = t[t[col].astype(str).str.startswith(str(day))]
    return int(len(t)), float(t["pnl"].sum()) if "pnl" in t and len(t) else 0.0


def record(card: dict, state_dir: Path) -> pd.DataFrame:
    """Add today's line to state/scorecard.csv (replacing a re-run of the same day); return the history."""
    path = Path(state_dir) / "scorecard.csv"
    hist = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame(columns=HISTORY_FIELDS)
    hist = hist[hist["day"].astype(str) != card["day"]]
    row = {k: card.get(k, np.nan) for k in HISTORY_FIELDS}
    row["bars"] = card.get("bars", 0)
    row["predictions"] = card.get("graded", 0)
    new = pd.DataFrame([row], columns=HISTORY_FIELDS)
    hist = (pd.concat([hist, new], ignore_index=True) if len(hist) else new).sort_values("day")
    path.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(path, index=False)
    return hist


def _f(x, fmt="%+.2f", none="n/a") -> str:
    try:
        return none if x is None or not np.isfinite(float(x)) else fmt % float(x)
    except (TypeError, ValueError):
        return none


def format_scorecard(card: dict, hist: Optional[pd.DataFrame] = None, model_ic: Optional[float] = None) -> List[str]:
    L: List[str] = []
    L.append("Scorecard %s  (%s)" % (card["day"], "every stock, every bar" if card.get("source") == "predictions"
                                      else "top ideas only - full prediction log starts with the next session"))
    if not card.get("graded"):
        L.append("  Nothing to grade (no completed predictions with a known outcome).")
        return L
    H = card["horizon_min"]
    L.append("  Graded %s predictions over %d bars and %d stocks; each against the next %d minutes."
             % (format(card["graded"], ","), card["bars"], card["symbols"], H))
    if np.isfinite(card.get("rank_ic", np.nan)):
        L.append("  Ranking:  rank IC %s (backtest average %s), positive in %d of %d bars"
                 % (_f(card["rank_ic"], "%+.3f"), _f(model_ic, "%+.3f"), card["ic_bars_positive"], card["ic_bars"]))
    L.append("  Direction right (vs the market): %s of calls" % _f(100 * card["hit_rate"], "%.1f%%"))
    L.append("  Top %d calls each bar (%d): predicted %s bp, got %s bp vs market, %s bp outright;"
             % (card["top_k"], card["top_n"], _f(card["top_expected_bps"], "%.1f"), _f(card["top_signed_rel_bps"]),
                _f(card["top_signed_raw_bps"])))
    L.append("    right %s of the time; after %s bp round-trip cost: %s bp per trade"
             % (_f(100 * card["top_hit_rate"], "%.0f%%"), _f(card["top_cost_bps"], "%.1f"), _f(card["top_net_bps"])))
    L.append("  By predicted size (bp):   n     predicted  got(vs mkt)  got(outright)  right")
    for r in card["by_size"]:
        L.append("    %-10s %10s   %8s   %10s   %12s   %5s" % (
            r["bucket"], format(r["n"], ","), _f(r["expected"], "%.2f"), _f(r["realised_rel"]),
            _f(r["realised_raw"]), _f(100 * r["hit"], "%.0f%%")))
    if card["near_n"]:
        L.append("  Near misses (within 1.5 bp of the hurdle): %d; outright %s bp, after cost %s bp"
                 % (card["near_n"], _f(card["near_signed_raw_bps"]), _f(card["near_net_bps"])))
    else:
        L.append("  Near misses (within 1.5 bp of the hurdle): none")
    L.append("  Strongest calls of the day:")
    for b in card["best"]:
        L.append("    %s  %-5s predicted %+6.1f bp  ->  %+7.1f bp outright (%+.1f vs market), cost %.1f"
                 % (str(b["time"])[11:16], b["symbol"], b["expected"], b["realised_raw"], b["realised_rel"], b["cost"]))
    L.append("  Trades today: %d, P&L %s   Median round-trip cost: %s bp"
             % (card.get("trades", 0), _f(card.get("pnl", 0.0), "$%+.2f"), _f(card["cost_median_bps"], "%.1f")))
    if hist is not None and len(hist) > 1:
        ic = pd.to_numeric(hist["rank_ic"], errors="coerce").dropna()
        top = pd.to_numeric(hist["top_signed_rel_bps"], errors="coerce").dropna()
        line = "  Since %s (%d days):" % (hist["day"].iloc[0], len(hist))
        if len(ic) > 1:
            line += "  mean IC %+.3f (t %.1f)" % (ic.mean(), ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
                                                  if ic.std(ddof=1) > 0 else 0.0)
        if len(top):
            line += "  top calls %+.2f bp vs market" % top.mean()
        L.append(line)
    L.append("  (One day is noise: judge the model on weeks of these, not on today alone.)")
    return L


def run_scorecard(day: date, state_dir: Path, store, symbols: List[str], cfg: Config,
                  cal: MarketCalendar, report_dir: Path, model_ic: Optional[float] = None) -> Optional[List[str]]:
    card = build_scorecard(day, state_dir, store, symbols, cfg, cal)
    if card is None:
        return None
    card["trades"], card["pnl"] = _trades_today(state_dir, day)
    hist = record(card, state_dir) if card.get("graded") else None
    lines = format_scorecard(card, hist, model_ic)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / ("scorecard_%s.txt" % card["day"])).write_text("\n".join(lines) + "\n")
    return lines

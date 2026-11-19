"""Training and backtesting on long histories without holding it all in memory.

Three years of 5-minute bars for ~200 stocks is about 13 million bars; computing all 77
features for all of it at once would need several gigabytes. Instead the history is cut
into blocks of sessions. Each block is loaded together with a warm-up of earlier sessions
(so rolling and multi-day features start from the same state they would in one long run),
its features are computed, and only the block's own sessions are kept. Live trading builds
its features from the last 32 sessions, so a warm-up of that length also makes training
features match live ones.

On short histories (Yahoo's ~60 days) this is a single block, identical to before.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import BacktestResult, performance_stats, run_backtest
from .broker import PaperBroker
from .config import Config
from .features import compute_features
from .market_calendar import ET, MarketCalendar
from .model import Dataset, make_dataset
from .panel import build_panel

BAR_ID = 1000          # rows carry t = session * BAR_ID + bar


@dataclass
class HistoryInfo:
    """What train() needs to know about the sessions (duck-types the parts of a Panel it reads)."""
    dates: List[date]
    symbols: List[str]
    blocks: List[Tuple[int, int]]                 # (first, last) global session owned by each block
    stride: int
    day: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=int))   # last bar's session
    k: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=int))     # last bar's index
    nbar: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=int))   # bars in that session


def _sessions(store, symbols: List[str], asof: Optional[datetime]) -> List[date]:
    days = store.session_days(symbols)
    out = [date(1970, 1, 1) + timedelta(days=int(d)) for d in days]
    if asof is not None:
        cut = pd.Timestamp(asof).tz_convert(ET).date()
        out = [d for d in out if d <= cut]
    return out


def _load_block(store, symbols, cfg: Config, cal: MarketCalendar, first: date, last: date,
                asof: Optional[datetime], with_labels: bool = True):
    start = datetime(first.year, first.month, first.day)
    end = datetime(last.year, last.month, last.day) + timedelta(days=1)
    bars = store.load(symbols, start=pd.Timestamp(start, tz="UTC"), end=pd.Timestamp(end, tz="UTC"))
    if len(bars) == 0:
        return None, None
    try:
        panel = build_panel(bars, cfg.data.interval_min, cal, asof=asof, symbols=symbols)
    except ValueError:
        return None, None
    del bars
    return panel, compute_features(panel, cfg, with_labels=with_labels)


def plan_blocks(n_sessions: int, block_sessions: int) -> List[Tuple[int, int]]:
    if n_sessions <= 0:
        return []
    size = max(1, block_sessions)
    return [(a, min(a + size, n_sessions) - 1) for a in range(0, n_sessions, size)]


def choose_stride(n_sessions: int, n_symbols: int, cfg: Config) -> int:
    """Keep every Nth bar when the full history would not fit the row budget."""
    est = n_sessions * cfg.bars_per_day * n_symbols * 0.9
    return max(1, int(math.ceil(est / max(cfg.model.max_dataset_rows, 1))))


def build_dataset(store, symbols: List[str], cfg: Config, cal: MarketCalendar,
                  asof: Optional[datetime] = None, log=print) -> Tuple[Dataset, HistoryInfo]:
    dates = _sessions(store, symbols, asof)
    if not dates:
        raise ValueError("No bars in the cache to train on.")
    gidx = {d: i for i, d in enumerate(dates)}
    blocks = plan_blocks(len(dates), cfg.model.block_sessions)
    stride = choose_stride(len(dates), len(symbols), cfg)
    warm = max(cfg.model.block_warmup_sessions, 0)
    if log and (len(blocks) > 1 or stride > 1):
        log("  Building features for %d sessions (%s to %s) in %d block(s)%s..."
            % (len(dates), dates[0], dates[-1], len(blocks),
               ", every %d bars" % stride if stride > 1 else ""))
    info = HistoryInfo(dates=dates, symbols=list(symbols), blocks=blocks, stride=stride)
    # Fill one pre-sized set of arrays instead of joining per-block pieces at the end, which
    # would briefly need twice the memory. The bound is exact-or-above: at most one row per
    # symbol per kept bar.
    cap = sum((b - a + 1) for a, b in blocks) * (cfg.bars_per_day // stride + 1) * len(symbols)
    out: Optional[Dataset] = None
    n = 0
    for bi, (a, b) in enumerate(blocks):
        panel, feats = _load_block(store, symbols, cfg, cal, dates[max(0, a - warm)], dates[b], asof)
        if panel is None:
            continue
        day_index = np.array([gidx.get(d, -1) for d in panel.dates])
        keep = (day_index >= a) & (day_index <= b)
        ds = make_dataset(panel, feats, cfg, keep_days=keep, day_index=np.maximum(day_index, 0), stride=stride)
        if len(ds.y):
            if out is None:
                out = Dataset(X=np.empty((cap, ds.X.shape[1]), np.float32), y=np.empty(cap, np.float32),
                              rel_bps=np.empty(cap, np.float32), sigH_bps=np.empty(cap, np.float32),
                              price=np.empty(cap, np.float32), day=np.empty(cap, np.int64),
                              t=np.empty(cap, np.int64), s=np.empty(cap, np.int64), k=np.empty(cap, np.int64),
                              names=ds.names, stride=stride)
            m = len(ds.y)
            if n + m > len(out.y):                 # safety net: never expected, but never fatal
                grow = max(n + m, int(len(out.y) * 1.25))
                for name in ("X", "y", "rel_bps", "sigH_bps", "price", "day", "t", "s", "k"):
                    arr = getattr(out, name)
                    new = np.empty((grow,) + arr.shape[1:], arr.dtype)
                    new[:n] = arr[:n]
                    setattr(out, name, new)
            for name in ("X", "y", "rel_bps", "sigH_bps", "price", "day", "t", "s", "k"):
                getattr(out, name)[n:n + m] = getattr(ds, name)
            n += m
        if bi == len(blocks) - 1:
            info.day = np.array([int(day_index[panel.day[-1]])])
            info.k = np.array([int(panel.k[-1])])
            info.nbar = np.array([int(panel.nbar[-1])])
        del panel, feats
        if log and len(blocks) > 1:
            log("    block %d/%d (%s to %s): %s rows" % (bi + 1, len(blocks), dates[a], dates[b],
                                                       format(len(ds.y), ",")))
        del ds
    if out is None or n == 0:
        raise ValueError("No usable training rows in the cache.")
    for name in ("X", "y", "rel_bps", "sigH_bps", "price", "day", "t", "s", "k"):
        setattr(out, name, getattr(out, name)[:n])
    return out, info


def _fold_grid(res, panel, feats, gidx_local: np.ndarray, chunk: int = 250_000) -> Tuple[np.ndarray, np.ndarray]:
    """Expected move on EVERY bar of the panel's out-of-sample sessions, each from its own fold."""
    from .model import fold_predict

    exp_z = np.full((panel.T, panel.S), np.nan)
    gday = gidx_local[panel.day]                       # global session of each panel row
    traded = []
    for fd in res.folds:
        rows = (gday >= fd["first"]) & (gday <= fd["last"])
        if not rows.any():
            continue
        traded += sorted({int(d) for d in np.unique(panel.day[rows])})
        mask = feats.usable & rows[:, None]
        t_idx, s_idx = np.nonzero(mask)
        for a in range(0, len(t_idx), chunk):
            sl = slice(a, a + chunk)
            X = np.column_stack([arr[t_idx[sl], s_idx[sl]] for arr in feats.arrays]).astype(np.float32)
            exp_z[t_idx[sl], s_idx[sl]] = fold_predict(fd, X, res.ckpts, res.haircut)
    return exp_z, np.array(sorted(set(traded)), dtype=int)


def backtest_history(store, info: HistoryInfo, oos, cfg: Config, cal: MarketCalendar,
                     asof: Optional[datetime] = None, capital: Optional[float] = None,
                     log=print) -> BacktestResult:
    """Walk-forward backtest over the out-of-sample sessions, one block at a time.

    `oos` is a TrainResult (preferred) or its walk-forward prediction table. With a
    TrainResult every bar is predicted by the fold model that had not seen that day, so the
    backtest decides on every bar exactly like live trading, however thinly the training
    rows were sampled. One simulated account is carried through every block."""
    res = None
    if hasattr(oos, "folds"):
        res, oos = oos, oos.oos
    full_res = res is not None and bool(res.folds)
    broker = PaperBroker(cfg, None, capital)
    o_day = (oos["t"].to_numpy() // BAR_ID).astype(int)
    o_k = (oos["t"].to_numpy() % BAR_ID).astype(int)
    daily, equity = [], []
    warm = max(cfg.model.block_warmup_sessions, 0)
    gidx = {d: i for i, d in enumerate(info.dates)}
    for a, b in info.blocks:
        m = (o_day >= a) & (o_day <= b)
        if not m.any():
            continue
        # labels are needed even here: some features look at what happened on earlier days
        panel, feats = _load_block(store, info.symbols, cfg, cal, info.dates[max(0, a - warm)], info.dates[b],
                                   asof, with_labels=full_res)
        if panel is None:
            continue
        if full_res:
            g = np.array([gidx.get(d, -1) for d in panel.dates])
            own = (g >= a) & (g <= b)
            exp_z, days = _fold_grid(res, panel, feats, np.where(own, g, -1))
            r = run_backtest(panel, feats, None, cfg, broker=broker, exp_z=exp_z, days=days)
            daily.append(r.daily)
            equity.append(r.equity)
            del panel, feats, exp_z
            continue
        local = {gidx.get(d, -1): i for i, d in enumerate(panel.dates)}
        offsets = np.flatnonzero(np.r_[True, panel.day[1:] != panel.day[:-1], True])
        ld = np.array([local.get(int(d), -1) for d in o_day[m]])
        ok = ld >= 0
        t_local = offsets[ld[ok]] + o_k[m][ok]
        valid = t_local < offsets[ld[ok] + 1]
        sub = pd.DataFrame({"t": t_local[valid], "s": oos["s"].to_numpy()[m][ok][valid],
                            "day": ld[ok][valid], "exp_z": oos["exp_z"].to_numpy()[m][ok][valid]})
        res = run_backtest(panel, feats, sub, cfg, broker=broker)
        daily.append(res.daily)
        equity.append(res.equity)
        del panel, feats
    daily_df = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()
    eq_df = pd.concat(equity, ignore_index=True) if equity else pd.DataFrame({"time": [], "equity": []})
    trades = pd.DataFrame(broker.trades)
    stats = performance_stats(daily_df, trades, broker.start_capital, eq_df["equity"] if len(eq_df) else None)
    stats["decision_every_bars"] = 1 if full_res else int(info.stride)
    return BacktestResult(daily=daily_df, trades=trades, equity=eq_df, stats=stats)

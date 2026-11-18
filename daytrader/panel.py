"""Turn long bar tables into a regular (time x symbol) grid.

Every session is laid out on a fixed grid of bars (78 five-minute bars for
a full day). Missing bars are filled with the previous close and zero
volume, so every rolling/cross-sectional computation downstream is a fast
array operation and all symbols line up on identical timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from .market_calendar import ET, OPEN_MIN, EARLY_CLOSE_MIN, MarketCalendar


@dataclass
class Panel:
    ts: pd.DatetimeIndex          # bar START times, US/Eastern, length T
    symbols: List[str]
    O: np.ndarray                 # (T, S)
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    present: np.ndarray           # (T, S) bool: a real bar existed
    day: np.ndarray               # (T,) index into `dates`
    k: np.ndarray                 # (T,) bar index within the session
    nbar: np.ndarray              # (T,) number of bars in that session
    dates: List[date]
    interval_min: int

    @property
    def T(self) -> int:
        return len(self.ts)

    @property
    def S(self) -> int:
        return len(self.symbols)

    def minute_of_day(self) -> np.ndarray:
        """Minute after midnight ET at which each bar ENDS."""
        return OPEN_MIN + (self.k + 1) * self.interval_min


def build_panel(bars: pd.DataFrame, interval_min: int, calendar: MarketCalendar,
                asof: Optional[datetime] = None, symbols: Optional[List[str]] = None,
                last_n_days: Optional[int] = None, min_coverage: float = 0.2) -> Panel:
    """Build the grid.

    asof: only bars that have fully completed by this moment are used (live).
    """
    if bars is None or len(bars) == 0:
        raise ValueError("No bars to build a panel from.")
    df = bars
    et = df["ts"].dt.tz_convert(ET)
    if asof is not None:
        cutoff = pd.Timestamp(asof).tz_convert("UTC") - pd.Timedelta(minutes=interval_min)
        keep = (df["ts"] <= cutoff).values
        df, et = df[keep], et[keep]
        if len(df) == 0:
            raise ValueError("No completed bars before %s." % asof)
    minute = (et.dt.hour * 60 + et.dt.minute).to_numpy()
    dates_col = et.dt.date.to_numpy()
    sym_col = df["symbol"].to_numpy()
    if symbols is None:
        symbols = sorted(pd.unique(sym_col).tolist())
    sym_index = {s: i for i, s in enumerate(symbols)}
    s_idx = np.fromiter((sym_index.get(s, -1) for s in sym_col), dtype=np.int64, count=len(sym_col))

    on_grid = ((minute - OPEN_MIN) % interval_min == 0) & (minute >= OPEN_MIN) & (s_idx >= 0)
    kk = (minute - OPEN_MIN) // interval_min

    # --- which days make up the grid -------------------------------------
    day_codes, day_inv = np.unique(dates_col, return_inverse=True)
    n_sym = max(len(symbols), 1)
    days: List[date] = []
    day_nbar: List[int] = []
    day_len: List[int] = []
    day_map = np.full(len(day_codes), -1, dtype=np.int64)
    asof_et = pd.Timestamp(asof).tz_convert(ET) if asof is not None else None
    for i, d in enumerate(day_codes):
        m = (day_inv == i) & on_grid
        if not m.any():
            continue
        coverage = len(np.unique(s_idx[m])) / n_sym
        if coverage < min_coverage and i != len(day_codes) - 1:
            continue  # a handful of stray bars: not a real session
        n_cal = calendar.bars_in_session(d, interval_min)
        max_k = int(kk[m].max())
        is_last = i == len(day_codes) - 1
        in_progress = is_last and asof_et is not None and asof_et.date() == d
        if in_progress:
            done = int((asof_et.hour * 60 + asof_et.minute - OPEN_MIN) // interval_min)
            length = max(1, min(n_cal, done))
            n_full = n_cal
        else:
            n_full = n_cal
            half = (EARLY_CLOSE_MIN - OPEN_MIN) // interval_min
            if not is_last and max_k < half and n_cal > half:
                n_full = half       # data says it was a half day the calendar missed
            length = n_full if not is_last else min(n_full, max_k + 1)
            if max_k + 1 > n_full:
                n_full = length = max_k + 1
        day_map[i] = len(days)
        days.append(d)
        day_nbar.append(n_full)
        day_len.append(length)
    if not days:
        raise ValueError("No usable sessions in the bar data.")
    if last_n_days is not None and len(days) > last_n_days:
        drop = len(days) - last_n_days
        days, day_nbar, day_len = days[drop:], day_nbar[drop:], day_len[drop:]
        day_map = np.where(day_map >= drop, day_map - drop, -1)

    offsets = np.concatenate([[0], np.cumsum(day_len)])
    T, S = int(offsets[-1]), len(symbols)
    day_arr = np.repeat(np.arange(len(days)), day_len)
    k_arr = np.concatenate([np.arange(n) for n in day_len])
    nbar_arr = np.repeat(np.array(day_nbar), day_len)
    stamps = []
    for d, n in zip(days, day_len):
        base = pd.Timestamp(datetime(d.year, d.month, d.day, 9, 30), tz=ET)
        stamps.append(base + pd.to_timedelta(np.arange(n) * interval_min, unit="min"))
    ts_index = pd.DatetimeIndex(np.concatenate([s.tz_convert("UTC").as_unit("ns").asi8 for s in stamps]),
                                tz="UTC").tz_convert(ET)

    # --- scatter the bars into the grid ----------------------------------
    d_idx = day_map[day_inv]
    ok = on_grid & (d_idx >= 0)
    len_arr = np.array(day_len)
    ok &= kk < np.where(d_idx >= 0, len_arr[np.clip(d_idx, 0, None)], 0)
    t_idx = offsets[np.clip(d_idx, 0, None)] + kk
    t_idx, s_i = t_idx[ok], s_idx[ok]
    O = np.full((T, S), np.nan)
    H = np.full((T, S), np.nan)
    L = np.full((T, S), np.nan)
    C = np.full((T, S), np.nan)
    V = np.zeros((T, S))
    present = np.zeros((T, S), dtype=bool)
    O[t_idx, s_i] = df["open"].to_numpy()[ok]
    H[t_idx, s_i] = df["high"].to_numpy()[ok]
    L[t_idx, s_i] = df["low"].to_numpy()[ok]
    C[t_idx, s_i] = df["close"].to_numpy()[ok]
    V[t_idx, s_i] = df["volume"].to_numpy()[ok]
    present[t_idx, s_i] = True

    # --- fill gaps: flat bar at the previous close, zero volume -----------
    Cf = pd.DataFrame(C).ffill().to_numpy()
    prev = np.vstack([np.full((1, S), np.nan), Cf[:-1]])
    miss = ~present
    O = np.where(miss, prev, O)
    H = np.where(miss, prev, H)
    L = np.where(miss, prev, L)
    C = np.where(miss, prev, C)
    return Panel(ts=ts_index, symbols=list(symbols), O=O, H=H, L=L, C=C, V=V, present=present,
                 day=day_arr, k=k_arr, nbar=nbar_arr, dates=days, interval_min=interval_min)

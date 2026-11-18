"""Synthetic market + replay provider.

`generate_bars` builds a realistic-looking intraday market (U-shaped volume
and volatility, a market factor, overnight gaps, heavy tails) with an
OPTIONAL planted pattern. It exists for two reasons:

  * tests: with no pattern planted the model must find ~zero edge
    (anything else would mean the pipeline leaks future information), and
    with a pattern planted it must find it;
  * the offline `demo` command.

`ReplayProvider` feeds any stored history (synthetic or real) through the
live engine as if it were happening now.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..market_calendar import ET, MarketCalendar
from .base import DataProvider, Quote, empty_bars


def _u_shape(n: int, open_amp: float, close_amp: float) -> np.ndarray:
    x = np.arange(n) / max(n - 1, 1)
    u = 0.75 + open_amp * np.exp(-x / 0.05) + close_amp * np.exp(-(1 - x) / 0.04)
    return u / np.sqrt(np.mean(u ** 2))


def generate_bars(n_symbols: int = 30, n_days: int = 45, interval_min: int = 5,
                  end_day: Optional[date] = None, seed: int = 7, signal: float = 1.0,
                  calendar: Optional[MarketCalendar] = None) -> pd.DataFrame:
    """Simulate `n_days` sessions of bars. `signal`=0 gives a pure random walk."""
    rng = np.random.default_rng(seed)
    cal = calendar or MarketCalendar()
    end_day = end_day or date(2026, 11, 13)
    days: List[date] = []
    d = end_day
    while len(days) < n_days:
        if cal.is_trading_day(d):
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()

    S = n_symbols
    symbols = ["SYN%02d" % i for i in range(S)]
    beta = rng.uniform(0.6, 1.6, S)
    sig_s = 0.0010 * np.exp(rng.normal(0, 0.35, S))        # idiosyncratic vol per 5-min bar
    scale = np.sqrt(interval_min / 5.0)
    sig_s = sig_s * scale
    sig_m = 0.0006 * scale
    base_vol = np.exp(rng.normal(11.0, 0.8, S)) * (interval_min / 5.0)
    price = np.exp(rng.normal(4.3, 0.7, S)).clip(8, 900)
    log_dayvol = np.zeros(S)

    rows_ts, rows = [], []
    for day in days:
        n = cal.bars_in_session(day, interval_min)
        u = _u_shape(n, 1.6, 0.6)
        uv = _u_shape(n, 2.2, 1.4)
        # persistent "in play" multiplier per symbol-day
        log_dayvol = 0.6 * log_dayvol + rng.normal(0, 0.30, S)
        dv = np.exp(log_dayvol)
        gap = beta * rng.normal(0, 0.004) + rng.normal(0, 0.006, S) * dv
        day_open = price * np.exp(gap)
        mkt = sig_m * u * rng.standard_t(5, n) / np.sqrt(5 / 3)
        eps = rng.standard_t(5, (n, S)) / np.sqrt(5 / 3)
        idio_hist = np.zeros((n, S))
        cum = np.zeros(S)          # log return since open
        o = day_open.copy()
        session_open = datetime.combine(day, time(9, 30), tzinfo=ET)
        for k in range(n):
            shock = eps[k]
            alpha = 0.0
            if signal > 0:
                # (a) 30-minute reversal of the idiosyncratic move
                lo = max(0, k - 6)
                z6 = idio_hist[lo:k].sum(axis=0) / np.sqrt(6.0) if k > 0 else 0.0
                alpha = -0.05 * signal * z6
                # (b) trend continuation on high-activity days
                alpha = alpha + 0.03 * signal * np.sign(cum) * np.clip(dv - 1.0, 0, 2.0) * (k >= 3)
            idio_hist[k] = shock
            r = beta * mkt[k] + sig_s * dv * u[k] * (shock + alpha)
            c = o * np.exp(r)
            wick = np.abs(rng.normal(0, 0.45, (2, S))) * sig_s * dv * u[k]
            hi = np.maximum(o, c) * np.exp(wick[0])
            lo_ = np.minimum(o, c) * np.exp(-wick[1])
            vol = base_vol * uv[k] * dv ** 1.5 * np.exp(rng.normal(0, 0.4, S)) * (1 + 2.0 * np.abs(shock))
            ts = session_open + timedelta(minutes=k * interval_min)
            rows_ts.append(np.full(S, pd.Timestamp(ts).tz_convert("UTC").as_unit("ns").value, dtype="int64"))
            rows.append(np.column_stack([np.arange(S), o, hi, lo_, c, np.round(vol)]))
            cum = cum + r
            o = c
        price = c
    arr = np.vstack(rows)
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(np.concatenate(rows_ts), utc=True),
            "symbol": np.array(symbols, dtype=object)[arr[:, 0].astype(int)],
            "open": arr[:, 1], "high": arr[:, 2], "low": arr[:, 3], "close": arr[:, 4],
            "volume": arr[:, 5],
        }
    )
    return df


class ReplayProvider(DataProvider):
    """Serves stored bars as though the clock's 'now' were the present."""

    name = "replay"
    max_history_days = 3650

    def __init__(self, bars: pd.DataFrame, clock, interval_min: int):
        self.clock = clock
        self.interval_ns = interval_min * 60 * 1_000_000_000
        self._by_sym: Dict[str, tuple] = {}
        ts = bars["ts"].dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").values
        val = bars[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        sym = bars["symbol"].to_numpy()
        for s in pd.unique(sym):
            m = sym == s
            o = np.argsort(ts[m], kind="stable")
            self._by_sym[str(s)] = (ts[m][o], val[m][o])

    def now(self) -> datetime:
        return self.clock.now()

    def _now_ns(self) -> int:
        return int(pd.Timestamp(self.clock.now()).tz_convert("UTC").as_unit("ns").value)

    def fetch_bars(self, symbols: List[str], start: datetime, end: Optional[datetime],
                   interval_min: int) -> pd.DataFrame:
        now_ns = self._now_ns()
        lo = int(pd.Timestamp(start).tz_convert("UTC").as_unit("ns").value)
        hi = now_ns if end is None else min(now_ns, int(pd.Timestamp(end).tz_convert("UTC").as_unit("ns").value))
        parts = []
        for s in symbols:
            cur = self._by_sym.get(s)
            if cur is None:
                continue
            ts, val = cur
            a = np.searchsorted(ts, lo, side="left")
            b = np.searchsorted(ts, hi - self.interval_ns, side="right")  # completed bars only
            if b <= a:
                continue
            part = pd.DataFrame(val[a:b], columns=["open", "high", "low", "close", "volume"])
            part.insert(0, "symbol", s)
            part.insert(0, "ts", pd.to_datetime(ts[a:b], utc=True))
            parts.append(part)
        if not parts:
            return empty_bars()
        return pd.concat(parts, ignore_index=True)

    def latest_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        now_ns = self._now_ns()
        out: Dict[str, Quote] = {}
        for s in symbols:
            cur = self._by_sym.get(s)
            if cur is None:
                continue
            ts, val = cur
            i = np.searchsorted(ts, now_ns, side="right") - 1
            if i < 0:
                continue
            o, c = val[i, 0], val[i, 3]
            frac = (now_ns - ts[i]) / self.interval_ns
            px = c if frac >= 1.0 else o + (c - o) * max(frac, 0.0)
            out[s] = Quote(price=float(px), ts=self.clock.now())
        return out

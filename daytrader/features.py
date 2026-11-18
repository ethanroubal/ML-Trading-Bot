"""Feature engineering and labels.

Everything here is CAUSAL: the value at bar t uses only information that
was available when bar t closed. The same function builds the training
set and the live feature row, so the model sees identical inputs in both.

Returns are measured in units of each stock's own expected volatility
(with the intraday U-shape removed), which makes a quiet utility and a
volatile growth stock comparable and lets one model serve the whole
universe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .panel import Panel

CLIP = 8.0
LABEL_CLIP = 5.0


# ----------------------------------------------------------------- helpers
def _lag(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=float)
    if n < len(a):
        out[n:] = a[:-n]
    return out


def _roll(a: np.ndarray, n: int, how: str, minp: Optional[int] = None) -> np.ndarray:
    r = pd.DataFrame(a).rolling(n, min_periods=minp or n)
    return getattr(r, how)().to_numpy()


def _ewm(a: np.ndarray, span: float, minp: int = 1) -> np.ndarray:
    return pd.DataFrame(a).ewm(span=span, min_periods=minp, adjust=True).mean().to_numpy()


def _xs_mean(a: np.ndarray) -> np.ndarray:
    m = np.isfinite(a)
    s = np.where(m, a, 0.0).sum(axis=1)
    c = m.sum(axis=1)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _xs_std(a: np.ndarray) -> np.ndarray:
    mu = _xs_mean(a)
    m = np.isfinite(a)
    d = np.where(m, a - mu[:, None], 0.0)
    c = m.sum(axis=1)
    return np.where(c > 1, np.sqrt((d ** 2).sum(axis=1) / np.maximum(c - 1, 1)), np.nan)


def _xs_rank(a: np.ndarray) -> np.ndarray:
    return pd.DataFrame(a).rank(axis=1, pct=True).to_numpy() - 0.5


def _prior_days_mean(a3: np.ndarray, n: int, minp: int) -> np.ndarray:
    """Mean over the previous n days (current day excluded) along axis 0, NaN-aware."""
    valid = np.isfinite(a3)
    cs = np.cumsum(np.where(valid, a3, 0.0), axis=0)
    cc = np.cumsum(valid, axis=0)
    zero = np.zeros((1,) + a3.shape[1:])
    cs = np.concatenate([zero, cs], axis=0)
    cc = np.concatenate([zero, cc], axis=0)
    d = np.arange(a3.shape[0])
    lo = np.maximum(d - n, 0)
    tot = cs[d] - cs[lo]
    cnt = cc[d] - cc[lo]
    return np.where(cnt >= minp, tot / np.maximum(cnt, 1), np.nan)


def default_tod_profile(n: int) -> np.ndarray:
    """Typical intraday volatility shape: busy open, quiet midday, busier close."""
    x = np.arange(n) / max(n - 1, 1)
    u = 0.75 + 1.5 * np.exp(-x / 0.05) + 0.5 * np.exp(-(1 - x) / 0.04)
    return u / np.sqrt(np.mean(u ** 2))


@dataclass
class FeatureSet:
    names: List[str]
    arrays: List[np.ndarray]      # each (T, S) float32
    sig: np.ndarray               # (T, S) per-bar volatility, intraday shape removed
    sigH: np.ndarray              # (T, S) expected volatility over the forward horizon
    tod: np.ndarray               # (T,) intraday volatility multiplier for each bar
    y: np.ndarray                 # (T, S) label in volatility units (NaN if unknown)
    fwd_ret: np.ndarray           # (T, S) raw forward log return
    fwd_rel: np.ndarray           # (T, S) forward log return relative to the universe
    usable: np.ndarray            # (T, S) bool: safe to act on this symbol at this bar

    def matrix(self, mask: np.ndarray) -> np.ndarray:
        n = int(mask.sum())
        X = np.empty((n, len(self.arrays)), dtype=np.float32)
        for i, a in enumerate(self.arrays):
            X[:, i] = a[mask]
        return X

    def row(self, t: int) -> np.ndarray:
        return np.column_stack([a[t] for a in self.arrays]).astype(np.float32)

    def get(self, name: str) -> np.ndarray:
        return self.arrays[self.names.index(name)]

    def beta(self, t: int) -> np.ndarray:
        """Market beta per symbol at bar t (1.0 where unknown), bounded to a sane range."""
        b = self.get("beta")[t].astype(float)
        return np.clip(np.where(np.isfinite(b), b, 1.0), 0.2, 3.0)


def compute_features(panel: Panel, cfg: Config, with_labels: bool = True) -> FeatureSet:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _compute(panel, cfg, with_labels)


def _compute(p: Panel, cfg: Config, with_labels: bool) -> FeatureSet:
    T, S = p.T, p.S
    B = cfg.bars_per_day
    H = cfg.model.horizon_bars
    O, Hi, Lo, C, V = p.O, p.H, p.L, p.C, p.V
    day, k, nbar = p.day, p.k, p.nbar
    D = len(p.dates)
    K = int(max(nbar.max(), k.max() + 1))
    offsets = np.flatnonzero(np.r_[True, day[1:] != day[:-1], True])
    day_len = np.diff(offsets)
    day_nbar = nbar[offsets[:-1]]
    day_end_row = (offsets[1:] - 1)[day]            # (T,) last grid row of the same session

    def to3(a: np.ndarray) -> np.ndarray:
        out = np.full((D, K) + a.shape[1:], np.nan)
        out[day, k] = a
        return out

    def from3(a3: np.ndarray) -> np.ndarray:
        return a3[day, k]

    def day_acc(a: np.ndarray, fn) -> np.ndarray:
        out = np.empty_like(a, dtype=float)
        for i in range(D):
            sl = slice(offsets[i], offsets[i + 1])
            out[sl] = fn(a[sl], axis=0)
        return out

    def prev_day(a_ds: np.ndarray, n: int = 1) -> np.ndarray:
        """(D, S) per-day values -> (T, S) value from n sessions earlier."""
        sh = np.full_like(a_ds, np.nan, dtype=float)
        if n < D:
            sh[n:] = a_ds[:-n]
        return sh[day]

    lc = np.log(C)
    first = (k == 0)[:, None]
    ret1 = np.where(first, np.log(C / O), lc - _lag(lc, 1))       # intraday bar return

    # ---- intraday volatility profile, estimated from PRIOR days only -----
    sig0 = np.sqrt(_ewm(ret1 ** 2, span=2 * B, minp=max(B // 4, 2)))
    a = np.abs(ret1) / _lag(sig0, 1)
    a[~np.isfinite(a)] = np.nan
    with_warn = np.where(np.isfinite(a).sum(axis=1) >= 3, 1.0, np.nan)
    med = np.array([np.nanmedian(r) if w == 1.0 else np.nan for r, w in zip(a, with_warn)])
    prof = _prior_days_mean(to3(med[:, None]), 20, 5)[:, :, 0]     # (D, K)
    for i in range(D):
        n = int(day_nbar[i])
        dflt = default_tod_profile(n)
        row = prof[i, :n]
        row = np.where(np.isfinite(row) & (row > 0), row, np.nan)
        if np.isfinite(row).sum() < n // 2:
            row = dflt
        else:
            scale = np.sqrt(np.nanmean(row ** 2))
            row = np.where(np.isfinite(row), row / scale, dflt)
            row = np.clip(row, 0.3, 5.0)
            row = row / np.sqrt(np.mean(row ** 2))
        prof[i, :n] = row
        prof[i, n:] = np.nan
    tod = prof[day, k]

    u = ret1 / tod[:, None]                                        # de-seasonalised return
    u2 = u ** 2
    sig = np.sqrt(_ewm(u2, span=2 * B, minp=max(B // 2, 2)))
    sig[~(sig > 1e-6)] = np.nan

    # ---- forward horizon: bars covered and expected volatility -----------
    bars_after = nbar - 1 - k                                      # bars left after this one
    hb = np.minimum(H, bars_after)
    prof2 = np.nan_to_num(prof, nan=0.0) ** 2
    cs = np.concatenate([np.zeros((D, 1)), np.cumsum(prof2, axis=1)], axis=1)   # cs[d, j] = sum prof2[:j]
    sumsq = cs[day, np.minimum(k + 1 + hb, K)] - cs[day, np.minimum(k + 1, K)]
    sigH = sig * np.sqrt(sumsq)[:, None]

    # ---- labels -----------------------------------------------------------
    y = np.full((T, S), np.nan)
    fwd = np.full((T, S), np.nan)
    rel = np.full((T, S), np.nan)
    if with_labels:
        t = np.arange(T)
        complete = (day_len == day_nbar)[day]
        full_h = (k + 1 + H) <= (nbar - 1)
        entry_row = np.minimum(t + 1, T - 1)
        exit_row = np.minimum(t + 1 + H, T - 1)
        valid = (t + 1 <= day_end_row) & (bars_after >= cfg.model.min_bars_left)
        valid &= np.where(full_h, (t + 1 + H) <= day_end_row, complete)
        exit_px = np.where(full_h[:, None], O[exit_row], C[day_end_row])
        fwd = np.log(exit_px / O[entry_row])
        fwd[~valid] = np.nan
        fwd[~np.isfinite(fwd)] = np.nan
        mkt_fwd = _xs_mean(fwd)
        rel = fwd - mkt_fwd[:, None] if cfg.model.target == "relative" else fwd.copy()
        y = np.clip(rel / sigH, -LABEL_CLIP, LABEL_CLIP)
        y[~np.isfinite(y)] = np.nan

    # ---- features ---------------------------------------------------------
    names: List[str] = []
    arrays: List[np.ndarray] = []

    def add(name: str, arr: np.ndarray, clip: float = CLIP) -> None:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = np.repeat(arr[:, None], S, axis=1)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        if clip:
            arr = np.clip(arr, -clip, clip)
        names.append(name)
        arrays.append(arr.astype(np.float32))

    sq = np.sqrt
    # momentum / reversal over several horizons, in volatility units
    add("u1", u / sig)
    z: Dict[int, np.ndarray] = {}
    for n in (1, 2, 3, 6, 12, 24, 48, B):
        z[n] = (lc - _lag(lc, n)) / (sig * sq(n))
        if n != 1:
            add("z%d" % n, z[n])
    for n in (1, 3, 6, 12, 24, 48):
        m = _xs_mean(z[n])
        add("xz%d" % n, z[n] - m[:, None])
        if n <= 24:
            add("mkt_z%d" % n, m)
    add("rank_z6", _xs_rank(z[6]), clip=0)
    add("rank_z24", _xs_rank(z[24]), clip=0)
    add("disp6", _xs_std(z[6]))

    # market beta, correlation and beta-adjusted returns
    m1 = np.nan_to_num(_xs_mean(ret1))[:, None]                    # (T, 1) market bar return
    W, wmin = 2 * B, max(B // 2, 4)
    ex, em = _roll(ret1, W, "mean", wmin), _roll(m1, W, "mean", wmin)
    cov = _roll(ret1 * m1, W, "mean", wmin) - ex * em
    var_m = _roll(m1 ** 2, W, "mean", wmin) - em ** 2
    var_x = _roll(ret1 ** 2, W, "mean", wmin) - ex ** 2
    beta = np.clip(cov / var_m, -1.0, 4.0)
    add("beta", beta, clip=0)
    add("corr_mkt", cov / sq(var_x * var_m), clip=1.0)
    for n in (3, 12):
        rn = _roll(ret1, n, "sum", 1)
        mn = _roll(m1, n, "sum", 1)
        add("res%d" % n, (rn - beta * mn) / (sig * sq(n)))

    # where are we relative to today's open / yesterday / recent days
    day_open = O[offsets[:-1]]                                     # (D, S) first bar's open
    day_close = C[offsets[1:] - 1]                                 # last bar present in each session
    day_high = np.array([np.nanmax(Hi[offsets[i]:offsets[i + 1]], axis=0) for i in range(D)])
    day_low = np.array([np.nanmin(Lo[offsets[i]:offsets[i + 1]], axis=0) for i in range(D)])
    kk = (k + 1.0)[:, None]
    dopen = day_open[day]
    pc1 = prev_day(day_close, 1)
    z_open = np.log(C / dopen) / (sig * sq(kk))
    add("z_open", z_open)
    add("xz_open", z_open - _xs_mean(z_open)[:, None])
    add("mkt_z_open", _xs_mean(z_open))
    add("rank_z_open", _xs_rank(z_open), clip=0)
    add("z_gap", np.log(dopen / pc1) / (sig * sq(B)))
    add("z_prev_close", np.log(C / pc1) / (sig * sq(B)))
    add("z_prev_day", np.log(pc1 / prev_day(day_close, 2)) / (sig * sq(B)))
    add("z_5d", np.log(pc1 / prev_day(day_close, 6)) / (sig * sq(5 * B)))
    add("z_10d", np.log(pc1 / prev_day(day_close, 11)) / (sig * sq(10 * B)))
    add("z_prev_oc", np.log(pc1 / prev_day(day_open, 1)) / (sig * sq(B)))
    hour = max(60 // p.interval_min, 1)
    c_before_close = np.full((D, S), np.nan)
    for i in range(D):
        n_i = int(day_len[i])
        if n_i > hour:
            c_before_close[i] = C[offsets[i] + n_i - 1 - hour]
    add("z_prev_last_hour", np.log(pc1 / prev_day(c_before_close, 1)) / (sig * sq(hour)))

    # VWAP and the day's range
    tp = (Hi + Lo + C) / 3.0
    cum_v = day_acc(V, np.nancumsum)
    cum_pv = day_acc(tp * V, np.nancumsum)
    vwap = np.where(cum_v > 0, cum_pv / cum_v, C)
    add("z_vwap", np.log(C / vwap) / (sig * sq(kk)))
    dh = day_acc(Hi, np.fmax.accumulate)
    dl = day_acc(Lo, np.fmin.accumulate)
    rng_day = dh - dl
    add("pos_day", np.where(rng_day > 0, (C - dl) / rng_day - 0.5, 0.0), clip=0)
    add("z_from_high", np.log(C / dh) / (sig * sq(kk)))
    add("z_from_low", np.log(C / dl) / (sig * sq(kk)))
    add("z_day_range", np.log(dh / dl) / (sig * sq(kk)))

    # opening range (first 30 minutes)
    n_or = max(30 // p.interval_min, 1)
    orh = np.full((D, S), np.nan)
    orl = np.full((D, S), np.nan)
    for i in range(D):
        if day_len[i] >= n_or:
            r = offsets[i] + n_or - 1
            orh[i], orl[i] = dh[r], dl[r]
    after_or = (k >= n_or - 1)[:, None]
    add("z_or_high", np.where(after_or, np.log(C / orh[day]) / (sig * sq(n_or)), np.nan))
    add("z_or_low", np.where(after_or, np.log(C / orl[day]) / (sig * sq(n_or)), np.nan))

    # prior-day and multi-day levels
    add("z_prev_high", np.log(C / prev_day(day_high, 1)) / (sig * sq(B)))
    add("z_prev_low", np.log(C / prev_day(day_low, 1)) / (sig * sq(B)))
    for n in (5, 20):
        hh = np.fmax(prev_day(_roll(day_high, n, "max", 1), 1), dh)   # incl. today so far
        ll = np.fmin(prev_day(_roll(day_low, n, "min", 1), 1), dl)
        add("pos_%dd" % n, np.where(hh > ll, (C - ll) / (hh - ll) - 0.5, 0.0), clip=0)

    # volume: is this stock "in play" today?
    V3 = to3(V)
    cumV3 = np.nancumsum(np.nan_to_num(V3, nan=0.0), axis=1)
    cumV3[~np.isfinite(V3)] = np.nan
    base_cum = from3(_prior_days_mean(cumV3, 10, 3))
    base_bar = from3(_prior_days_mean(V3, 10, 3))
    rvol = np.log((cum_v + 1.0) / (base_cum + 1.0))
    add("rvol_day", rvol, clip=5.0)
    add("rank_rvol", _xs_rank(rvol), clip=0)
    add("rvol_bar", np.log((V + 1.0) / (base_bar + 1.0)), clip=5.0)
    add("vol_trend", np.log((_roll(V, 3, "mean", 1) + 1.0) / (_roll(V, 12, "mean", 1) + 1.0)), clip=5.0)
    sv = np.sign(ret1) * V
    for n in (6, 24):
        den = _roll(V, n, "sum", 1)
        add("flow%d" % n, np.where(den > 0, _roll(sv, n, "sum", 1) / den, 0.0), clip=0)
    dollar = np.array([np.nansum(C[offsets[i]:offsets[i + 1]] * V[offsets[i]:offsets[i + 1]], axis=0)
                       for i in range(D)])
    add("log_dollar_vol", np.log10(_prior_days_mean(dollar[:, None, :], 10, 2)[:, 0, :][day] + 1.0), clip=0)
    add("log_price", lc, clip=0)

    # volatility state
    add("rv6", sq(_roll(u2, 6, "mean", 3)) / sig)
    rv24 = sq(_roll(u2, 24, "mean", 6)) / sig
    add("rv24", rv24)
    add("mkt_rv24", _xs_mean(rv24))
    add("vol_fast", np.log(sq(_ewm(u2, span=max(B // 2, 2), minp=max(B // 4, 2))) / sig), clip=4.0)
    add("vol_regime", np.log(sig / sq(_ewm(u2, span=10 * B, minp=B))), clip=4.0)
    add("log_sig_bps", np.log(sig * 1e4), clip=0)

    # shape of the latest bars
    rng = Hi - Lo
    clv = np.where(rng > 0, ((C - Lo) - (Hi - C)) / rng, 0.0)
    add("clv", clv, clip=0)
    add("clv6", _roll(clv, 6, "mean", 1), clip=0)
    add("body", np.where(rng > 0, (C - O) / rng, 0.0), clip=0)
    add("bar_range", np.log(Hi / Lo) / (sig * tod[:, None]))

    # classic oscillators
    gain = _ewm(np.where(ret1 > 0, ret1, 0.0), span=27, minp=14)   # Wilder RSI(14)
    loss = _ewm(np.where(ret1 < 0, -ret1, 0.0), span=27, minp=14)
    add("rsi14", np.where(gain + loss > 0, gain / (gain + loss) - 0.5, 0.0), clip=0)
    for n in (20, B):
        sd = _roll(lc, n, "std", n // 2)
        add("bb%d" % n, np.where(sd > 0, (lc - _roll(lc, n, "mean", n // 2)) / sd, 0.0), clip=5.0)
    macd = (_ewm(lc, 12, 12) - _ewm(lc, 26, 26)) / (sig * sq(26.0))
    add("macd", macd)
    add("macd_hist", macd - _ewm(macd, 9, 9))

    # seasonality: time of day, and what this stock did at this time on recent days
    add("tod_frac", k / np.maximum(nbar - 1, 1), clip=0)
    add("bars_left", bars_after / float(B), clip=0)
    add("dow", np.array([d.weekday() for d in p.dates], dtype=float)[day], clip=0)
    Y3 = to3(y)
    add("seas_fwd5", from3(_prior_days_mean(Y3, 5, 3)))
    add("seas_fwd20", from3(_prior_days_mean(Y3, 20, 8)))
    U3 = _prior_days_mean(to3(u / sig), 10, 5)
    nxt = np.full_like(U3, np.nan)
    nxt[:, :-1] = U3[:, 1:]
    add("seas_next", from3(nxt))

    # ---- which (bar, symbol) cells are safe to act on ---------------------
    recent = _roll(p.present.astype(float), 6, "sum", 1)
    usable = (np.isfinite(sig) & np.isfinite(sigH) & (sigH > 0) & p.present
              & (recent >= 3) & (C >= cfg.strategy.min_price))
    usable &= (bars_after >= cfg.model.min_bars_left)[:, None]

    return FeatureSet(names=names, arrays=arrays, sig=sig, sigH=sigH, tod=tod, y=y,
                      fwd_ret=fwd, fwd_rel=rel, usable=usable)

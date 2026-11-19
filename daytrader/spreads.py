"""Measured bid/ask spreads, per stock and time of day.

The cost of a trade is mostly the spread you cross. Instead of one assumed
number for every stock, this keeps a table of what each stock's spread has
actually been in each part of the session:

  * from Alpaca's historical quotes and trades it samples a few moments in each
    part of the day for each recent session it has not measured yet (at startup
    and after every close).

What is stored is the EFFECTIVE spread: twice the distance between each trade
and the national best mid-quote at that moment, averaged over the ordinary
trades on exchanges in the window. That is what a marketable order actually
paid, including any price improvement inside the quote. It is never taken as
less than half the quoted spread, and windows with fewer than MIN_TRADES
trades use the quoted spread. The quoted spread is kept too, for comparison.

Each (stock, time-of-day bucket, day) keeps the median of that day's samples.
The estimate is the median of the last `spread_lookback_days` daily medians,
needing at least `spread_min_days` of them - otherwise the assumed model is
used for that stock and bucket.

Measurements come from Alpaca's full-market (SIP) quotes, which show the
national best bid and offer. The free plan serves them with a 15-minute delay,
so on the free plan the table is filled from quote history - at startup and
after each close - rather than live. (The free real-time feed only shows
IEX's own order book, often dollars wide for stocks whose real spread is a
cent, so it is never used for spreads.)
Spreads are kept both in basis points and in dollars; applied to a different
price (older history), the wider of the two readings is used, because for
penny-tick stocks the dollar spread, not the percentage, is what stays fixed.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Bucket edges in minutes after midnight ET. Spreads are widest just after the open.
BUCKET_EDGES = [570, 585, 600, 630, 690, 810, 870, 930, 960]
BUCKET_NAMES = ["09:30", "09:45", "10:00", "10:30", "11:30", "13:30", "14:30", "15:30"]
# one moment inside each bucket, used when sampling historical quotes
SAMPLE_MINUTES = [577, 592, 615, 660, 750, 840, 900, 945]
MAX_SPREAD_BPS = 200.0      # quotes wider than this are treated as junk (crossed, stub or halted)
VERSION = 4          # older tables (IEX quotes; quoted rather than effective spreads) are discarded
# trade conditions of ordinary trades; anything else (average-price, derivatively priced, opening/
# closing prints, out-of-sequence, extended hours...) does not show what a marketable order pays
PLAIN_CONDITIONS = {"@", " ", "I", "F", "E"}
OFF_EXCHANGE = {"D"}   # FINRA trade reporting: dark pools and wholesalers, not reachable by a routed order
MIN_TRADES = 3         # fewer lit trades than this in a window: use the quoted spread instead


def bucket_of(minute) -> np.ndarray:
    """Time-of-day bucket for minutes after midnight ET (array or scalar)."""
    m = np.asarray(minute, dtype=float)
    return np.clip(np.searchsorted(BUCKET_EDGES, m, side="right") - 1, 0, len(BUCKET_NAMES) - 1)


def spread_of(bid, ask) -> Optional[Tuple[float, float]]:
    """(spread in bp of the mid, spread in dollars), or None for an unusable quote."""
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if not (bid > 0 and ask > bid):
        return None
    mid = 0.5 * (bid + ask)
    bp = (ask - bid) / mid * 1e4
    if bp > MAX_SPREAD_BPS:
        return None
    return bp, ask - bid


class SpreadBook:
    def __init__(self, path: Optional[Path] = None, lookback_days: int = 20, min_days: int = 3,
                 feed: str = "sip"):
        self.path = Path(path) if path else None
        self.feed = feed
        self.sampled: set = set()      # sessions whose quote history has been fully sampled
        self.lookback_days = lookback_days
        self.min_days = min_days
        # done[sym][bucket][day] = [median bp, median $, n samples]: the spread a trade pays
        self.done: Dict[str, Dict[str, Dict[str, list]]] = {}
        # quoted[sym][bucket][day]: the quoted national best spread, kept for comparison
        self.quoted: Dict[str, Dict[str, Dict[str, list]]] = {}
        # today's raw samples: open[sym][bucket][day] = [[bp, $], ...]
        self.open: Dict[str, Dict[str, Dict[str, list]]] = {}
        self._est: Optional[Dict[Tuple[str, int], Tuple[float, float]]] = None
        self._mat = None
        self._idx_cache = None

    # ------------------------------------------------------------ storage
    @classmethod
    def load(cls, path: Path, lookback_days: int = 20, min_days: int = 3, feed: str = "sip") -> "SpreadBook":
        book = cls(path, lookback_days, min_days, feed)
        try:
            raw = json.loads(Path(path).read_text())
            if raw.get("version") == VERSION and raw.get("feed", "sip") == feed:
                book.done = raw.get("done") or {}
                book.open = raw.get("open") or {}
                book.quoted = raw.get("quoted") or {}
                book.sampled = set(raw.get("sampled") or [])
        except FileNotFoundError:
            pass
        except Exception:
            pass        # a damaged file is rebuilt from fresh samples
        return book

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": VERSION, "feed": self.feed, "sampled": sorted(self.sampled),
                                   "done": self.done, "quoted": self.quoted, "open": self.open}))
        tmp.replace(self.path)

    # ------------------------------------------------------------ samples
    def add(self, symbol: str, day: date, minute: int, bid, ask) -> bool:
        sp = spread_of(bid, ask)
        if sp is None or not (BUCKET_EDGES[0] <= minute < BUCKET_EDGES[-1]):
            return False
        b = str(int(bucket_of(minute)))
        self.open.setdefault(symbol, {}).setdefault(b, {}).setdefault(str(day), []).append(
            [round(sp[0], 3), round(sp[1], 5)])
        self._est = self._mat = None
        return True

    def add_day_median(self, symbol: str, day: date, bucket: int, samples: List[Tuple[float, float]],
                       quoted: bool = False) -> None:
        """Store finished samples for one (symbol, bucket, day) directly (historical bootstrap)."""
        if not samples:
            return
        arr = np.asarray(samples, dtype=float)
        cell = (self.quoted if quoted else self.done).setdefault(symbol, {}).setdefault(str(int(bucket)), {})
        old = cell.get(str(day))
        n = len(arr) + (old[2] if old else 0)
        if old:     # merge: weight the old median by its sample count
            arr = np.vstack([arr, np.repeat([[old[0], old[1]]], min(old[2], 50), axis=0)])
        cell[str(day)] = [round(float(np.median(arr[:, 0])), 3), round(float(np.median(arr[:, 1])), 5), int(n)]
        self._est = self._mat = None

    def close_days(self, before: Optional[date] = None) -> None:
        """Collapse raw samples into daily medians (all days, or only days before `before`)."""
        for sym, buckets in list(self.open.items()):
            for b, days in list(buckets.items()):
                for d, samples in list(days.items()):
                    if before is not None and d >= str(before):
                        continue
                    self.add_day_median(sym, d, int(b), [tuple(x) for x in samples])
                    del days[d]
                if not days:
                    del buckets[b]
            if not buckets:
                del self.open[sym]
        self._trim()
        self._est = self._mat = None

    def _trim(self) -> None:
        keep = self.lookback_days * 2
        for buckets in list(self.done.values()) + list(self.quoted.values()):
            for b, days in buckets.items():
                if len(days) > keep:
                    for d in sorted(days)[:-keep]:
                        del days[d]

    # ----------------------------------------------------------- estimates
    def improvement(self) -> Optional[float]:
        """Median ratio of the spread trades paid to the quoted spread, over all stocks and buckets."""
        ratios = []
        for sym, buckets in self.done.items():
            for b, days in buckets.items():
                q = self.quoted.get(sym, {}).get(b, {})
                for d, v in days.items():
                    if d in q and q[d][1] > 0:
                        ratios.append(v[1] / q[d][1])
        return float(np.median(ratios)) if ratios else None

    def _estimates(self) -> Dict[Tuple[str, int], Tuple[float, float]]:
        if self._est is not None:
            return self._est
        est: Dict[Tuple[str, int], Tuple[float, float]] = {}
        for sym in set(self.done) | set(self.open):
            for b in range(len(BUCKET_NAMES)):
                rows = dict(self.done.get(sym, {}).get(str(b), {}))
                for d, samples in self.open.get(sym, {}).get(str(b), {}).items():
                    arr = np.asarray(samples, dtype=float)
                    if len(arr) and d not in rows:
                        rows[d] = [float(np.median(arr[:, 0])), float(np.median(arr[:, 1])), len(arr)]
                recent = [rows[d] for d in sorted(rows)[-self.lookback_days:]]
                if len(recent) >= self.min_days:
                    arr = np.asarray([[r[0], r[1]] for r in recent], dtype=float)
                    est[(sym, b)] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
        self._est = est
        return est

    def _matrix(self):
        """(bp[S, B], $[S, B], {symbol: row}) of every symbol with any estimate; NaN = unknown."""
        if self._mat is None:
            est = self._estimates()
            names = sorted({s for s, _ in est})
            pos = {s: i for i, s in enumerate(names)}
            bp = np.full((len(names) + 1, len(BUCKET_NAMES)), np.nan)    # last row: unknown symbol
            usd = np.full_like(bp, np.nan)
            for (s, b), (e_bp, e_usd) in est.items():
                bp[pos[s], b], usd[pos[s], b] = e_bp, e_usd
            self._mat = (bp, usd, pos)
            self._idx_cache = None
        return self._mat

    def _indices(self, symbols) -> np.ndarray:
        _, _, pos = self._matrix()
        unknown = len(pos)
        cache = self._idx_cache
        if isinstance(symbols, list) and cache is not None and cache[0] is symbols and cache[1] == len(symbols):
            return cache[2]
        if isinstance(symbols, str):
            return np.asarray(pos.get(symbols, unknown))
        arr = np.asarray(symbols).astype(str)
        uniq, inv = np.unique(arr, return_inverse=True)
        idx = np.array([pos.get(u, unknown) for u in uniq], dtype=int)[inv].reshape(arr.shape)
        if isinstance(symbols, list):
            self._idx_cache = (symbols, len(symbols), idx)
        return idx

    def spread_bps(self, symbols, minute, price) -> np.ndarray:
        """Estimated full spread in bp at `price` (NaN where there is not enough data).

        symbols, minute and price broadcast together; minute is minutes after midnight ET."""
        bp_m, usd_m, _ = self._matrix()
        idx = self._indices(symbols)
        price = np.asarray(price, dtype=float)
        idx, b, price = np.broadcast_arrays(idx, bucket_of(minute), price)
        e_bp, e_usd = bp_m[idx, b], usd_m[idx, b]
        with np.errstate(divide="ignore", invalid="ignore"):
            usd_bps = np.where(price > 0, e_usd / price * 1e4, np.nan)
        return np.where(np.isfinite(usd_bps), np.fmax(e_bp, usd_bps), e_bp) * np.where(np.isfinite(e_bp), 1.0, np.nan)

    # ------------------------------------------------------------ summary
    def coverage(self, symbols: Iterable[str]) -> Tuple[int, int]:
        """(symbols with an estimate in at least one bucket, symbols asked about)."""
        est = self._estimates()
        have = {s for s, _ in est}
        syms = list(symbols)
        return sum(1 for s in syms if s in have), len(syms)

    def days_measured(self) -> int:
        days = set()
        for buckets in list(self.done.values()) + list(self.open.values()):
            for d in buckets.values():
                days.update(d)
        return len(days)

    def table(self, symbols: Iterable[str]) -> List[dict]:
        """Per-symbol estimates (bp) by bucket, for reports."""
        est = self._estimates()
        out = []
        for s in symbols:
            row = {"symbol": s}
            for b, name in enumerate(BUCKET_NAMES):
                e = est.get((s, b))
                row[name] = round(e[0], 2) if e else None
            out.append(row)
        return out


def nbbo_spreads(rows, warmup_until=None) -> List[Tuple[float, float]]:
    """National best bid/offer spreads from a stream of per-exchange quotes.

    rows: (time, bid exchange, bid, bid size, ask exchange, ask, ask size), in time order, or
    plain (bid, ask) pairs that are already the best quote. Keeps each exchange's latest bid
    and ask (a zero price or size withdraws it); after `warmup_until` (an RFC 3339 time, so
    every active exchange has been seen) records the best ask minus best bid at each update.
    Locked or crossed moments are skipped."""
    out: List[Tuple[float, float]] = []
    bids: Dict[str, float] = {}
    asks: Dict[str, float] = {}
    rows = list(rows)
    full = [r for r in rows if len(r) == 7]
    if full and sum(1 for r in full if r[1] != r[4]) > 0.5 * len(full):
        # bid and ask mostly on different exchanges: these are national best quotes already
        rows = [(r[2], r[5]) if len(r) == 7 else r for r in rows
                if len(r) != 7 or warmup_until is None or r[0] is None or str(r[0]) >= warmup_until]
    for r in rows:
        if len(r) == 2:
            sp = spread_of(r[0], r[1])
            if sp is not None:
                out.append(sp)
            continue
        t, bx, bp, bs, ax, ap, as_ = r
        if bx is not None:
            if bp and bs:
                bids[bx] = float(bp)
            else:
                bids.pop(bx, None)
        if ax is not None:
            if ap and as_:
                asks[ax] = float(ap)
            else:
                asks.pop(ax, None)
        if warmup_until is not None and t is not None and str(t) < warmup_until:
            continue
        if bids and asks:
            sp = spread_of(max(bids.values()), min(asks.values()))
            if sp is not None:
                out.append(sp)
    return out


def effective_spreads(quote_rows, trade_rows, warmup_until=None) -> List[Tuple[float, float]]:
    """What trades actually paid: 2 x |trade price - national best mid| (bp of mid, $), per trade.

    quote_rows as for nbbo_spreads; trade_rows: (time, exchange, price, size, conditions).
    Only ordinary trades on exchanges count - off-exchange prints (dark pools, wholesalers)
    are left out, so price improvement only a wholesaler gives is not assumed."""
    quote_rows, trade_rows = list(quote_rows), list(trade_rows)
    full = [r for r in quote_rows if len(r) == 7]
    nbbo_mode = bool(full) and sum(1 for r in full if r[1] != r[4]) > 0.5 * len(full)
    events = [(str(r[0]) if len(r) == 7 and r[0] is not None else "", 0, r) for r in quote_rows]
    events += [(str(r[0]) if r[0] is not None else "", 1, r) for r in trade_rows]
    events.sort(key=lambda e: (e[0], e[1]))
    bids: Dict[str, float] = {}
    asks: Dict[str, float] = {}
    best: Optional[Tuple[float, float]] = None
    out: List[Tuple[float, float]] = []
    for t, kind, r in events:
        if kind == 0:
            if len(r) == 2 or nbbo_mode:
                bid, ask = (r[0], r[1]) if len(r) == 2 else (r[2], r[5])
                best = (float(bid), float(ask)) if bid and ask else best
            else:
                _, bx, bp, bs, ax, ap, as_ = r
                if bx is not None:
                    if bp and bs:
                        bids[bx] = float(bp)
                    else:
                        bids.pop(bx, None)
                if ax is not None:
                    if ap and as_:
                        asks[ax] = float(ap)
                    else:
                        asks.pop(ax, None)
                best = (max(bids.values()), min(asks.values())) if bids and asks else None
            continue
        if warmup_until is not None and t and t < warmup_until:
            continue
        _, x, p, sz, cond = r
        if x in OFF_EXCHANGE or not p or best is None or spread_of(*best) is None:
            continue
        if cond and any(c not in PLAIN_CONDITIONS for c in cond):
            continue
        mid = 0.5 * (best[0] + best[1])
        eff = 2.0 * abs(float(p) - mid)
        out.append((eff / mid * 1e4, eff))
    return out


def bootstrap(book: SpreadBook, provider, symbols: List[str], days: List[date],
              log: Callable[[str], None] = print, window_seconds: int = 8,
              now: Optional[datetime] = None, delay_minutes: float = 15.5, feed: Optional[str] = None) -> int:
    """Fill the table from historical quotes: for each session in `days` and each part of the day,
    take the quotes of every symbol during a short window. Windows less than `delay_minutes` old
    are skipped (the free plan serves full-market quotes 15 minutes late); a session counts as
    sampled once all its windows are done. Returns the samples stored."""
    from .market_calendar import ET, UTC

    fetch = getattr(provider, "historical_quotes", None)
    if fetch is None:
        return 0
    stored = 0
    for k, day in enumerate(days):
        complete = True
        for b, minute in enumerate(SAMPLE_MINUTES):
            start = datetime(day.year, day.month, day.day, minute // 60, minute % 60, tzinfo=ET)
            end = start + timedelta(seconds=window_seconds)
            warm = (start + timedelta(seconds=window_seconds / 2)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")
            if now is not None and end > now - timedelta(minutes=delay_minutes):
                complete = False
                continue
            trades_fetch = getattr(provider, "historical_trades", None)
            try:
                got = fetch(symbols, start, end, feed=feed) if feed else fetch(symbols, start, end)
                trades = (trades_fetch(symbols, start, end, feed=feed) if feed else trades_fetch(symbols, start, end)) \
                    if trades_fetch else {}
            except Exception as exc:
                log("  ! spread sample %s %s failed (%s)" % (day, BUCKET_NAMES[b], exc))
                complete = False
                if "rejected" in str(exc).lower() or "subscription" in str(exc).lower():
                    raise
                continue
            for sym, pairs in got.items():
                sps = nbbo_spreads(pairs, warm)
                if not sps:
                    continue
                # one sample per window, so busy stocks do not dominate
                q = np.median(np.asarray(sps), axis=0)
                book.add_day_median(sym, day, b, [(float(q[0]), float(q[1]))], quoted=True)
                cost = q
                effs = effective_spreads(pairs, trades.get(sym, []), warm)
                if len(effs) >= MIN_TRADES:
                    # the average trade's cost, but never below half the quoted spread
                    cost = np.maximum(np.mean(np.asarray(effs), axis=0), 0.5 * q)
                book.add_day_median(sym, day, b, [(float(cost[0]), float(cost[1]))])
                stored += 1
        if complete:
            book.sampled.add(str(day))
        log("  spreads: sampled %s (%d/%d sessions)" % (day, k + 1, len(days)))
    book._trim()
    return stored

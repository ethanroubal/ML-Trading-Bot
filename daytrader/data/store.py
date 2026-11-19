"""Local bar cache: one compact .npz file per symbol, plus an in-memory copy.

The cache lets history grow beyond what a free data source serves in one
request (Yahoo only returns ~60 days of 5-minute bars, but every day the
trader runs adds another day to the cache).
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BAR_COLUMNS, empty_bars

_FIELDS = ("open", "high", "low", "close", "volume")


def _to_ns(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return int(t.tz_convert("UTC").as_unit("ns").value)


class BarStore:
    def __init__(self, root: Path, interval_min: int, persist: bool = True):
        self.dir = Path(root) / ("%dm" % interval_min)
        self.persist = persist
        if persist:
            self.dir.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._dirty: set = set()        # symbols changed in memory but not yet written (batch mode)
        self._batch = 0
        self._meta: Optional[dict] = None

    # ---- internals -------------------------------------------------------
    def _file(self, symbol: str) -> Path:
        return self.dir / (symbol.replace("/", "_") + ".npz")

    def _get(self, symbol: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if symbol in self._mem:
            return self._mem[symbol]
        if self.persist:
            f = self._file(symbol)
            if f.exists():
                try:
                    with np.load(f) as z:
                        ts = z["ts"].astype("int64")
                        val = z["ohlcv"].astype("float32")     # float32 in memory: half the RAM
                    self._mem[symbol] = (ts, val)
                    return self._mem[symbol]
                except Exception:
                    return None
        return None

    def _put(self, symbol: str, ts: np.ndarray, val: np.ndarray) -> None:
        self._mem[symbol] = (ts, np.asarray(val, dtype="float32"))
        if self.persist:
            if self._batch:
                self._dirty.add(symbol)
            else:
                self._write(symbol)

    def _write(self, symbol: str) -> None:
        ts, val = self._mem[symbol]
        f = self._file(symbol)
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez_compressed(tmp, ts=ts, ohlcv=val)
        os.replace(tmp, f)

    def flush(self) -> None:
        """Write every symbol changed since the last flush."""
        for sym in sorted(self._dirty):
            if sym in self._mem:
                self._write(sym)
        self._dirty.clear()

    @contextmanager
    def batch(self):
        """Group many appends (a long download) into fewer disk writes; flush() saves progress."""
        self._batch += 1
        try:
            yield self
        finally:
            self._batch -= 1
            if not self._batch:
                self.flush()

    # ---- small metadata file (how far back history has been fetched) -----
    def _meta_file(self) -> Path:
        return self.dir / "_meta.json"

    def get_meta(self, key: str, default=None):
        if self._meta is None:
            self._meta = {}
            if self.persist and self._meta_file().exists():
                try:
                    self._meta = json.loads(self._meta_file().read_text())
                except Exception:
                    self._meta = {}
        return self._meta.get(key, default)

    def set_meta(self, key: str, value) -> None:
        self.get_meta(key)
        self._meta[key] = value
        if self.persist:
            tmp = self._meta_file().with_suffix(".tmp")
            tmp.write_text(json.dumps(self._meta))
            os.replace(tmp, self._meta_file())

    def session_days(self, symbols: Iterable[str]) -> np.ndarray:
        """Sorted unique UTC dates (as int days since 1970) on which any of `symbols` has bars.
        Regular-session bars (13:30-21:00 UTC) never cross UTC midnight, so this is the ET date."""
        days = set()
        for sym in symbols:
            cur = self._get(sym)
            if cur is not None and len(cur[0]):
                days.update(np.unique(cur[0] // 86_400_000_000_000).tolist())
        return np.array(sorted(days), dtype="int64")

    # ---- public ----------------------------------------------------------
    def symbols(self) -> List[str]:
        names = set(self._mem)
        if self.persist and self.dir.exists():
            names.update(p.stem for p in self.dir.glob("*.npz") if not p.stem.endswith(".tmp"))
        return sorted(names)

    def append(self, bars: pd.DataFrame) -> int:
        """Merge new bars in; rows with the same timestamp are overwritten."""
        if bars is None or len(bars) == 0:
            return 0
        n = 0
        ts_all = bars["ts"].dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").values
        vals_all = bars[list(_FIELDS)].to_numpy(dtype="float64")
        sym_all = bars["symbol"].to_numpy()
        order = np.argsort(sym_all, kind="stable")
        sym_sorted = sym_all[order]
        bounds = np.flatnonzero(np.r_[True, sym_sorted[1:] != sym_sorted[:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            idx = order[a:b]
            sym = str(sym_sorted[a])
            ts_new, val_new = ts_all[idx], vals_all[idx]
            cur = self._get(sym)
            if cur is not None and len(cur[0]):
                ts = np.concatenate([cur[0], ts_new])
                val = np.vstack([cur[1].astype("float64"), val_new])
            else:
                ts, val = ts_new, val_new
            o = np.argsort(ts, kind="stable")
            ts, val = ts[o], val[o]
            keep = np.r_[ts[1:] != ts[:-1], True]  # last occurrence wins
            ts, val = ts[keep], val[keep]
            if cur is None or len(ts) != len(cur[0]) or not np.array_equal(val.astype("float32"), cur[1]):
                self._put(sym, ts, val)
            n += len(ts_new)
        return n

    def conflicts(self, bars: pd.DataFrame, tol: float = 0.05) -> List[str]:
        """Symbols whose incoming bars disagree with cached bars at the same timestamps.

        A big mismatch means the vendor re-based the price history (a stock split),
        so the cached history for that symbol is on the wrong scale and must be re-downloaded.
        """
        out: List[str] = []
        if bars is None or len(bars) == 0:
            return out
        ts_all = bars["ts"].dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").values
        close_all = bars["close"].to_numpy(dtype="float64")
        sym_all = bars["symbol"].to_numpy()
        for sym in pd.unique(sym_all):
            cur = self._get(str(sym))
            if cur is None or len(cur[0]) == 0:
                continue
            m = sym_all == sym
            ts_new, c_new = ts_all[m], close_all[m]
            pos = np.searchsorted(cur[0], ts_new)
            ok = (pos < len(cur[0]))
            ok[ok] &= cur[0][pos[ok]] == ts_new[ok]
            if not ok.any():
                continue
            ratio = c_new[ok] / cur[1][pos[ok], 3]
            if np.nanmedian(np.abs(ratio - 1.0)) > tol:
                out.append(str(sym))
        return out

    def drop(self, symbol: str) -> None:
        self._mem.pop(symbol, None)
        self._dirty.discard(symbol)
        if self.persist:
            f = self._file(symbol)
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    np.savez_compressed(f, ts=np.zeros(0, dtype="int64"), ohlcv=np.zeros((0, 5)))

    def last_ts(self, symbol: str) -> Optional[pd.Timestamp]:
        cur = self._get(symbol)
        if cur is None or len(cur[0]) == 0:
            return None
        return pd.Timestamp(int(cur[0][-1]), tz="UTC")

    def first_ts(self, symbol: str) -> Optional[pd.Timestamp]:
        cur = self._get(symbol)
        if cur is None or len(cur[0]) == 0:
            return None
        return pd.Timestamp(int(cur[0][0]), tz="UTC")

    def load(self, symbols: Iterable[str], start: Optional[datetime] = None,
             end: Optional[datetime] = None) -> pd.DataFrame:
        lo = _to_ns(start) if start is not None else None
        hi = _to_ns(end) if end is not None else None
        ts_parts, val_parts, sym_parts = [], [], []
        for sym in symbols:
            cur = self._get(sym)
            if cur is None or len(cur[0]) == 0:
                continue
            ts, val = cur
            a = np.searchsorted(ts, lo, side="left") if lo is not None else 0
            b = np.searchsorted(ts, hi, side="left") if hi is not None else len(ts)
            if b <= a:
                continue
            ts_parts.append(ts[a:b])
            val_parts.append(val[a:b])
            sym_parts.append(np.full(b - a, sym, dtype=object))
        if not ts_parts:
            return empty_bars()
        ts = np.concatenate(ts_parts)
        val = np.vstack(val_parts)
        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(ts, utc=True),
                "symbol": np.concatenate(sym_parts),
            }
        )
        for i, col in enumerate(_FIELDS):
            df[col] = val[:, i].astype("float64")
        return df[BAR_COLUMNS]

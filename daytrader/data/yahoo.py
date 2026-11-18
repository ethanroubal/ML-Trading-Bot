"""Yahoo Finance provider (via yfinance). Free, no account needed.

Limits to know about: ~60 days of 5-minute history, one request per symbol,
and unofficial rate limits - so keep the universe modest (the 'core' list).
"""
from __future__ import annotations

import time as _time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from ..market_calendar import UTC
from .base import DataError, DataProvider, Quote, empty_bars, normalize_bars

_FIELD_NAMES = {"open", "high", "low", "close", "volume"}


def parse_yf_frame(df, tickers: List[str]) -> pd.DataFrame:
    """Flatten whatever shape yfinance.download returned into long bars."""
    if df is None or len(df) == 0:
        return empty_bars()
    frames = []
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = {str(x).lower() for x in df.columns.get_level_values(0)}
        if lvl0 & _FIELD_NAMES:  # (field, ticker) layout -> make it (ticker, field)
            df = df.swaplevel(axis=1)
        for tkr in dict.fromkeys(df.columns.get_level_values(0)):
            sub = df[tkr]
            frames.append((str(tkr), sub))
    else:
        frames.append((tickers[0], df))
    out = []
    for tkr, sub in frames:
        sub = sub.copy()
        sub.columns = [str(c).lower() for c in sub.columns]
        if not {"open", "high", "low", "close"} <= set(sub.columns):
            continue
        sub = sub.dropna(subset=["open", "high", "low", "close"], how="any")
        if len(sub) == 0:
            continue
        idx = pd.DatetimeIndex(sub.index)
        if idx.tz is None:  # yfinance returned naive exchange-local times
            idx = idx.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
        part = pd.DataFrame(
            {
                "ts": idx.tz_convert("UTC"),
                "symbol": tkr.upper(),
                "open": sub["open"].values,
                "high": sub["high"].values,
                "low": sub["low"].values,
                "close": sub["close"].values,
                "volume": sub["volume"].values if "volume" in sub.columns else 0.0,
            }
        )
        out.append(part[part["ts"].notna()])
    if not out:
        return empty_bars()
    return normalize_bars(pd.concat(out, ignore_index=True))


class YahooProvider(DataProvider):
    name = "yahoo"

    def __init__(self, cfg):
        try:
            import yfinance as yf
        except ImportError:
            raise DataError("The 'yfinance' package is not installed. Run:  pip install yfinance")
        self.yf = yf
        import logging

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # we report missing symbols ourselves
        self.chunk = max(1, int(cfg.data.yahoo_chunk))
        self.pause = float(cfg.data.request_pause)
        self.max_history_days = 7 if cfg.data.interval_min == 1 else 58
        self.verbose = True

    def _download(self, tickers: List[str], **kw):
        last_err = None
        attempts = 2 if self.fast_fail else 4
        for attempt in range(attempts):
            try:
                df = self.yf.download(
                    tickers, group_by="ticker", auto_adjust=True, prepost=False,
                    threads=True, progress=False, **kw
                )
                if df is not None and len(df) > 0:
                    return df
                last_err = "empty response"
            except Exception as exc:  # rate limits, network hiccups
                last_err = "%s: %s" % (type(exc).__name__, exc)
            if attempt < attempts - 1:
                _time.sleep(2.0 if self.fast_fail else 3.0 * (2 ** attempt))
        if self.verbose:
            print("  [yahoo] no data for %s (%s)" % (",".join(tickers[:4]) + ("..." if len(tickers) > 4 else ""), last_err))
        return None

    def fetch_bars(self, symbols: List[str], start: datetime, end: Optional[datetime],
                   interval_min: int) -> pd.DataFrame:
        now = datetime.now(tz=UTC)
        floor = now - timedelta(days=self.max_history_days)
        start = max(start.astimezone(UTC), floor)
        kw = {"interval": "%dm" % interval_min, "start": start}
        if end is not None:
            kw["end"] = end.astimezone(UTC)
        parts = []
        for i in range(0, len(symbols), self.chunk):
            batch = symbols[i:i + self.chunk]
            df = self._download(batch, **kw)
            parts.append(parse_yf_frame(df, batch))
            if i + self.chunk < len(symbols):
                _time.sleep(self.pause)
        parts = [p for p in parts if len(p)]
        if not parts:
            return empty_bars()
        return pd.concat(parts, ignore_index=True)

    def latest_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        out: Dict[str, Quote] = {}
        for i in range(0, len(symbols), self.chunk):
            batch = symbols[i:i + self.chunk]
            df = self._download(batch, interval="1m", period="1d")
            bars = parse_yf_frame(df, batch)
            if len(bars) == 0:
                continue
            last = bars.groupby("symbol").tail(1)
            for sym, px, ts in zip(last["symbol"], last["close"], last["ts"]):
                out[str(sym)] = Quote(price=float(px), ts=ts.to_pydatetime())
        return out

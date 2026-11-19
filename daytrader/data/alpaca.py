"""Alpaca Market Data provider (free API key; real-time IEX feed).

Recommended for live use: multi-symbol requests, real-time quotes with
bid/ask, and years of intraday history. Uses plain HTTPS - no SDK needed.

Docs: https://docs.alpaca.markets/us/reference/stockbars
"""
from __future__ import annotations

import time as _time
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd

from ..config import alpaca_keys
from ..market_calendar import UTC
from .base import DataError, DataProvider, Quote, empty_bars, normalize_bars

BASE = "https://data.alpaca.markets"


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class AlpacaProvider(DataProvider):
    name = "alpaca"
    max_history_days = 4200      # Alpaca's history starts in 2016 (full-market feed) or 2020 (IEX)
    fetch_window_days = 30       # long downloads are fetched (and saved) a month at a time
    # Earliest intraday bars per feed. The free IEX feed starts in 2020; the paid full-market
    # (SIP) feed in January 2016. Training must use the same feed as live trading: IEX reports
    # a few percent of full-market volume, so mixing the two would distort the volume features.
    EARLIEST = {"iex": date(2020, 1, 1), "sip": date(2016, 1, 4)}

    def __init__(self, cfg):
        keys = alpaca_keys()
        if not keys:
            raise DataError(
                "Alpaca keys not found. Put APCA_API_KEY_ID and APCA_API_SECRET_KEY in the .env file "
                "(free keys: https://alpaca.markets), or set data.provider: yahoo."
            )
        import requests

        self.feed = cfg.data.alpaca_feed
        self.earliest_date = self.EARLIEST.get(self.feed)
        # IEX quotes are IEX's own book, not the national best bid/offer: fine for a price,
        # useless as a spread (often a dollar wide on a one-cent stock)
        self.quotes_full_market = self.feed == "sip"
        self.session = requests.Session()
        self.session.headers.update(
            {"APCA-API-KEY-ID": keys[0], "APCA-API-SECRET-KEY": keys[1], "Accept": "application/json"}
        )
        self.chunk = 100
        self.verbose = True

    def _get(self, path: str, params: dict) -> dict:
        last = None
        for attempt in range(6):
            try:
                resp = self.session.get(BASE + path, params=params, timeout=30)
            except Exception as exc:
                last = "%s: %s" % (type(exc).__name__, exc)
                _time.sleep(2.0 * (attempt + 1))
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                reset = resp.headers.get("X-RateLimit-Reset")
                wait = 5.0
                try:
                    wait = max(1.0, min(65.0, float(reset) - _time.time() + 1.0))
                except (TypeError, ValueError):
                    pass
                _time.sleep(wait)
                last = "rate limited"
                continue
            if resp.status_code >= 500:
                last = "server error %d" % resp.status_code
                _time.sleep(2.0 * (attempt + 1))
                continue
            try:
                msg = resp.json().get("message", resp.text[:200])
            except Exception:
                msg = resp.text[:200]
            if resp.status_code in (401, 403):
                hint = ""
                if "SIP" in str(msg).upper() or "subscription" in str(msg).lower():
                    hint = " (the free plan only serves real-time data from the IEX feed: set data.alpaca_feed: iex)"
                raise DataError("Alpaca rejected the request [%d]: %s%s" % (resp.status_code, msg, hint))
            raise DataError("Alpaca error [%d]: %s" % (resp.status_code, msg))
        raise DataError("Alpaca request failed after retries (%s)" % last)

    def fetch_bars(self, symbols: List[str], start: datetime, end: Optional[datetime],
                   interval_min: int) -> pd.DataFrame:
        parts, n_rows = [], 0
        for i in range(0, len(symbols), self.chunk):
            batch = symbols[i:i + self.chunk]
            params = {
                "symbols": ",".join(batch),
                "timeframe": "%dMin" % interval_min,
                "start": _rfc3339(start),
                "limit": 10000,
                "adjustment": "split",
                "feed": self.feed,
                "sort": "asc",
            }
            if end is not None:
                params["end"] = _rfc3339(end)
            pages = 0
            while True:
                data = self._get("/v2/stocks/bars", params)
                for sym, bars in (data.get("bars") or {}).items():
                    if not bars:
                        continue
                    parts.append(pd.DataFrame({
                        "ts": [b["t"] for b in bars], "symbol": sym,
                        "open": [b["o"] for b in bars], "high": [b["h"] for b in bars],
                        "low": [b["l"] for b in bars], "close": [b["c"] for b in bars],
                        "volume": [b.get("v", 0) for b in bars],
                    }))
                    n_rows += len(bars)
                token = data.get("next_page_token")
                pages += 1
                if not token:
                    break
                params["page_token"] = token
                if self.verbose and not self.fast_fail and pages % 25 == 0:
                    print("  [alpaca] %d pages, %s bars so far..." % (pages, format(n_rows, ",")), flush=True)
        if not parts:
            return empty_bars()
        return normalize_bars(pd.concat(parts, ignore_index=True))

    def latest_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        out: Dict[str, Quote] = {}
        for i in range(0, len(symbols), self.chunk):
            batch = symbols[i:i + self.chunk]
            data = self._get("/v2/stocks/snapshots", {"symbols": ",".join(batch), "feed": self.feed})
            if "snapshots" in data and isinstance(data["snapshots"], dict):
                data = data["snapshots"]
            for sym, snap in data.items():
                if not isinstance(snap, dict):
                    continue
                trade = snap.get("latestTrade") or {}
                quote = snap.get("latestQuote") or {}
                mbar = snap.get("minuteBar") or {}
                price = trade.get("p") or mbar.get("c")
                bid, ask = quote.get("bp") or None, quote.get("ap") or None
                if bid and ask and ask >= bid > 0:
                    mid = 0.5 * (bid + ask)
                    # prefer the mid when the last trade is far from the current market
                    if not price or abs(price / mid - 1.0) > 0.002:
                        price = mid
                else:
                    bid = ask = None
                if not price:
                    continue
                ts = qts = None
                try:
                    ts = pd.Timestamp(trade.get("t") or quote.get("t")).to_pydatetime()
                except Exception:
                    pass
                try:
                    qts = pd.Timestamp(quote["t"]).to_pydatetime() if quote.get("t") else None
                except Exception:
                    pass
                out[sym] = Quote(price=float(price), bid=bid, ask=ask, ts=ts, quote_ts=qts)
        return out

    def raw_quotes(self, symbol: str, start: datetime, end: datetime, feed: str = "sip", limit: int = 1000) -> list:
        """The raw quote messages Alpaca returns (diagnostics)."""
        data = self._get("/v2/stocks/quotes", {"symbols": symbol, "start": _rfc3339(start), "end": _rfc3339(end),
                                               "limit": limit, "feed": feed, "sort": "asc"})
        return (data.get("quotes") or {}).get(symbol) or []

    def historical_trades(self, symbols: List[str], start: datetime, end: datetime,
                          max_pages: int = 60, feed: Optional[str] = None) -> Dict[str, List[tuple]]:
        """Every trade for each symbol in [start, end), as (time, exchange, price, size, conditions)."""
        out: Dict[str, List[tuple]] = {}
        for i in range(0, len(symbols), self.chunk):
            batch = symbols[i:i + self.chunk]
            params = {"symbols": ",".join(batch), "start": _rfc3339(start), "end": _rfc3339(end),
                      "limit": 10000, "feed": feed or self.feed, "sort": "asc"}
            for _ in range(max_pages):
                data = self._get("/v2/stocks/trades", params)
                for sym, rows in (data.get("trades") or {}).items():
                    out.setdefault(sym, []).extend(
                        (r.get("t"), r.get("x"), r.get("p"), r.get("s"), tuple(r.get("c") or ())) for r in rows or [])
                token = data.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
        return out

    def historical_quotes(self, symbols: List[str], start: datetime, end: datetime,
                          max_pages: int = 60, feed: Optional[str] = None) -> Dict[str, List[tuple]]:
        """Every quote for each symbol in [start, end), as (time, bid exchange, bid, bid size,
        ask exchange, ask, ask size) - used to measure spreads. Each full-market (feed="sip")
        message is one exchange's best bid/offer; spreads.nbbo_spreads() combines them into
        the national best. Free when more than 15 minutes old."""
        out: Dict[str, List[tuple]] = {}
        for i in range(0, len(symbols), self.chunk):
            batch = symbols[i:i + self.chunk]
            params = {"symbols": ",".join(batch), "start": _rfc3339(start), "end": _rfc3339(end),
                      "limit": 10000, "feed": feed or self.feed, "sort": "asc"}
            for _ in range(max_pages):
                data = self._get("/v2/stocks/quotes", params)
                for sym, rows in (data.get("quotes") or {}).items():
                    out.setdefault(sym, []).extend(
                        (r.get("t"), r.get("bx"), r.get("bp"), r.get("bs"), r.get("ax"), r.get("ap"), r.get("as"))
                        for r in rows or [])
                token = data.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
        return out

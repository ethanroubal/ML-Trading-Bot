"""Alpaca Market Data provider (free API key; real-time IEX feed).

Recommended for live use: multi-symbol requests, real-time quotes with
bid/ask, and years of intraday history. Uses plain HTTPS - no SDK needed.

Docs: https://docs.alpaca.markets/us/reference/stockbars
"""
from __future__ import annotations

import time as _time
from datetime import datetime
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
    max_history_days = 400

    def __init__(self, cfg):
        keys = alpaca_keys()
        if not keys:
            raise DataError(
                "Alpaca keys not found. Put APCA_API_KEY_ID and APCA_API_SECRET_KEY in the .env file "
                "(free keys: https://alpaca.markets), or set data.provider: yahoo."
            )
        import requests

        self.feed = cfg.data.alpaca_feed
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
                ts = None
                try:
                    ts = pd.Timestamp(trade.get("t") or quote.get("t")).to_pydatetime()
                except Exception:
                    pass
                out[sym] = Quote(price=float(price), bid=bid, ask=ask, ts=ts)
        return out

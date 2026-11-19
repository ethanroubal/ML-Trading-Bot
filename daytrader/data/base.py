"""Common types for market-data providers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from ..market_calendar import ET, OPEN_MIN, CLOSE_MIN

BAR_COLUMNS = ["ts", "symbol", "open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    """A provider could not deliver data (network, auth, rate limit...)."""


@dataclass
class Quote:
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    ts: Optional[datetime] = None
    quote_ts: Optional[datetime] = None   # when the bid/ask was last updated (None if unknown)


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.Series([], dtype="datetime64[ns, UTC]"),
            "symbol": pd.Series([], dtype=object),
            "open": pd.Series([], dtype=float),
            "high": pd.Series([], dtype=float),
            "low": pd.Series([], dtype=float),
            "close": pd.Series([], dtype=float),
            "volume": pd.Series([], dtype=float),
        }
    )


def to_utc_ns(ts) -> pd.Series:
    """Coerce any datetime-like series/index to tz-aware UTC with ns resolution."""
    s = pd.Series(pd.to_datetime(ts, utc=True))
    return s.dt.as_unit("ns")


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce schema, drop junk rows, keep regular-session bars only."""
    if df is None or len(df) == 0:
        return empty_bars()
    out = pd.DataFrame(
        {
            "ts": to_utc_ns(df["ts"]).values,
            "symbol": df["symbol"].astype(str).values,
        }
    )
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(pd.Series(df[col]).values, errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[(out["close"] > 0) & (out["open"] > 0)]
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    # tidy up bad ticks: make high/low bracket open/close
    out["high"] = out[["open", "high", "close"]].max(axis=1)
    out["low"] = out[["open", "low", "close"]].min(axis=1)
    # regular session only
    et = out["ts"].dt.tz_convert(ET)
    minute = et.dt.hour * 60 + et.dt.minute
    keep = (minute >= OPEN_MIN) & (minute < CLOSE_MIN) & (et.dt.dayofweek < 5)
    out = out[keep.values]
    out = out.drop_duplicates(subset=["symbol", "ts"], keep="last")
    return out.sort_values(["symbol", "ts"]).reset_index(drop=True)[BAR_COLUMNS]


def drop_incomplete(df: pd.DataFrame, interval_min: int, now: datetime) -> pd.DataFrame:
    """Remove bars that are still forming at `now` and bars off the interval grid."""
    if len(df) == 0:
        return df
    end = df["ts"] + pd.Timedelta(minutes=interval_min)
    now_ts = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
    on_grid = (df["ts"].dt.minute % interval_min == 0) & (df["ts"].dt.second == 0)
    return df[(end <= now_ts).values & on_grid.values].reset_index(drop=True)


class DataProvider:
    """Interface every data source implements."""

    name = "base"
    max_history_days = 365
    fast_fail = False     # set by the engine during market hours: give up quickly, the next bar is soon

    def fetch_bars(self, symbols: List[str], start: datetime, end: Optional[datetime],
                   interval_min: int) -> pd.DataFrame:
        """Completed + forming bars in [start, end]; UTC timestamps = bar START."""
        raise NotImplementedError

    def latest_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        raise NotImplementedError

    def now(self) -> datetime:
        from ..market_calendar import now_et

        return now_et()

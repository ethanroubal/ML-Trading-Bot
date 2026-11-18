"""US equity market calendar (NYSE/Nasdaq core session), computed locally.

Rules follow the NYSE holiday schedule: weekend holidays are observed on the
nearest weekday (except New Year's Day falling on a Saturday, which is not
observed), and the market closes at 1:00 p.m. ET on the day after
Thanksgiving, on Christmas Eve, and on July 3rd when those are regular
weekdays next to the holiday. Exceptional closures can be added in
config.yaml under `calendar:`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Iterable, List, Optional, Set

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9
    raise

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

OPEN_MIN = 9 * 60 + 30
CLOSE_MIN = 16 * 60
EARLY_CLOSE_MIN = 13 * 60


def _easter(year: int) -> date:
    # Anonymous Gregorian algorithm
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> Optional[date]:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def _holidays(year: int) -> frozenset:
    out: Set[date] = set()
    ny = date(year, 1, 1)
    if ny.weekday() == 6:
        out.add(ny + timedelta(days=1))
    elif ny.weekday() != 5:  # Saturday New Year's Day is not observed
        out.add(ny)
    out.add(_nth_weekday(year, 1, 0, 3))        # Martin Luther King Jr. Day
    out.add(_nth_weekday(year, 2, 0, 3))        # Washington's Birthday
    out.add(_easter(year) - timedelta(days=2))  # Good Friday
    out.add(_last_weekday(year, 5, 0))          # Memorial Day
    if year >= 2022:
        out.add(_observed(date(year, 6, 19)))   # Juneteenth
    out.add(_observed(date(year, 7, 4)))        # Independence Day
    out.add(_nth_weekday(year, 9, 0, 1))        # Labor Day
    out.add(_nth_weekday(year, 11, 3, 4))       # Thanksgiving
    out.add(_observed(date(year, 12, 25)))      # Christmas
    return frozenset(out)


@lru_cache(maxsize=64)
def _early_closes(year: int) -> frozenset:
    hol = _holidays(year)
    out: Set[date] = set()
    out.add(_nth_weekday(year, 11, 3, 4) + timedelta(days=1))  # day after Thanksgiving
    for d in (date(year, 12, 24), date(year, 7, 3)):
        nxt = d + timedelta(days=1)
        # early close only when both the eve and the holiday itself are weekdays
        if d.weekday() < 5 and nxt.weekday() < 5 and d not in hol:
            out.add(d)
    return frozenset(out)


@dataclass(frozen=True)
class Session:
    day: date
    open: datetime    # tz-aware, US/Eastern
    close: datetime

    @property
    def minutes(self) -> int:
        return int((self.close - self.open).total_seconds() // 60)


class MarketCalendar:
    def __init__(self, extra_holidays: Iterable[str] = (), extra_early_closes: Iterable[str] = ()):
        self._extra_h = {date.fromisoformat(str(x)) for x in extra_holidays}
        self._extra_e = {date.fromisoformat(str(x)) for x in extra_early_closes}

    def is_trading_day(self, d: date) -> bool:
        if d.weekday() >= 5:
            return False
        return d not in _holidays(d.year) and d not in self._extra_h

    def close_minute(self, d: date) -> int:
        if d in _early_closes(d.year) or d in self._extra_e:
            return EARLY_CLOSE_MIN
        return CLOSE_MIN

    def session(self, d: date) -> Optional[Session]:
        if not self.is_trading_day(d):
            return None
        cm = self.close_minute(d)
        o = datetime.combine(d, time(9, 30), tzinfo=ET)
        c = datetime.combine(d, time(cm // 60, cm % 60), tzinfo=ET)
        return Session(d, o, c)

    def bars_in_session(self, d: date, interval_min: int) -> int:
        return (self.close_minute(d) - OPEN_MIN) // interval_min

    def next_session(self, after: datetime) -> Session:
        """First session whose close is after the given moment."""
        d = after.astimezone(ET).date()
        for _ in range(15):
            s = self.session(d)
            if s is not None and s.close > after:
                return s
            d += timedelta(days=1)
        raise RuntimeError("No trading session found within two weeks of %s" % after)

    def previous_trading_day(self, d: date) -> date:
        d -= timedelta(days=1)
        while not self.is_trading_day(d):
            d -= timedelta(days=1)
        return d

    def trading_days_between(self, start: date, end: date) -> List[date]:
        out = []
        d = start
        while d <= end:
            if self.is_trading_day(d):
                out.append(d)
            d += timedelta(days=1)
        return out


def now_et() -> datetime:
    return datetime.now(tz=ET)

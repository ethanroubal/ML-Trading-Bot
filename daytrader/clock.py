"""Clocks: the real one for live trading, a simulated one for replay/demo."""
from __future__ import annotations

import time as _time
from datetime import datetime, timedelta

from .market_calendar import ET


class RealClock:
    simulated = False

    def now(self) -> datetime:
        return datetime.now(tz=ET)

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            _time.sleep(seconds)


class SimClock:
    """Virtual time. `speed` = simulated seconds per real second (0 = as fast as possible)."""

    simulated = True

    def __init__(self, start: datetime, speed: float = 0.0):
        self._t = start.astimezone(ET)
        self.speed = float(speed)

    def now(self) -> datetime:
        return self._t

    def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self._t = self._t + timedelta(seconds=seconds)
        if self.speed > 0:
            _time.sleep(min(seconds / self.speed, 2.0))

    def jump(self, seconds: float) -> None:
        """Advance without pausing (used to skip the hours when the market is closed)."""
        if seconds > 0:
            self._t = self._t + timedelta(seconds=seconds)

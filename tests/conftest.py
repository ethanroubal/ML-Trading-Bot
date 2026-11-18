import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daytrader.config import Config  # noqa: E402
from daytrader.market_calendar import MarketCalendar  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.root = str(tmp_path)
    c.model.max_iter = 80          # keep tests quick
    c.model.seeds = 1
    return c


@pytest.fixture(scope="session")
def cal():
    return MarketCalendar()

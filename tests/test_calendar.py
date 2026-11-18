from datetime import date, datetime

from daytrader.market_calendar import ET, MarketCalendar


def test_2026_holidays_match_nyse_schedule(cal):
    closed = ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
              "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]
    for d in closed:
        assert not cal.is_trading_day(date.fromisoformat(d)), d
    for d in ["2026-01-02", "2026-07-02", "2026-11-16", "2026-11-27", "2026-12-24"]:
        assert cal.is_trading_day(date.fromisoformat(d)), d
    n_closed = sum(1 for d in cal.trading_days_between(date(2026, 1, 1), date(2026, 12, 31)))
    assert n_closed == 251          # 261 weekdays - 10 holidays


def test_early_closes(cal):
    assert cal.close_minute(date(2026, 11, 27)) == 13 * 60      # day after Thanksgiving
    assert cal.close_minute(date(2026, 12, 24)) == 13 * 60      # Christmas Eve (Thursday)
    assert cal.close_minute(date(2026, 7, 2)) == 16 * 60        # July 3 is the holiday itself in 2026
    assert cal.close_minute(date(2025, 7, 3)) == 13 * 60
    assert cal.close_minute(date(2028, 7, 3)) == 13 * 60
    assert cal.bars_in_session(date(2026, 11, 27), 5) == 42
    assert cal.bars_in_session(date(2026, 11, 16), 5) == 78


def test_2027_and_2028_edge_cases(cal):
    assert not cal.is_trading_day(date(2027, 12, 24))           # Christmas observed on Friday
    assert cal.close_minute(date(2027, 11, 26)) == 13 * 60
    assert not cal.is_trading_day(date(2027, 7, 5))             # July 4 (Sunday) observed Monday
    assert not cal.is_trading_day(date(2027, 6, 18))            # Juneteenth (Saturday) observed Friday
    assert cal.is_trading_day(date(2027, 12, 31))               # New Year's 2028 on Saturday: not observed
    assert cal.is_trading_day(date(2028, 1, 3))


def test_next_session_skips_weekends_and_respects_time(cal):
    fri_evening = datetime(2026, 11, 13, 18, 0, tzinfo=ET)
    s = cal.next_session(fri_evening)
    assert s.day == date(2026, 11, 16) and s.open.hour == 9 and s.open.minute == 30
    during = datetime(2026, 11, 16, 11, 0, tzinfo=ET)
    assert cal.next_session(during).day == date(2026, 11, 16)
    at_close = datetime(2026, 11, 16, 16, 0, tzinfo=ET)
    assert cal.next_session(at_close).day == date(2026, 11, 17)


def test_extra_closures_from_config():
    c = MarketCalendar(extra_holidays=["2026-11-18"], extra_early_closes=["2026-11-19"])
    assert not c.is_trading_day(date(2026, 11, 18))
    assert c.close_minute(date(2026, 11, 19)) == 13 * 60

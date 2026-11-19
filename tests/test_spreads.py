from datetime import date, datetime, timedelta

import numpy as np
import pytest

from daytrader import costs as costmod
from daytrader.config import Config
from daytrader.data.base import Quote
from daytrader.market_calendar import ET
from daytrader.spreads import SpreadBook, bootstrap, bucket_of, spread_of
from daytrader.strategy import make_signals

DAYS = [date(2026, 11, d) for d in (9, 10, 11, 12, 13)]


def _book(tmp_path=None, days=DAYS, bp_by_sym=None):
    book = SpreadBook(tmp_path / "s.json" if tmp_path else None, lookback_days=20, min_days=3)
    bp_by_sym = bp_by_sym or {"AAA": 2.0, "BBB": 10.0}
    for d in days:
        for sym, bp in bp_by_sym.items():
            mid = 100.0
            for minute in (600, 700, 800, 900):
                half = mid * bp / 2e4
                book.add(sym, d, minute, mid - half, mid + half)
    book.close_days()
    return book


def test_spread_of_rejects_bad_quotes():
    assert spread_of(100.0, 100.02)[0] == pytest.approx(2.0, rel=1e-3)
    assert spread_of(100.0, 99.0) is None           # crossed
    assert spread_of(0, 1) is None
    assert spread_of(None, 1) is None
    assert spread_of(100.0, 103.0) is None          # 300 bp: junk


def test_buckets():
    assert int(bucket_of(570)) == 0 and int(bucket_of(584)) == 0 and int(bucket_of(585)) == 1
    assert int(bucket_of(959)) == 7
    assert list(bucket_of([600, 720])) == [2, 4]


def test_estimates_need_min_days_and_fall_back(tmp_path):
    book = _book(tmp_path, days=DAYS[:2])
    assert np.isnan(book.spread_bps(["AAA"], 700, 100.0)).all()       # only 2 sessions < 3 needed
    book = _book(tmp_path)
    est = book.spread_bps(["AAA", "BBB", "ZZZ"], 700, 100.0)
    assert est[0] == pytest.approx(2.0, rel=1e-2) and est[1] == pytest.approx(10.0, rel=1e-2)
    assert np.isnan(est[2])
    assert np.isnan(book.spread_bps(["AAA"], 575, 100.0)).all()        # no data in the opening bucket


def test_dollar_spread_applies_at_lower_prices(tmp_path):
    book = _book(tmp_path)
    # 2 bp at $100 = 2 cents; at $50 the same 2 cents is 4 bp - use the wider reading
    assert float(book.spread_bps("AAA", 700, 50.0)) == pytest.approx(4.0, rel=1e-2)
    assert float(book.spread_bps("AAA", 700, 200.0)) == pytest.approx(2.0, rel=1e-2)


def test_vectorised_lookup_matches_rows(tmp_path):
    book = _book(tmp_path)
    syms = np.array(["AAA", "BBB", "AAA", "QQQ"], dtype=object)
    est = book.spread_bps(syms, np.array([700, 700, 900, 700]), np.array([100.0, 100.0, 100.0, 100.0]))
    assert est[0] == pytest.approx(2.0, rel=1e-2) and est[2] == pytest.approx(2.0, rel=1e-2)
    assert np.isnan(est[3])
    lst = ["AAA", "BBB"]
    a = book.spread_bps(lst, 700, [100.0, 100.0])
    b = book.spread_bps(lst, 700, [100.0, 100.0])      # cached index path
    assert np.allclose(a, b)


def test_save_load_roundtrip_and_open_samples(tmp_path):
    book = _book(tmp_path)
    book.add("AAA", date(2026, 11, 16), 700, 99.99, 100.01)
    book.save()
    again = SpreadBook.load(tmp_path / "s.json", 20, 3)
    assert again.days_measured() == 6
    assert float(again.spread_bps("AAA", 700, 100.0)) == pytest.approx(2.0, rel=1e-2)
    again.close_days(before=date(2026, 11, 16))
    assert "AAA" in again.open                          # today's samples stay raw
    again.close_days()
    assert not again.open


def test_costs_use_measured_spreads_with_floor_and_fallback(tmp_path):
    cfg = Config()
    book = _book(tmp_path, bp_by_sym={"AAA": 0.1, "BBB": 10.0})
    costmod.attach_book(cfg.costs, book)
    c = cfg.costs
    assumed = 2 * (costmod.assumed_half_spread_bps(500.0, c) + c.slippage_bps)
    rt = costmod.round_trip_cost_bps(np.array([500.0, 100.0, 100.0]), c,
                                     symbols=["AAA", "BBB", "NEW"], minute=700)
    # AAA: 0.1 bp measured at $100 is 0.1 cents; at $500 the half-cent tick floor (0.1 bp) binds
    floor = max(c.measured_min_half_spread_bps, c.tick_half_spread / 500.0 * 1e4)
    assert rt[0] == pytest.approx(2 * (floor + c.slippage_bps), rel=1e-6)
    assert rt[1] == pytest.approx(2 * (5.0 + c.slippage_bps), rel=1e-2)
    assert rt[2] == pytest.approx(float(2 * (costmod.assumed_half_spread_bps(100.0, c) + c.slippage_bps)))
    assert rt[0] < assumed
    c.spread_source = "assumed"
    rt2 = costmod.round_trip_cost_bps(np.array([500.0]), c, symbols=["AAA"], minute=700)
    assert rt2[0] == pytest.approx(float(assumed))


def test_signals_and_backtest_fills_use_measured_costs(tmp_path):
    from daytrader.broker import Order, PaperBroker

    cfg = Config()
    costmod.attach_book(cfg.costs, _book(tmp_path, bp_by_sym={"AAA": 0.5}))
    sig = make_signals(["AAA"], np.array([100.0]), np.array([0.1]), np.array([0.003]), np.array([True]), cfg,
                       minute=700)
    expect = 2 * (max(0.25, cfg.costs.measured_min_half_spread_bps, 0.5) + cfg.costs.slippage_bps)
    assert sig.cost_bps[0] == pytest.approx(expect, rel=1e-2)
    br = PaperBroker(cfg, None, 100000.0)
    od = Order("AAA", 1, 100, 100.0, 0.01, 0.02, 1.0, 1.0, 30.0)
    pos = br.open(od, "2026-11-18 11:40", price=100.0)
    one_way = expect / 2
    assert pos.entry_price == pytest.approx(100.0 * (1 + one_way / 1e4), rel=1e-6)


class FakeQuotes:
    name = "alpaca"

    def __init__(self):
        self.calls = 0

    def historical_quotes(self, symbols, start, end, feed=None):
        self.calls += 1
        self.feed = feed
        return {s: [(99.99, 100.01), (99.98, 100.02), (99.99, 100.01)] for s in symbols}


def test_bootstrap_fills_every_bucket():
    book = SpreadBook(None, min_days=3)
    fake = FakeQuotes()
    n = bootstrap(book, fake, ["AAA", "BBB"], DAYS[:3], log=lambda m: None, feed="sip")
    assert fake.calls == 3 * 8 and n == 3 * 8 * 2 and fake.feed == "sip"
    assert book.sampled == {str(d) for d in DAYS[:3]}
    have, total = book.coverage(["AAA", "BBB", "CCC"])
    assert (have, total) == (2, 3)
    assert float(book.spread_bps("AAA", 577, 100.0)) == pytest.approx(2.0, rel=1e-2)


def test_bootstrap_skips_quotes_younger_than_the_delay():
    book = SpreadBook(None)
    fake = FakeQuotes()
    day = DAYS[-1]
    now = datetime(day.year, day.month, day.day, 16, 1, 30, tzinfo=ET)     # 90 s after the close
    bootstrap(book, fake, ["AAA"], [day], log=lambda m: None, now=now)
    assert fake.calls == 8 and str(day) in book.sampled                    # 15:45 window is 16 min old
    book2, fake2 = SpreadBook(None), FakeQuotes()
    bootstrap(book2, fake2, ["AAA"], [day], log=lambda m: None, now=now - timedelta(minutes=5))
    assert fake2.calls == 7 and str(day) not in book2.sampled              # last window skipped: not done


def test_old_iex_tables_and_other_feeds_are_discarded(tmp_path):
    import json
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"version": 1, "done": {"AAA": {"4": {"2026-11-12": [30.0, 1.5, 1]}}}, "open": {}}))
    assert not SpreadBook.load(path).done
    book = _book(tmp_path)
    book.save()
    assert SpreadBook.load(tmp_path / "s.json", feed="sip").done
    assert not SpreadBook.load(tmp_path / "s.json", feed="iex").done


def test_sim_broker_ignores_quotes_that_are_not_full_market():
    from daytrader.broker import Order, PaperBroker

    cfg = Config()
    br = PaperBroker(cfg, None, 100000.0)
    br.quote_spreads = False
    od = Order("AAA", 1, 10, 100.0, 0.01, 0.02, 1.0, 1.0, 30.0)
    pos = br.open(od, "2026-11-18 11:40", price=100.0, bid=99.0, ask=101.0)      # a dollar-wide IEX quote
    assumed = float(costmod.one_way_cost_bps(100.0, cfg.costs))
    assert pos.entry_price == pytest.approx(100.0 * (1 + assumed / 1e4))


def test_nbbo_is_rebuilt_from_per_exchange_quotes():
    from daytrader.spreads import nbbo_spreads

    T = "2026-11-18T17:00:0%dZ"
    rows = [
        (T % 0, "N", 100.00, 5, "N", 100.04, 5),    # NYSE 4 cents wide
        (T % 1, "Q", 100.01, 3, "Q", 100.05, 3),    # Nasdaq: best bid 100.01
        (T % 2, "Z", 99.99, 2, "Z", 100.02, 2),     # Cboe: best ask 100.02
        (T % 5, "Q", 100.01, 3, "Q", 100.05, 3),    # after warm-up: NBBO 100.01 x 100.02
        (T % 6, "Z", 0, 0, "Z", 0, 0),              # Cboe withdraws: NBBO 100.01 x 100.04
    ]
    sps = nbbo_spreads(rows, "2026-11-18T17:00:04")
    assert [round(u, 4) for _, u in sps] == [0.01, 0.03]
    # each per-exchange message alone would have said 3-4 cents
    assert min(spread_of(r[2], r[5])[1] for r in rows[:4]) == pytest.approx(0.03)


def test_quotes_that_are_already_national_best_are_used_as_is():
    from daytrader.spreads import nbbo_spreads

    rows = [("2026-11-18T17:00:05Z", "Q", 100.00, 1, "N", 100.01, 1),
            ("2026-11-18T17:00:06Z", "Z", 99.99, 1, "Q", 100.02, 1)]
    assert [round(u, 4) for _, u in nbbo_spreads(rows, "2026-11-18T17:00:04")] == [0.01, 0.03]


def test_effective_spread_from_trades_against_the_mid():
    from daytrader.spreads import effective_spreads

    T = "2026-11-18T17:00:0%dZ"
    quotes = [(T % 0, "Q", 100.00, 1, "N", 100.10, 1)] * 3     # national best 100.00 x 100.10 (mid 100.05)
    trades = [(T % 5, "Q", 100.10, 100, ("@",)),            # at the ask: paid the full 10 cents
              (T % 5, "N", 100.07, 100, ("@", "I")),        # improved: 2 x 2 cents = 4 cents
              (T % 6, "D", 100.05, 100, ("@",)),            # off-exchange midpoint: ignored
              (T % 6, "Q", 100.00, 100, ("W",)),            # average-price print: ignored
              (T % 1, "Q", 100.10, 100, ("@",))]            # during warm-up: ignored
    effs = effective_spreads(quotes, trades, "2026-11-18T17:00:04")
    assert [round(u, 4) for _, u in effs] == [0.10, 0.04]


def test_bootstrap_stores_effective_spread_floored_at_half_quoted():
    class Fake:
        def historical_quotes(self, symbols, start, end, feed=None):
            return {s: [(100.00, 100.10)] for s in symbols}         # 10 cents quoted

        def historical_trades(self, symbols, start, end, feed=None):
            px = {"AAA": [100.06, 100.04, 100.06],                  # 2 cents effective -> floor 5
                  "BBB": [100.10, 100.00, 100.08]}                  # 10, 10, 6 cents -> mean 8.67
            return {s: [(None, "Q", p, 100, ()) for p in px[s]] for s in symbols}

    book = SpreadBook(None, min_days=3)
    bootstrap(book, Fake(), ["AAA", "BBB"], DAYS[:3], log=lambda m: None)
    a, b = book.spread_bps(["AAA", "BBB"], 700, 100.05)
    assert a == pytest.approx(5.0, rel=0.01) and b == pytest.approx(8.67, rel=0.01)
    assert book.improvement() == pytest.approx((0.5 + 0.867) / 2, rel=0.01)


def test_bootstrap_without_trades_uses_quoted_spread():
    book = SpreadBook(None, min_days=3)
    bootstrap(book, FakeQuotes(), ["AAA"], DAYS[:3], log=lambda m: None)
    assert float(book.spread_bps("AAA", 700, 100.0)) == pytest.approx(2.0, rel=0.02)

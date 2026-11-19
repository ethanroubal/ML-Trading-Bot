"""Run the real live engine on a simulated clock and compare it with the backtester."""
from datetime import date, datetime, time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from daytrader.backtest import run_backtest  # noqa: E402
from daytrader.clock import SimClock  # noqa: E402
from daytrader.data.store import BarStore  # noqa: E402
from daytrader.data.synthetic import ReplayProvider, generate_bars  # noqa: E402
from daytrader.engine import Engine  # noqa: E402
from daytrader.features import compute_features  # noqa: E402
from daytrader.market_calendar import ET  # noqa: E402
from daytrader.panel import build_panel  # noqa: E402
from daytrader.report import load_state  # noqa: E402

LAST = date(2026, 11, 13)


def _engine(cfg, tmp_path, bars, start, trade=True, hh=9, mm=20):
    clock = SimClock(datetime.combine(start, time(hh, mm), tzinfo=ET))
    provider = ReplayProvider(bars, clock, 5)
    symbols = sorted(bars["symbol"].unique())
    return Engine(cfg, provider, symbols, clock=clock, trade=trade, capital=50_000,
                  store=BarStore(tmp_path / "cache", 5, persist=False), state_dir=tmp_path / "state",
                  model_path=tmp_path / "model.joblib", report_dir=tmp_path / "reports",
                  fresh_account=True, allow_retrain=False, force_train=True, label="test")


@pytest.fixture(scope="module")
def bars():
    return generate_bars(n_symbols=24, n_days=40, signal=3.0, seed=21, end_day=LAST)


def test_live_session_end_to_end_and_parity_with_backtest(cfg, cal, tmp_path, bars, capsys):
    eng = _engine(cfg, tmp_path, bars, LAST)
    eng.run(max_sessions=1)
    out = capsys.readouterr().out
    assert "Buy candidates" in out and "Day summary" in out

    state = load_state(tmp_path / "state")
    daily, trades = state["daily"], state["trades"]
    assert len(daily) == 1 and str(daily["date"].iloc[0]) == str(LAST)
    assert len(trades) > 10
    assert not state["portfolio"]["positions"]                       # flat overnight
    assert state["portfolio"]["cash"] == pytest.approx(float(daily["end_equity"].iloc[0]), abs=0.01)
    assert float(daily["pnl"].iloc[0]) == pytest.approx(float(trades["pnl"].sum()), abs=0.5)
    last_exit = pd.to_datetime(trades["exit_time"]).max()
    assert (last_exit.hour, last_exit.minute) <= (15, 56)
    first_entry = pd.to_datetime(trades["entry_time"]).min()
    assert (first_entry.hour, first_entry.minute) >= (9, 40)
    assert (tmp_path / "reports" / "dashboard.html").exists()
    html = (tmp_path / "reports" / "dashboard.html").read_text()
    assert "__DATA__" not in html and '"daily": [{' in html

    # The backtester, given the same model and data, should take the same trades.
    panel = build_panel(bars, 5, cal)
    feats = compute_features(panel, cfg)
    d = len(panel.dates) - 1
    rows = []
    for t in np.flatnonzero(panel.day == d):
        ez = eng.model.expected_z(feats.row(t))
        rows += [(t, s, d, ez[s]) for s in np.flatnonzero(feats.usable[t])]
    bt = run_backtest(panel, feats, pd.DataFrame(rows, columns=["t", "s", "day", "exp_z"]), cfg, capital=50_000)
    key = lambda df: set(zip(df["symbol"], df["side"], pd.to_datetime(df["entry_time"]).dt.strftime("%H:%M")))
    live, sim = key(trades), key(bt.trades)
    assert len(live & sim) >= 0.85 * max(len(live), len(sim))
    assert float(trades["pnl"].sum()) == pytest.approx(float(bt.trades["pnl"].sum()), abs=0.004 * 50_000)


def test_advise_only_mode_never_opens_an_account(cfg, tmp_path, bars, capsys):
    eng = _engine(cfg, tmp_path, bars, LAST, trade=False, hh=15, mm=11)   # join late in the day
    eng.run(max_sessions=1)
    out = capsys.readouterr().out
    assert "Joined mid-session" in out and "recommendations only" in out
    assert eng.broker is None and not (tmp_path / "state" / "portfolio.json").exists()
    assert (tmp_path / "state" / "signals.csv").exists()


def test_restart_resumes_account_and_closes_leftover_positions(cfg, tmp_path, bars, capsys):
    from daytrader.broker import Order, PaperBroker

    prev = date(2026, 11, 12)
    br = PaperBroker(cfg, tmp_path / "state", 50_000)
    br.start_day(str(prev))
    br.open(Order("SYN03", 1, 10, 50.0, 0.02, 0.0, 5.0, 2.0, 40.0), "2026-11-12 15:00:00")
    br.save()
    eng = _engine(cfg, tmp_path, bars, LAST, hh=15, mm=41)
    eng.fresh_account = False
    eng.run(max_sessions=1)
    out = capsys.readouterr().out
    assert "Paper account resumed" in out and "left over" in out
    trades = load_state(tmp_path / "state")["trades"]
    assert "carryover" in set(trades["reason"])
    assert not eng.broker.positions and eng.broker.day == str(LAST)


def _fake_yfinance(bars, clock, dead=("DEAD",)):
    """A stand-in for the yfinance module that returns frames shaped like the real library's."""
    import types

    wide = bars.pivot(index="ts", columns="symbol")
    calls = []

    def download(tickers, start=None, end=None, period=None, interval="1d", group_by="column", **kw):
        tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        calls.append({"tickers": tickers, "interval": interval, "start": start, "period": period, **kw})
        assert group_by == "ticker" and kw.get("auto_adjust") is True and kw.get("prepost") is False
        now = pd.Timestamp(clock.now()).tz_convert("UTC")
        if interval == "1m":                                    # quotes: last traded price, minute bars
            lo = now - pd.Timedelta(minutes=30)
        else:
            assert start is not None and start.tzinfo is not None   # the adapter must pass aware datetimes
            lo = pd.Timestamp(start).tz_convert("UTC")
        idx = wide.index[(wide.index >= lo) & (wide.index <= now)]     # includes the still-forming bar
        frames = {}
        for t in tickers:
            if t in dead or t not in wide["close"].columns:
                frames[t] = pd.DataFrame(np.nan, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])
            else:
                frames[t] = pd.DataFrame({"Open": wide["open"][t].reindex(idx), "High": wide["high"][t].reindex(idx),
                                          "Low": wide["low"][t].reindex(idx), "Close": wide["close"][t].reindex(idx),
                                          "Volume": wide["volume"][t].reindex(idx)})
        out = pd.concat(frames, axis=1, names=["Ticker", "Price"])
        out.index = out.index.tz_convert("America/New_York")
        out.index.name = "Datetime"
        return out

    mod = types.ModuleType("yfinance")
    mod.download = download
    mod.calls = calls
    return mod


def test_yahoo_adapter_drives_a_full_session(cfg, tmp_path, bars, monkeypatch, capsys):
    """Same engine, but fed through the Yahoo adapter with a library stand-in (no network)."""
    import sys
    import time as _time

    from daytrader.data.yahoo import YahooProvider

    clock = SimClock(datetime.combine(LAST, time(9, 20), tzinfo=ET))
    fake = _fake_yfinance(bars, clock)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    # the adapter clamps history to the last 58 days of the REAL calendar; the fixture ends in the recent past
    cfg.data.request_pause = 0.0
    provider = YahooProvider(cfg)
    provider.max_history_days = 3650
    symbols = sorted(bars["symbol"].unique()) + ["DEAD"]
    eng = Engine(cfg, provider, symbols, clock=clock, trade=True, capital=40_000,
                 store=BarStore(tmp_path / "cache", 5, persist=False), state_dir=tmp_path / "state",
                 model_path=tmp_path / "model.joblib", report_dir=tmp_path / "reports",
                 fresh_account=True, allow_retrain=False, force_train=True)
    eng.run(max_sessions=1)
    out = capsys.readouterr().out
    assert "no data for 1 symbol(s), skipping them: DEAD" in out
    assert "DEAD" not in eng.active and len(eng.active) == len(symbols) - 1
    assert "%d/%d symbols fresh" % (len(eng.active), len(eng.active)) in out
    state = load_state(tmp_path / "state")
    assert len(state["trades"]) > 10 and len(state["daily"]) == 1
    assert not state["portfolio"]["positions"]
    # no forming bar ever reached the cache: every stored bar had finished when it was stored
    last = max(eng.store.last_ts(s) for s in eng.active).tz_convert(ET)
    assert (last.hour, last.minute) == (15, 55)
    assert {c["interval"] for c in fake.calls} == {"5m", "1m"}
    assert all(len(c["tickers"]) <= cfg.data.yahoo_chunk for c in fake.calls)
    assert "DEAD" not in {t for c in fake.calls[-50:] for t in c["tickers"]}    # stopped asking for it


def test_switching_data_source_uses_its_own_cache_and_retrains(cfg, tmp_path, bars, capsys):
    """Moving from Yahoo to Alpaca must not mix the two caches, and must retrain the model."""
    cfg.root = str(tmp_path)
    clock = SimClock(datetime.combine(LAST, time(18, 0), tzinfo=ET))
    symbols = sorted(bars["symbol"].unique())

    def engine(name):
        prov = ReplayProvider(bars, clock, 5)
        prov.name = name
        return Engine(cfg, prov, symbols, clock=clock, trade=False, model_path=tmp_path / "model.joblib")

    yahoo = engine("yahoo")
    yahoo.sync_history(verbose=False)
    yahoo.ensure_model()
    assert yahoo.model.meta["provider"] == "yahoo"
    assert (tmp_path / "data_cache" / "5m").exists()

    alpaca = engine("alpaca")
    assert alpaca.store.dir == tmp_path / "data_cache" / "alpaca" / "5m"
    assert alpaca.store.last_ts(symbols[0]) is None            # starts empty: full download, no Yahoo bars
    alpaca.sync_history(verbose=False)
    capsys.readouterr()
    alpaca.ensure_model()
    assert "data source changed from yahoo to alpaca" in capsys.readouterr().out
    assert alpaca.model.meta["provider"] == "alpaca"

    again = engine("alpaca")                                     # same source next time: no needless retrain
    again.sync_history(verbose=False)
    capsys.readouterr()
    again.ensure_model()
    assert "Training model" not in capsys.readouterr().out


class QuotingReplay(ReplayProvider):
    """Replay with a 2-cent bid/ask on every quote and a quote history to bootstrap from."""

    def latest_quotes(self, symbols):
        from daytrader.data.base import Quote
        out = {}
        for s, q in super().latest_quotes(symbols).items():
            out[s] = Quote(q.price, q.price - 0.01, q.price + 0.01, q.ts, quote_ts=self.clock.now())
        return out

    def historical_quotes(self, symbols, start, end, feed=None):
        return {s: [(99.99, 100.01)] for s in symbols}

    def historical_trades(self, symbols, start, end, feed=None):
        # trades at the ask and bid: they pay the whole quoted spread
        return {s: [(None, "Q", 100.01, 100, ("@",)), (None, "N", 99.99, 100, ()), (None, "Q", 100.01, 100, ())]
                for s in symbols}


class IexOnlyReplay(QuotingReplay):
    """Live quotes are a single exchange's (a dollar wide); the history is junk too."""
    quotes_full_market = False

    def historical_quotes(self, symbols, start, end, feed=None):
        return {s: [(99.0, 101.0)] for s in symbols}

    def historical_trades(self, symbols, start, end, feed=None):
        return {s: [(None, "Q", 101.0, 100, ()), (None, "Q", 99.0, 100, ()), (None, "Q", 101.0, 100, ())]
                for s in symbols}


def test_engine_measures_spreads_and_writes_scorecard(cfg, tmp_path, bars, capsys):
    clock = SimClock(datetime.combine(LAST, time(9, 20), tzinfo=ET))
    provider = QuotingReplay(bars, clock, 5)
    symbols = sorted(bars["symbol"].unique())
    eng = Engine(cfg, provider, symbols, clock=clock, trade=True, capital=50_000,
                 store=BarStore(tmp_path / "cache", 5, persist=False), state_dir=tmp_path / "state",
                 model_path=tmp_path / "model.joblib", report_dir=tmp_path / "reports",
                 fresh_account=True, allow_retrain=False, force_train=True, label="test")
    eng.run(max_sessions=1)
    out = capsys.readouterr().out
    assert "Costs: measured for 24/24 symbols" in out and "paid 100% of the quoted spread" in out
    assert "Scorecard %s" % LAST in out and "rank IC" in out
    assert eng.spreads.days_measured() == cfg.costs.spread_bootstrap_days + 1     # history + today after the close
    assert eng.spreads.improvement() == pytest.approx(1.0)                        # trades paid the full spread
    assert (cfg.bar_cache_dir("replay") / "spreads.json").exists()
    preds = pd.read_csv(tmp_path / "state" / "predictions" / ("%s.csv" % LAST))
    assert preds["symbol"].nunique() == 24 and preds["time"].nunique() > 60
    assert (tmp_path / "state" / "scorecard.csv").exists()
    sig = pd.read_csv(tmp_path / "state" / "signals.csv")
    # every quote was 2 cents wide (2 bp at $100 in the quote history): half spread = max(1 bp, 1 cent)
    px = sig["price"].astype(float)
    expect = 2 * (np.maximum(1.0, 100.0 / px) + cfg.costs.slippage_bps)
    assert np.allclose(sig["cost_bps"], expect, atol=0.11)


def test_implausible_spreads_fall_back_and_iex_quotes_are_not_used(cfg, tmp_path, bars, capsys):
    from daytrader import costs as costmod

    clock = SimClock(datetime.combine(LAST, time(9, 20), tzinfo=ET))
    provider = IexOnlyReplay(bars, clock, 5)
    eng = Engine(cfg, provider, sorted(bars["symbol"].unique()), clock=clock, trade=True, capital=50_000,
                 store=BarStore(tmp_path / "cache", 5, persist=False), state_dir=tmp_path / "state",
                 model_path=tmp_path / "model.joblib", report_dir=tmp_path / "reports",
                 fresh_account=True, allow_retrain=False, force_train=True, label="test")
    eng.startup()
    out = capsys.readouterr().out
    assert "the quote data is broken" in out
    assert costmod.book_of(cfg.costs) is None              # assumed model in use
    assert eng.broker.quote_spreads is False                # no fills at IEX bid/ask

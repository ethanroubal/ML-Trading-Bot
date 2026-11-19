"""Command-line entry points, run against a temporary project folder."""
from datetime import date, datetime, time

import pytest

pytest.importorskip("sklearn")

import daytrader.cli as cli  # noqa: E402
from daytrader.clock import SimClock  # noqa: E402
from daytrader.data.base import normalize_bars  # noqa: E402
from daytrader.data.store import BarStore  # noqa: E402
from daytrader.data.synthetic import ReplayProvider, generate_bars  # noqa: E402
from daytrader.engine import Engine  # noqa: E402
from daytrader.market_calendar import ET, MarketCalendar, now_et  # noqa: E402


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / "config.yaml").write_text("model:\n  max_iter: 60\n  seeds: 1\n")
    for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def _fake_live_engine(monkeypatch, project, n_days=40):
    """Stand in for the real data source with a replay of synthetic history that ends 'yesterday'."""
    cal = MarketCalendar()
    end = cal.previous_trading_day(now_et().date())
    bars = normalize_bars(generate_bars(n_symbols=20, n_days=n_days, signal=3.0, seed=4, end_day=end))

    def make(cfg, trade, capital=None):
        clock = SimClock(datetime.combine(end, time(18, 0), tzinfo=ET))
        return Engine(cfg, ReplayProvider(bars, clock, 5), sorted(bars["symbol"].unique()), clock=clock,
                      trade=trade, capital=capital)

    monkeypatch.setattr(cli, "_engine", make)
    return end


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "backtest" in capsys.readouterr().out


def test_demo_runs_offline_and_writes_dashboard(project, capsys):
    assert cli.main(["demo", "--speed", "0", "--days", "1", "--capital", "25000"]) == 0
    out = capsys.readouterr().out
    assert "DEMO" in out and "Day summary" in out and "$25,000.00" in out
    assert (project / "demo" / "dashboard.html").exists()
    assert (project / "demo" / "state" / "daily.csv").read_text().count("\n") == 2
    assert not (project / "state").exists()                  # the demo never touches the real account


def test_backtest_command(project, monkeypatch, capsys):
    _fake_live_engine(monkeypatch, project)
    assert cli.main(["backtest"]) == 0
    out = capsys.readouterr().out
    assert "Walk-forward backtest" in out and "Out-of-sample results" in out and "SMALL sample" in out
    for name in ("backtest.html", "backtest_daily.csv", "backtest_trades.csv"):
        assert (project / "reports" / name).exists()
    assert (project / "models" / "model.joblib").exists()


def test_replay_trains_only_on_earlier_data(project, monkeypatch, capsys):
    end = _fake_live_engine(monkeypatch, project)
    assert cli.main(["replay", "--days", "2", "--speed", "0"]) == 0
    out = capsys.readouterr().out
    assert out.count("Day summary") == 2
    first_replayed = MarketCalendar().previous_trading_day(end)
    trained_through = MarketCalendar().previous_trading_day(first_replayed)
    assert "-> %s" % trained_through in out                  # model saw nothing from the replayed sessions
    assert (project / "replay" / "state" / "trades.csv").exists()
    assert not (project / "state" / "portfolio.json").exists()


def test_replay_needs_enough_history(project, monkeypatch, capsys):
    _fake_live_engine(monkeypatch, project, n_days=20)
    assert cli.main(["replay", "--days", "3", "--speed", "0"]) == 1
    assert "Not enough cached history" in capsys.readouterr().out


def test_status_report_and_reset(project, capsys):
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "Data provider: yahoo" in out and "Paper account: none yet" in out
    assert cli.main(["report"]) == 0
    assert "No paper account recorded yet" in capsys.readouterr().out
    assert cli.main(["reset", "--capital", "5000"]) == 0
    assert cli.main(["reset", "--capital", "7500"]) == 0     # archives the first account
    out = capsys.readouterr().out
    assert "archived" in out and "$7,500.00" in out
    assert len(list(project.glob("state_archive_*"))) == 1
    assert cli.main(["report"]) == 0
    assert (project / "reports" / "dashboard.html").exists()
    assert cli.main(["status"]) == 0
    assert "equity $7,500.00" in capsys.readouterr().out


def test_bad_settings_give_a_clear_error(project, capsys):
    (project / "config.yaml").write_text("data:\n  intervl_min: 5\n")
    assert cli.main(["status"]) == 1
    assert "intervl_min" in capsys.readouterr().err
    (project / "config.yaml").write_text("")
    assert cli.main(["status", "--universe", "AAPL,MSFT"]) == 0     # status only describes
    assert cli.main(["fetch", "--universe", "AAPL,MSFT"]) == 1
    assert "at least 8 symbols" in capsys.readouterr().err


def test_tune_compares_horizons(project, monkeypatch, capsys):
    _fake_live_engine(monkeypatch, project)
    assert cli.main(["tune", "--horizons", "6,12"]) == 0
    out = capsys.readouterr().out
    assert "30 min" in out and "60 min" in out and "<- current setting" in out
    assert "Best after costs" in out and "horizon_bars" in out          # the planted pattern is tradable
    assert not (project / "models" / "model.joblib").exists()            # tuning never replaces the live model


_KEY_VARS = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET_KEY",
             "ALPACA_LIVE_KEY_ID", "ALPACA_LIVE_SECRET_KEY")


def test_alpaca_keys_from_the_command_line(project, monkeypatch, capsys):
    from daytrader.config import alpaca_keys, alpaca_trading_keys

    for k in _KEY_VARS:
        monkeypatch.delenv(k, raising=False)
    (project / ".env").write_text("APCA_API_KEY_ID=fromfile\nAPCA_API_SECRET_KEY=filesecret\n")
    assert cli.main(["status", "--alpaca-key", "PKCLI", "--alpaca-secret", "clisecret"]) == 0
    out = capsys.readouterr().out
    assert "Data provider: alpaca" in out
    assert "PKCLI" not in out and "clisecret" not in out                 # never echoed
    assert alpaca_keys() == ("PKCLI", "clisecret")                     # command line beats .env
    assert alpaca_trading_keys(live=False) == ("PKCLI", "clisecret")   # also used for alpaca-paper
    assert alpaca_trading_keys(live=True) is None                      # never used for real money
    assert "clisecret" not in (project / ".env").read_text()           # nothing written to disk


def test_alpaca_secret_is_prompted_for_when_left_out(project, monkeypatch, capsys):
    import getpass
    import sys

    from daytrader.config import alpaca_keys

    for k in _KEY_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "typedsecret")
    assert cli.main(["status", "--alpaca-key", "PKCLI"]) == 0
    assert alpaca_keys() == ("PKCLI", "typedsecret")

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert cli.main(["status", "--alpaca-key", "PKCLI"]) == 1
    assert cli.main(["status", "--alpaca-secret", "x"]) == 1
    assert "needs --alpaca-key" in capsys.readouterr().err


def test_scorecard_command(project, capsys):
    import numpy as np
    from daytrader.config import load_config
    from daytrader import scorecard

    cfg = load_config(root=str(project))
    cal = MarketCalendar()
    day = date(2026, 11, 13)
    bars = normalize_bars(generate_bars(n_symbols=12, n_days=3, signal=3.0, seed=4, end_day=day))
    store = BarStore(cfg.bar_cache_dir("yahoo"), 5)
    store.append(bars)
    syms = sorted(bars["symbol"].unique())
    sess = cal.session(day)
    from datetime import timedelta
    for k in range(3, 60):
        stamp = (sess.open + timedelta(minutes=5 * k)).strftime("%Y-%m-%d %H:%M ET")
        scorecard.write_predictions(cfg.state_dir, stamp, syms, np.full(12, 50.0), np.linspace(-3, 3, 12),
                                    np.full(12, 30.0), np.full(12, 3.0))
    assert cli.main(["scorecard", "--date", str(day), "--universe", ",".join(syms)]) == 0
    out = capsys.readouterr().out
    assert "Scorecard %s" % day in out and "rank IC" in out
    assert (cfg.report_dir / ("scorecard_%s.txt" % day)).exists()
    assert cli.main(["scorecard", "--date", "2026-11-12", "--universe", ",".join(syms)]) == 1


def test_spreadcheck_command(project, monkeypatch, capsys):
    import daytrader.data as data

    class Fake:
        def raw_quotes(self, sym, start, end, feed="sip", limit=1000):
            t = start.astimezone(__import__("datetime").timezone.utc)
            ts = lambda s: (t + __import__("datetime").timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            return [{"t": ts(0), "bx": "N", "bp": 100.0, "bs": 1, "ax": "N", "ap": 100.04, "as": 1, "c": ["R"], "z": "A"},
                    {"t": ts(1), "bx": "Q", "bp": 100.01, "bs": 1, "ax": "Q", "ap": 100.02, "as": 1, "c": ["R"], "z": "A"},
                    {"t": ts(5), "bx": "N", "bp": 100.0, "bs": 1, "ax": "N", "ap": 100.04, "as": 1, "c": ["R"], "z": "A"}]

    monkeypatch.setattr(data, "get_provider", lambda cfg: Fake())
    assert cli.main(["spreadcheck", "--symbols", "SPY"]) == 0
    out = capsys.readouterr().out
    assert "SPY: 3 messages" in out and "rebuilt NBBO spread $: median 0.01" in out
    assert (project / "reports" / "spread_check.txt").exists()

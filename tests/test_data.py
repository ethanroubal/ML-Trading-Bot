from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from daytrader.data.base import drop_incomplete, normalize_bars
from daytrader.data.store import BarStore
from daytrader.data.yahoo import parse_yf_frame
from daytrader.market_calendar import ET, UTC


def _bars(sym, start, n, px=100.0, step=5):
    ts = [start + timedelta(minutes=step * i) for i in range(n)]
    return pd.DataFrame({"ts": pd.to_datetime(ts, utc=True), "symbol": sym, "open": px, "high": px + 1,
                         "low": px - 1, "close": px + 0.5, "volume": 1000.0})


def test_normalize_keeps_only_valid_regular_session_bars():
    start = datetime(2026, 11, 16, 8, 0, tzinfo=ET)          # 08:00 ET -> pre-market
    raw = _bars("AAA", start, 120)                           # runs to 17:55 ET
    raw.loc[30, "close"] = np.nan
    raw.loc[31, "open"] = -1
    out = normalize_bars(raw)
    et = out["ts"].dt.tz_convert(ET)
    mins = et.dt.hour * 60 + et.dt.minute
    assert mins.min() >= 570 and mins.max() < 960
    assert len(out) == 78 - 2
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert str(out["ts"].dt.tz) == "UTC"


def test_drop_incomplete_removes_forming_and_off_grid_bars():
    start = datetime(2026, 11, 16, 9, 30, tzinfo=ET)
    df = normalize_bars(_bars("AAA", start, 6))
    odd = df.iloc[[0]].copy()
    odd["ts"] = odd["ts"] + pd.Timedelta(minutes=2, seconds=13)   # a "live" row off the 5-minute grid
    df = pd.concat([df, odd], ignore_index=True)
    now = datetime(2026, 11, 16, 9, 52, tzinfo=ET)                # 09:50 bar still forming
    out = drop_incomplete(df, 5, now)
    assert len(out) == 4
    assert out["ts"].max() == pd.Timestamp(datetime(2026, 11, 16, 9, 45, tzinfo=ET)).tz_convert("UTC")


def test_store_roundtrip_overwrite_and_persistence(tmp_path):
    start = datetime(2026, 11, 16, 9, 30, tzinfo=ET)
    store = BarStore(tmp_path, 5)
    a = normalize_bars(pd.concat([_bars("AAA", start, 10), _bars("BBB", start, 5, px=50)]))
    assert store.append(a) == 15
    newer = normalize_bars(_bars("AAA", start + timedelta(minutes=40), 4, px=200))   # overlaps 2 bars
    store.append(newer)
    again = BarStore(tmp_path, 5)                              # fresh object reads from disk
    got = again.load(["AAA", "BBB", "MISSING"])
    assert len(got[got.symbol == "AAA"]) == 12 and len(got[got.symbol == "BBB"]) == 5
    row = got[(got.symbol == "AAA") & (got.ts == pd.Timestamp(start + timedelta(minutes=40)).tz_convert("UTC"))]
    assert float(row["open"].iloc[0]) == 200.0                 # later write wins
    assert again.last_ts("AAA") == pd.Timestamp(start + timedelta(minutes=55)).tz_convert("UTC")
    assert again.last_ts("MISSING") is None
    part = again.load(["AAA"], start=start + timedelta(minutes=45))
    assert len(part) == 3


@pytest.mark.parametrize("layout", ["ticker_first", "field_first", "single", "naive_index"])
def test_parse_yfinance_frames_in_every_layout(layout):
    idx = pd.date_range("2026-11-16 09:30", periods=4, freq="5min", tz="America/New_York")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    vals = np.array([[10, 11, 9, 10.5, 100]] * 4, dtype=float)
    if layout == "single":
        df = pd.DataFrame(vals, index=idx, columns=fields)
        out = parse_yf_frame(df, ["AAA"])
        assert set(out.symbol) == {"AAA"} and len(out) == 4
        return
    if layout == "naive_index":
        df = pd.DataFrame(vals, index=idx.tz_localize(None), columns=fields)
        out = parse_yf_frame(df, ["AAA"])
        assert out["ts"].iloc[0] == pd.Timestamp("2026-11-16 14:30", tz="UTC")
        return
    cols = [("AAA", f) for f in fields] + [("BBB", f) for f in fields]
    df = pd.DataFrame(np.hstack([vals, vals * 2]), index=idx, columns=pd.MultiIndex.from_tuples(cols))
    df.loc[idx[-1], ("BBB", "Close")] = np.nan                 # a missing bar for one ticker
    if layout == "field_first":
        df = df.swaplevel(axis=1)
    out = parse_yf_frame(df, ["AAA", "BBB"])
    assert len(out[out.symbol == "AAA"]) == 4 and len(out[out.symbol == "BBB"]) == 3
    assert out["ts"].iloc[0] == pd.Timestamp("2026-11-16 14:30", tz="UTC")
    assert float(out[out.symbol == "BBB"]["close"].iloc[0]) == 21.0


def test_parse_yfinance_empty():
    assert len(parse_yf_frame(None, ["AAA"])) == 0
    assert len(parse_yf_frame(pd.DataFrame(), ["AAA"])) == 0


def test_alpaca_pagination_and_quotes(monkeypatch, cfg):
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    from daytrader.data.alpaca import AlpacaProvider

    prov = AlpacaProvider(cfg)
    calls = []
    pages = [
        {"bars": {"AAA": [{"t": "2026-11-16T14:30:00Z", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100, "n": 5, "vw": 10.2},
                          {"t": "2026-11-16T13:00:00Z", "o": 9, "h": 9, "l": 9, "c": 9, "v": 5, "n": 1, "vw": 9}]},
         "next_page_token": "abc"},
        {"bars": {"BBB": [{"t": "2026-11-16T14:35:00Z", "o": 20, "h": 21, "l": 19, "c": 20.5, "v": 200, "n": 7, "vw": 20.1}]},
         "next_page_token": None},
    ]

    def fake_get(path, params):
        calls.append((path, dict(params)))
        if path.endswith("/bars"):
            return pages[len([c for c in calls if c[0].endswith("/bars")]) - 1]
        return {"AAA": {"latestTrade": {"p": 10.52, "t": "2026-11-16T14:36:01Z"},
                        "latestQuote": {"bp": 10.51, "ap": 10.53}},
                "BBB": {"latestTrade": {"p": 20.0, "t": "2026-11-16T14:36:01Z"}, "latestQuote": {"bp": 0, "ap": 0}},
                "CCC": {}}

    monkeypatch.setattr(prov, "_get", fake_get)
    start = datetime(2026, 11, 16, 9, 30, tzinfo=ET)
    bars = prov.fetch_bars(["AAA", "BBB"], start, None, 5)
    assert len(bars) == 2                                       # the 08:00 ET pre-market bar is dropped
    assert calls[1][1]["page_token"] == "abc"
    first = calls[0][1]
    assert first["timeframe"] == "5Min" and first["feed"] == "iex" and first["start"] == "2026-11-16T14:30:00Z"
    q = prov.latest_quotes(["AAA", "BBB", "CCC"])
    assert q["AAA"].bid == 10.51 and q["AAA"].ask == 10.53 and abs(q["AAA"].price - 10.52) < 1e-9
    assert q["BBB"].bid is None and q["BBB"].price == 20.0
    assert "CCC" not in q


def test_alpaca_requires_keys(monkeypatch, cfg):
    for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    from daytrader.data import resolve_provider_name
    from daytrader.data.alpaca import AlpacaProvider
    from daytrader.data.base import DataError

    assert resolve_provider_name(cfg) == "yahoo"
    with pytest.raises(DataError):
        AlpacaProvider(cfg)


def test_store_detects_rebased_history_after_a_split(tmp_path):
    start = datetime(2026, 11, 16, 9, 30, tzinfo=ET)
    store = BarStore(tmp_path, 5)
    store.append(normalize_bars(pd.concat([_bars("AAA", start, 10, px=200), _bars("BBB", start, 10, px=50)])))
    incoming = normalize_bars(pd.concat([
        _bars("AAA", start + timedelta(minutes=40), 4, px=100),      # same bars, half the price: 2-for-1 split
        _bars("BBB", start + timedelta(minutes=40), 4, px=50.1),     # ordinary small revision
        _bars("NEW", start, 3, px=10)]))
    assert store.conflicts(incoming) == ["AAA"]
    store.drop("AAA")
    assert store.last_ts("AAA") is None and BarStore(tmp_path, 5).last_ts("AAA") is None
    assert store.last_ts("BBB") is not None


def test_shipped_config_file_matches_built_in_defaults():
    from dataclasses import asdict
    from pathlib import Path

    from daytrader.config import Config, load_config

    root = Path(__file__).resolve().parent.parent
    loaded = asdict(load_config(str(root / "config.yaml"), root=str(root)))
    default = asdict(Config())
    loaded.pop("root"), default.pop("root")
    assert loaded == default


def test_config_rejects_typos(tmp_path):
    from daytrader.config import ConfigError, load_config

    (tmp_path / "config.yaml").write_text("strategy:\n  max_postions: 3\n")
    with pytest.raises(ConfigError, match="max_postions"):
        load_config(None, root=str(tmp_path))
    (tmp_path / "config.yaml").write_text("strategy:\n  max_positions: 3\n  capital: 5000\n")
    c = load_config(None, root=str(tmp_path))
    assert c.strategy.max_positions == 3 and c.strategy.capital == 5000.0


def test_universes_are_clean():
    from daytrader.universe import CORE, LIQUID, check_universe, resolve_universe

    core, liquid = resolve_universe("auto", "yahoo"), resolve_universe("auto", "alpaca")
    assert 60 <= len(core) <= 90 and 180 <= len(liquid) <= 260
    assert len(set(liquid)) == len(liquid) and set(core) <= set(liquid)
    assert all(s.isalpha() and s.isupper() for s in liquid)
    assert resolve_universe(["aapl", "MSFT", "aapl"], "yahoo") == ["AAPL", "MSFT"]
    with pytest.raises(ValueError):
        check_universe(["AAPL", "MSFT"])

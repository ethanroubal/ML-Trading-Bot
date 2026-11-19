"""Long histories: block-wise features, the memory budget, and chunked, resumable downloads."""
from datetime import date, datetime, time, timedelta

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
from daytrader.history import BAR_ID, backtest_history, build_dataset  # noqa: E402
from daytrader.market_calendar import ET  # noqa: E402
from daytrader.model import train  # noqa: E402
from daytrader.panel import build_panel  # noqa: E402

LAST = date(2026, 11, 13)


@pytest.fixture(scope="module")
def bars():
    return generate_bars(n_symbols=16, n_days=70, signal=3.0, seed=8, end_day=LAST)


def _store(bars):
    st = BarStore("/nonexistent", 5, persist=False)
    st.append(bars)
    return st, sorted(bars["symbol"].unique())


def _index(ds):
    return pd.Series(np.arange(len(ds.y)), index=ds.t.astype(np.int64) * 100 + ds.s)


def test_blocks_give_the_same_rows_labels_and_features_as_one_pass(cfg, cal, bars):
    st, syms = _store(bars)
    cfg.model.block_sessions = 1000
    one, _ = build_dataset(st, syms, cfg, cal, log=None)
    cfg.model.block_sessions = 12
    blk, info = build_dataset(st, syms, cfg, cal, log=None)
    assert len(info.blocks) == 6
    ia, ib = _index(one), _index(blk)
    assert ia.index.sort_values().equals(ib.index.sort_values())
    a, b = ia.sort_index().values, ib.sort_index().values
    assert np.allclose(one.y[a], blk.y[b], rtol=1e-3, atol=1e-4)   # labels: same up to float rounding
    xa, xb = one.X[a], blk.X[b]
    assert (np.isfinite(xa) == np.isfinite(xb)).all()
    both = np.isfinite(xa)
    assert np.abs(xa[both] - xb[both]).max() < 0.05        # only long-memory averages differ, slightly


def test_block_backtest_matches_one_pass_backtest(cfg, cal, bars):
    st, syms = _store(bars)
    cfg.model.block_sessions = 12
    ds, info = build_dataset(st, syms, cfg, cal, log=None)
    res = train(ds, info, cfg, verbose=False, fit_final=False)
    bt = backtest_history(st, info, res.oos, cfg, cal, capital=50_000, log=None)

    panel = build_panel(bars, 5, cal, symbols=syms)          # the same predictions, traded in one pass
    feats = compute_features(panel, cfg)
    offsets = np.flatnonzero(np.r_[True, panel.day[1:] != panel.day[:-1], True])
    gday = (res.oos["t"] // BAR_ID).to_numpy()
    local = pd.DataFrame({"t": offsets[gday] + (res.oos["t"] % BAR_ID).to_numpy(), "s": res.oos["s"].to_numpy(),
                          "day": gday, "exp_z": res.oos["exp_z"].to_numpy()})
    ref = run_backtest(panel, feats, local, cfg, capital=50_000)
    assert len(bt.trades) > 20
    assert len(bt.daily) == len(ref.daily)
    assert bt.stats["end_equity"] == pytest.approx(ref.stats["end_equity"], rel=2e-4)   # within ~0.02%


def test_memory_budget_thins_bars_but_still_trains_and_backtests(cfg, cal, bars):
    st, syms = _store(bars)
    cfg.model.block_sessions = 20
    cfg.model.max_dataset_rows = 20_000
    ds, info = build_dataset(st, syms, cfg, cal, log=None)
    assert info.stride >= 3 and ds.stride == info.stride
    assert len(ds.y) <= 1.3 * 20_000
    assert np.all(ds.k % info.stride == ds.day % info.stride)       # rotates through every time of day
    assert len(np.unique(ds.k)) > 60
    res = train(ds, info, cfg, verbose=False, fit_final=False)
    bt = backtest_history(st, info, res.oos, cfg, cal, capital=50_000, log=None)
    assert bt.stats["decision_every_bars"] == info.stride and bt.stats["days"] > 10


class WindowedProvider(ReplayProvider):
    """Imitates Alpaca: month-sized requests, with an optional failure part-way through."""
    name = "alpaca"
    fetch_window_days = 30

    def __init__(self, *a, fail_after=None, **k):
        super().__init__(*a, **k)
        self.calls, self.fail_after = [], fail_after

    def fetch_bars(self, symbols, start, end, interval_min):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("connection dropped")
        self.calls.append((pd.Timestamp(start), None if end is None else pd.Timestamp(end)))
        return super().fetch_bars(symbols, start, end, interval_min)


def _engine(cfg, tmp_path, bars, days, fail_after=None):
    clock = SimClock(datetime.combine(LAST, time(18, 0), tzinfo=ET))
    cfg.data.history_days = days
    prov = WindowedProvider(bars, clock, 5, fail_after=fail_after)
    store = BarStore(tmp_path / "cache", 5)
    eng = Engine(cfg, prov, sorted(bars["symbol"].unique()), clock=clock, trade=False, store=store,
                 model_path=tmp_path / "m.joblib")
    eng.log = lambda m="": None
    return eng, prov, store


def _held(store, sym):
    return store.first_ts(sym).tz_convert(ET).date(), store.last_ts(sym)


def test_long_download_is_chunked_extends_back_and_does_not_repeat(cfg, tmp_path, bars):
    eng, prov, store = _engine(cfg, tmp_path, bars, 45)
    eng.sync_history()
    first, last = _held(store, "SYN00")
    assert first >= LAST - timedelta(days=46) and len(prov.calls) == 2

    eng, prov, store = _engine(cfg, tmp_path, bars, 95)       # ask for more history: backfill once
    eng.sync_history()
    first2, last2 = _held(store, "SYN00")
    assert first2 <= LAST - timedelta(days=90) and last2 == last
    back = [c for c in prov.calls if c[1] is not None]                  # backfill windows (the rest is today's tail)
    assert len(back) >= 2 and all(c[0] < pd.Timestamp(datetime.combine(first, time(0)), tz=ET) for c in back)
    assert store.get_meta("history_from") <= str(LAST - timedelta(days=94))

    eng, prov, store = _engine(cfg, tmp_path, bars, 95)       # nothing more to fetch
    eng.sync_history()
    assert all(c[0] > pd.Timestamp(LAST - timedelta(days=5), tz=ET) for c in prov.calls)


def test_interrupted_download_resumes_where_it_stopped(cfg, tmp_path, bars):
    eng, prov, store = _engine(cfg, tmp_path, bars, 95, fail_after=2)
    with pytest.raises(RuntimeError):
        eng.sync_history()
    saved = store.last_ts("SYN00")
    assert saved is not None and saved < pd.Timestamp(LAST - timedelta(days=30), tz="UTC")   # progress kept

    eng, prov, store = _engine(cfg, tmp_path, bars, 95)
    eng.sync_history()
    assert all(c[0] >= saved - pd.Timedelta(minutes=5) for c in prov.calls)      # nothing re-fetched
    got = store.load(["SYN00"])
    want = bars[(bars.symbol == "SYN00") & (bars.ts >= got.ts.min())]
    assert len(got) == len(want) and got.ts.is_unique


def test_backtest_predicts_every_bar_with_the_fold_that_had_not_seen_it(cfg, cal, bars):
    """The fold models reproduce the walk-forward predictions exactly, so the backtest can
    decide on every bar even when training kept only some bars."""
    from daytrader.history import _fold_grid, _load_block

    st, syms = _store(bars)
    cfg.model.block_sessions = 1000
    ds, info = build_dataset(st, syms, cfg, cal, log=None)
    res = train(ds, info, cfg, verbose=False, fit_final=False)
    assert res.folds and all(fd["c"] is not None for fd in res.folds)
    panel, feats = _load_block(st, syms, cfg, cal, info.dates[0], info.dates[-1], None)
    g = np.arange(len(panel.dates))
    grid, days = _fold_grid(res, panel, feats, g)
    offsets = np.flatnonzero(np.r_[True, panel.day[1:] != panel.day[:-1], True])
    gd = (res.oos["t"] // BAR_ID).to_numpy()
    t_local = offsets[gd] + (res.oos["t"] % BAR_ID).to_numpy()
    assert np.allclose(grid[t_local, res.oos["s"].to_numpy()], res.oos["exp_z"].to_numpy(), atol=1e-6)
    assert set(days) == set(np.unique(gd))

    cfg.model.max_dataset_rows = 20_000                  # thin training rows: backtest still every bar
    ds, info = build_dataset(st, syms, cfg, cal, log=None)
    res = train(ds, info, cfg, verbose=False, fit_final=False)
    bt = backtest_history(st, info, res, cfg, cal, capital=50_000, log=None)
    assert info.stride > 1 and bt.stats["decision_every_bars"] == 1


def test_history_start_respects_max_and_the_feed_start(cfg, tmp_path, bars):
    eng, prov, store = _engine(cfg, tmp_path, bars, 0)
    now = eng.clock.now()
    prov.earliest_date = date(2020, 1, 1)
    eng.cfg.data.history_days = "max"
    assert eng.history_start(now).date() == date(2020, 1, 1)
    eng.cfg.data.history_days = 30
    assert eng.history_start(now).date() == (now - timedelta(days=30)).date()
    prov.earliest_date = (now - timedelta(days=10)).date()
    assert eng.history_start(now).date() == prov.earliest_date


def test_extending_history_backwards_retrains_the_model(cfg, tmp_path, bars, capsys):
    eng, prov, store = _engine(cfg, tmp_path, bars, 45)
    eng.sync_history()
    eng.ensure_model()
    first_start = eng.model.meta["train_start"]
    eng, prov, store = _engine(cfg, tmp_path, bars, 95)      # history grows backwards, no new days
    eng.log = print
    eng.sync_history(verbose=False)
    capsys.readouterr()
    eng.ensure_model()
    out = capsys.readouterr().out
    assert "bar history now starts" in out
    assert eng.model.meta["train_start"] < first_start
    eng.ensure_model()                                           # and only once
    assert "bar history now starts" not in capsys.readouterr().out

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from daytrader.data.synthetic import generate_bars
from daytrader.features import compute_features
from daytrader.market_calendar import ET
from daytrader.panel import build_panel


@pytest.fixture(scope="module")
def bars():
    return generate_bars(n_symbols=20, n_days=36, signal=1.0, seed=3, end_day=date(2026, 11, 13))


def test_panel_grid_shape_and_gap_filling(bars, cal):
    b = bars.copy()
    sym = b["symbol"].iloc[0]
    hole = (b["symbol"] == sym) & (b["ts"] == b["ts"].iloc[500])
    hole_ts = b.loc[hole, "ts"].iloc[0]
    b = b[~hole]
    p = build_panel(b, 5, cal)
    assert p.T == 36 * 78 and p.S == 20 and len(p.dates) == 36
    assert (p.nbar == 78).all() and p.k.max() == 77
    t = p.ts.get_loc(hole_ts.tz_convert(ET))
    s = p.symbols.index(sym)
    assert not p.present[t, s] and p.V[t, s] == 0
    assert p.O[t, s] == p.C[t, s] == p.H[t, s] == p.L[t, s] == p.C[t - 1, s]   # flat bar at prior close
    assert p.present.sum() == p.T * p.S - 1


def test_panel_asof_uses_only_completed_bars(bars, cal):
    asof = datetime(2026, 11, 13, 10, 37, 30, tzinfo=ET)
    p = build_panel(bars, 5, cal, asof=asof)
    assert p.ts[-1] == pd.Timestamp("2026-11-13 10:30", tz=ET)     # 10:35 bar is still forming
    assert p.k[-1] == 12 and p.nbar[-1] == 78
    assert p.dates[-1] == date(2026, 11, 13)


def test_half_day_is_recognised(cal):
    hb = generate_bars(n_symbols=10, n_days=8, signal=0.0, seed=1, end_day=date(2026, 12, 1))
    p = build_panel(hb, 5, cal)
    i = p.dates.index(date(2026, 11, 27))                           # day after Thanksgiving
    rows = p.day == i
    assert rows.sum() == 42 and (p.nbar[rows] == 42).all()
    assert date(2026, 11, 26) not in p.dates


def test_features_never_look_ahead(bars, cal, cfg):
    """Features computed with the future removed must equal features computed with it present."""
    full = build_panel(bars, 5, cal)
    ff = compute_features(full, cfg)
    for d, hh, mm in ((date(2026, 11, 13), 11, 35), (date(2026, 11, 11), 9, 40), (date(2026, 11, 6), 15, 30)):
        asof = datetime(d.year, d.month, d.day, hh, mm, 7, tzinfo=ET)
        part = build_panel(bars, 5, cal, asof=asof)
        fp = compute_features(part, cfg)
        t = full.ts.get_loc(part.ts[-1])
        for name, a, b in zip(ff.names, ff.arrays, fp.arrays):
            x, y = a[t], b[-1]
            assert (np.isfinite(x) == np.isfinite(y)).all(), name
            ok = np.isfinite(x)
            assert np.allclose(x[ok], y[ok], atol=1e-5), "look-ahead in feature %s" % name
        assert np.allclose(ff.sigH[t], fp.sigH[-1], equal_nan=True)
        assert (ff.usable[t] == fp.usable[-1]).all()


def test_live_window_matches_full_history(bars, cal, cfg):
    ff = compute_features(build_panel(bars, 5, cal), cfg)
    fw = compute_features(build_panel(bars, 5, cal, last_n_days=cfg.engine.feature_window_days), cfg)
    checked = 0
    for row in (-1, -38, -70):                                        # close, midday and morning of the last day
        for name, a, b in zip(ff.names, ff.arrays, fw.arrays):
            ok = np.isfinite(a[row]) & np.isfinite(b[row])
            assert (np.isfinite(a[row]) == np.isfinite(b[row])).all(), name
            if ok.any():
                checked += 1
                assert np.abs(a[row][ok] - b[row][ok]).max() < 0.03, name
    assert checked > 200


def test_labels_are_the_forward_return_from_next_open(bars, cal, cfg):
    cfg.model.target = "raw"
    p = build_panel(bars, 5, cal)
    f = compute_features(p, cfg)
    H = cfg.model.horizon_bars
    t = int(np.flatnonzero((p.day == 20) & (p.k == 10))[0])
    expect = np.log(p.O[t + 1 + H] / p.O[t + 1])
    assert np.allclose(f.fwd_ret[t], expect)
    assert np.allclose(f.y[t], np.clip(expect / f.sigH[t], -5, 5))
    late = int(np.flatnonzero((p.day == 20) & (p.k == 70))[0])       # horizon runs past the close
    last = int(np.flatnonzero(p.day == 20)[-1])
    assert np.allclose(f.fwd_ret[late], np.log(p.C[last] / p.O[late + 1]))
    too_late = int(np.flatnonzero((p.day == 20) & (p.k == 76))[0])   # fewer than min_bars_left remain
    assert np.isnan(f.y[too_late]).all() and not f.usable[too_late].any()


def test_relative_labels_are_market_neutral(bars, cal, cfg):
    f = compute_features(build_panel(bars, 5, cal), cfg)
    rows = f.fwd_rel[200:2000]
    rows = rows[np.isfinite(rows).any(axis=1)]
    assert len(rows) > 1000 and np.abs(np.nanmean(rows, axis=1)).max() < 1e-12
    assert 0.5 < np.nanstd(f.y) < 1.3                                 # sensible volatility scaling

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from daytrader.config import Config
from daytrader.data.store import BarStore
from daytrader.market_calendar import ET, UTC, MarketCalendar
from daytrader import scorecard

DAY = date(2026, 11, 18)


def _market(tmp_path, n_sym=20, seed=0):
    """A session of 5-minute bars where each stock drifts at its own steady rate."""
    rng = np.random.default_rng(seed)
    cal = MarketCalendar()
    sess = cal.session(DAY)
    syms = ["S%02d" % i for i in range(n_sym)]
    drift = np.linspace(-2.0, 2.0, n_sym)              # bp per bar
    rows = []
    for i, s in enumerate(syms):
        px = 100.0
        for k in range(78):
            ts = (sess.open + timedelta(minutes=5 * k)).astimezone(UTC)
            o = px
            px = px * np.exp((drift[i] + rng.normal(0, 0.2)) / 1e4)
            rows.append({"ts": ts, "symbol": s, "open": o, "high": max(o, px), "low": min(o, px),
                         "close": px, "volume": 1000})
    store = BarStore(tmp_path / "bars", 5, persist=False)
    store.append(pd.DataFrame(rows))
    return store, syms, drift, cal, sess


def _write_preds(state, syms, pred, sess, cost=3.0):
    for k in range(3, 70):
        stamp = (sess.open + timedelta(minutes=5 * k)).strftime("%Y-%m-%d %H:%M ET")
        scorecard.write_predictions(state, stamp, syms, np.full(len(syms), 100.0), pred,
                                    np.full(len(syms), 30.0), np.full(len(syms), cost))


def test_perfect_ranking_scores_high(tmp_path):
    store, syms, drift, cal, sess = _market(tmp_path)
    cfg = Config()
    state = tmp_path / "state"
    _write_preds(state, syms, drift * 12, sess)          # predicts exactly the 60-minute drift
    card = scorecard.build_scorecard(DAY, state, store, syms, cfg, cal)
    assert card["source"] == "predictions"
    assert card["rank_ic"] > 0.9 and card["hit_rate"] > 0.9
    assert card["top_signed_rel_bps"] > 15 and card["top_net_bps"] > 10
    lines = scorecard.run_scorecard(DAY, state, store, syms, cfg, cal, tmp_path / "rep", model_ic=0.01)
    assert any("rank IC" in ln for ln in lines)
    assert (tmp_path / "rep" / ("scorecard_%s.txt" % DAY)).exists()
    hist = pd.read_csv(state / "scorecard.csv")
    assert len(hist) == 1
    scorecard.run_scorecard(DAY, state, store, syms, cfg, cal, tmp_path / "rep")     # re-run replaces the day
    assert len(pd.read_csv(state / "scorecard.csv")) == 1


def test_backwards_ranking_scores_negative(tmp_path):
    store, syms, drift, cal, sess = _market(tmp_path)
    state = tmp_path / "state"
    _write_preds(state, syms, -drift * 12, sess)
    card = scorecard.build_scorecard(DAY, state, store, syms, Config(), cal)
    assert card["rank_ic"] < -0.9 and card["top_signed_rel_bps"] < -15


def test_labels_match_training_definition(tmp_path):
    """Entry at the open after the decision, exit at the open 12 bars later."""
    store, syms, drift, cal, sess = _market(tmp_path, n_sym=5)
    state = tmp_path / "state"
    stamp = (sess.open + timedelta(minutes=5 * 10)).strftime("%Y-%m-%d %H:%M ET")   # bar ending 10:20
    scorecard.write_predictions(state, stamp, syms, np.full(5, 100.0), np.ones(5), np.full(5, 30.0), np.full(5, 3.0))
    df, _ = scorecard._load_predictions(state, DAY)
    moves, n = scorecard._forward_moves(store, syms, DAY, cal, Config())
    g = scorecard.grade(df, moves, n, DAY, cal, Config())
    bars = store.load([syms[0]])
    o = bars["open"].to_numpy()
    assert g.loc[g.symbol == syms[0], "raw_bps"].iloc[0] == pytest.approx(np.log(o[22] / o[10]) * 1e4, rel=1e-4)


def test_falls_back_to_signals_csv(tmp_path):
    store, syms, drift, cal, sess = _market(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    rows = []
    for k in range(3, 70):
        stamp = (sess.open + timedelta(minutes=5 * k)).strftime("%Y-%m-%d %H:%M ET")
        for s, d in [(syms[-1], drift[-1]), (syms[0], drift[0])]:
            rows.append({"time": stamp, "action": "watch", "symbol": s, "price": 100.0, "expected_bps": d * 12,
                         "cost_bps": 3.0, "net_edge_bps": abs(d * 12) - 3, "hit_rate": 0.5, "stop": 99})
    pd.DataFrame(rows).to_csv(state / "signals.csv", index=False)
    card = scorecard.build_scorecard(DAY, state, store, syms, Config(), cal)
    assert card["source"] == "signals" and np.isnan(card["rank_ic"])
    assert card["hit_rate"] > 0.9
    assert scorecard.build_scorecard(date(2026, 11, 17), state, store, syms, Config(), cal) is None

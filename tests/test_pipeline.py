"""End-to-end checks of the learning pipeline on synthetic markets.

The two key properties:
  * on a pure random walk the pipeline must NOT find an edge and must not trade
    (if it did, information from the future would be leaking in somewhere);
  * when a pattern is planted, it must find it and trade it profitably after costs.
"""
from datetime import date

import numpy as np
import pytest

pytest.importorskip("sklearn")

from daytrader.backtest import run_backtest  # noqa: E402
from daytrader.data.synthetic import generate_bars  # noqa: E402
from daytrader.features import compute_features  # noqa: E402
from daytrader.model import Model, NotEnoughData, evidence_factor, make_dataset, train  # noqa: E402
from daytrader.panel import build_panel  # noqa: E402


def _fit(cfg, cal, signal, seed, n_sym=30, n_days=42):
    bars = generate_bars(n_symbols=n_sym, n_days=n_days, signal=signal, seed=seed, end_day=date(2026, 11, 13))
    panel = build_panel(bars, 5, cal)
    feats = compute_features(panel, cfg)
    ds = make_dataset(panel, feats, cfg)
    return panel, feats, ds, train(ds, panel, cfg, verbose=False)


def test_random_walk_gives_no_edge_and_no_trades(cfg, cal):
    panel, feats, ds, res = _fit(cfg, cal, signal=0.0, seed=100)
    r = res.report
    assert not r["has_edge"]
    assert abs(r["oos_rank_ic"]) < 0.03 and abs(r["oos_ic_tstat"]) < 2.5
    bt = run_backtest(panel, feats, res.oos, cfg)
    assert bt.stats["trades"] <= 5                       # costs must keep it out of the market
    assert abs(bt.stats["total_return_pct"]) < 0.3


def test_planted_pattern_is_found_and_traded_profitably(cfg, cal):
    panel, feats, ds, res = _fit(cfg, cal, signal=3.0, seed=11)
    r = res.report
    assert r["has_edge"] and r["oos_rank_ic"] > 0.08 and r["oos_ic_tstat"] > 5
    assert 0.3 < r["calibration_slope"] < 2.0            # predictions are roughly the right size
    top = r["edge_table"][-1]
    assert top["net_bps"] > 0 and top["hit_rate"] > 0.55
    bt = run_backtest(panel, feats, res.oos, cfg)
    st = bt.stats
    assert st["trades"] > 100 and st["total_return_pct"] > 2.0 and st["win_rate_pct"] > 55
    assert st["avg_trade_bps"] > 0
    # every position is closed before the end of each session
    assert (bt.trades["reason"].isin(["signal", "eod", "stop", "time", "target", "loss-limit"])).all()
    by_day = bt.trades.groupby("date")["pnl"].sum()
    assert np.allclose(by_day.reindex(bt.daily["date"]).fillna(0).to_numpy(), bt.daily["pnl"].to_numpy(), atol=0.5)   # per-trade P&L is rounded to cents


def test_walk_forward_predictions_only_cover_later_folds(cfg, cal):
    panel, feats, ds, res = _fit(cfg, cal, signal=1.0, seed=5)
    days = np.unique(ds.day)
    first_oos_day = res.oos["day"].min()
    n_train_only = int((days < first_oos_day).sum())
    assert n_train_only >= max(15, int(0.4 * len(days)))  # train-only block + the calibration fold
    assert np.isfinite(res.oos["exp_z"]).all()


def test_model_save_load_and_prediction_consistency(cfg, cal, tmp_path):
    panel, feats, ds, res = _fit(cfg, cal, signal=3.0, seed=11)
    path = tmp_path / "models" / "model.joblib"
    res.model.save(path)
    again = Model.load(path)
    assert again is not None and again.compatible(cfg, feats.names)
    X = ds.X[:500]
    assert np.allclose(again.expected_z(X), res.model.expected_z(X))
    assert np.all((again.p_win(np.array([0.0, 0.1, 1.0])) >= 0.3) & (again.p_win(np.array([0.0, 0.1, 1.0])) <= 0.95))
    assert again.p_win(np.array([1.0]))[0] >= again.p_win(np.array([0.0]))[0]
    cfg.model.horizon_bars = 6
    assert not again.compatible(cfg, feats.names)
    assert Model.load(tmp_path / "nope.joblib") is None


def test_too_little_history_is_refused(cfg, cal):
    bars = generate_bars(n_symbols=12, n_days=14, signal=0.0, seed=1, end_day=date(2026, 11, 13))
    panel = build_panel(bars, 5, cal)
    feats = compute_features(panel, cfg)
    with pytest.raises(NotEnoughData):
        train(make_dataset(panel, feats, cfg), panel, cfg, verbose=False)


def test_evidence_factor_shrinks_weak_results():
    assert evidence_factor(-1.0) == 0.0 and evidence_factor(0.0) == 0.0
    assert evidence_factor(1.0) == pytest.approx(0.5)
    assert evidence_factor(2.0) == pytest.approx(0.8)
    assert evidence_factor(6.0) > 0.97

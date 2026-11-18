import numpy as np
import pytest

from daytrader.broker import Order, PaperBroker
from daytrader.costs import one_way_cost_bps
from daytrader.strategy import make_signals, plan, position_notional, recommendations, session_gates


def _order(sym="AAA", side=1, shares=100, px=100.0, **kw):
    d = dict(symbol=sym, side=side, shares=shares, ref_price=px, stop_frac=0.01, target_frac=0.0,
             predicted_bps=10.0, edge_bps=6.0, sigH_bps=40.0)
    d.update(kw)
    return Order(**d)


def test_long_round_trip_pays_costs_both_ways(cfg):
    br = PaperBroker(cfg, None, 100_000)
    br.start_day("2026-11-16")
    c = float(one_way_cost_bps(100.0, cfg.costs)) / 1e4
    pos = br.open(_order(), "t0")
    assert pos.entry_price == pytest.approx(100.0 * (1 + c))
    assert br.equity() == pytest.approx(100_000 - 100 * 100.0 * c)      # marked at mid: down by the cost
    br.mark({"AAA": 101.0})
    tr = br.close("AAA", "t1", "signal", price=101.0)
    assert tr["pnl"] == pytest.approx(100 * (101.0 * (1 - c) - 100.0 * (1 + c)), abs=0.01)
    assert br.equity() == pytest.approx(100_000 + tr["pnl"], abs=0.01) and not br.positions


def test_short_round_trip_and_quote_fills(cfg):
    br = PaperBroker(cfg, None, 50_000)
    br.start_day("2026-11-16")
    slip = cfg.costs.slippage_bps / 1e4
    pos = br.open(_order(side=-1, shares=50), "t0", price=100.0, bid=99.98, ask=100.02)
    assert pos.entry_price == pytest.approx(99.98 * (1 - slip))           # shorts sell at the bid
    tr = br.close("AAA", "t1", "signal", price=98.0, bid=97.99, ask=98.01)
    assert tr["exit_price"] == pytest.approx(98.01 * (1 + slip), abs=1e-4)  # covers buy at the ask
    assert tr["pnl"] > 0 and tr["side"] == "SHORT"
    assert br.equity() == pytest.approx(50_000 + tr["pnl"], abs=0.01)
    wide = br.open(_order(sym="BBB"), "t2", price=100.0, bid=99.0, ask=101.0)   # absurd quote -> ignored
    assert wide.entry_price == pytest.approx(100.0 * (1 + float(one_way_cost_bps(100.0, cfg.costs)) / 1e4))


def test_stops_targets_and_gaps(cfg):
    br = PaperBroker(cfg, None, 100_000)
    br.start_day("2026-11-16")
    pos = br.open(_order(stop_frac=0.01, target_frac=0.02), "t0")
    assert pos.stop == pytest.approx(99.0) and pos.target == pytest.approx(102.0)
    assert br.stop_or_target_hit(pos, 99.5, 100.5, open_px=100.0) is None
    assert br.stop_or_target_hit(pos, 98.7, 100.2, open_px=100.0) == ("stop", 99.0)
    assert br.stop_or_target_hit(pos, 97.0, 98.0, open_px=98.0) == ("stop", 98.0)     # gapped through
    assert br.stop_or_target_hit(pos, 98.5, 102.5, open_px=100.0)[0] == "stop"          # both touched: assume worst
    assert br.stop_or_target_hit(pos, 100.1, 102.4, open_px=100.2) == ("target", 102.0)
    assert br.stop_or_target_hit(pos, 98.9, 98.9) == ("stop", 98.9)                     # live: last price only
    br.close("AAA", "t1", "stop", price=99.0)
    assert br.cooldown["AAA"] == cfg.strategy.cooldown_bars
    sh = br.open(_order(sym="CCC", side=-1), "t2")
    assert sh.stop == pytest.approx(101.0)
    assert br.stop_or_target_hit(sh, 100.0, 101.2, open_px=100.1) == ("stop", 101.0)


def test_daily_loss_limit_and_day_summary(cfg):
    br = PaperBroker(cfg, None, 100_000)
    br.start_day("2026-11-16")
    br.open(_order(shares=150), "t0")
    br.mark({"AAA": 79.0})                                              # -$3,150 on $100k > 3% limit
    assert br.daily_loss_breached()
    br.close("AAA", "t1", "loss-limit", price=79.0)
    row = br.day_summary()
    assert row["trades"] == 1 and row["wins"] == 0 and row["pnl"] < -3000 and row["ret_pct"] < -3


def test_state_survives_restart(cfg, tmp_path):
    br = PaperBroker(cfg, tmp_path, 25_000)
    br.start_day("2026-11-16")
    br.open(_order(shares=10, beta=1.4), "09:45")
    br.end_bar()
    br.save()
    again = PaperBroker.load(cfg, tmp_path)
    assert again.start_capital == 25_000 and again.day == "2026-11-16"
    pos = again.positions["AAA"]
    assert pos.shares == 10 and pos.bars_held == 1 and pos.beta == 1.4
    assert again.equity() == pytest.approx(br.equity())
    again.close("AAA", "10:00", "signal", price=101.0)
    again.record_day()
    assert (tmp_path / "trades.csv").read_text().count("\n") == 2       # header + one trade
    assert "2026-11-16" in (tmp_path / "daily.csv").read_text()


def _signals(cfg, exp_bps, price=100.0, sig_bps=40.0, beta=None):
    n = len(exp_bps)
    syms = ["S%02d" % i for i in range(n)]
    exp_z = np.array(exp_bps, dtype=float) / sig_bps
    return make_signals(syms, np.full(n, price), exp_z, np.full(n, sig_bps / 1e4), np.ones(n, bool), cfg,
                        beta=np.array(beta, dtype=float) if beta is not None else None)


def test_plan_only_trades_edges_that_beat_costs(cfg):
    br = PaperBroker(cfg, None, 100_000)
    br.start_day("d")
    cost = 2 * float(one_way_cost_bps(100.0, cfg.costs))
    sig = _signals(cfg, [cost + 5, cost + 0.5, -(cost + 8), 1.0, np.nan])
    exits, orders = plan(sig, br, cfg, can_enter=True)
    assert not exits and [o.symbol for o in orders] == ["S02", "S00"]    # best expected dollars first
    assert orders[0].side == -1 and orders[1].side == 1
    assert plan(sig, br, cfg, can_enter=False)[1] == []
    cfg.strategy.allow_short = False
    assert [o.symbol for o in plan(sig, br, cfg, True)[1]] == ["S00"]


def test_plan_respects_position_gross_and_market_exposure_limits(cfg):
    s = cfg.strategy
    br = PaperBroker(cfg, None, 100_000)
    br.start_day("d")
    sig = _signals(cfg, [30.0] * 12)                                     # twelve strong longs
    _, orders = plan(sig, br, cfg, True)
    total = sum(o.shares * o.ref_price for o in orders)
    assert total <= s.max_net_exposure * 100_000 + 1                     # all long -> capped by market exposure
    assert all(o.shares * o.ref_price <= s.max_position_pct * 100_000 + 1 for o in orders)
    mixed = _signals(cfg, [30.0, -30.0] * 8, beta=[1.0] * 16)
    _, orders = plan(mixed, br, cfg, True)
    gross = sum(o.shares * o.ref_price for o in orders)
    net = sum(o.side * o.shares * o.ref_price for o in orders)
    assert len(orders) <= s.max_positions and gross <= s.max_gross_exposure * 100_000 + 1
    assert abs(net) <= s.max_net_exposure * 100_000 + 1
    high_beta = _signals(cfg, [30.0] * 6, beta=[2.5] * 6)                # high-beta longs use up exposure faster
    _, hb = plan(high_beta, br, cfg, True)
    assert sum(o.shares * o.ref_price * o.beta for o in hb) <= s.max_net_exposure * 100_000 + 1
    assert sum(o.shares * o.ref_price for o in hb) < total


def test_plan_exits_on_signal_flip_time_and_respects_cooldown(cfg):
    br = PaperBroker(cfg, None, 100_000)
    br.start_day("d")
    sig = _signals(cfg, [30.0, -30.0, 30.0])
    for o in plan(sig, br, cfg, True)[1]:
        br.open(o, "t")
    assert set(br.positions) == {"S00", "S01", "S02"}
    flipped = _signals(cfg, [-5.0, -30.0, 30.0])
    assert plan(flipped, br, cfg, True)[0] == []                         # min_hold_bars not reached yet
    for _ in range(cfg.strategy.min_hold_bars):
        br.end_bar()
    assert plan(flipped, br, cfg, True)[0] == [("S00", "signal")]
    for _ in range(3 * cfg.model.horizon_bars):
        br.end_bar()
    assert {r for _, r in plan(sig, br, cfg, True)[0]} == {"time"}
    br.close("S00", "t", "stop", price=99.0)
    assert "S00" not in [o.symbol for o in plan(sig, br, cfg, True)[1]]  # cooling off after a stop


def test_sizing_and_gates(cfg):
    s = cfg.strategy
    quiet = position_notional(100_000, 20.0, cfg)                        # low vol -> capped by max position
    wild = position_notional(100_000, 200.0, cfg)                        # high vol -> sized by risk
    assert quiet == pytest.approx(s.max_position_pct * 100_000)
    assert wild == pytest.approx(100_000 * s.risk_per_trade_pct / (s.stop_sigma * 0.02))
    assert session_gates(cfg) == (9 * 60 + 40, 15 * 60 + 30, 15 * 60 + 55)
    assert session_gates(cfg, 13 * 60) == (9 * 60 + 40, 12 * 60 + 30, 12 * 60 + 55)


def test_recommendations_are_ranked_and_labelled(cfg):
    cost = 2 * float(one_way_cost_bps(100.0, cfg.costs))
    recs = recommendations(_signals(cfg, [cost + 9, 2.0, -(cost + 4), -1.0, 0.5, cost + 2]), cfg, top_n=2)
    buys = [r for r in recs if r["side"] == 1]
    sells = [r for r in recs if r["side"] == -1]
    assert [r["symbol"] for r in buys] == ["S00", "S05"] and all(r["action"] == "BUY" for r in buys)
    assert [r["symbol"] for r in sells] == ["S02", "S03"]
    assert sells[0]["action"] == "SELL/SHORT" and sells[1]["action"] == "watch"
    assert buys[0]["stop"] < buys[0]["price"] and sells[0]["stop"] > sells[0]["price"]

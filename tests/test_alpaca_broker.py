"""The Alpaca brokerage connection, exercised against a fake Alpaca server.

Nothing here touches the network: FakeAlpaca imitates the Trading API's
account, orders and positions endpoints, keeps its own books, and lets the
tests check that the program's ledger always agrees with the 'broker'.
"""
from datetime import date, datetime, time

import pytest

from daytrader.alpaca_broker import AlpacaBroker, BrokerError
from daytrader.broker import Order
from daytrader.config import alpaca_trading_keys


class FakeAlpaca:
    def __init__(self, equity=100_000.0, shorting=True, fill="full"):
        self.cash = equity
        self.shorting = shorting
        self.fill = fill              # full | none | half
        self.pos = {}                 # symbol -> signed shares
        self.last = {}                # symbol -> last price
        self.orders = {}
        self.log = []
        self.n = 0

    # -- helpers
    def equity(self):
        return self.cash + sum(q * self.last[s] for s, q in self.pos.items())

    def _apply(self, o, qty, px):
        sign = 1 if o["side"] == "buy" else -1
        self.cash -= sign * qty * px
        self.pos[o["symbol"]] = self.pos.get(o["symbol"], 0) + sign * qty
        if self.pos[o["symbol"]] == 0:
            del self.pos[o["symbol"]]
        self.last[o["symbol"]] = px
        done = float(o["filled_qty"]) + qty
        tot = float(o["filled_avg_price"] or 0) * float(o["filled_qty"]) + qty * px
        o["filled_qty"], o["filled_avg_price"] = str(done), str(tot / done)
        o["status"] = "filled" if done >= float(o["qty"]) else "partially_filled"

    def trigger_stop(self, symbol, px):
        for o in self.orders.values():
            if o["symbol"] == symbol and o["type"] == "stop" and o["status"] == "new":
                self._apply(o, int(float(o["qty"])), px)

    # -- the three verbs the client uses
    def get(self, path, params=None):
        if path == "/v2/account":
            return {"status": "ACTIVE", "equity": str(self.equity()), "cash": str(self.cash),
                    "buying_power": str(max(self.equity(), 0) * 2), "shorting_enabled": self.shorting,
                    "trading_blocked": False, "account_blocked": False}
        if path == "/v2/positions":
            return [{"symbol": s, "qty": str(abs(q)), "side": "long" if q > 0 else "short",
                     "current_price": str(self.last[s]), "asset_class": "us_equity"} for s, q in self.pos.items()]
        if path.startswith("/v2/orders/"):
            return dict(self.orders[path.rsplit("/", 1)[1]])
        raise AssertionError(path)

    def post(self, path, body):
        assert path == "/v2/orders"
        self.n += 1
        o = dict(body, id="o%d" % self.n, status="new", filled_qty="0", filled_avg_price=None)
        self.orders[o["id"]] = o
        self.log.append(dict(body))
        qty = int(float(o["qty"]))
        held = self.pos.get(o["symbol"], 0)
        reserved = sum(int(float(x["qty"])) for x in self.orders.values()
                       if x is not o and x["symbol"] == o["symbol"] and x["status"] == "new" and x["side"] == o["side"])
        closing = (o["side"] == "sell" and held > 0) or (o["side"] == "buy" and held < 0)
        if closing and qty + reserved > abs(held):
            raise BrokerError("insufficient qty available for order", 403)
        if o["type"] == "stop":
            return dict(o)
        if o["type"] == "limit":
            px = float(o["limit_price"])
            share = {"full": qty, "none": 0, "half": qty // 2}[self.fill]
        else:
            px = self.last.get(o["symbol"], 100.0)
            share = qty
        if share:
            self._apply(o, share, px)
        return dict(o)

    def delete(self, path):
        o = self.orders[path.rsplit("/", 1)[1]]
        if o["status"] in ("filled", "canceled"):
            raise BrokerError("order is not cancelable", 422)
        o["status"] = "canceled"
        return None


def _order(sym="AAA", side=1, shares=100, px=50.0):
    return Order(symbol=sym, side=side, shares=shares, ref_price=px, stop_frac=0.01, target_frac=0.0,
                 predicted_bps=10.0, edge_bps=6.0, sigH_bps=40.0)


def _broker(cfg, tmp_path, fake, capital=20_000.0, **kw):
    msgs = []
    br = AlpacaBroker(cfg, tmp_path / "state", capital, api=fake, sleep=lambda s: None, log=msgs.append, **kw)
    br.connect(fresh=True)
    br.start_day("2026-11-16")
    br.msgs = msgs
    return br


def test_capital_caps_how_much_of_the_account_is_used(cfg, tmp_path):
    br = _broker(cfg, tmp_path, FakeAlpaca(equity=100_000.0), capital=20_000.0)
    assert br.start_capital == 20_000.0 and br.equity() == 20_000.0
    assert br.sizing_equity() == 20_000.0
    small = _broker(cfg, tmp_path / "b", FakeAlpaca(equity=5_000.0), capital=20_000.0)
    assert small.start_capital == 5_000.0             # cannot use more than the account holds


def test_round_trip_matches_the_brokers_books(cfg, tmp_path):
    fake = FakeAlpaca()
    br = _broker(cfg, tmp_path, fake)
    pos = br.open(_order(), "t0", price=50.0, bid=49.99, ask=50.01)
    assert pos.shares == 100 and fake.pos == {"AAA": 100}
    entry = fake.log[0]
    assert entry["type"] == "limit" and entry["side"] == "buy" and entry["time_in_force"] == "day"
    assert float(entry["limit_price"]) == pytest.approx(50.01 * 1.0005, abs=0.01)   # never far past the ask
    assert pos.entry_price == pytest.approx(float(entry["limit_price"]))
    stop = fake.log[1]
    assert stop["type"] == "stop" and stop["side"] == "sell" and float(stop["stop_price"]) == pytest.approx(49.50)
    assert "AAA" in br.stop_orders

    tr = br.close("AAA", "t1", "signal", price=51.0, bid=50.99, ask=51.01)
    assert fake.pos == {} and not br.positions and not br.stop_orders
    assert fake.orders["o2"]["status"] == "canceled"                    # the resting stop was pulled first
    assert tr["pnl"] == pytest.approx(100_000.0 - 100_000.0 + fake.cash - 100_000.0, abs=0.01)
    assert br.equity() - 20_000.0 == pytest.approx(fake.cash - 100_000.0, abs=0.01)   # ledger == broker


def test_short_round_trip_and_long_only_fallback(cfg, tmp_path):
    fake = FakeAlpaca()
    br = _broker(cfg, tmp_path, fake)
    br.open(_order(side=-1), "t0", price=50.0, bid=49.99, ask=50.01)
    assert fake.pos == {"AAA": -100} and fake.log[1]["side"] == "buy"     # protective buy-stop above
    assert float(fake.log[1]["stop_price"]) == pytest.approx(50.50)
    br.close("AAA", "t1", "signal", price=49.0, bid=48.99, ask=49.01)
    assert fake.pos == {} and br.equity() > 20_000.0

    nos = _broker(cfg, tmp_path / "n", FakeAlpaca(shorting=False))
    assert cfg.strategy.allow_short is False
    assert nos.open(_order(side=-1), "t0", price=50.0) is None


def test_unfilled_entry_is_cancelled_not_chased(cfg, tmp_path):
    fake = FakeAlpaca(fill="none")
    br = _broker(cfg, tmp_path, fake)
    assert br.open(_order(), "t0", price=50.0) is None
    assert fake.orders["o1"]["status"] == "canceled" and not br.positions and fake.pos == {}
    assert br.cash == 20_000.0


def test_partial_entry_books_only_what_filled_and_exit_finishes_at_market(cfg, tmp_path):
    fake = FakeAlpaca(fill="half")
    br = _broker(cfg, tmp_path, fake)
    pos = br.open(_order(), "t0", price=50.0)
    assert pos.shares == 50 and fake.pos == {"AAA": 50}
    assert float(fake.log[1]["qty"]) == 50                               # stop covers what is actually held
    br.close("AAA", "t1", "signal", price=50.0)
    kinds = [o["type"] for o in fake.log[2:]]
    assert kinds == ["limit", "market"] and fake.pos == {} and not br.positions


def test_stop_filled_at_the_broker_is_picked_up(cfg, tmp_path):
    fake = FakeAlpaca()
    br = _broker(cfg, tmp_path, fake)
    br.open(_order(), "t0", price=50.0)
    fake.trigger_stop("AAA", 49.40)
    trades = br.sync("t1")
    assert len(trades) == 1 and trades[0]["reason"] == "stop" and trades[0]["exit_price"] == 49.40
    assert not br.positions and not br.stop_orders and "AAA" in br.cooldown
    assert br.equity() - 20_000.0 == pytest.approx(fake.cash - 100_000.0, abs=0.01)


def test_stop_that_fills_while_we_try_to_exit(cfg, tmp_path):
    fake = FakeAlpaca()
    br = _broker(cfg, tmp_path, fake)
    br.open(_order(), "t0", price=50.0)
    fake.trigger_stop("AAA", 49.45)
    tr = br.close("AAA", "t1", "signal", price=49.4)
    assert tr["reason"] == "stop" and fake.pos == {} and len(fake.log) == 2     # no second sell was sent


def test_never_touches_positions_it_did_not_open(cfg, tmp_path):
    fake = FakeAlpaca()
    fake.pos, fake.last = {"MINE": 10}, {"MINE": 200.0}
    with pytest.raises(BrokerError, match="did not open"):
        _broker(cfg, tmp_path, fake)
    cfg.broker.allow_shared_account = True
    br = _broker(cfg, tmp_path / "s", fake)
    br.open(_order(), "t0", price=50.0)
    br.sync("t1")
    br.close("AAA", "t2", "eod", price=50.0)
    assert fake.pos == {"MINE": 10}
    assert all(o["symbol"] != "MINE" for o in fake.log)


def test_position_closed_by_hand_is_dropped_from_the_ledger(cfg, tmp_path):
    fake = FakeAlpaca()
    br = _broker(cfg, tmp_path, fake)
    br.open(_order(), "t0", price=50.0)
    fake.orders[br.stop_orders["AAA"]]["status"] = "canceled"
    del fake.pos["AAA"]                                   # the user sold it in the Alpaca app
    trades = br.sync("t1")
    assert trades[0]["reason"] == "external" and not br.positions


def test_hard_limits(cfg, tmp_path):
    fake = FakeAlpaca()
    cfg.broker.max_order_notional = 2_000.0
    br = _broker(cfg, tmp_path, fake)
    pos = br.open(_order(shares=500), "t0", price=50.0)
    assert pos.shares == 40                               # $2,000 cap at $50
    (tmp_path / "STOP").write_text("")
    cfg.root = str(tmp_path)
    assert br.open(_order(sym="BBB"), "t0", price=50.0) is None
    assert br.close("AAA", "t1", "eod", price=50.0) is not None      # exits still work with STOP present


def test_state_survives_a_restart(cfg, tmp_path):
    fake = FakeAlpaca()
    br = _broker(cfg, tmp_path, fake)
    br.open(_order(), "t0", price=50.0)
    again = AlpacaBroker.load(cfg, tmp_path / "state", None, api=fake, sleep=lambda s: None, log=lambda m: None)
    again.connect(fresh=False)
    assert again.positions["AAA"].shares == 100 and again.stop_orders == br.stop_orders
    assert again.start_capital == 20_000.0
    again.close("AAA", "t1", "eod", price=50.0)
    assert fake.pos == {}


def test_live_keys_are_never_guessed(monkeypatch):
    for k in ("ALPACA_LIVE_KEY_ID", "ALPACA_LIVE_SECRET_KEY", "ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    assert alpaca_trading_keys(live=False) == ("k", "s")
    assert alpaca_trading_keys(live=True) is None


def test_live_mode_is_refused_unless_enabled_and_confirmed(capsys, monkeypatch):
    from daytrader import cli

    monkeypatch.setattr(cli, "_engine", lambda *a, **k: pytest.fail("must not start"))
    assert cli.main(["run", "--broker", "alpaca-live"]) == 1
    assert "switched off" in capsys.readouterr().out


def test_full_session_through_the_engine_keeps_ledger_and_broker_in_step(cfg, tmp_path):
    pytest.importorskip("sklearn")
    from daytrader.clock import SimClock
    from daytrader.data.store import BarStore
    from daytrader.data.synthetic import ReplayProvider, generate_bars
    from daytrader.engine import Engine
    from daytrader.market_calendar import ET
    from daytrader.report import load_state

    last = date(2026, 11, 13)
    bars = generate_bars(n_symbols=24, n_days=40, signal=3.0, seed=21, end_day=last)
    clock = SimClock(datetime.combine(last, time(9, 20), tzinfo=ET))
    fake = FakeAlpaca(equity=80_000.0)
    cfg.root = str(tmp_path)
    eng = Engine(cfg, ReplayProvider(bars, clock, 5), sorted(bars["symbol"].unique()), clock=clock,
                 trade=True, capital=50_000, store=BarStore(tmp_path / "cache", 5, persist=False),
                 state_dir=tmp_path / "state", model_path=tmp_path / "m.joblib", report_dir=tmp_path / "rep",
                 fresh_account=True, allow_retrain=False, force_train=True, broker_mode="alpaca-paper",
                 broker_api=fake)
    eng.broker_sleep = None
    import daytrader.alpaca_broker as ab
    orig = ab._time.sleep
    ab._time.sleep = lambda s: None
    try:
        eng.run(max_sessions=1)
    finally:
        ab._time.sleep = orig
    state = load_state(tmp_path / "state")
    assert len(state["trades"]) > 10
    assert fake.pos == {} and not eng.broker.positions                 # flat at the broker overnight
    assert not [o for o in fake.orders.values() if o["status"] == "new"]   # no orders left resting
    assert eng.broker.equity() - 50_000.0 == pytest.approx(fake.cash - 80_000.0, abs=0.05)
    assert all(o["type"] in ("limit", "stop", "market") and float(o["qty"]) == int(float(o["qty"])) for o in fake.log)

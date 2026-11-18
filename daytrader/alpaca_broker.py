"""Alpaca brokerage connection: the same account bookkeeping as the simulator,
but every entry and exit is a real order sent to an Alpaca account.

Two modes:
  alpaca-paper  Alpaca's own paper-trading service (fake money, real order plumbing)
  alpaca-live   a funded Alpaca account - REAL MONEY

Safety design:
  * The program keeps its own ledger and trades only what it opened itself. It
    never touches a position it does not recognise, and by default refuses to
    start in an account that already holds other positions.
  * `--capital` caps how much of the account it may use.
  * Orders are whole-share marketable LIMIT orders, so a fill can never be far
    worse than the quote the decision was based on. Unfilled entries are
    cancelled. Exits fall back to a market order so a position is never stranded.
  * Every position gets a stop order resting at Alpaca, so the stop still works
    if this program or the computer dies.
  * Every single order is capped in dollars (broker.max_order_notional).
  * Creating a file named STOP in the project folder blocks all new entries.

API reference: https://docs.alpaca.markets/us/reference/postorder
"""
from __future__ import annotations

import json
import time as _time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .broker import Order, PaperBroker, Position
from .config import Config, alpaca_trading_keys

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
TERMINAL = {"filled", "canceled", "expired", "rejected", "replaced", "done_for_day"}


class BrokerError(RuntimeError):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class AlpacaAPI:
    """Thin HTTPS client for the Alpaca Trading API."""

    def __init__(self, live: bool, session=None):
        keys = alpaca_trading_keys(live)
        if not keys:
            if live:
                raise BrokerError(
                    "Live trading keys not found. Add ALPACA_LIVE_KEY_ID and ALPACA_LIVE_SECRET_KEY to the "
                    ".env file (they are deliberately separate from the data/paper keys).")
            raise BrokerError(
                "Alpaca paper-trading keys not found. Add ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY "
                "(or APCA_API_KEY_ID / APCA_API_SECRET_KEY) to the .env file.")
        self.base = LIVE_URL if live else PAPER_URL
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.headers = {"APCA-API-KEY-ID": keys[0], "APCA-API-SECRET-KEY": keys[1],
                        "Accept": "application/json"}

    def _call(self, method: str, path: str, params: Optional[dict] = None, body: Optional[dict] = None) -> Any:
        # Reads are retried; anything that could create or cancel an order is sent exactly once.
        attempts = 3 if method == "GET" else 1
        last = "no response"
        for i in range(attempts):
            try:
                resp = self.session.request(method, self.base + path, params=params, json=body,
                                            headers=self.headers, timeout=15)
            except Exception as exc:
                last = "%s: %s" % (type(exc).__name__, exc)
                if i + 1 < attempts:
                    _time.sleep(1.0 + i)
                continue
            if resp.status_code in (200, 201, 204, 207):
                if resp.status_code == 204 or not resp.content:
                    return None
                return resp.json()
            try:
                msg = resp.json().get("message", "")
            except Exception:
                msg = (resp.text or "")[:200]
            if resp.status_code in (429, 500, 502, 503, 504) and i + 1 < attempts:
                last = "HTTP %d %s" % (resp.status_code, msg)
                _time.sleep(1.0 + i)
                continue
            if resp.status_code == 401:
                msg = (msg or "unauthorized") + " - check that the keys in .env belong to this kind of account " \
                                                "(paper keys for alpaca-paper, live keys for alpaca-live)"
            raise BrokerError("Alpaca %s %s failed [%d]: %s" % (method, path, resp.status_code, msg),
                              resp.status_code)
        raise BrokerError("Alpaca %s %s failed: %s" % (method, path, last))

    def get(self, path: str, params: Optional[dict] = None) -> Any:
        return self._call("GET", path, params=params)

    def post(self, path: str, body: dict) -> Any:
        return self._call("POST", path, body=body)

    def delete(self, path: str) -> Any:
        return self._call("DELETE", path)


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class AlpacaBroker(PaperBroker):
    is_real = True

    def __init__(self, cfg: Config, state_dir: Optional[Path] = None, capital: Optional[float] = None,
                 live: bool = False, api: Optional[AlpacaAPI] = None,
                 sleep: Callable[[float], None] = _time.sleep, log: Callable[[str], None] = print):
        super().__init__(cfg, state_dir, capital)
        self.live = live
        self.api = api or AlpacaAPI(live)
        self._sleep = sleep
        self._log = log
        self._capital_cap = float(capital) if capital else None
        self.stop_orders: Dict[str, str] = {}       # symbol -> id of its resting stop order
        self.account_equity = 0.0
        self.buying_power = 0.0
        self._warned: set = set()

    # ---- labels ----------------------------------------------------------
    @property
    def name(self) -> str:
        return "Alpaca LIVE account (real money)" if self.live else "Alpaca paper account"

    def _warn(self, msg: str, key: Optional[str] = None) -> None:
        if key is not None:
            if key in self._warned:
                return
            self._warned.add(key)
        self._log("  ! " + msg)

    # ---- connection and reconciliation -----------------------------------
    def refresh_account(self) -> dict:
        acct = self.api.get("/v2/account") or {}
        self.account_equity = _f(acct.get("equity"))
        self.buying_power = _f(acct.get("buying_power"))
        return acct

    def connect(self, fresh: bool) -> dict:
        """Check the account can trade, set the starting ledger, and reconcile positions."""
        acct = self.refresh_account()
        if str(acct.get("status", "")).upper() != "ACTIVE":
            raise BrokerError("The Alpaca account is not active (status: %s)." % acct.get("status"))
        if acct.get("trading_blocked") or acct.get("account_blocked"):
            raise BrokerError("Alpaca reports that trading is blocked on this account.")
        if self.cfg.strategy.allow_short and not acct.get("shorting_enabled", False):
            self.cfg.strategy.allow_short = False
            self._warn("short selling is not enabled on this account: running long-only")
        if fresh:
            cap = self._capital_cap or self.account_equity
            self.start_capital = min(cap, self.account_equity)
            if self.start_capital <= 0:
                raise BrokerError("The Alpaca account has no equity to trade with.")
            self.cash = self.start_capital
            self.day_start_equity = self.start_capital
        self._reconcile("", startup=True)
        return acct

    def _broker_positions(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for p in self.api.get("/v2/positions") or []:
            if p.get("asset_class", "us_equity") != "us_equity":
                continue
            qty = abs(_f(p.get("qty")))
            side = -1 if str(p.get("side", "long")).lower() == "short" else 1
            out[str(p.get("symbol"))] = {"qty": qty, "side": side, "price": _f(p.get("current_price"))}
        return out

    def _reconcile(self, ts: str, startup: bool = False) -> List[dict]:
        """Make the local ledger agree with what Alpaca says is actually held."""
        trades: List[dict] = []
        held = self._broker_positions()
        for sym, pos in list(self.positions.items()):
            b = held.get(sym)
            if b is None or b["side"] != pos.side:
                self._warn("%s is no longer held at Alpaca (closed outside this program?) - removing it from the ledger" % sym)
                tr = self._book_close(sym, pos.shares, pos.last_price, ts, "external")
                self.stop_orders.pop(sym, None)
                if tr:
                    trades.append(tr)
            elif int(b["qty"]) != pos.shares:
                self._warn("%s: ledger says %d shares, Alpaca says %d - using Alpaca's number"
                           % (sym, pos.shares, int(b["qty"])))
                if int(b["qty"]) < pos.shares:
                    tr = self._book_close(sym, pos.shares - int(b["qty"]), pos.last_price, ts, "external")
                    if tr:
                        trades.append(tr)
                # more shares than we opened: the extra ones are not ours; leave them alone
        foreign = sorted(s for s in held if s not in self.positions)
        if foreign:
            if startup and not self.cfg.broker.allow_shared_account:
                raise BrokerError(
                    "This Alpaca account already holds positions this program did not open (%s). Use a "
                    "dedicated account, close them, or set broker.allow_shared_account: true. The program "
                    "never touches positions it does not recognise." % ", ".join(foreign[:10]))
            self._warn("positions not opened by this program are being left alone: %s" % ", ".join(foreign[:10]),
                       key="foreign:" + ",".join(foreign))
        return trades

    def sync(self, ts: str = "") -> List[dict]:
        """Pick up stop orders that filled at the broker, then reconcile holdings."""
        trades: List[dict] = []
        for sym, oid in list(self.stop_orders.items()):
            try:
                o = self.api.get("/v2/orders/%s" % oid) or {}
            except BrokerError as exc:
                self._warn("could not check the stop order for %s (%s)" % (sym, exc))
                continue
            trades += self._absorb_stop(sym, o, ts)
        try:
            trades += self._reconcile(ts)
            self.refresh_account()
        except BrokerError as exc:
            self._warn("could not reconcile with Alpaca (%s)" % exc)
        return trades

    def _absorb_stop(self, sym: str, o: dict, ts: str) -> List[dict]:
        out: List[dict] = []
        status = str(o.get("status", ""))
        filled = int(_f(o.get("filled_qty")))
        if status in TERMINAL:
            self.stop_orders.pop(sym, None)
            if filled >= 1 and sym in self.positions:
                tr = self._book_close(sym, filled, _f(o.get("filled_avg_price")), ts, "stop")
                if tr:
                    out.append(tr)
        return out

    # ---- order plumbing --------------------------------------------------
    def sizing_equity(self) -> float:
        # never size against more than the real account is worth
        eq = self.equity()
        return min(eq, self.account_equity) if self.account_equity > 0 else eq

    def _kill_switch(self) -> bool:
        return self.cfg.path("STOP").exists()

    def _await(self, order_id: str, timeout: float) -> dict:
        waited, o = 0.0, {}
        while True:
            try:
                o = self.api.get("/v2/orders/%s" % order_id) or {}
            except BrokerError as exc:
                self._warn("order status check failed (%s)" % exc)
            if str(o.get("status", "")) in TERMINAL or waited >= timeout:
                return o
            self._sleep(0.5)
            waited += 0.5

    def _cancel(self, order_id: str) -> dict:
        """Cancel and return the order's final state (it may have filled in the meantime)."""
        try:
            self.api.delete("/v2/orders/%s" % order_id)
        except BrokerError:
            pass        # already filled or already cancelled: the status check below tells us which
        return self._await(order_id, 5.0)

    def _submit(self, symbol: str, is_buy: bool, qty: int, limit: Optional[float], timeout: float) -> tuple:
        """Send one order and wait. Returns (filled shares, average fill price)."""
        body = {"symbol": symbol, "qty": str(int(qty)), "side": "buy" if is_buy else "sell",
                "time_in_force": "day", "client_order_id": "mldt-" + uuid.uuid4().hex[:20]}
        if limit is not None:
            body.update({"type": "limit", "limit_price": "%.2f" % limit})
        else:
            body["type"] = "market"
        o = self.api.post("/v2/orders", body) or {}
        oid = o.get("id")
        if not oid:
            raise BrokerError("Alpaca accepted the order but returned no order id.")
        o = self._await(oid, timeout)
        if str(o.get("status", "")) not in TERMINAL:
            o = self._cancel(oid)
        filled = int(_f(o.get("filled_qty")))
        return filled, _f(o.get("filled_avg_price"))

    def _limit_price(self, is_buy: bool, ref: float, bid: Optional[float], ask: Optional[float]) -> Optional[float]:
        b = self.cfg.broker
        if b.order_type != "limit":
            return None
        c = self.cfg.costs
        good = bid and ask and ask >= bid > 0 and (ask - bid) / ((ask + bid) / 2) * 1e4 <= c.max_quote_spread_bps
        base = (ask if is_buy else bid) if good else ref
        px = base * (1 + b.limit_offset_bps / 1e4) if is_buy else base * (1 - b.limit_offset_bps / 1e4)
        return round(px, 2)

    def _place_stop(self, pos: Position) -> None:
        if not self.cfg.broker.resting_stops:
            return
        body = {"symbol": pos.symbol, "qty": str(pos.shares), "side": "sell" if pos.side > 0 else "buy",
                "type": "stop", "stop_price": "%.2f" % pos.stop, "time_in_force": "day",
                "client_order_id": "mldt-stop-" + uuid.uuid4().hex[:16]}
        try:
            o = self.api.post("/v2/orders", body) or {}
            if o.get("id"):
                self.stop_orders[pos.symbol] = o["id"]
        except BrokerError as exc:
            self._warn("could not place the resting stop for %s (%s); the program will still watch the stop itself"
                       % (pos.symbol, exc))

    # ---- entries and exits -----------------------------------------------
    def open(self, order: Order, ts: str, price: Optional[float] = None,
             bid: Optional[float] = None, ask: Optional[float] = None) -> Optional[Position]:
        if order.symbol in self.positions or order.shares < 1:
            return None
        if self._kill_switch():
            self._warn("STOP file present: no new positions will be opened", key="stopfile")
            return None
        if order.side < 0 and not self.cfg.strategy.allow_short:
            return None
        ref = float(price if price else order.ref_price)
        if ref <= 0:
            return None
        shares = min(int(order.shares), int(self.cfg.broker.max_order_notional // ref))
        if self.buying_power > 0:
            shares = min(shares, int(0.95 * self.buying_power // ref))
        if shares < 1:
            self._warn("%s skipped: order would exceed the per-order cap or the account's buying power" % order.symbol)
            return None
        is_buy = order.side > 0
        try:
            filled, avg = self._submit(order.symbol, is_buy, shares, self._limit_price(is_buy, ref, bid, ask),
                                       self.cfg.broker.fill_timeout_seconds)
        except BrokerError as exc:
            self._warn("%s order not placed (%s)" % (order.symbol, exc))
            return None
        if filled < 1 or avg <= 0:
            return None
        self.buying_power = max(self.buying_power - filled * avg, 0.0)
        pos = self._book_open(order, filled, avg, ref, ts)
        self._place_stop(pos)
        self.save()
        return pos

    def close(self, symbol: str, ts: str, reason: str, price: Optional[float] = None,
              bid: Optional[float] = None, ask: Optional[float] = None) -> Optional[dict]:
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        last_trade: Optional[dict] = None
        oid = self.stop_orders.get(symbol)
        if oid:                                   # free the shares held by the resting stop
            for tr in self._absorb_stop(symbol, self._cancel(oid), ts):
                last_trade = tr
            self.stop_orders.pop(symbol, None)
            pos = self.positions.get(symbol)
            if pos is None:                       # the stop had already filled
                self.save()
                return last_trade
        ref = float(price if price else pos.last_price)
        is_buy = pos.side < 0
        remaining, got, value = pos.shares, 0, 0.0
        timeout = self.cfg.broker.fill_timeout_seconds
        # first a marketable limit, then a market order for whatever is left
        for limit in (self._limit_price(is_buy, ref, bid, ask), None):
            if remaining < 1:
                break
            try:
                filled, avg = self._submit(symbol, is_buy, remaining, limit, timeout)
            except BrokerError as exc:
                self._warn("%s exit order failed (%s)" % (symbol, exc))
                filled, avg = 0, 0.0
            if filled >= 1 and avg > 0:
                got += filled
                value += filled * avg
                remaining -= filled
            if limit is None:
                break
        if got >= 1:
            last_trade = self._book_close(symbol, got, value / got, ts, reason) or last_trade
        if remaining >= 1 and symbol in self.positions:
            self._warn("%s: %d share(s) could not be closed and are STILL HELD at Alpaca - check the account"
                       % (symbol, remaining))
            self._place_stop(self.positions[symbol])
        self.save()
        return last_trade

    def cancel_resting_orders(self) -> None:
        for sym, oid in list(self.stop_orders.items()):
            self._cancel(oid)
            self.stop_orders.pop(sym, None)

    # ---- persistence -----------------------------------------------------
    def save(self) -> None:
        super().save()
        if self.dir:
            (self.dir / "alpaca.json").write_text(json.dumps(
                {"mode": "alpaca-live" if self.live else "alpaca-paper", "stop_orders": self.stop_orders}))

    @classmethod
    def load(cls, cfg: Config, state_dir: Path, capital: Optional[float] = None, **kw) -> "AlpacaBroker":
        broker = cls(cfg, state_dir, capital, **kw)
        base = PaperBroker.load(cfg, state_dir, capital)
        for attr in ("start_capital", "cash", "day", "day_start_equity", "halted", "cooldown",
                     "traded_notional", "fees", "day_trades", "positions"):
            setattr(broker, attr, getattr(base, attr))
        extra = Path(state_dir) / "alpaca.json"
        if extra.exists() and extra.stat().st_size > 0:
            try:
                broker.stop_orders = dict(json.loads(extra.read_text()).get("stop_orders", {}))
            except Exception:
                pass
        return broker

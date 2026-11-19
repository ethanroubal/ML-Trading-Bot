"""Paper broker: a simulated account with realistic fills.

No real orders are ever sent anywhere. Buys fill at the ask (or at the
last price plus an assumed half-spread), sells at the bid, and every fill
pays extra slippage - so the simulated P&L is deliberately a little
pessimistic rather than flattering.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .costs import minute_of, one_way_cost_bps

TRADE_FIELDS = ["date", "symbol", "side", "shares", "entry_time", "entry_price", "exit_time",
                "exit_price", "pnl", "ret_bps", "bars_held", "reason", "predicted_bps"]
EQUITY_FIELDS = ["time", "equity", "cash", "gross", "net", "positions"]
DAILY_FIELDS = ["date", "start_equity", "end_equity", "pnl", "ret_pct", "trades", "wins",
                "gross_pnl_wins", "gross_pnl_losses", "traded_notional"]


@dataclass
class Position:
    symbol: str
    side: int               # +1 long, -1 short
    shares: int
    entry_price: float      # actual fill, costs included
    entry_time: str
    stop: float
    target: float           # 0 = none
    predicted_bps: float
    sigH_bps: float
    last_price: float
    bars_held: int = 0
    beta: float = 1.0

    @property
    def value(self) -> float:
        return self.side * self.shares * self.last_price

    @property
    def unrealized(self) -> float:
        return self.side * self.shares * (self.last_price - self.entry_price)


@dataclass
class Order:
    symbol: str
    side: int
    shares: int
    ref_price: float
    stop_frac: float
    target_frac: float
    predicted_bps: float
    edge_bps: float
    sigH_bps: float
    beta: float = 1.0


class PaperBroker:
    def __init__(self, cfg: Config, state_dir: Optional[Path] = None, capital: Optional[float] = None):
        self.cfg = cfg
        self.dir = Path(state_dir) if state_dir else None
        self.start_capital = float(capital if capital is not None else cfg.strategy.capital)
        self.cash = self.start_capital
        self.positions: Dict[str, Position] = {}
        self.cooldown: Dict[str, int] = {}
        self.day: str = ""
        self.day_start_equity = self.start_capital
        self.halted = False
        self.trades: List[dict] = []         # closed trades (this process)
        self.day_trades: List[dict] = []
        self.traded_notional = 0.0
        self.fees = 0.0
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    # ---- account maths ---------------------------------------------------
    def equity(self) -> float:
        return self.cash + sum(p.value for p in self.positions.values())

    def gross(self) -> float:
        return sum(abs(p.value) for p in self.positions.values())

    def net(self) -> float:
        return sum(p.value for p in self.positions.values())

    def net_beta(self) -> float:
        """Market exposure in dollars: each position weighted by its beta."""
        return sum(p.value * p.beta for p in self.positions.values())

    def day_pnl(self) -> float:
        return self.equity() - self.day_start_equity

    def mark(self, prices: Dict[str, float]) -> None:
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is not None and px > 0:
                pos.last_price = float(px)

    # ---- fills -----------------------------------------------------------
    quote_spreads = True     # fill at the live bid/ask when one is given (off when quotes are not full-market)

    def _fill_price(self, is_buy: bool, ref: float, bid: Optional[float], ask: Optional[float],
                    symbol: Optional[str] = None, ts: Optional[str] = None) -> float:
        c = self.cfg.costs
        if self.quote_spreads and bid and ask and ask >= bid > 0 and (ask - bid) / ((ask + bid) / 2) * 1e4 <= c.max_quote_spread_bps:
            base = ask if is_buy else bid
            cost = c.slippage_bps
        else:       # no usable quote (backtests, IEX-only quotes): the spread model, measured where possible
            base = ref
            minute = minute_of(ts) if ts else None
            cost = float(one_way_cost_bps(ref, c, symbols=symbol if minute is not None else None, minute=minute))
        return base * (1 + cost / 1e4) if is_buy else base * (1 - cost / 1e4)

    def open(self, order: Order, ts: str, price: Optional[float] = None,
             bid: Optional[float] = None, ask: Optional[float] = None) -> Optional[Position]:
        if order.symbol in self.positions or order.shares < 1:
            return None
        ref = float(price if price else order.ref_price)
        fill = self._fill_price(order.side > 0, ref, bid, ask, order.symbol, ts)
        return self._book_open(order, int(order.shares), fill, ref, ts)

    def close(self, symbol: str, ts: str, reason: str, price: Optional[float] = None,
              bid: Optional[float] = None, ask: Optional[float] = None) -> Optional[dict]:
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        ref = float(price if price else pos.last_price)
        fill = self._fill_price(pos.side < 0, ref, bid, ask, symbol, ts)
        return self._book_close(symbol, pos.shares, fill, ts, reason)

    # ---- bookkeeping shared by the simulated and the real-broker accounts ----
    def _book_open(self, order: Order, shares: int, fill: float, ref: float, ts: str) -> Position:
        fee = self.cfg.costs.commission_per_share * shares
        self.cash -= order.side * shares * fill + fee
        self.fees += fee
        self.traded_notional += shares * fill
        pos = Position(
            symbol=order.symbol, side=order.side, shares=int(shares), entry_price=fill,
            entry_time=ts, stop=ref * (1 - order.side * order.stop_frac),
            target=ref * (1 + order.side * order.target_frac) if order.target_frac > 0 else 0.0,
            predicted_bps=order.predicted_bps, sigH_bps=order.sigH_bps, last_price=ref,
            beta=float(order.beta),
        )
        self.positions[order.symbol] = pos
        return pos

    def _book_close(self, symbol: str, shares: int, fill: float, ts: str, reason: str) -> Optional[dict]:
        """Record the exit of `shares` (all or part of the position) at `fill`."""
        pos = self.positions.get(symbol)
        if pos is None or shares < 1:
            return None
        shares = min(int(shares), pos.shares)
        fee = self.cfg.costs.commission_per_share * shares
        self.cash += pos.side * shares * fill - fee
        self.fees += fee
        self.traded_notional += shares * fill
        pnl = pos.side * shares * (fill - pos.entry_price) - 2 * fee
        trade = {
            "date": self.day, "symbol": symbol, "side": "LONG" if pos.side > 0 else "SHORT",
            "shares": shares, "entry_time": pos.entry_time, "entry_price": round(pos.entry_price, 4),
            "exit_time": ts, "exit_price": round(fill, 4), "pnl": round(pnl, 2),
            "ret_bps": round(pnl / (shares * pos.entry_price) * 1e4, 2),
            "bars_held": pos.bars_held, "reason": reason, "predicted_bps": round(pos.predicted_bps, 2),
        }
        if shares >= pos.shares:
            del self.positions[symbol]
        else:
            pos.shares -= shares
        self.trades.append(trade)
        self.day_trades.append(trade)
        if reason == "stop":
            self.cooldown[symbol] = self.cfg.strategy.cooldown_bars
        self._append_csv("trades.csv", TRADE_FIELDS, trade)
        return trade

    def sizing_equity(self) -> float:
        """Equity the strategy sizes positions against."""
        return self.equity()

    def sync(self, ts: str = "") -> List[dict]:
        """Reconcile with the outside world. Nothing to do for a simulated account."""
        return []

    def stop_or_target_hit(self, pos: Position, low: float, high: float,
                           open_px: Optional[float] = None) -> Optional[tuple]:
        """Given the price range seen since the last check: (reason, reference price) or None.

        If the price gapped straight through a level, the fill reference is the
        first price seen (the open), not the level itself.
        """
        first = open_px if open_px is not None else (high if pos.side > 0 else low)
        if pos.side > 0:
            if low <= pos.stop:
                return "stop", min(pos.stop, first)
            if pos.target and high >= pos.target:
                return "target", max(pos.target, first) if open_px is not None else pos.target
        else:
            if high >= pos.stop:
                return "stop", max(pos.stop, first)
            if pos.target and low <= pos.target:
                return "target", min(pos.target, first) if open_px is not None else pos.target
        return None

    def end_bar(self) -> None:
        for pos in self.positions.values():
            pos.bars_held += 1
        for sym in list(self.cooldown):
            self.cooldown[sym] -= 1
            if self.cooldown[sym] <= 0:
                del self.cooldown[sym]

    def daily_loss_breached(self) -> bool:
        limit = self.cfg.strategy.max_daily_loss_pct
        return limit > 0 and self.day_pnl() <= -limit * self.day_start_equity

    # ---- day bookkeeping -------------------------------------------------
    def start_day(self, day: str) -> None:
        if day == self.day:
            return
        self.day = day
        self.day_start_equity = self.equity()
        self.halted = False
        self.cooldown = {}
        self.day_trades = []
        self.traded_notional = 0.0

    def day_summary(self) -> dict:
        wins = [t["pnl"] for t in self.day_trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in self.day_trades if t["pnl"] <= 0]
        eq = self.equity()
        return {
            "date": self.day, "start_equity": round(self.day_start_equity, 2), "end_equity": round(eq, 2),
            "pnl": round(eq - self.day_start_equity, 2),
            "ret_pct": round((eq / self.day_start_equity - 1) * 100, 4) if self.day_start_equity else 0.0,
            "trades": len(self.day_trades), "wins": len(wins),
            "gross_pnl_wins": round(sum(wins), 2), "gross_pnl_losses": round(sum(losses), 2),
            "traded_notional": round(self.traded_notional, 2),
        }

    def record_day(self) -> dict:
        row = self.day_summary()
        if self.dir:
            path = self.dir / "daily.csv"
            rows = []
            if path.exists() and path.stat().st_size > 0:
                with open(path, newline="") as fh:
                    rows = [r for r in csv.DictReader(fh) if r.get("date") != row["date"]]
            rows.append(row)
            rows.sort(key=lambda r: r["date"])
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=DAILY_FIELDS)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in DAILY_FIELDS})
            os.replace(tmp, path)
        return row

    def log_equity(self, ts: str) -> None:
        eq = self.equity()
        self._append_csv("equity.csv", EQUITY_FIELDS, {
            "time": ts, "equity": round(eq, 2), "cash": round(self.cash, 2),
            "gross": round(self.gross(), 2), "net": round(self.net(), 2), "positions": len(self.positions),
        })

    # ---- persistence -----------------------------------------------------
    def _append_csv(self, name: str, fields: List[str], row: dict) -> None:
        if not self.dir:
            return
        path = self.dir / name
        new = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})

    def save(self) -> None:
        if not self.dir:
            return
        state = {
            "start_capital": self.start_capital, "cash": self.cash, "day": self.day,
            "day_start_equity": self.day_start_equity, "halted": self.halted,
            "cooldown": self.cooldown, "traded_notional": self.traded_notional, "fees": self.fees,
            "positions": [asdict(p) for p in self.positions.values()],
            "day_trades": self.day_trades,
        }
        path = self.dir / "portfolio.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, path)

    @classmethod
    def load(cls, cfg: Config, state_dir: Path, capital: Optional[float] = None) -> "PaperBroker":
        """Resume the saved account, or open a new one with `capital`."""
        broker = cls(cfg, state_dir, capital)
        path = Path(state_dir) / "portfolio.json"
        if path.exists() and path.stat().st_size > 0:
            st = json.loads(path.read_text())
            broker.start_capital = float(st["start_capital"])
            broker.cash = float(st["cash"])
            broker.day = st.get("day", "")
            broker.day_start_equity = float(st.get("day_start_equity", broker.cash))
            broker.halted = bool(st.get("halted", False))
            broker.cooldown = {k: int(v) for k, v in st.get("cooldown", {}).items()}
            broker.traded_notional = float(st.get("traded_notional", 0.0))
            broker.fees = float(st.get("fees", 0.0))
            broker.day_trades = list(st.get("day_trades", []))
            for p in st.get("positions", []):
                broker.positions[p["symbol"]] = Position(**p)
        return broker

    @staticmethod
    def exists(state_dir: Path) -> bool:
        p = Path(state_dir) / "portfolio.json"
        return p.exists() and p.stat().st_size > 0

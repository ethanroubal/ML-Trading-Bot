"""From predictions to decisions.

The rule is simple on purpose: act only when the model's expected move,
after shrinkage, is bigger than the full round-trip cost of trading it -
then size by risk, keep the book diversified, cut losers at a volatility
stop, and be flat before the close.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .broker import Order, PaperBroker
from .config import Config, parse_hhmm
from .costs import one_way_cost_bps
from .market_calendar import CLOSE_MIN


@dataclass
class Signals:
    """Per-symbol view at one decision bar (arrays aligned with `symbols`)."""
    symbols: List[str]
    price: np.ndarray
    exp_bps: np.ndarray        # expected move over the horizon (+ = up), NaN if unknown
    sigH_bps: np.ndarray       # expected volatility over the horizon
    cost_bps: np.ndarray       # round-trip cost
    edge_bps: np.ndarray       # |expected move| - round-trip cost
    ok: np.ndarray             # tradable right now
    beta: Optional[np.ndarray] = None   # market beta per symbol (None = treat as 1)

    def index(self) -> Dict[str, int]:
        return {s: i for i, s in enumerate(self.symbols)}


def make_signals(symbols: List[str], price: np.ndarray, exp_z: np.ndarray, sigH: np.ndarray,
                 usable: np.ndarray, cfg: Config, beta: Optional[np.ndarray] = None) -> Signals:
    price = np.asarray(price, dtype=float)
    sigH_bps = np.asarray(sigH, dtype=float) * 1e4
    exp_bps = np.asarray(exp_z, dtype=float) * sigH_bps
    cost = 2.0 * one_way_cost_bps(np.where(np.isfinite(price), price, 1.0), cfg.costs)
    ok = np.asarray(usable, dtype=bool) & np.isfinite(exp_bps) & np.isfinite(price) & (price >= cfg.strategy.min_price)
    edge = np.where(ok, np.abs(exp_bps) - cost, np.nan)
    return Signals(list(symbols), price, np.where(ok, exp_bps, np.nan), sigH_bps, cost, edge, ok, beta)


def session_gates(cfg: Config, close_min: int = CLOSE_MIN) -> Tuple[int, int, int]:
    """(first entry, last entry, flatten) as minutes after midnight ET, adjusted for half days."""
    s = cfg.strategy
    shift = CLOSE_MIN - close_min            # 180 on a 1pm-close day
    return parse_hhmm(s.entry_start), parse_hhmm(s.entry_end) - shift, parse_hhmm(s.flatten_time) - shift


def position_notional(equity: float, sigH_bps: float, cfg: Config) -> float:
    s = cfg.strategy
    stop_frac = max(s.stop_sigma * sigH_bps / 1e4, 1e-4)
    return min(equity * s.risk_per_trade_pct / stop_frac, equity * s.max_position_pct)


def plan(sig: Signals, broker: PaperBroker, cfg: Config, can_enter: bool) -> Tuple[List[Tuple[str, str]], List[Order]]:
    """Decide exits and entries for this bar. Returns ([(symbol, reason)], [orders])."""
    s = cfg.strategy
    idx = sig.index()
    exits: List[Tuple[str, str]] = []
    max_hold = s.max_hold_bars or 3 * cfg.model.horizon_bars
    for sym, pos in broker.positions.items():
        i = idx.get(sym)
        if pos.bars_held >= max_hold:
            exits.append((sym, "time"))
        elif i is not None and np.isfinite(sig.exp_bps[i]) and pos.bars_held >= s.min_hold_bars \
                and pos.side * sig.exp_bps[i] < s.exit_edge_bps:
            exits.append((sym, "signal"))
    orders: List[Order] = []
    if not can_enter or broker.halted:
        return exits, orders

    exiting = {sym for sym, _ in exits}
    held = {sym: p for sym, p in broker.positions.items() if sym not in exiting}
    equity = broker.sizing_equity()
    if equity <= 0:
        return exits, orders
    gross = sum(abs(p.value) for p in held.values())
    net = sum(p.value * p.beta for p in held.values())       # beta-weighted market exposure
    net_cap = (s.max_net_exposure if s.allow_short else max(s.max_net_exposure, s.max_gross_exposure)) * equity
    slots = s.max_positions - len(held)
    if slots <= 0:
        return exits, orders

    cand = np.flatnonzero(sig.ok & (sig.edge_bps >= s.min_edge_bps))
    scored = []
    for i in cand:
        sym = sig.symbols[i]
        if sym in broker.positions or sym in broker.cooldown:
            continue
        side = 1 if sig.exp_bps[i] > 0 else -1
        if side < 0 and not s.allow_short:
            continue
        notional = position_notional(equity, sig.sigH_bps[i], cfg)
        scored.append((notional * sig.edge_bps[i], i, side, notional))
    scored.sort(reverse=True)

    for _, i, side, notional in scored:
        if slots <= 0:
            break
        beta = float(sig.beta[i]) if sig.beta is not None else 1.0
        room = s.max_gross_exposure * equity - gross
        # keep the book's market exposure inside +/- net_cap so the result reflects
        # stock selection (what the model predicts), not the market's direction
        room = min(room, ((net_cap - net) if side > 0 else (net_cap + net)) / beta)
        size = min(notional, room)
        if size < 0.25 * notional:
            continue
        shares = int(size // sig.price[i])
        if shares < 1:
            continue
        orders.append(Order(
            symbol=sig.symbols[i], side=side, shares=shares, ref_price=float(sig.price[i]),
            stop_frac=s.stop_sigma * sig.sigH_bps[i] / 1e4,
            target_frac=s.target_sigma * sig.sigH_bps[i] / 1e4,
            predicted_bps=float(sig.exp_bps[i]), edge_bps=float(sig.edge_bps[i]),
            sigH_bps=float(sig.sigH_bps[i]), beta=beta,
        ))
        value = shares * sig.price[i]
        gross += value
        net += side * value * beta
        slots -= 1
    return exits, orders


def recommendations(sig: Signals, cfg: Config, model=None, top_n: Optional[int] = None) -> List[dict]:
    """Ranked buy and sell/short ideas for display (strongest first on each side)."""
    n = top_n or cfg.strategy.top_n
    s = cfg.strategy
    out: List[dict] = []
    valid = np.flatnonzero(sig.ok)
    if len(valid) == 0:
        return out
    order = valid[np.argsort(-sig.exp_bps[valid])]
    picks = [(i, 1) for i in order[:n] if sig.exp_bps[i] > 0]
    picks += [(i, -1) for i in order[::-1][:n] if sig.exp_bps[i] < 0]
    horizon_min = cfg.model.horizon_bars * cfg.data.interval_min
    for i, side in picks:
        exp = float(sig.exp_bps[i])
        z = exp / sig.sigH_bps[i] if sig.sigH_bps[i] > 0 else 0.0
        px = float(sig.price[i])
        stop = px * (1 - side * s.stop_sigma * sig.sigH_bps[i] / 1e4)
        actionable = bool(sig.edge_bps[i] >= s.min_edge_bps) and (side > 0 or s.allow_short)
        out.append({
            "symbol": sig.symbols[i],
            "action": ("BUY" if side > 0 else "SELL/SHORT") if actionable else "watch",
            "side": side, "price": round(px, 2),
            "expected_bps": round(exp, 1), "cost_bps": round(float(sig.cost_bps[i]), 1),
            "net_edge_bps": round(float(sig.edge_bps[i]), 1),
            "hit_rate": round(float(model.p_win(np.array([z]))[0]), 3) if model is not None else None,
            "strength": int(model.strength(np.array([z]))[0]) if model is not None else None,
            "stop": round(stop, 2), "horizon_min": horizon_min, "actionable": actionable,
        })
    return out

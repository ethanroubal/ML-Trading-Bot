"""Trading-cost model shared by training diagnostics, the backtest and the paper broker.

Per side, a trade pays half the bid/ask spread plus slippage. The half spread is
measured per stock and time of day when a spread table is attached (see
spreads.py and costs.spread_source); otherwise, or where the table has too
little data, it falls back to the assumed model: max(min_half_spread_bps, half
a cent).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import CostCfg


def attach_book(costs: CostCfg, book) -> None:
    """Use a measured spread table for every cost computed with these settings."""
    costs.__dict__["_book"] = book


def book_of(costs: CostCfg):
    return costs.__dict__.get("_book") if costs.spread_source == "measured" else None


def assumed_half_spread_bps(price, costs: CostCfg) -> np.ndarray:
    price = np.asarray(price, dtype=float)
    return np.maximum(costs.min_half_spread_bps, costs.tick_half_spread / np.maximum(price, 0.01) * 1e4)


def half_spread_bps(price, costs: CostCfg, symbols=None, minute=None) -> np.ndarray:
    """Half the bid/ask spread in bp: measured where possible, assumed elsewhere."""
    assumed = assumed_half_spread_bps(price, costs)
    book = book_of(costs)
    if book is None or symbols is None or minute is None:
        return assumed
    price = np.asarray(price, dtype=float)
    measured = 0.5 * book.spread_bps(symbols, minute, price)
    floor = np.maximum(costs.measured_min_half_spread_bps, costs.tick_half_spread / np.maximum(price, 0.01) * 1e4)
    return np.where(np.isfinite(measured), np.maximum(measured, floor), assumed)


def one_way_cost_bps(price, costs: CostCfg, symbols=None, minute: Optional[object] = None):
    """Cost of crossing the spread plus slippage, per side, in basis points."""
    return half_spread_bps(price, costs, symbols, minute) + costs.slippage_bps


def round_trip_cost_bps(price, costs: CostCfg, symbols=None, minute=None):
    return 2.0 * one_way_cost_bps(price, costs, symbols, minute)


def minute_of(stamp) -> Optional[int]:
    """Minutes after midnight ET from 'YYYY-MM-DD HH:MM[...]' or a datetime (None if unknown)."""
    try:
        if hasattr(stamp, "hour"):
            return int(stamp.hour) * 60 + int(stamp.minute)
        hh, mm = str(stamp).split(" ")[1].split(":")[:2]
        return int(hh) * 60 + int(mm)
    except Exception:
        return None

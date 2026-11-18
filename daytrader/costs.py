"""Trading-cost model shared by training diagnostics, the backtest and the paper broker."""
from __future__ import annotations

import numpy as np

from .config import CostCfg


def one_way_cost_bps(price, costs: CostCfg):
    """Assumed cost of crossing the spread plus slippage, per side, in basis points."""
    price = np.asarray(price, dtype=float)
    half_spread = np.maximum(costs.min_half_spread_bps, costs.tick_half_spread / np.maximum(price, 0.01) * 1e4)
    return half_spread + costs.slippage_bps


def round_trip_cost_bps(price, costs: CostCfg):
    return 2.0 * one_way_cost_bps(price, costs)

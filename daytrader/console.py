"""Terminal output helpers (plain ANSI, no dependencies)."""
from __future__ import annotations

import os
import sys
from typing import List

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return "\033[%sm%s\033[0m" % (code, text) if _COLOR else text


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def money(v: float) -> str:
    return "${:,.2f}".format(v)


def signed_money(v: float) -> str:
    s = ("+" if v >= 0 else "-") + "${:,.2f}".format(abs(v))
    return green(s) if v > 0 else red(s) if v < 0 else s


def signed_pct(v: float, digits: int = 2) -> str:
    s = ("%+." + str(digits) + "f%%") % v
    return green(s) if v > 0 else red(s) if v < 0 else s


def rule(title: str = "", width: int = 78) -> str:
    if not title:
        return dim("-" * width)
    pad = max(width - len(title) - 4, 2)
    return dim("-- ") + bold(title) + " " + dim("-" * pad)


def recommendations_table(recs: List[dict]) -> List[str]:
    """Two blocks: buys, then sells/shorts."""
    lines: List[str] = []
    if not recs:
        return [dim("  no usable signals this bar")]
    head = "  %-11s %-6s %9s %9s %7s %9s %6s %9s" % (
        "action", "symbol", "price", "expected", "cost", "net edge", "hit", "stop")
    for side, title in ((1, "Buy candidates"), (-1, "Sell / short candidates")):
        rows = [r for r in recs if r["side"] == side]
        if not rows:
            continue
        lines.append("  " + bold(title) + dim("  (next %d min, relative to the market)" % rows[0]["horizon_min"]))
        lines.append(dim(head))
        for r in rows:
            act = r["action"]
            label = "%-11s" % act
            if r["actionable"]:
                label = green(label) if side > 0 else red(label)
            else:
                label = dim(label)
            hit = "%5.1f%%" % (100 * r["hit_rate"]) if r.get("hit_rate") is not None else "    -"
            lines.append("  %s %-6s %9.2f %+8.1fbp %5.1fbp %+7.1fbp %s %9.2f" % (
                label, r["symbol"], r["price"], r["expected_bps"], r["cost_bps"], r["net_edge_bps"], hit, r["stop"]))
    return lines


def duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return "%dd %dh" % (d, h)
    if h:
        return "%dh %02dm" % (h, m)
    return "%dm" % m

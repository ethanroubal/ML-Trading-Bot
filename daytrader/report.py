"""Performance reporting: reads the paper account's logs and writes an HTML dashboard."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .backtest import performance_stats

_HERE = Path(__file__).resolve().parent


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_state(state_dir: Path) -> Dict[str, Any]:
    state_dir = Path(state_dir)
    out: Dict[str, Any] = {
        "daily": _read_csv(state_dir / "daily.csv"),
        "trades": _read_csv(state_dir / "trades.csv"),
        "equity": _read_csv(state_dir / "equity.csv"),
        "portfolio": {},
        "signals": {},
    }
    for key, name in (("portfolio", "portfolio.json"), ("signals", "latest_signals.json")):
        p = state_dir / name
        if p.exists() and p.stat().st_size > 0:
            try:
                out[key] = json.loads(p.read_text())
            except Exception:
                pass
    return out


def account_stats(state: Dict[str, Any], default_capital: float) -> Dict[str, Any]:
    port = state.get("portfolio") or {}
    start = float(port.get("start_capital", default_capital))
    daily, trades = state["daily"], state["trades"]
    return performance_stats(daily, trades, start)


def _downsample(rows: List[dict], limit: int = 700) -> List[dict]:
    if len(rows) <= limit:
        return rows
    step = len(rows) / float(limit)
    idx = sorted({int(i * step) for i in range(limit)} | {len(rows) - 1})
    return [rows[i] for i in idx]


def build_payload(title: str, subtitle: str, daily: pd.DataFrame, trades: pd.DataFrame,
                  equity: pd.DataFrame, start_capital: float, current_equity: Optional[float] = None,
                  positions: Optional[List[dict]] = None, signals: Optional[dict] = None,
                  model: Optional[dict] = None, note: str = "", day_pnl: Optional[float] = None) -> dict:
    stats = performance_stats(daily, trades, start_capital)
    eq_now = float(current_equity) if current_equity is not None else float(stats.get("end_equity", start_capital))
    eq_rows: List[dict] = []
    if len(equity):
        for t, v in zip(equity["time"].astype(str), equity["equity"].astype(float)):
            eq_rows.append({"t": t[:16], "v": round(float(v), 2)})
    elif len(daily):
        for t, v in zip(daily["date"].astype(str), daily["end_equity"].astype(float)):
            eq_rows.append({"t": t, "v": round(float(v), 2)})
    daily_rows = []
    if len(daily):
        for r in daily.to_dict("records"):
            daily_rows.append({
                "date": str(r["date"]), "pnl": float(r["pnl"]), "ret": float(r["ret_pct"]),
                "end": float(r["end_equity"]), "trades": int(r["trades"]), "wins": int(r["wins"]),
            })
    trade_rows = []
    if len(trades):
        for r in trades.tail(60).iloc[::-1].to_dict("records"):
            trade_rows.append({
                "exit": str(r["exit_time"])[:16], "symbol": str(r["symbol"]), "side": str(r["side"]),
                "shares": int(r["shares"]), "entry": float(r["entry_price"]), "exitpx": float(r["exit_price"]),
                "pnl": float(r["pnl"]), "bps": float(r["ret_bps"]), "reason": str(r["reason"]),
                "pred": float(r["predicted_bps"]),
            })
    return {
        "title": title, "subtitle": subtitle, "note": note,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start": float(start_capital), "equity_now": eq_now, "day_pnl": day_pnl,
        "stats": {k: v for k, v in stats.items() if k != "exit_reasons"},
        "equity": _downsample(eq_rows), "daily": daily_rows, "trades": trade_rows,
        "positions": positions or [], "signals": signals or {}, "model": model or {},
    }


def render_html(payload: dict, refresh_seconds: int = 0) -> str:
    template = (_HERE / "dashboard_template.html").read_text()
    meta = '<meta http-equiv="refresh" content="%d">' % refresh_seconds if refresh_seconds > 0 else ""
    data = json.dumps(payload, default=str).replace("</", "<\\/")
    return template.replace("__REFRESH__", meta).replace("__DATA__", data)


def write_dashboard(state_dir: Path, out_path: Path, default_capital: float, title: str,
                    subtitle: str, model_meta: Optional[dict] = None, note: str = "",
                    refresh_seconds: int = 30) -> Path:
    state = load_state(state_dir)
    port = state.get("portfolio") or {}
    start = float(port.get("start_capital", default_capital))
    positions, cur_eq, day_pnl = [], None, None
    if port:
        cur_eq = float(port.get("cash", start))
        for p in port.get("positions", []):
            val = p["side"] * p["shares"] * p["last_price"]
            cur_eq += val
            positions.append({
                "symbol": p["symbol"], "side": "LONG" if p["side"] > 0 else "SHORT", "shares": p["shares"],
                "entry": p["entry_price"], "last": p["last_price"], "stop": p["stop"],
                "pnl": p["side"] * p["shares"] * (p["last_price"] - p["entry_price"]),
                "since": str(p["entry_time"])[:16],
            })
        day_pnl = cur_eq - float(port.get("day_start_equity", start))
    payload = build_payload(title, subtitle, state["daily"], state["trades"], state["equity"], start,
                            current_equity=cur_eq, positions=positions, signals=state.get("signals"),
                            model=model_meta, note=note, day_pnl=day_pnl)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(payload, refresh_seconds))
    return out_path


def write_backtest_report(result, out_dir: Path, start_capital: float, model_meta: dict, note: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(out_dir / "backtest_daily.csv", index=False)
    result.trades.to_csv(out_dir / "backtest_trades.csv", index=False)
    eq = result.equity.copy()
    if len(eq):
        eq["time"] = pd.to_datetime(eq["time"]).dt.strftime("%Y-%m-%d %H:%M")
    payload = build_payload("ML Day Trader - walk-forward backtest",
                            "Every day shown was traded by a model that had never seen it",
                            result.daily, result.trades, eq, start_capital, model=model_meta, note=note)
    path = out_dir / "backtest.html"
    path.write_text(render_html(payload, 0))
    return path

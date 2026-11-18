"""The live loop.

Every time a bar closes during market hours the engine:
  1. pulls the newest bars for the whole universe,
  2. recomputes features and asks the model for each stock's expected move,
  3. prints ranked buy / sell recommendations,
  4. (autonomous mode) manages the paper account: exits, new entries, sizing.
Between bars it polls prices for open positions so stops trigger promptly.
After the close it records the day's result and retrains the model.

The same engine runs on a simulated clock for `replay` and `demo`.
"""
from __future__ import annotations

import csv
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import console as con
from .backtest import format_stats, performance_stats
from .broker import PaperBroker
from .clock import RealClock
from .config import Config
from .data.base import DataProvider, Quote, drop_incomplete
from .data.store import BarStore
from .features import compute_features
from .market_calendar import ET, UTC, MarketCalendar, Session
from .model import Model, NotEnoughData, format_report, make_dataset, train
from .panel import build_panel
from .report import load_state, write_dashboard
from .strategy import make_signals, plan, recommendations, session_gates

DEFAULT_HISTORY_DAYS = {"yahoo": 58, "alpaca": 180, "replay": 3650}
SIGNAL_FIELDS = ["time", "action", "symbol", "price", "expected_bps", "cost_bps", "net_edge_bps",
                 "hit_rate", "stop"]


class Engine:
    def __init__(self, cfg: Config, provider: DataProvider, symbols: List[str], clock=None,
                 calendar: Optional[MarketCalendar] = None, trade: bool = True,
                 capital: Optional[float] = None, store: Optional[BarStore] = None,
                 state_dir: Optional[Path] = None, model_path: Optional[Path] = None,
                 report_dir: Optional[Path] = None, fresh_account: bool = False,
                 allow_retrain: bool = True, force_train: bool = False, label: str = "", note: str = "",
                 broker_mode: str = "sim", broker_api=None):
        self.cfg = cfg
        self.provider = provider
        self.symbols = list(symbols)
        self.active: List[str] = []
        self.clock = clock or RealClock()
        self.cal = calendar or MarketCalendar(cfg.calendar.extra_holidays, cfg.calendar.extra_early_closes)
        self.trade = trade
        self.capital = capital
        self.store = store or BarStore(cfg.bar_cache_dir(provider.name), cfg.data.interval_min)
        self.state_dir = Path(state_dir) if state_dir else cfg.state_dir
        self.model_path = Path(model_path) if model_path else cfg.model_dir / "model.joblib"
        self.report_dir = Path(report_dir) if report_dir else cfg.report_dir
        self.fresh_account = fresh_account
        self.allow_retrain = allow_retrain
        self.force_train = force_train
        self.label = label
        self.note = note
        self.broker_mode = broker_mode      # sim | alpaca-paper | alpaca-live
        self.broker_api = broker_api        # injectable Alpaca client (tests)
        self.model: Optional[Model] = None
        self.broker: Optional[PaperBroker] = None
        self.last_recs: List[dict] = []
        self._warned: set = set()
        self._dead: set = set()        # symbols the data source has nothing for (this run)

    # ------------------------------------------------------------ utilities
    def log(self, msg: str = "") -> None:
        print(msg, flush=True)

    def _now_str(self) -> str:
        return self.clock.now().strftime("%H:%M:%S")

    def _stamp(self, dt: Optional[datetime] = None) -> str:
        return (dt or self.clock.now()).strftime("%Y-%m-%d %H:%M:%S")

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.log(con.yellow("  ! " + msg))

    def quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        if not symbols:
            return {}
        try:
            return self.provider.latest_quotes(list(dict.fromkeys(symbols)))
        except Exception as exc:
            self.log(con.yellow("  ! could not fetch quotes (%s)" % exc))
            return {}

    # ------------------------------------------------------------- history
    def sync_history(self, verbose: bool = True) -> None:
        """Bring the local bar cache up to date."""
        iv = self.cfg.data.interval_min
        now = self.clock.now()
        days = self.cfg.data.history_days or DEFAULT_HISTORY_DAYS.get(self.provider.name, 58)
        days = min(days, self.provider.max_history_days)
        target_start = now - timedelta(days=days)
        full, lasts = [], {}
        for s in self.symbols:
            last = self.store.last_ts(s)
            if last is None:
                if s not in self._dead:
                    full.append(s)
            else:
                lasts[s] = last
        # symbols that are up to date together, and stragglers (so one stale ticker
        # does not force weeks of re-downloading for everything else)
        groups = []
        if lasts:
            newest = max(lasts.values())
            current = [s for s, t in lasts.items() if newest - t <= pd.Timedelta(days=4)]
            stale = [s for s in lasts if s not in set(current)]
            groups = [g for g in (current, stale) if g]
        added = 0
        if full:
            if verbose and self.provider.name == "replay":
                self.log("  Loading history for %d symbols..." % len(full))
            elif verbose:
                self.log("  Downloading %d days of %d-minute bars for %d symbols from %s..."
                         % (days, iv, len(full), self.provider.name))
            df = drop_incomplete(self.provider.fetch_bars(full, target_start, None, iv), iv, now)
            added += self.store.append(df)
            got = set(df["symbol"].unique()) if len(df) else set()
            retry = [s for s in full if s not in got]
            if retry and len(retry) < len(full):
                if not getattr(self.clock, "simulated", False):
                    self.clock.sleep(5)      # transient rate limits usually clear quickly: one more try
                df = drop_incomplete(self.provider.fetch_bars(retry, target_start, None, iv), iv, now)
                added += self.store.append(df)
                got |= set(df["symbol"].unique()) if len(df) else set()
                self._dead |= {s for s in retry if s not in got}   # don't keep asking for these
        for tail in groups:
            oldest = min(lasts[s] for s in tail)
            start = max(oldest.to_pydatetime(), target_start.astimezone(UTC))
            if (now - start).total_seconds() >= 2 * iv * 60:      # a newer bar can have completed
                df = drop_incomplete(self.provider.fetch_bars(tail, start, None, iv), iv, now)
                rebased = self._handle_rebased(df, target_start, now)
                if rebased:
                    df = df[~df["symbol"].isin(rebased)]
                added += self.store.append(df)
        self.active = [s for s in self.symbols if self.store.last_ts(s) is not None]
        missing = [s for s in self.symbols if s not in set(self.active)]
        if missing:
            self._warn_once("missing:" + ",".join(missing),
                            "no data for %d symbol(s), skipping them: %s"
                            % (len(missing), ", ".join(missing[:12]) + ("..." if len(missing) > 12 else "")))
        if not self.active:
            raise RuntimeError("No market data could be downloaded. Check your internet connection"
                               " and data provider settings.")
        if verbose:
            first = min(self.store.first_ts(s) for s in self.active)
            last = max(self.store.last_ts(s) for s in self.active)
            self.log("  Bar cache: %d symbols, %s to %s"
                     % (len(self.active), first.tz_convert(ET).strftime("%Y-%m-%d"),
                        last.tz_convert(ET).strftime("%Y-%m-%d %H:%M ET")))

    def _handle_rebased(self, df: pd.DataFrame, target_start: datetime, now: datetime) -> List[str]:
        """Re-download symbols whose price history changed scale (stock splits)."""
        rebased = self.store.conflicts(df)
        if not rebased:
            return []
        self.log(con.yellow("  ! price history changed scale (split?) for %s - re-downloading"
                            % ", ".join(rebased)))
        iv = self.cfg.data.interval_min
        for sym in rebased:
            self.store.drop(sym)
        try:
            fresh = drop_incomplete(self.provider.fetch_bars(rebased, target_start, None, iv), iv, now)
            self.store.append(fresh)
        except Exception as exc:
            self.log(con.yellow("  ! re-download failed (%s)" % exc))
        return rebased

    def _latest_complete_day(self):
        """Most recent session in the cache that has finished."""
        now = self.clock.now()
        last = max(self.store.last_ts(s) for s in self.active).tz_convert(ET)
        day = last.date()
        sess = self.cal.session(day)
        if sess is not None and now < sess.close and day == now.date():
            day = self.cal.previous_trading_day(day)
        return day

    # --------------------------------------------------------------- model
    def train_model(self, reason: str = ""):
        self.log(con.rule("Training model" + (" (%s)" % reason if reason else "")))
        bars = self.store.load(self.active)
        panel = build_panel(bars, self.cfg.data.interval_min, self.cal, asof=self.clock.now(),
                            symbols=self.active)
        feats = compute_features(panel, self.cfg)
        ds = make_dataset(panel, feats, self.cfg)
        res = train(ds, panel, self.cfg, verbose=True, provider_name=self.provider.name)
        res.model.save(self.model_path)
        self.model = res.model
        self.log(format_report(res.report))
        return res, panel, feats

    def ensure_model(self) -> None:
        model = self.model or Model.load(self.model_path)
        reason = ""
        if model is None:
            reason = "no saved model"
        else:
            mt = model.meta
            if (mt.get("interval_min") != self.cfg.data.interval_min
                    or mt.get("horizon_bars") != self.cfg.model.horizon_bars
                    or mt.get("target") != self.cfg.model.target):
                reason = "settings changed"
            elif mt.get("provider") not in (None, "", "replay") and self.provider.name != "replay" \
                    and mt.get("provider") != self.provider.name:
                reason = "data source changed from %s to %s" % (mt.get("provider"), self.provider.name)
            elif self.allow_retrain:
                try:
                    end = datetime.strptime(mt["train_end"], "%Y-%m-%d").date()
                    latest = self._latest_complete_day()
                    new_days = [d for d in self.cal.trading_days_between(end + timedelta(days=1), latest)]
                    if len(new_days) >= max(1, self.cfg.model.retrain_every_days):
                        reason = "%d new session(s) of data" % len(new_days)
                except Exception:
                    reason = "model metadata unreadable"
        if reason:
            self.train_model(reason)
        else:
            self.model = model
            mt = model.meta
            self.log("  Model: trained through %s, out-of-sample rank IC %+.4f (t-stat %.1f)%s"
                     % (mt["train_end"], mt["oos_rank_ic"], mt["oos_ic_tstat"],
                        "" if mt.get("has_edge") else con.yellow("  - no reliable edge detected")))

    # -------------------------------------------------------------- broker
    def _open_account(self) -> None:
        if not self.trade:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self.fresh_account:
            for name in ("portfolio.json", "trades.csv", "equity.csv", "daily.csv", "signals.csv",
                         "latest_signals.json"):
                p = self.state_dir / name
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        p.write_text("")
        port = self.state_dir / "portfolio.json"
        if self.broker_mode != "sim":
            from .alpaca_broker import AlpacaBroker

            existing = port.exists() and port.stat().st_size > 0
            br = AlpacaBroker.load(self.cfg, self.state_dir, self.capital, live=self.broker_mode == "alpaca-live",
                                   api=self.broker_api, log=self.log)
            br.connect(fresh=not existing)
            br.save()
            self.broker = br
            self.log("  Connected to the %s: account equity %s, buying power %s"
                     % (con.bold(br.name), con.money(br.account_equity), con.money(br.buying_power)))
            self.log("  This program's ledger: %s equity (%s with %s)"
                     % (con.money(br.equity()), "resumed, started" if existing else "starting",
                        con.money(br.start_capital)))
            if br.stop_orders or br.positions:
                self.log("  Resumed %d open position(s) with %d resting stop order(s)"
                         % (len(br.positions), len(br.stop_orders)))
            return
        if port.exists() and port.stat().st_size > 0:
            self.broker = PaperBroker.load(self.cfg, self.state_dir, self.capital)
            self.log("  Paper account resumed: equity %s (started with %s)"
                     % (con.money(self.broker.equity()), con.money(self.broker.start_capital)))
            if self.capital and abs(self.capital - self.broker.start_capital) > 0.005:
                self.log(con.yellow("  ! --capital ignored: an account already exists. Run the 'reset' command"
                                    " to start over with a new amount."))
        else:
            self.broker = PaperBroker(self.cfg, self.state_dir, self.capital)
            self.broker.save()
            self.log("  Paper account opened with %s of simulated money" % con.money(self.broker.start_capital))

    def _mark_and_stops(self, quotes: Dict[str, Quote], fallback: Optional[Dict[str, float]] = None) -> List[dict]:
        """Mark positions to the latest prices; close any that hit a stop or target."""
        br = self.broker
        closed: List[dict] = []
        if fallback:
            br.mark(fallback)
        br.mark({s: q.price for s, q in quotes.items()})
        for sym, pos in list(br.positions.items()):
            q = quotes.get(sym)
            if q is None:
                continue
            hit = br.stop_or_target_hit(pos, q.price, q.price)
            if hit:
                tr = br.close(sym, self._stamp(), hit[0], price=q.price, bid=q.bid, ask=q.ask)
                if tr:
                    closed.append(tr)
        return closed

    def _flatten(self, reason: str, quotes: Optional[Dict[str, Quote]] = None) -> List[dict]:
        br = self.broker
        if not br.positions:
            return []
        quotes = quotes if quotes is not None else self.quotes(list(br.positions))
        out = []
        for sym in list(br.positions):
            q = quotes.get(sym)
            tr = br.close(sym, self._stamp(), reason, price=q.price if q else None,
                          bid=q.bid if q else None, ask=q.ask if q else None)
            if tr:
                out.append(tr)
        return out

    def _print_trade(self, tr: dict) -> None:
        verb = "SOLD" if tr["side"] == "LONG" else "COVERED"
        self.log("  %s %s %d %s @ %.2f  (%s)  P&L %s"
                 % (con.bold("<<"), verb, tr["shares"], tr["symbol"], tr["exit_price"], tr["reason"],
                    con.signed_money(tr["pnl"])))

    def _print_account(self) -> None:
        br = self.broker
        eq = br.equity()
        day = br.day_pnl()
        pct = day / br.day_start_equity * 100 if br.day_start_equity else 0.0
        self.log("  Account: equity %s  today %s (%s)  cash %s  %d open  gross %.0f%%  market exposure %+.0f%%%s"
                 % (con.bold(con.money(eq)), con.signed_money(day), con.signed_pct(pct), con.money(br.cash),
                    len(br.positions), br.gross() / eq * 100 if eq else 0, br.net_beta() / eq * 100 if eq else 0,
                    con.red("  [halted: daily loss limit]") if br.halted else ""))
        for pos in br.positions.values():
            self.log(con.dim("    %-5s %-5s %5d @ %8.2f  now %8.2f  ") % (
                "LONG" if pos.side > 0 else "SHORT", pos.symbol, pos.shares, pos.entry_price, pos.last_price)
                + con.signed_money(pos.unrealized))

    # ----------------------------------------------------------- one cycle
    def refresh_bars(self, sess: Session, b: datetime, wait: bool) -> float:
        """Pull today's bars; return the share of symbols that have the bar ending at `b`."""
        iv = self.cfg.data.interval_min
        target = pd.Timestamp(b - timedelta(minutes=iv)).tz_convert("UTC")
        # start from the previous session so the response is never empty right at the open
        prev = self.cal.session(self.cal.previous_trading_day(sess.day))
        start = (prev.open if prev else sess.open).astimezone(UTC)
        attempts = 3 if wait else 1
        frac = 0.0
        for i in range(attempts):
            try:
                df = drop_incomplete(self.provider.fetch_bars(self.active, start, None, iv), iv, self.clock.now())
                rebased = self.store.conflicts(df)
                if rebased:      # leave these alone until the next full sync re-downloads them
                    self._warn_once("rebased:" + ",".join(rebased),
                                    "price history changed scale for %s - ignoring until the next data sync"
                                    % ", ".join(rebased))
                    df = df[~df["symbol"].isin(rebased)]
                self.store.append(df)
            except Exception as exc:
                self.log(con.yellow("  ! data refresh failed (%s)" % exc))
            have = sum(1 for s in self.active if self.store.last_ts(s) == target)
            frac = have / max(len(self.active), 1)
            if frac >= 0.3 or i == attempts - 1:      # the new bar has landed for a meaningful share
                break
            self.clock.sleep(self.cfg.engine.max_bar_wait_seconds / attempts)
        return frac

    def on_bar(self, sess: Session, b: datetime, trade: bool) -> None:
        cfg = self.cfg
        iv = cfg.data.interval_min
        fresh = self.refresh_bars(sess, b, wait=trade)
        window_start = b - timedelta(days=int(cfg.engine.feature_window_days * 1.6) + 6)
        bars = self.store.load(self.active, start=window_start)
        try:
            panel = build_panel(bars, iv, self.cal, asof=b, symbols=self.active,
                                last_n_days=cfg.engine.feature_window_days)
        except ValueError:
            self.log("[%s ET] no completed bars available yet" % self._now_str())
            return
        bar_end = (panel.ts[-1] + pd.Timedelta(minutes=iv)).to_pydatetime()
        if panel.dates[-1] != sess.day or bar_end != b:
            self.log("[%s ET] waiting for today's data (latest bar: %s)"
                     % (self._now_str(), panel.ts[-1].strftime("%Y-%m-%d %H:%M")))
            return
        feats = compute_features(panel, cfg, with_labels=True)
        if not self.model.compatible(cfg, feats.names):
            self.train_model("feature set changed")
        t = panel.T - 1
        exp_z = self.model.expected_z(feats.row(t))
        sig = make_signals(panel.symbols, panel.C[t], exp_z, feats.sigH[t], feats.usable[t], cfg,
                           beta=feats.beta(t))
        recs = recommendations(sig, cfg, self.model)
        self.last_recs = recs
        g_start, g_end, g_flat = session_gates(cfg, sess.close.hour * 60 + sess.close.minute)
        b_min = b.hour * 60 + b.minute
        mt = self.model.meta
        head = "[%s ET] bar %s-%s  %d/%d symbols fresh  model through %s (IC %+.3f, t %.1f)" % (
            self._now_str(), (b - timedelta(minutes=iv)).strftime("%H:%M"), b.strftime("%H:%M"),
            int(round(fresh * len(self.active))), len(self.active), mt["train_end"], mt["oos_rank_ic"],
            mt["oos_ic_tstat"])
        self.log("")
        self.log(con.cyan(head))
        if b_min < g_flat:
            for line in con.recommendations_table(recs):
                self.log(line)
            if recs and not any(r["actionable"] for r in recs):
                self.log(con.dim("  Nothing clears the cost hurdle right now: the best ideas are shown as 'watch'."))

        if self.trade and trade:
            br = self.broker
            closes = {s: float(panel.C[t][i]) for i, s in enumerate(panel.symbols) if s in br.positions}
            for tr in br.sync(self._stamp()):      # real broker: stops filled there, manual changes
                self._print_trade(tr)
            quotes = self.quotes(list(br.positions))
            for tr in self._mark_and_stops(quotes, closes):
                self._print_trade(tr)
            br.end_bar()
            if b_min >= g_flat:
                for tr in self._flatten("eod", quotes):
                    self._print_trade(tr)
            elif not br.halted and br.daily_loss_breached():
                for tr in self._flatten("loss-limit", quotes):
                    self._print_trade(tr)
                br.halted = True
                self.log(con.red("  Daily loss limit reached: flat and standing aside until tomorrow."))
            else:
                can_enter = (g_start <= b_min <= g_end) and fresh >= 0.3
                if g_start <= b_min <= g_end and fresh < 0.3:
                    self.log(con.yellow("  ! fresh data for only %.0f%% of symbols - no new entries this bar" % (fresh * 100)))
                exits, orders = plan(sig, br, cfg, can_enter)
                q2 = self.quotes([o.symbol for o in orders]) if orders else {}
                for sym, reason in exits:
                    q = quotes.get(sym)
                    tr = br.close(sym, self._stamp(), reason, price=q.price if q else None,
                                  bid=q.bid if q else None, ask=q.ask if q else None)
                    if tr:
                        self._print_trade(tr)
                for od in orders:
                    q = q2.get(od.symbol)
                    if q is None:
                        continue
                    # skip if the live price has already run away from the bar we decided on
                    if abs(q.price / od.ref_price - 1.0) * 1e4 > 1.5 * od.sigH_bps:
                        self.log(con.dim("  (skipped %s: price moved too far since the bar closed)" % od.symbol))
                        continue
                    pos = br.open(od, self._stamp(), price=q.price, bid=q.bid, ask=q.ask)
                    if pos:
                        self.log("  %s %s %d %s @ %.2f  expected %+.1fbp, net edge %+.1fbp, stop %.2f"
                                 % (con.bold(">>"), con.green("BOUGHT") if od.side > 0 else con.red("SHORTED"),
                                    pos.shares, pos.symbol, pos.entry_price, od.predicted_bps, od.edge_bps, pos.stop))
            br.log_equity(self._stamp())
            br.save()
            self._print_account()
        self._save_signals(b, recs)
        self._dashboard()

    def poll(self, sess: Session) -> None:
        """Between bars: watch open positions for stops, targets, the flatten time and the loss limit."""
        br = self.broker
        if not br or not br.positions:
            return
        changed = False
        for tr in br.sync(self._stamp()):
            self.log("[%s ET]" % self._now_str())
            self._print_trade(tr)
            changed = True
        if not br.positions:
            br.save()
            return
        quotes = self.quotes(list(br.positions))
        for tr in self._mark_and_stops(quotes):
            self.log("[%s ET]" % self._now_str())
            self._print_trade(tr)
            changed = True
        now = self.clock.now()
        _, _, g_flat = session_gates(self.cfg, sess.close.hour * 60 + sess.close.minute)
        if br.positions and now.hour * 60 + now.minute >= g_flat:
            for tr in self._flatten("eod", quotes):
                self._print_trade(tr)
                changed = True
        elif br.positions and not br.halted and br.daily_loss_breached():
            for tr in self._flatten("loss-limit", quotes):
                self._print_trade(tr)
            br.halted = True
            changed = True
            self.log(con.red("  Daily loss limit reached: flat and standing aside until tomorrow."))
        if changed:
            br.log_equity(self._stamp())
        br.save()

    def _save_signals(self, b: datetime, recs: List[dict]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        stamp = b.strftime("%Y-%m-%d %H:%M ET")
        latest = self.state_dir / "latest_signals.json"
        if recs or not latest.exists():      # keep the last real list on screen after the close
            latest.write_text(json.dumps({"time": stamp, "recs": recs}))
        path = self.state_dir / "signals.csv"
        new = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SIGNAL_FIELDS)
            if new:
                w.writeheader()
            for r in recs:
                row = {k: r.get(k, "") for k in SIGNAL_FIELDS}
                row["time"] = stamp
                w.writerow(row)

    def _dashboard(self) -> None:
        if not self.cfg.engine.dashboard:
            return
        try:
            title = "ML Day Trader" + (" - " + self.label if self.label else "")
            kind = {"alpaca-paper": "Autonomous trading - Alpaca paper account",
                    "alpaca-live": "Autonomous trading - LIVE Alpaca account (real money)"}.get(
                        self.broker_mode, "Autonomous paper trading")
            sub = (kind if self.trade else "Recommendations only") + \
                  " · %d symbols · data: %s" % (len(self.active), self.provider.name)
            write_dashboard(self.state_dir, self.report_dir / "dashboard.html", self.cfg.strategy.capital,
                            title, sub, model_meta=self.model.meta if self.model else None, note=self.note,
                            refresh_seconds=0 if getattr(self.clock, "simulated", False) else 30)
        except Exception as exc:  # the dashboard must never take the trader down
            self._warn_once("dash", "dashboard not updated (%s)" % exc)

    # ------------------------------------------------------------ sessions
    def run_session(self, sess: Session) -> None:
        cfg = self.cfg
        iv = cfg.data.interval_min
        delay = cfg.engine.bar_delay_seconds
        _, _, g_flat = session_gates(cfg, sess.close.hour * 60 + sess.close.minute)
        self.log(con.rule("Session %s  (%s - %s ET)" % (sess.day.strftime("%a %Y-%m-%d"),
                                                         sess.open.strftime("%H:%M"), sess.close.strftime("%H:%M"))))
        if self.trade:
            br = self.broker
            if br.positions and br.day != str(sess.day):
                self.log(con.yellow("  ! closing %d position(s) left over from %s" % (len(br.positions), br.day)))
                for tr in self._flatten("carryover"):
                    self._print_trade(tr)
            br.start_day(str(sess.day))
            br.save()
        n = sess.minutes // iv
        boundaries = [sess.open + timedelta(minutes=iv * i) for i in range(1, n + 1)]
        self.provider.fast_fail = True     # during the session a late answer is worse than none
        try:
            self._session_loop(sess, boundaries, delay, g_flat)
        finally:
            self.provider.fast_fail = False
        remaining = (sess.close - self.clock.now()).total_seconds()
        if remaining > 0:
            self.clock.sleep(remaining)

    def _session_loop(self, sess: Session, boundaries: List[datetime], delay: float, g_flat: int) -> None:
        cfg = self.cfg
        now = self.clock.now()
        past = [b for b in boundaries
                if (now - (b + timedelta(seconds=delay))).total_seconds() > cfg.engine.max_bar_wait_seconds]
        if past:
            self.log(con.dim("  Joined mid-session: showing the latest completed bar; trading starts at the next bar."))
            self._safe_bar(sess, past[-1], trade=False)
        for b in boundaries:
            due = b + timedelta(seconds=delay)
            if (self.clock.now() - due).total_seconds() > cfg.engine.max_bar_wait_seconds:
                continue
            while True:
                remaining = (due - self.clock.now()).total_seconds()
                if remaining <= 0:
                    break
                self.clock.sleep(min(cfg.engine.poll_seconds, remaining))
                if self.trade and self.broker.positions and (due - self.clock.now()).total_seconds() > 1:
                    try:
                        self.poll(sess)
                    except Exception as exc:
                        self.log(con.yellow("  ! position check failed (%s)" % exc))
            self._safe_bar(sess, b, trade=True)
            if b.hour * 60 + b.minute >= g_flat:
                break

    def _safe_bar(self, sess: Session, b: datetime, trade: bool) -> None:
        try:
            self.on_bar(sess, b, trade)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.log(con.red("  ! error while processing the %s bar: %s" % (b.strftime("%H:%M"), exc)))
            self.log(con.dim(traceback.format_exc(limit=4)))

    def end_of_day(self, sess: Session) -> None:
        if self.trade:
            br = self.broker
            for tr in self._flatten("eod"):
                self._print_trade(tr)
            br.log_equity(self._stamp())
            row = br.record_day()
            br.save()
            self.log("")
            self.log(con.rule("Day summary %s" % sess.day))
            self.log("  P&L %s (%s)   %d trades, %d winners   equity %s"
                     % (con.signed_money(row["pnl"]), con.signed_pct(row["ret_pct"], 3), row["trades"],
                        row["wins"], con.bold(con.money(row["end_equity"]))))
            state = load_state(self.state_dir)
            st = performance_stats(state["daily"], state["trades"], br.start_capital)
            if st.get("days", 0) > 1:
                self.log(format_stats(st, "Since inception"))
        if not getattr(self.clock, "simulated", False):
            self.clock.sleep(90)       # let the data vendor finalise the closing bars
        try:
            self.sync_history(verbose=False)
        except Exception as exc:
            self.log(con.yellow("  ! end-of-day data refresh failed (%s)" % exc))
        if self.allow_retrain:
            try:
                self.ensure_model()
            except NotEnoughData as exc:
                self.log(con.yellow("  ! retrain skipped: %s" % exc))
            except Exception as exc:
                self.log(con.yellow("  ! retrain failed, keeping the current model (%s)" % exc))
        self._dashboard()
        if self.cfg.engine.dashboard:
            self.log(con.dim("  Dashboard: %s" % (self.report_dir / "dashboard.html")))

    def wait_for_open(self, sess: Session) -> None:
        now = self.clock.now()
        self.log("")
        self.log("Market is closed. Next session: %s %s ET (in %s). Waiting - press Ctrl+C to stop."
                 % (sess.day.strftime("%a %Y-%m-%d"), sess.open.strftime("%H:%M"),
                    con.duration((sess.open - now).total_seconds())))
        prep = sess.open - timedelta(minutes=10)
        slept = False
        jump = getattr(self.clock, "jump", None)      # simulated clocks skip the closed hours instantly
        while self.clock.now() < prep:
            wait = (prep - self.clock.now()).total_seconds()
            if jump:
                jump(wait)
            else:
                self.clock.sleep(min(300.0, wait))
            slept = True
        if slept:   # refresh data and the model shortly before the bell
            try:
                self.sync_history(verbose=False)
                self.ensure_model()
            except Exception as exc:
                self.log(con.yellow("  ! pre-open refresh failed (%s)" % exc))
        remaining = (sess.open - self.clock.now()).total_seconds()
        if remaining > 0:
            self.clock.sleep(remaining)

    # ---------------------------------------------------------------- main
    def startup(self) -> None:
        cfg = self.cfg
        mode = "AUTONOMOUS paper trading" if self.trade else "recommendations only"
        if self.trade and self.broker_mode == "alpaca-paper":
            mode = "AUTONOMOUS trading in an Alpaca PAPER account"
        elif self.trade and self.broker_mode == "alpaca-live":
            mode = con.red("AUTONOMOUS trading in a LIVE Alpaca account - REAL MONEY")
        self.log(con.rule("ML Day Trader" + (" - " + self.label if self.label else "")))
        self.log("  Mode: %s   Data: %s   Universe: %d symbols   Bars: %d min   Horizon: %d min"
                 % (con.bold(mode), self.provider.name, len(self.symbols), cfg.data.interval_min,
                    cfg.model.horizon_bars * cfg.data.interval_min))
        if self.note:
            self.log(con.yellow("  " + self.note))
        self.sync_history()
        if self.force_train:
            self.train_model("fresh model for this run")
        else:
            self.ensure_model()
        self._open_account()
        self._dashboard()
        if self.cfg.engine.dashboard:
            self.log("  Dashboard: %s  (open it in a browser; it refreshes itself)"
                     % (self.report_dir / "dashboard.html"))

    def run(self, max_sessions: Optional[int] = None) -> None:
        self.startup()
        done = 0
        while True:
            now = self.clock.now()
            sess = self.cal.next_session(now)
            if now < sess.open:
                self.wait_for_open(sess)
            self.run_session(sess)
            self.end_of_day(sess)
            done += 1
            if max_sessions and done >= max_sessions:
                break

    def shutdown(self) -> None:
        if self.broker:
            self.broker.save()
            self._dashboard()
            self.log("")
            self.log("Stopped. Account saved: equity %s, %d open position(s)."
                     % (con.money(self.broker.equity()), len(self.broker.positions)))
            if self.broker_mode != "sim" and self.broker.positions:
                self.log(con.red("  ! %d position(s) are STILL OPEN at Alpaca and nothing is watching them now, apart"
                                 % len(self.broker.positions)))
                self.log(con.red("    from their resting stop orders (which expire at today's close). To close them:"))
                self.log(con.red("    ./trader flatten --broker %s" % self.broker_mode))

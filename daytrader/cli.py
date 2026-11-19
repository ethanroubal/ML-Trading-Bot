"""Command line interface:  python -m daytrader <command>"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, time

from .market_calendar import UTC
from pathlib import Path
from typing import List, Optional

from . import __version__
from . import console as con
from .config import Config, ConfigError, load_config

ROOT = Path(__file__).resolve().parent.parent


def _apply_alpaca_args(args) -> None:
    """--alpaca-key / --alpaca-secret: Alpaca data (and paper-trading) keys for this run only.

    They take priority over .env and the environment, and are never written to disk or printed.
    Live-trading keys are deliberately NOT accepted here: they are only read from .env.
    """
    key = getattr(args, "alpaca_key", None)
    secret = getattr(args, "alpaca_secret", None)
    if not key and not secret:
        return
    if secret and not key:
        raise ConfigError("--alpaca-secret needs --alpaca-key as well.")
    if not secret:
        if not sys.stdin.isatty():
            raise ConfigError("--alpaca-key was given without --alpaca-secret, and there is no terminal to ask for it.")
        import getpass

        secret = getpass.getpass("Alpaca secret key (not shown as you type): ").strip()
    key, secret = key.strip(), secret.strip()
    if not key or not secret:
        raise ConfigError("Both the Alpaca key and secret must be non-empty.")
    for name in ("APCA_API_KEY_ID", "ALPACA_PAPER_KEY_ID"):
        os.environ[name] = key
    for name in ("APCA_API_SECRET_KEY", "ALPACA_PAPER_SECRET_KEY"):
        os.environ[name] = secret


def _load(args) -> Config:
    cfg = load_config(args.config, root=str(ROOT))
    _apply_alpaca_args(args)          # after .env is read, so the command line wins
    if getattr(args, "provider", None):
        cfg.data.provider = args.provider
    if getattr(args, "universe", None):
        u = args.universe
        cfg.data.universe = [s for s in u.replace(",", " ").split()] if ("," in u or " " in u or u.isupper()) else u
    if getattr(args, "capital", None):
        cfg.strategy.capital = float(args.capital)
    return cfg


def _live_setup(cfg: Config):
    from .data import get_provider, resolve_provider_name
    from .universe import check_universe, resolve_universe

    symbols = check_universe(resolve_universe(cfg.data.universe, resolve_provider_name(cfg), cfg.root))
    return get_provider(cfg), symbols


def _broker_dirs(cfg: Config, mode: str):
    """Each kind of account keeps its own ledger and dashboard."""
    if mode == "sim":
        return cfg.state_dir, cfg.report_dir
    tag = mode.replace("-", "_")
    return cfg.path("state_" + tag), cfg.report_dir / tag


def _engine(cfg: Config, trade: bool, capital: Optional[float] = None, broker_mode: str = "sim"):
    from .engine import Engine

    provider, symbols = _live_setup(cfg)
    state_dir, report_dir = _broker_dirs(cfg, broker_mode)
    return Engine(cfg, provider, symbols, trade=trade, capital=capital, broker_mode=broker_mode,
                  state_dir=state_dir, report_dir=report_dir)


LIVE_PHRASE = "TRADE REAL MONEY"


def _confirm_live(cfg: Config, capital: Optional[float]) -> bool:
    """Live trading needs a setting changed by hand AND a typed confirmation, every time."""
    if not cfg.broker.allow_live:
        print(con.red("Live trading is switched off. To allow it, set this in config.yaml:"))
        print("    broker:\n      allow_live: true")
        print("Try it against Alpaca's paper service first:  ./trader run --broker alpaca-paper")
        return False
    if not sys.stdin.isatty():
        print(con.red("Live trading must be started from an interactive terminal so it can be confirmed."))
        return False
    print(con.red(con.bold("You are about to let this program place REAL orders with REAL money.")))
    print("  Account: your live Alpaca account    Money it may use: %s"
          % (con.money(capital) if capital else "the whole account (pass --capital to limit it)"))
    print("  Per-order cap: %s    Daily loss limit: %.1f%%    Shorting: %s"
          % (con.money(cfg.broker.max_order_notional), cfg.strategy.max_daily_loss_pct * 100,
             "allowed" if cfg.strategy.allow_short else "off"))
    print("  It can lose money. Nothing about the model's past results guarantees future ones.")
    try:
        typed = input('  Type "%s" to continue, anything else to cancel: ' % LIVE_PHRASE)
    except EOFError:
        typed = ""
    if typed.strip() != LIVE_PHRASE:
        print("Cancelled. Nothing was sent.")
        return False
    return True


# ------------------------------------------------------------------ commands
def cmd_run(args) -> int:
    cfg = _load(args)
    mode = args.broker or cfg.broker.mode
    if args.advise_only:
        mode = "sim"
    if mode == "alpaca-live" and not _confirm_live(cfg, args.capital):
        return 1
    eng = _engine(cfg, trade=not args.advise_only, capital=args.capital, broker_mode=mode)
    try:
        eng.run(max_sessions=args.sessions)
    except KeyboardInterrupt:
        eng.shutdown()
    return 0


def cmd_flatten(args) -> int:
    """Close every position this program holds at Alpaca and cancel its resting orders."""
    from .alpaca_broker import AlpacaBroker
    from .broker import PaperBroker

    cfg = _load(args)
    mode = args.broker or cfg.broker.mode
    if mode == "sim":
        print("Nothing to do: the built-in simulator has no real orders. Use --broker alpaca-paper or alpaca-live.")
        return 0
    state_dir, _ = _broker_dirs(cfg, mode)
    if not PaperBroker.exists(state_dir):
        print("This program has no recorded %s account, so it has no positions there." % mode)
        return 0
    br = AlpacaBroker.load(cfg, state_dir, None, live=mode == "alpaca-live")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    br.refresh_account()
    for tr in br.sync(stamp):
        print("  closed earlier at the broker: %s %d %s  P&L %s" % (tr["side"], tr["shares"], tr["symbol"], con.signed_money(tr["pnl"])))
    if not br.positions:
        br.cancel_resting_orders()
        br.save()
        print("No open positions. Any resting orders were cancelled.")
        return 0
    for sym in list(br.positions):
        tr = br.close(sym, stamp, "manual")
        if tr:
            print("  closed %s %d %s @ %.2f  P&L %s" % (tr["side"], tr["shares"], tr["symbol"], tr["exit_price"], con.signed_money(tr["pnl"])))
    br.cancel_resting_orders()
    br.save()
    left = len(br.positions)
    print("Done. %s" % ("%d position(s) could NOT be closed - check the Alpaca account (is the market open?)." % left
                         if left else "This program holds nothing at Alpaca now."))
    return 1 if left else 0


def cmd_fetch(args) -> int:
    cfg = _load(args)
    eng = _engine(cfg, trade=False)
    eng.sync_history()
    return 0


def cmd_train(args) -> int:
    cfg = _load(args)
    eng = _engine(cfg, trade=False)
    eng.sync_history()
    eng.load_spreads()
    eng.train_model("requested")
    return 0


def cmd_backtest(args) -> int:
    from .backtest import format_stats
    from .history import backtest_history
    from .report import write_backtest_report

    cfg = _load(args)
    eng = _engine(cfg, trade=False)
    eng.sync_history()
    eng.load_spreads()
    res, info = eng.train_model("walk-forward backtest")
    print("  Running the walk-forward backtest...")
    bt = backtest_history(eng.store, info, res, cfg, eng.cal, asof=eng.clock.now(), capital=cfg.strategy.capital)
    print(con.rule("Walk-forward backtest"))
    print(format_stats(bt.stats, "Out-of-sample results (after simulated spread + slippage)"))
    n = bt.stats.get("days", 0)
    note = ("%d out-of-sample sessions. Each was traded by a model fitted only on earlier data, with "
            "fills at the next bar's open plus spread and slippage." % n)
    if bt.stats.get("decision_every_bars", 1) > 1:
        k = bt.stats["decision_every_bars"]
        note += (" Predictions were made every %d bars, so the backtest re-decides every %d minutes"
                 " (live trading re-decides every bar)." % (k, k * cfg.data.interval_min))
    if n < 60:
        note += " This is a SMALL sample: day-to-day luck dominates over a few weeks, so treat it as a sanity check, not a forecast."
    print("  " + con.dim(note))
    path = write_backtest_report(bt, cfg.report_dir, cfg.strategy.capital, res.report, note)
    print("  Report: %s" % path)
    return 0


def cmd_tune(args) -> int:
    """Compare prediction horizons on the same history, after costs."""
    import copy

    from .history import backtest_history, build_dataset
    from .model import SELECTION_ALLOWANCE, NotEnoughData, train

    cfg = _load(args)
    eng = _engine(cfg, trade=False)
    eng.sync_history()
    eng.load_spreads()
    iv = cfg.data.interval_min
    horizons = [int(h) for h in args.horizons.split(",")] if args.horizons else \
        sorted({max(1, 30 // iv), max(1, 60 // iv), max(2, 120 // iv), max(3, 180 // iv)})
    print(con.rule("Horizon comparison (walk-forward, after simulated costs)"))
    print("  %-9s %8s %7s %14s %8s %11s %9s" % ("horizon", "rank IC", "t-stat", "top-1% net", "trades", "avg trade", "return"))
    rows = []
    for h in horizons:
        c = copy.deepcopy(cfg)
        c.model.horizon_bars = h
        try:
            ds, info = build_dataset(eng.store, eng.active, c, eng.cal, asof=eng.clock.now(), log=None)
            res = train(ds, info, c, verbose=False, fit_final=False)
            del ds
        except NotEnoughData as exc:
            print("  %-9s %s" % ("%d min" % (h * iv), exc))
            continue
        bt = backtest_history(eng.store, info, res, c, eng.cal, asof=eng.clock.now(), capital=c.strategy.capital)
        r, st = res.report, bt.stats
        top = r["edge_table"][-1]["net_bps"] if r.get("edge_table") else float("nan")
        rows.append((h, r, st))
        print("  %-9s %+8.4f %7.1f %11.1f bp %8d %8.1f bp %+8.2f%%%s"
              % ("%d min" % (h * iv), r["oos_rank_ic"], r["oos_ic_tstat"], top, st.get("trades", 0),
                 st.get("avg_trade_bps", 0.0), st.get("total_return_pct", 0.0),
                 "   <- current setting" if h == cfg.model.horizon_bars else ""), flush=True)
    good = [(h, r, st) for h, r, st in rows
            if r["oos_ic_tstat"] - SELECTION_ALLOWANCE >= 2.0 and st.get("trades", 0) >= 20
            and st.get("total_return_pct", 0.0) > 0]
    days = rows[0][2].get("days", 0) if rows else 0
    print()
    if good:
        best = max(good, key=lambda x: x[2]["total_return_pct"])
        print("  Best after costs: %d minutes. To use it, set in config.yaml:   model:  horizon_bars: %d"
              % (best[0] * iv, best[0]))
    else:
        print("  No horizon shows a dependable, tradable edge on this history (needs t-stat >= 2.5,")
        print("  20+ trades and a positive result). Leave the setting as it is.")
    print(con.dim("  Based on %d out-of-sample sessions%s." % (
        days, " - too few to settle this; re-run as the bar cache grows" if days < 60 else "")))
    return 0


def _sim_engine(cfg: Config, bars, symbols: List[str], start_day, speed: float, capital, sub: str,
                label: str, note: str):
    from .clock import SimClock
    from .data.store import BarStore
    from .data.synthetic import ReplayProvider
    from .engine import Engine
    from .market_calendar import ET

    clock = SimClock(datetime.combine(start_day, time(9, 20), tzinfo=ET), speed=speed)
    provider = ReplayProvider(bars, clock, cfg.data.interval_min)
    out = Path(cfg.root) / sub
    store = BarStore(cfg.data_dir, cfg.data.interval_min, persist=False)
    return Engine(cfg, provider, symbols, clock=clock, trade=True, capital=capital, store=store,
                  state_dir=out / "state", model_path=out / "model.joblib", report_dir=out,
                  fresh_account=True, allow_retrain=False, force_train=True, label=label, note=note)


REPLAY_HISTORY_DAYS = 400     # calendar days of history a replay trains on


def cmd_replay(args) -> int:
    """Run past sessions from the bar cache through the live engine."""
    from .data.store import BarStore
    from .market_calendar import ET, MarketCalendar, now_et

    cfg = _load(args)
    eng = _engine(cfg, trade=False)
    eng.sync_history()
    from datetime import date as _date, timedelta as _td

    import pandas as pd

    cal = MarketCalendar(cfg.calendar.extra_holidays, cfg.calendar.extra_early_closes)
    days = [_date(1970, 1, 1) + _td(days=int(d)) for d in eng.store.session_days(eng.active)]
    now = now_et()
    sess = cal.session(now.date())
    if days and days[-1] == now.date() and sess is not None and now < sess.close:
        days = days[:-1]            # today is still in progress
    n = max(1, int(args.days))
    if len(days) < n + 22:
        print("Not enough cached history to replay %d session(s) (have %d sessions)." % (n, len(days)))
        return 1
    start = days[-n]
    # A replay only needs the year before it: keeps memory modest with a multi-year Alpaca cache.
    since = pd.Timestamp(start - _td(days=REPLAY_HISTORY_DAYS), tz="UTC")
    bars = eng.store.load(eng.active, start=since)
    sim = _sim_engine(cfg, bars, eng.active, start, args.speed, cfg.strategy.capital, "replay",
                      "replay of %d past session(s)" % n,
                      "Replay of real past sessions. The model was trained only on data before %s." % start)
    try:
        sim.run(max_sessions=n)
    except KeyboardInterrupt:
        sim.shutdown()
    return 0


def cmd_demo(args) -> int:
    """Offline end-to-end run on a synthetic market (no network or account needed)."""
    from .data.synthetic import generate_bars
    from .market_calendar import MarketCalendar, now_et

    cfg = _load(args)
    cal = MarketCalendar()
    n = max(1, int(args.days))
    end_day = cal.previous_trading_day(now_et().date())
    start_day = end_day
    for _ in range(n - 1):
        start_day = cal.previous_trading_day(start_day)
    cfg.model.max_iter = min(cfg.model.max_iter, 150)
    cfg.model.seeds = 1
    print("Generating a synthetic market (36 symbols, %d sessions)..." % (45 + n))
    bars = generate_bars(n_symbols=36, n_days=45 + n, interval_min=cfg.data.interval_min, end_day=end_day,
                         seed=args.seed, signal=3.0, calendar=cal)
    symbols = sorted(bars["symbol"].unique().tolist())
    sim = _sim_engine(cfg, bars, symbols, start_day, args.speed, cfg.strategy.capital, "demo",
                      "DEMO on synthetic data",
                      "DEMO: synthetic market with an artificial pattern planted in it so you can watch the "
                      "machinery work. Its profits say nothing about real markets.")
    try:
        sim.run(max_sessions=n)
    except KeyboardInterrupt:
        sim.shutdown()
    return 0


def cmd_report(args) -> int:
    from .backtest import format_stats
    from .model import Model
    from .report import account_stats, load_state, write_dashboard

    cfg = _load(args)
    mode = getattr(args, "broker", None) or cfg.broker.mode
    state_dir, report_dir = _broker_dirs(cfg, mode)
    state = load_state(state_dir)
    if not state["portfolio"]:
        print("No %s account recorded yet. Start one with:  python -m daytrader run --capital 100000%s"
              % ("paper" if mode == "sim" else mode, "" if mode == "sim" else " --broker " + mode))
        return 0
    label = {"sim": "Paper account", "alpaca-paper": "Alpaca paper account", "alpaca-live": "LIVE Alpaca account"}[mode]
    print(format_stats(account_stats(state, cfg.strategy.capital), label + " since inception"))
    model = Model.load(cfg.model_dir / "model.joblib")
    path = write_dashboard(state_dir, report_dir / "dashboard.html", cfg.strategy.capital,
                           "ML Day Trader", label,
                           model_meta=model.meta if model else None, refresh_seconds=0)
    print("  Dashboard: %s" % path)
    return 0


def cmd_status(args) -> int:
    from .config import alpaca_keys
    from .data import resolve_provider_name
    from .data.store import BarStore
    from .market_calendar import ET, MarketCalendar, now_et
    from .model import Model, format_report
    from .report import load_state
    from .universe import resolve_universe

    cfg = _load(args)
    name = resolve_provider_name(cfg)
    symbols = resolve_universe(cfg.data.universe, name, cfg.root)
    print(con.rule("ML Day Trader %s" % __version__))
    print("  Data provider: %s%s" % (name, "" if name == "alpaca" or not alpaca_keys() else " (Alpaca keys found but provider set to yahoo)"))
    print("  Universe: %d symbols   Bars: %d min   Horizon: %d min"
          % (len(symbols), cfg.data.interval_min, cfg.model.horizon_bars * cfg.data.interval_min))
    cal = MarketCalendar(cfg.calendar.extra_holidays, cfg.calendar.extra_early_closes)
    now = now_et()
    sess = cal.next_session(now)
    if sess.open <= now < sess.close:
        print("  Market: OPEN until %s ET" % sess.close.strftime("%H:%M"))
    else:
        print("  Market: closed. Next session %s %s ET (in %s)"
              % (sess.day.strftime("%a %Y-%m-%d"), sess.open.strftime("%H:%M"),
                 con.duration((sess.open - now).total_seconds())))
    store = BarStore(cfg.bar_cache_dir(name), cfg.data.interval_min)
    have = [s for s in symbols if store.last_ts(s) is not None]
    if have:
        print("  Bar cache: %d symbols, %s to %s" % (
            len(have), min(store.first_ts(s) for s in have).tz_convert(ET).strftime("%Y-%m-%d"),
            max(store.last_ts(s) for s in have).tz_convert(ET).strftime("%Y-%m-%d %H:%M ET")))
    else:
        print("  Bar cache: empty (it fills on the first run)")
    model = Model.load(cfg.model_dir / "model.joblib")
    if model:
        print(format_report(model.meta))
    else:
        print("  Model: not trained yet (it trains on the first run)")
    state = load_state(cfg.state_dir)
    port = state["portfolio"]
    if port:
        eq = port["cash"] + sum(p["side"] * p["shares"] * p["last_price"] for p in port.get("positions", []))
        print("  Paper account: equity %s (started with %s), %d open position(s), %d session(s) recorded"
              % (con.money(eq), con.money(port["start_capital"]), len(port.get("positions", [])),
                 len(state["daily"])))
    else:
        print("  Paper account: none yet")
    from .config import alpaca_trading_keys
    for mode in ("alpaca-paper", "alpaca-live"):
        sd, _ = _broker_dirs(cfg, mode)
        st = load_state(sd)["portfolio"]
        keys = "keys found" if alpaca_trading_keys(mode == "alpaca-live") else "no keys in .env"
        if st:
            eq = st["cash"] + sum(p["side"] * p["shares"] * p["last_price"] for p in st.get("positions", []))
            print("  %s ledger: equity %s (started with %s), %d open position(s); %s"
                  % (mode, con.money(eq), con.money(st["start_capital"]), len(st.get("positions", [])), keys))
        else:
            print("  %s: not used yet; %s%s" % (mode, keys,
                  "" if mode == "alpaca-paper" or cfg.broker.allow_live else "; live trading switched off in config"))
    return 0


def cmd_scorecard(args) -> int:
    """Grade a day's predictions against what happened (default: the latest session)."""
    from .data import resolve_provider_name
    from .data.store import BarStore
    from .market_calendar import MarketCalendar, now_et
    from .scorecard import run_scorecard
    from .universe import resolve_universe

    cfg = _load(args)
    name = resolve_provider_name(cfg)
    cal = MarketCalendar(cfg.calendar.extra_holidays, cfg.calendar.extra_early_closes)
    if args.date:
        day = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        now = now_et()
        day = now.date()
        sess = cal.session(day)
        if sess is None or now < sess.close:
            day = cal.previous_trading_day(day)
    state_dir, report_dir = cfg.state_dir, cfg.report_dir
    if args.broker and args.broker != "sim":
        state_dir, report_dir = _broker_dirs(cfg, args.broker)
    store = BarStore(cfg.bar_cache_dir(name), cfg.data.interval_min)
    symbols = [s for s in resolve_universe(cfg.data.universe, name, cfg.root) if store.last_ts(s) is not None]
    model_ic = None
    try:        # the validation report is plain JSON - no need to load the model itself
        import json
        model_ic = json.loads((cfg.model_dir / "model.json").read_text()).get("oos_rank_ic")
    except Exception:
        pass
    lines = run_scorecard(day, state_dir, store, symbols, cfg, cal, report_dir, model_ic)
    if not lines:
        print("No predictions recorded for %s." % day)
        return 1
    print(con.rule(lines[0]))
    for line in lines[1:]:
        print(line)
    return 0


def cmd_spreadcheck(args) -> int:
    """Dump raw full-market quotes for a few stocks so the spread measurement can be checked."""
    import json
    from collections import Counter
    from datetime import timedelta

    from .data import get_provider
    from .market_calendar import ET, MarketCalendar, now_et
    from .spreads import nbbo_spreads, spread_of

    cfg = _load(args)
    cfg.data.provider = "alpaca"
    prov = get_provider(cfg)
    cal = MarketCalendar(cfg.calendar.extra_holidays, cfg.calendar.extra_early_closes)
    day = now_et().date()
    if cal.session(day) is None or now_et() < cal.session(day).close + timedelta(minutes=16):
        day = cal.previous_trading_day(day)
    start = datetime.combine(day, time(12, 30), tzinfo=ET)
    end = start + timedelta(seconds=8)
    warm = (start + timedelta(seconds=4)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    lines = ["Spread check: %s 12:30:00-12:30:08 ET, feed=sip" % day]
    for sym in (args.symbols or "SPY,AAPL,NVDA,KO,AMD").split(","):
        rows = prov.raw_quotes(sym.strip().upper(), start, end)
        n = len(rows)
        same = sum(1 for r in rows if r.get("bx") == r.get("ax"))
        per = sorted(round(sp[1], 4) for sp in (spread_of(r.get("bp"), r.get("ap")) for r in rows) if sp)
        nb = sorted(round(u, 4) for _, u in nbbo_spreads(
            [(r.get("t"), r.get("bx"), r.get("bp"), r.get("bs"), r.get("ax"), r.get("ap"), r.get("as")) for r in rows], warm))
        med = lambda v: v[len(v) // 2] if v else None
        lines.append("")
        lines.append("%s: %d messages, bid and ask on the same exchange in %d; exchanges %s; conditions %s; tapes %s"
                     % (sym, n, same, dict(Counter(r.get("bx") for r in rows)),
                        dict(Counter(tuple(r.get("c") or []) for r in rows)), dict(Counter(r.get("z") for r in rows))))
        lines.append("  per-message spread $: median %s, min %s, max %s" % (med(per), per[0] if per else None, per[-1] if per else None))
        lines.append("  rebuilt NBBO spread $: median %s, min %s, max %s (%d points)" % (med(nb), nb[0] if nb else None, nb[-1] if nb else None, len(nb)))
        for r in rows[:12]:
            lines.append("  " + json.dumps(r, sort_keys=True))
    text = "\n".join(lines) + "\n"
    out = cfg.report_dir / "spread_check.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print("Saved to %s" % out)
    return 0


def cmd_reset(args) -> int:
    from .broker import PaperBroker

    cfg = _load(args)
    sd = cfg.state_dir
    if sd.exists() and any(sd.iterdir()):
        archive = sd.with_name("state_archive_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        sd.rename(archive)
        print("Previous account archived to %s" % archive)
    broker = PaperBroker(cfg, sd, cfg.strategy.capital)
    broker.save()
    print("New paper account opened with %s." % con.money(broker.start_capital))
    return 0


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="path to a config file (default: config.yaml next to this program)")
    common.add_argument("--provider", choices=["auto", "alpaca", "yahoo"], help="market data source")
    common.add_argument("--universe", help="'core', 'liquid', a tickers file, or a comma-separated list like AAPL,MSFT,NVDA")
    common.add_argument("--capital", type=float, help="simulated starting money for the paper account")
    common.add_argument("--alpaca-key", metavar="KEY_ID",
                        help="Alpaca API key ID for market data (and --broker alpaca-paper); overrides .env")
    common.add_argument("--alpaca-secret", metavar="SECRET",
                        help="Alpaca secret key; leave this out to be asked for it without it showing on screen")

    p = argparse.ArgumentParser(
        prog="python -m daytrader",
        description="ML Day Trader: machine-learning intraday stock recommendations with a paper-trading simulator.",
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", metavar="command")

    r = sub.add_parser("run", parents=[common], help="live loop: recommendations + autonomous paper trading")
    r.add_argument("--advise-only", action="store_true", help="only print recommendations; no simulated trading")
    r.add_argument("--sessions", type=int, help="stop after this many trading sessions (default: run until Ctrl+C)")
    r.add_argument("--broker", choices=["sim", "alpaca-paper", "alpaca-live"],
                   help="where trades go: sim = built-in simulator (default), alpaca-paper = your Alpaca paper "
                        "account, alpaca-live = your funded Alpaca account (real money)")
    r.set_defaults(fn=cmd_run)

    sub.add_parser("backtest", parents=[common], help="walk-forward backtest on downloaded history").set_defaults(fn=cmd_backtest)
    sub.add_parser("train", parents=[common], help="download data and (re)train the model").set_defaults(fn=cmd_train)
    tn = sub.add_parser("tune", parents=[common], help="compare prediction horizons (30 min to 3 hours) after costs")
    tn.add_argument("--horizons", help="comma-separated horizons in bars, e.g. 6,12,24,36")
    tn.set_defaults(fn=cmd_tune)

    rp = sub.add_parser("replay", parents=[common], help="re-run recent real sessions through the live engine")
    rp.add_argument("--days", type=int, default=1, help="how many past sessions to replay (default 1)")
    rp.add_argument("--speed", type=float, default=900.0, help="simulated seconds per real second (0 = instant)")
    rp.set_defaults(fn=cmd_replay)

    d = sub.add_parser("demo", parents=[common], help="offline demo on a synthetic market (no internet needed)")
    d.add_argument("--days", type=int, default=2, help="how many simulated sessions to run (default 2)")
    d.add_argument("--speed", type=float, default=1800.0, help="simulated seconds per real second (0 = instant)")
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(fn=cmd_demo)

    rpt = sub.add_parser("report", parents=[common], help="print account performance and refresh the dashboard")
    rpt.add_argument("--broker", choices=["sim", "alpaca-paper", "alpaca-live"], help="which account (default: sim)")
    rpt.set_defaults(fn=cmd_report)
    sub.add_parser("status", parents=[common], help="show data, model and account status").set_defaults(fn=cmd_status)
    fl = sub.add_parser("flatten", parents=[common], help="close everything this program holds at Alpaca")
    fl.add_argument("--broker", choices=["alpaca-paper", "alpaca-live"], help="which Alpaca account")
    fl.set_defaults(fn=cmd_flatten)
    sub.add_parser("fetch", parents=[common], help="download/refresh market history only").set_defaults(fn=cmd_fetch)
    sc = sub.add_parser("scorecard", parents=[common], help="grade a day's predictions against what actually happened")
    sc.add_argument("--date", help="session to grade, YYYY-MM-DD (default: the latest finished session)")
    sc.add_argument("--broker", choices=["sim", "alpaca-paper", "alpaca-live"], help="which account (default: sim)")
    sc.set_defaults(fn=cmd_scorecard)
    spc = sub.add_parser("spreadcheck", parents=[common], help="dump raw full-market quotes to check spread measurement")
    spc.add_argument("--symbols", help="comma-separated (default SPY,AAPL,NVDA,KO,AMD)")
    spc.set_defaults(fn=cmd_spreadcheck)
    sub.add_parser("reset", parents=[common], help="archive the paper account and start a new one").set_defaults(fn=cmd_reset)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    from .alpaca_broker import BrokerError
    from .data.base import DataError
    from .model import NotEnoughData

    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return 0
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except (ConfigError, DataError, NotEnoughData, BrokerError) as exc:
        print(con.red("Error: %s" % exc), file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(con.red("Error: %s" % exc), file=sys.stderr)
        return 1

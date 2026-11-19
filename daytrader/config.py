"""Configuration: typed defaults, overridden by config.yaml and CLI flags.

Every tunable lives here so the live loop, the backtester and the trainer
always agree on what the strategy is.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataCfg:
    provider: str = "auto"          # auto | alpaca | yahoo  (auto = alpaca when keys are present)
    interval_min: int = 5           # bar size in minutes
    history_days: Any = 0           # calendar days of history to train on; 0 = provider default; "max" = all
    alpaca_feed: str = "iex"        # iex (free, real-time) | sip (paid, full market)
    universe: Any = "auto"          # auto | core | liquid | [list of tickers] | path to a text file
    request_pause: float = 0.3      # seconds between Yahoo request batches
    yahoo_chunk: int = 20           # tickers per Yahoo batch


@dataclass
class ModelCfg:
    horizon_bars: int = 12          # prediction horizon (12 x 5min = 60 minutes)
    min_bars_left: int = 4          # never enter with fewer bars than this left in the session
    target: str = "relative"        # relative (vs. universe average) | raw
    sample_every: int = 2           # use every Nth bar for training (labels overlap heavily)
    max_train_rows: int = 1_500_000
    max_dataset_rows: int = 2_500_000  # memory budget: long histories keep every Nth bar to fit
    block_sessions: int = 60        # long histories are processed this many sessions at a time
    block_warmup_sessions: int = 35  # earlier sessions loaded with each block so features start warm
    n_folds: int = 5                # walk-forward test blocks
    min_train_days: int = 0         # 0 = auto (40% of history, at least 15 days)
    warmup_days: int = 5            # skip the first days (rolling features still warming up)
    max_iter: int = 300             # boosting rounds upper bound (actual count chosen out-of-sample)
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 200
    l2: float = 1.0
    max_features: float = 0.6
    max_bins: int = 63
    seeds: int = 3                  # bagged models in the final ensemble
    edge_haircut: float = 1.0       # multiply calibrated edges by this (below 1 = margin for live decay)
    tail_calibration: bool = True   # calibrate the strongest signals on their own validation evidence
    retrain_every_days: int = 1     # retrain when the model is older than this many sessions


@dataclass
class StrategyCfg:
    capital: float = 100_000.0      # starting fake money for the paper account
    allow_short: bool = True
    max_positions: int = 10
    max_position_pct: float = 0.15  # max notional per position, fraction of equity
    max_gross_exposure: float = 1.0  # sum |positions| / equity
    max_net_exposure: float = 0.3   # |beta-weighted long - short| / equity (market exposure)
    risk_per_trade_pct: float = 0.005  # equity risked to the stop per trade
    stop_sigma: float = 2.5         # stop distance in units of expected horizon volatility
    target_sigma: float = 0.0       # profit target in the same units (0 = let the model decide)
    min_edge_bps: float = 1.0       # required predicted edge AFTER round-trip costs
    exit_edge_bps: float = 0.0      # exit when predicted move for our side drops below this
    min_hold_bars: int = 2
    max_hold_bars: int = 0          # 0 = auto: three times the prediction horizon
    cooldown_bars: int = 6          # no re-entry in a symbol for this long after a stop-out
    max_daily_loss_pct: float = 0.03  # circuit breaker: flatten and stop for the day
    entry_start: str = "09:40"      # US/Eastern
    entry_end: str = "15:30"
    flatten_time: str = "15:55"     # all positions closed by this time, every day
    min_price: float = 5.0
    top_n: int = 5                  # recommendations shown per side


@dataclass
class CostCfg:
    min_half_spread_bps: float = 0.75  # floor on assumed half bid/ask spread
    tick_half_spread: float = 0.005    # half of a one-cent spread, in dollars
    slippage_bps: float = 0.75         # extra slippage per side on top of the spread
    commission_per_share: float = 0.0
    max_quote_spread_bps: float = 30.0  # ignore live quotes wider than this (use the model instead)
    spread_source: str = "measured"     # measured = per-stock, time-of-day spreads from Alpaca quotes | assumed
    measured_min_half_spread_bps: float = 0.2  # floor on a measured half spread
    spread_lookback_days: int = 20      # sessions of spread history behind each estimate
    spread_min_days: int = 3            # sessions needed before a measured estimate replaces the assumed one
    spread_bootstrap_days: int = 5      # recent sessions whose quote history is sampled (missing ones only)
    spread_feed: str = "sip"            # quotes to measure spreads from: sip = national best bid/offer
    spread_sanity_max_bps: float = 30.0  # a measured median round trip above this means broken data: use the assumed model


@dataclass
class BrokerCfg:
    mode: str = "sim"               # sim (built-in simulator) | alpaca-paper | alpaca-live
    allow_live: bool = False        # must be set to true by hand before alpaca-live will start
    order_type: str = "limit"       # limit = marketable limit orders (recommended) | market
    limit_offset_bps: float = 5.0   # how far past the quote a limit order may reach
    fill_timeout_seconds: float = 20.0  # unfilled entries are cancelled after this long
    max_order_notional: float = 25_000.0  # hard cap on any single order, in dollars
    resting_stops: bool = True      # keep a stop order at the broker for every position
    allow_shared_account: bool = False  # start even if the account holds positions this program did not open


@dataclass
class EngineCfg:
    poll_seconds: int = 30          # position/stop checks between bars
    bar_delay_seconds: int = 6      # wait after a bar closes before pulling it
    max_bar_wait_seconds: int = 45  # how long to wait for a late bar before deciding without it
    feature_window_days: int = 32   # sessions of history loaded for live features
    dashboard: bool = True


@dataclass
class CalendarCfg:
    extra_holidays: List[str] = field(default_factory=list)   # "YYYY-MM-DD" full closures
    extra_early_closes: List[str] = field(default_factory=list)  # "YYYY-MM-DD" 1pm closes


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    costs: CostCfg = field(default_factory=CostCfg)
    engine: EngineCfg = field(default_factory=EngineCfg)
    broker: BrokerCfg = field(default_factory=BrokerCfg)
    calendar: CalendarCfg = field(default_factory=CalendarCfg)
    root: str = "."

    # ---- paths -----------------------------------------------------------
    def path(self, *parts: str) -> Path:
        p = Path(self.root).joinpath(*parts)
        return p

    def bar_cache_dir(self, provider: str) -> Path:
        """Each data source keeps its own bar cache: Alpaca's free IEX feed reports far less
        volume than Yahoo's consolidated tape, so mixing the two would distort volume features.
        Yahoo keeps the original location so existing caches stay valid."""
        return self.data_dir if provider in ("yahoo", "replay") else self.data_dir / provider

    @property
    def data_dir(self) -> Path:
        return self.path("data_cache")

    @property
    def model_dir(self) -> Path:
        return self.path("models")

    @property
    def state_dir(self) -> Path:
        return self.path("state")

    @property
    def report_dir(self) -> Path:
        return self.path("reports")

    @property
    def bars_per_day(self) -> int:
        return 390 // self.data.interval_min


class ConfigError(ValueError):
    pass


def _apply(obj: Any, values: Dict[str, Any], where: str) -> None:
    valid = {f.name: f for f in fields(obj)}
    for key, val in (values or {}).items():
        if key not in valid:
            raise ConfigError(
                "Unknown setting '%s%s'. Valid settings here: %s"
                % (where, key, ", ".join(sorted(valid)))
            )
        cur = getattr(obj, key)
        if is_dataclass(cur):
            if not isinstance(val, dict):
                raise ConfigError("Setting '%s%s' must be a section (mapping)." % (where, key))
            _apply(cur, val, where + key + ".")
        else:
            if isinstance(cur, bool):
                val = bool(val)
            elif isinstance(cur, int) and not isinstance(val, bool) and isinstance(val, (int, float)):
                val = int(val)
            elif isinstance(cur, float) and isinstance(val, (int, float)):
                val = float(val)
            setattr(obj, key, val)


def load_env_file(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def load_config(path: Optional[str] = None, root: Optional[str] = None) -> Config:
    """Load config.yaml (if present) on top of the defaults."""
    cfg = Config()
    root_path = Path(root) if root else Path.cwd()
    cfg_path = Path(path) if path else root_path / "config.yaml"
    if path and not root:
        root_path = cfg_path.resolve().parent
    cfg.root = str(root_path)
    if cfg_path.exists():
        import yaml

        raw = yaml.safe_load(cfg_path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ConfigError("%s must contain a mapping of settings." % cfg_path)
        raw.pop("root", None)
        _apply(cfg, raw, "")
    elif path:
        raise ConfigError("Config file not found: %s" % cfg_path)
    load_env_file(root_path / ".env")
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if 390 % cfg.data.interval_min != 0:
        raise ConfigError("data.interval_min must divide 390 (e.g. 1, 5, 15, 30).")
    hd = cfg.data.history_days
    if not (isinstance(hd, int) and hd >= 0) and str(hd).strip().lower() != "max":
        raise ConfigError("data.history_days must be a number of days, 0 (the default) or \"max\".")
    if cfg.model.horizon_bars < 1:
        raise ConfigError("model.horizon_bars must be at least 1.")
    if cfg.model.target not in ("relative", "raw"):
        raise ConfigError("model.target must be 'relative' or 'raw'.")
    if cfg.costs.spread_source not in ("measured", "assumed"):
        raise ConfigError("costs.spread_source must be 'measured' or 'assumed'.")
    if cfg.broker.mode not in BROKER_MODES:
        raise ConfigError("broker.mode must be one of: %s." % ", ".join(BROKER_MODES))
    if cfg.broker.order_type not in ("limit", "market"):
        raise ConfigError("broker.order_type must be 'limit' or 'market'.")
    if cfg.strategy.capital <= 0:
        raise ConfigError("strategy.capital must be positive.")
    if not (0 < cfg.strategy.max_position_pct <= 1.0):
        raise ConfigError("strategy.max_position_pct must be in (0, 1].")
    for name in ("entry_start", "entry_end", "flatten_time"):
        parse_hhmm(getattr(cfg.strategy, name))


def parse_hhmm(text: str) -> int:
    """'09:40' -> minutes after midnight."""
    try:
        hh, mm = str(text).split(":")
        return int(hh) * 60 + int(mm)
    except Exception:
        raise ConfigError("Times must look like '09:40' (got %r)." % (text,))


BROKER_MODES = ("sim", "alpaca-paper", "alpaca-live")


def alpaca_trading_keys(live: bool) -> Optional[tuple]:
    """Keys for ORDER placement. Live keys must be named explicitly; they are never guessed."""
    env = os.environ
    if live:
        key, secret = env.get("ALPACA_LIVE_KEY_ID"), env.get("ALPACA_LIVE_SECRET_KEY")
    else:
        key = env.get("ALPACA_PAPER_KEY_ID") or env.get("APCA_API_KEY_ID") or env.get("ALPACA_API_KEY")
        secret = env.get("ALPACA_PAPER_SECRET_KEY") or env.get("APCA_API_SECRET_KEY") or env.get("ALPACA_SECRET_KEY")
    if key and secret:
        return key, secret
    return None


def alpaca_keys() -> Optional[tuple]:
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if key and secret:
        return key, secret
    return None

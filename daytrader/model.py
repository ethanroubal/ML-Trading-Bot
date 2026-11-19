"""The predictive model: gradient-boosted trees + ridge, validated walk-forward.

What it predicts: each stock's return over the next hour (by default),
relative to the rest of the universe, in units of that stock's expected
volatility.

How it is validated: strictly forward in time. The model is trained on days
1..N and tested on the days that follow, repeatedly. Every number the
trader relies on (how many boosting rounds, how to blend the two models,
how much to trust a prediction) is estimated on those out-of-sample
predictions - never on data the model was fitted to.
"""
from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .costs import round_trip_cost_bps
from .market_calendar import OPEN_MIN
from .features import FeatureSet
from .panel import Panel

MODEL_VERSION = 5


class NotEnoughData(RuntimeError):
    pass


# ------------------------------------------------------------------ dataset
@dataclass
class Dataset:
    X: np.ndarray           # (n, F) float32
    y: np.ndarray           # (n,) label, volatility units
    rel_bps: np.ndarray     # (n,) realised forward relative return, basis points
    sigH_bps: np.ndarray    # (n,) expected horizon volatility, basis points
    price: np.ndarray       # (n,)
    day: np.ndarray         # (n,) session index
    t: np.ndarray           # (n,) panel row
    s: np.ndarray           # (n,) symbol index
    k: np.ndarray           # (n,) bar index within session
    names: List[str]
    stride: int = 1         # only every `stride`-th bar was kept (long histories, to fit in memory)


def make_dataset(panel: Panel, feats: FeatureSet, cfg: Config, keep_days: Optional[np.ndarray] = None,
                 day_index: Optional[np.ndarray] = None, stride: int = 1) -> Dataset:
    """Labelled rows of a panel.

    keep_days: bool per panel day - which sessions to take rows from (block-wise building);
    day_index: global session number of each panel day; rows then get t = day * 1000 + k,
               a bar id that is unique across blocks;
    stride:    keep only bars with k % stride == day % stride (rotates through the day).
    """
    mask = feats.usable & np.isfinite(feats.y)
    gday = panel.day if day_index is None else np.asarray(day_index)[panel.day]
    rows = gday >= cfg.model.warmup_days
    if keep_days is not None:
        rows &= np.asarray(keep_days)[panel.day]
    if stride > 1:
        rows &= (panel.k % stride) == (gday % stride)
    mask &= rows[:, None]
    t_idx, s_idx = np.nonzero(mask)
    if day_index is not None:
        return Dataset(
            X=feats.matrix(mask), y=feats.y[mask].astype(np.float32),
            rel_bps=(feats.fwd_rel[mask] * 1e4).astype(np.float32),
            sigH_bps=(feats.sigH[mask] * 1e4).astype(np.float32),
            price=panel.C[mask].astype(np.float32),
            day=gday[t_idx], t=gday[t_idx] * 1000 + panel.k[t_idx], s=s_idx, k=panel.k[t_idx],
            names=list(feats.names), stride=stride)
    return Dataset(
        X=feats.matrix(mask),
        y=feats.y[mask].astype(np.float32),
        rel_bps=(feats.fwd_rel[mask] * 1e4).astype(np.float32),
        sigH_bps=(feats.sigH[mask] * 1e4).astype(np.float32),
        price=panel.C[mask].astype(np.float32),
        day=panel.day[t_idx],
        t=t_idx,
        s=s_idx,
        k=panel.k[t_idx],
        names=list(feats.names),
    )


# -------------------------------------------------------------------- ridge
RIDGE_ALPHAS = (1e3, 1e4, 3e4, 1e5, 3e5, 1e6)


class Ridge:
    """Closed-form ridge regression on standardised features (NaN -> 0).

    Fits every penalty in `alphas` at once (they share the same X'X), so the
    best amount of shrinkage can be picked out-of-sample for free. With targets
    as noisy as short-horizon returns, heavy shrinkage is usually what wins.
    """

    def __init__(self, alphas=RIDGE_ALPHAS):
        self.alphas = tuple(float(a) for a in alphas)
        self.mu = self.sd = None
        self.coefs = None            # (n_alphas, n_features)
        self.intercept = 0.0
        self.use = 0                 # index of the penalty used by predict()

    def _prep(self, X: np.ndarray) -> np.ndarray:
        Z = (X.astype(np.float64) - self.mu) / self.sd
        np.clip(Z, -5.0, 5.0, out=Z)
        Z[~np.isfinite(Z)] = 0.0
        return Z

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        with np.errstate(invalid="ignore"):
            self.mu = np.nan_to_num(np.nanmean(X, axis=0, dtype=np.float64))
            sd = np.nanstd(X, axis=0, dtype=np.float64)
        self.sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        n, f = X.shape
        xtx = np.zeros((f, f))
        xty = np.zeros(f)
        ybar = float(np.mean(y))
        for a in range(0, n, 200_000):
            Z = self._prep(X[a:a + 200_000])
            xtx += Z.T @ Z
            xty += Z.T @ (y[a:a + 200_000].astype(np.float64) - ybar)
        eye = np.eye(f)
        self.coefs = np.vstack([np.linalg.solve(xtx + lam * eye, xty) for lam in self.alphas])
        self.intercept = ybar
        return self

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """Predictions for every penalty: shape (n, n_alphas)."""
        out = np.empty((len(X), len(self.alphas)))
        for a in range(0, len(X), 200_000):
            out[a:a + 200_000] = self._prep(X[a:a + 200_000]) @ self.coefs.T + self.intercept
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X))
        w = self.coefs[self.use]
        for a in range(0, len(X), 200_000):
            out[a:a + 200_000] = self._prep(X[a:a + 200_000]) @ w + self.intercept
        return out


def _new_gb(cfg: Config, n_iter: int, seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = cfg.model
    kw = dict(
        loss="squared_error", learning_rate=m.learning_rate, max_iter=int(n_iter),
        max_leaf_nodes=m.max_leaf_nodes, min_samples_leaf=m.min_samples_leaf,
        l2_regularization=m.l2, max_bins=m.max_bins, early_stopping=False, random_state=seed,
    )
    try:
        return HistGradientBoostingRegressor(max_features=m.max_features, **kw)
    except TypeError:  # scikit-learn < 1.4 has no max_features
        return HistGradientBoostingRegressor(**kw)


# -------------------------------------------------------------------- model
class Model:
    def __init__(self):
        self.version = MODEL_VERSION
        self.names: List[str] = []
        self.gbs: list = []
        self.ridge: Optional[Ridge] = None
        self.w_gb = 0.5
        self.w_lin = 0.5
        self.slope = 0.0            # calibrated: realised label per unit of raw prediction
        self.haircut = 1.0
        self.evidence = 0.0         # 0..1 shrink: how statistically solid the validation edge is
        self.calib_x: Optional[np.ndarray] = None   # tail calibration knots: |raw prediction| ...
        self.calib_g: Optional[np.ndarray] = None   # ... -> expected move in volatility units
        self.pwin_x = np.array([0.0, 1.0])
        self.pwin_y = np.array([0.5, 0.5])
        self.abs_quantiles = np.linspace(0, 1, 101)   # |expected z| percentiles out-of-sample
        self.meta: Dict[str, Any] = {}

    def raw_predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        out = np.zeros(len(X))
        if self.gbs and self.w_gb > 0:
            out += self.w_gb * np.mean([g.predict(X) for g in self.gbs], axis=0)
        if self.ridge is not None and self.w_lin > 0:
            out += self.w_lin * self.ridge.predict(X)
        return out

    def expected_z(self, X: np.ndarray) -> np.ndarray:
        """Expected forward return in volatility units, after calibration and haircut."""
        return apply_calibration(self.raw_predict(X), self.slope * self.evidence, self.haircut,
                                 self.calib_x, self.calib_g)

    def use_haircut(self, haircut: float) -> None:
        """Apply a different haircut without retraining (the calibration does not depend on it)."""
        if not hasattr(self, "haircut_trained"):
            self.haircut_trained = self.haircut
        self.haircut = float(haircut)

    def _as_trained(self, exp_z: np.ndarray) -> np.ndarray:
        """|exp_z| on the scale the hit-rate and strength tables were built with."""
        ht = getattr(self, "haircut_trained", self.haircut)
        scale = ht / self.haircut if self.haircut > 0 and ht > 0 else 1.0
        return np.abs(exp_z) * scale

    def p_win(self, exp_z: np.ndarray) -> np.ndarray:
        return np.interp(self._as_trained(exp_z), self.pwin_x, self.pwin_y)

    def strength(self, exp_z: np.ndarray) -> np.ndarray:
        """Percentile (0-100) of this signal's size among validation signals."""
        return np.searchsorted(self.abs_quantiles, self._as_trained(exp_z), side="right") - 1.0

    @property
    def has_edge(self) -> bool:
        return bool(self.meta.get("has_edge", False))

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        joblib.dump(self, tmp, compress=3)
        tmp.replace(path)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2, default=str))

    @staticmethod
    def load(path: Path) -> Optional["Model"]:
        if not path.exists():
            return None
        import joblib

        try:
            m = joblib.load(path)
        except Exception as exc:
            print("  (could not load saved model: %s - will retrain)" % exc)
            return None
        if not isinstance(m, Model) or getattr(m, "version", 0) != MODEL_VERSION:
            return None
        return m

    def compatible(self, cfg: Config, names: List[str]) -> bool:
        mt = self.meta
        return (mt.get("interval_min") == cfg.data.interval_min
                and mt.get("horizon_bars") == cfg.model.horizon_bars
                and mt.get("target") == cfg.model.target
                and list(self.names) == list(names))


# ------------------------------------------------------------- diagnostics
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def daily_rank_ic(pred: np.ndarray, y: np.ndarray, t: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Per-session rank correlation between prediction and outcome (ranked within each bar)."""
    df = pd.DataFrame({"p": pred, "y": y, "t": t, "d": day})
    g = df.groupby("t")
    df["rp"] = g["p"].rank(pct=True) - 0.5
    df["ry"] = g["y"].rank(pct=True) - 0.5
    out = []
    for _, sub in df.groupby("d"):
        out.append(_corr(sub["rp"].to_numpy(), sub["ry"].to_numpy()))
    return np.array(out)


def _slope(pred: np.ndarray, y: np.ndarray) -> float:
    den = float((pred * pred).sum())
    if den <= 0:
        return 0.0
    return float(np.clip((pred * y).sum() / den, 0.0, 3.0))


# Choosing the best of several model sizes/penalties on the same validation rows flatters the
# t-statistic by roughly half a point (measured on pure-noise markets), so that much is deducted.
SELECTION_ALLOWANCE = 0.5


def evidence_factor(t_stat: float) -> float:
    """Shrink predicted edges by how convincing the validation evidence is.

    t = 1 (could easily be luck) -> 0.5;  t = 2 -> 0.8;  t = 4 -> 0.94;  t <= 0 -> 0.
    A model with no demonstrated edge therefore predicts ~zero and does not trade.
    """
    t = max(float(t_stat), 0.0)
    return t * t / (1.0 + t * t)


def _ic_stats(pred: np.ndarray, y: np.ndarray, t: np.ndarray, day: np.ndarray) -> tuple:
    ics = daily_rank_ic(pred, y, t, day)
    if len(ics) == 0:
        return 0.0, 0.0, 0
    mean = float(np.mean(ics))
    sd = float(np.std(ics, ddof=1)) if len(ics) > 2 else 0.0
    return mean, (mean / (sd / np.sqrt(len(ics))) if sd > 0 else 0.0), len(ics)


# ------------------------------------------------------- tail calibration
# Signal-strength groups, as percentiles of |raw prediction|. Finer at the top, where it matters.
TAIL_PCTS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0)
TAIL_MIN_DAYS = 15       # a group seen on fewer sessions than this keeps the plain scaling
TAIL_SHRINK = 9.0        # positive-part James-Stein: a difference must exceed 3 standard errors to count (8 groups tested)
TAIL_MAX_Z = 1.0         # never expect more than one horizon-volatility of edge


def apply_calibration(raw: np.ndarray, base_scale: float, haircut: float,
                      knots_x: Optional[np.ndarray], knots_g: Optional[np.ndarray]) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if knots_x is None or len(knots_x) < 2:
        return base_scale * haircut * raw
    return haircut * np.sign(raw) * np.interp(np.abs(raw), knots_x, knots_g)


def tail_calibration(raw: np.ndarray, y: np.ndarray, day: np.ndarray, base_scale: float,
                     sigH_bps: Optional[np.ndarray] = None) -> tuple:
    """Calibrate the strongest signals on their own evidence.

    The plain calibration scales every prediction by one number (base_scale), fitted mostly
    on the weak signals that make up the bulk. Here each strength group's realized move (in
    the predicted direction) is compared with what that number implies. The difference is
    measured per session - a session's overlapping predictions are not independent - and is
    shrunk by (1 - 9/t^2): under three standard errors it is ignored entirely (eight groups are
    tested, so a two-standard-error difference would turn up by chance far too often). The groups are
    then forced to be monotone (stronger signal -> no smaller edge) and never reverse a
    signal's direction. With no real difference this reproduces the plain calibration.

    Returns (knots_x, knots_g, diagnostics).
    """
    from sklearn.isotonic import IsotonicRegression

    a = np.abs(np.asarray(raw, dtype=float))
    s = np.sign(raw) * np.asarray(y, dtype=float)
    edges = np.quantile(a, TAIL_PCTS)
    xs, gs, ws, diag = [], [], [], []
    for i in range(len(TAIL_PCTS) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (a >= lo) & ((a < hi) if i < len(TAIL_PCTS) - 2 else (a <= hi))
        if m.sum() < 20:
            continue
        am = float(a[m].mean())
        target = base_scale * am
        per_day = pd.Series(s[m]).groupby(np.asarray(day)[m]).mean()
        nd = len(per_day)
        obs = float(per_day.mean())
        t, k = 0.0, 0.0
        if nd >= TAIL_MIN_DAYS:
            sd = float(per_day.std(ddof=1))
            if sd > 0:
                t = (obs - target) / (sd / np.sqrt(nd))
                k = max(0.0, 1.0 - TAIL_SHRINK / (t * t)) if t != 0 else 0.0
        g = float(np.clip(target + k * (obs - target), 0.0, TAIL_MAX_Z))
        xs.append(am)
        gs.append(g)
        ws.append(float(m.sum()))
        bps = float(np.mean(sigH_bps[m])) if sigH_bps is not None else float("nan")
        diag.append({"group": "%g-%g%%" % (100 * TAIL_PCTS[i], 100 * TAIL_PCTS[i + 1]),
                     "rows": int(m.sum()), "sessions": int(nd),
                     "plain_z": target, "observed_z": obs, "t": float(t), "weight": float(k),
                     "calibrated_z": g, "sigH_bps": bps})
    if len(xs) < 2:
        return None, None, diag
    g_iso = IsotonicRegression(increasing=True, y_min=0.0).fit(xs, gs, sample_weight=ws).predict(xs)
    for d, g in zip(diag, g_iso):
        d["calibrated_z"] = float(g)
    return np.r_[0.0, xs], np.r_[0.0, g_iso], diag


def _blend_weights(ic_gb: float, ic_lin: float) -> tuple:
    a, b = max(ic_gb, 0.0), max(ic_lin, 0.0)
    if a + b <= 0:
        return 0.5, 0.5
    return a / (a + b), b / (a + b)


def edge_table(exp_bps: np.ndarray, rel_bps: np.ndarray, cost_bps: np.ndarray,
               day: Optional[np.ndarray] = None) -> List[Dict[str, float]]:
    """How the strongest signals actually did, before and after trading costs.

    `sessions_up` counts the sessions in which the group made money after costs on average:
    a big average that comes from a handful of sessions is luck, not an edge."""
    rows = []
    mag = np.abs(exp_bps)
    signed = np.sign(exp_bps) * rel_bps
    if len(mag) == 0:
        return rows
    for label, q in (("all signals", 0.0), ("top 20%", 0.80), ("top 10%", 0.90), ("top 5%", 0.95),
                     ("top 2%", 0.98), ("top 1%", 0.99)):
        thr = np.quantile(mag, q) if q > 0 else -1.0
        m = mag > thr
        if m.sum() < 20:
            continue
        rows.append({
            "bucket": label, "n": int(m.sum()),
            "predicted_bps": float(mag[m].mean()),
            "realized_bps": float(signed[m].mean()),
            "cost_bps": float(cost_bps[m].mean()),
            "net_bps": float(signed[m].mean() - cost_bps[m].mean()),
            "hit_rate": float((signed[m] > 0).mean()),
        })
        if day is not None:
            per = pd.Series(signed[m] - cost_bps[m]).groupby(np.asarray(day)[m]).mean()
            rows[-1]["sessions_up"] = int((per > 0).sum())
            rows[-1]["sessions"] = int(len(per))
    return rows


# ---------------------------------------------------------------- training
@dataclass
class TrainResult:
    model: Model
    oos: pd.DataFrame            # walk-forward predictions: t, s, day, exp_z (NaN in the calibration fold)
    report: Dict[str, Any] = field(default_factory=dict)
    folds: List[Dict[str, Any]] = field(default_factory=list)   # walk-forward models, for full-resolution backtests
    ckpts: Optional[np.ndarray] = None
    haircut: float = 1.0


def fold_predict(fold: Dict[str, Any], X: np.ndarray, ckpts: np.ndarray, haircut: float) -> np.ndarray:
    """Walk-forward expected move for rows of fold `fold`, using only that fold's models and the
    settings chosen on earlier folds - what the model would have said on those days, nothing more."""
    c = fold["c"]
    raw = np.zeros(len(X))
    if len(X) == 0:
        return raw
    if c["w_gb"] > 0:
        target = int(ckpts[c["j"]])
        for it, pred in enumerate(fold["gb"].staged_predict(X), start=1):
            if it == target:
                raw += c["w_gb"] * pred
                break
    if c["w_lin"] > 0:
        raw += c["w_lin"] * fold["lin"].predict_all(X)[:, c["a"]]
    return apply_calibration(raw, c["scale"], haircut, c["kx"], c["kg"])


def _train_mask(ds: Dataset, rows: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(rows)
    se = max(1, cfg.model.sample_every)
    if se > 1 and ds.stride < se:
        idx = idx[(ds.k[idx] % se) == 0]
    if len(idx) > cfg.model.max_train_rows:
        idx = np.sort(rng.choice(idx, cfg.model.max_train_rows, replace=False))
    return idx


def train(ds: Dataset, panel: Panel, cfg: Config, verbose: bool = True,
          provider_name: str = "", fit_final: bool = True) -> TrainResult:
    """Walk-forward validate, then (unless fit_final=False) fit the model used for live trading."""
    m = cfg.model
    t0 = _time.time()
    rng = np.random.default_rng(0)
    days = np.unique(ds.day)
    nd = len(days)
    min_train = m.min_train_days or max(15, int(round(0.4 * nd)))
    n_folds = max(2, min(m.n_folds, nd - min_train))
    if nd < 22 or nd - min_train < 4:
        raise NotEnoughData(
            "Only %d usable sessions of history - need at least 22 to train and validate. "
            "Let the bar cache grow, or lower model.warmup_days." % nd
        )
    blocks = np.array_split(days[min_train:], n_folds)
    # candidate model sizes, from "barely fitted" to the full budget; the winner is chosen out-of-sample
    ckpts = np.unique(np.clip((np.array([0.03, 0.07, 0.13, 0.25, 0.37, 0.5, 0.75, 1.0]) * m.max_iter).astype(int),
                              1, m.max_iter))
    if verbose:
        print("  Training set: %s rows, %d features, %d sessions (%s to %s), %d symbols"
              % (format(len(ds.y), ","), ds.X.shape[1], nd, panel.dates[int(days[0])],
                 panel.dates[int(days[-1])], len(np.unique(ds.s))))
        print("  Walk-forward validation: %d folds, first %d sessions train-only" % (n_folds, min_train))

    fold_id = np.full(len(ds.y), -1, dtype=np.int16)
    folds: List[Dict[str, Any]] = []
    p_gb = np.full((len(ds.y), len(ckpts)), np.nan, dtype=np.float32)
    p_lin = np.full((len(ds.y), len(RIDGE_ALPHAS)), np.nan, dtype=np.float32)
    for f, block in enumerate(blocks):
        tr = _train_mask(ds, ds.day < block[0], cfg, rng)
        te = np.flatnonzero((ds.day >= block[0]) & (ds.day <= block[-1]))
        if len(tr) < 2000 or len(te) == 0:
            continue
        gb = _new_gb(cfg, m.max_iter, seed=f)
        gb.fit(ds.X[tr], ds.y[tr])
        want = {int(c): j for j, c in enumerate(ckpts)}
        for it, pred in enumerate(gb.staged_predict(ds.X[te]), start=1):
            j = want.get(it)
            if j is not None:
                p_gb[te, j] = pred
        lin = Ridge().fit(ds.X[tr], ds.y[tr])
        p_lin[te] = lin.predict_all(ds.X[te])
        fold_id[te] = f
        folds.append({"f": f, "first": int(block[0]), "last": int(block[-1]), "gb": gb, "lin": lin, "c": None})
        if verbose:
            yt = ds.y[te].astype(float)
            c_gb = max(_corr(p_gb[te, j].astype(float), yt) for j in range(len(ckpts)))
            c_lin = max(_corr(p_lin[te, a].astype(float), yt) for a in range(len(RIDGE_ALPHAS)))
            print("    fold %d/%d: train %s rows -> test %s rows (%s to %s)  best-tree corr %+.4f, best-ridge corr %+.4f"
                  % (f + 1, n_folds, format(len(tr), ","), format(len(te), ","),
                     panel.dates[int(block[0])], panel.dates[int(block[-1])], c_gb, c_lin), flush=True)

    oos_rows = fold_id >= 0
    if oos_rows.sum() < 500:
        raise NotEnoughData("Not enough out-of-sample rows to validate the model.")
    y64 = ds.y.astype(np.float64)

    def choose(rows: np.ndarray):
        """Pick tree count, ridge penalty, blend weights and calibration from `rows` only."""
        cors = [_corr(p_gb[rows, j].astype(float), y64[rows]) for j in range(len(ckpts))]
        j = int(np.argmax(cors))
        lcors = [_corr(p_lin[rows, a].astype(float), y64[rows]) for a in range(len(RIDGE_ALPHAS))]
        a = int(np.argmax(lcors))
        ic_gb, ic_lin = cors[j], lcors[a]
        w_gb, w_lin = _blend_weights(ic_gb, ic_lin)
        blend = w_gb * p_gb[rows, j].astype(float) + w_lin * p_lin[rows, a].astype(float)
        ic_mean, ic_t, n_days = _ic_stats(blend, y64[rows], ds.t[rows], ds.day[rows])
        slope = _slope(blend, y64[rows])
        evidence = evidence_factor(ic_t - SELECTION_ALLOWANCE) if ic_mean > 0 else 0.0
        kx = kg = None
        diag: list = []
        if m.tail_calibration:
            kx, kg, diag = tail_calibration(blend, y64[rows], ds.day[rows], slope * evidence,
                                            ds.sigH_bps[rows].astype(float))
        return {"j": j, "a": a, "w_gb": w_gb, "w_lin": w_lin, "slope": slope,
                "ic_gb": ic_gb, "ic_lin": ic_lin, "ic": ic_mean, "ic_t": ic_t, "days": n_days,
                "evidence": evidence, "blend": blend, "kx": kx, "kg": kg, "tail": diag}

    # Honest walk-forward expectations: fold f uses settings chosen on folds < f only.
    exp_z = np.full(len(ds.y), np.nan)
    for f in range(1, n_folds):
        prev, cur = (fold_id >= 0) & (fold_id < f), fold_id == f
        if prev.sum() < 500 or cur.sum() == 0:
            continue
        c = choose(prev)
        for fd in folds:
            if fd["f"] == f:
                fd["c"] = {"j": c["j"], "a": c["a"], "w_gb": c["w_gb"], "w_lin": c["w_lin"],
                           "scale": c["slope"] * c["evidence"], "kx": c["kx"], "kg": c["kg"]}
        raw = c["w_gb"] * p_gb[cur, c["j"]].astype(float) + c["w_lin"] * p_lin[cur, c["a"]].astype(float)
        exp_z[cur] = apply_calibration(raw, c["slope"] * c["evidence"], m.edge_haircut, c["kx"], c["kg"])

    # Final settings from every out-of-sample row.
    c = choose(oos_rows)
    j, w_gb, w_lin, slope, ic_gb, ic_lin = c["j"], c["w_gb"], c["w_lin"], c["slope"], c["ic_gb"], c["ic_lin"]
    blend, ic_mean, ic_t, evidence = c["blend"], c["ic"], c["ic_t"], c["evidence"]
    n_ic_days = c["days"]
    exp_all = apply_calibration(blend, slope * evidence, m.edge_haircut, c["kx"], c["kg"])
    exp_bps = exp_all * ds.sigH_bps[oos_rows].astype(float)
    syms = getattr(panel, "symbols", None)
    cost = round_trip_cost_bps(ds.price[oos_rows].astype(float), cfg.costs,
                               symbols=np.asarray(syms, dtype=object)[ds.s[oos_rows]] if syms is not None else None,
                               minute=OPEN_MIN + (ds.k[oos_rows].astype(int) + 1) * cfg.data.interval_min)
    table = edge_table(exp_bps, ds.rel_bps[oos_rows].astype(float), cost, ds.day[oos_rows])

    model = Model()
    model.names = list(ds.names)
    model.w_gb, model.w_lin, model.slope, model.haircut = w_gb, w_lin, slope, m.edge_haircut
    model.evidence = evidence
    model.haircut_trained = m.edge_haircut
    model.calib_x, model.calib_g = c["kx"], c["kg"]
    n_iter = int(min(m.max_iter, round(ckpts[j] * 1.1)))
    if fit_final:
        full = _train_mask(ds, np.ones(len(ds.y), dtype=bool), cfg, rng)
        for seed in range(max(1, m.seeds)):
            model.gbs.append(_new_gb(cfg, n_iter, seed=100 + seed).fit(ds.X[full], ds.y[full]))
        model.ridge = Ridge().fit(ds.X[full], ds.y[full])
        model.ridge.use = int(c["a"])

    # hit-rate curve and signal-size percentiles (from out-of-sample predictions)
    mag = np.abs(exp_all)
    win = (np.sign(blend) * y64[oos_rows]) > 0
    if len(mag) >= 200 and mag.max() > 0:
        from sklearn.isotonic import IsotonicRegression

        edges = np.unique(np.quantile(mag, np.linspace(0, 1, 41)))
        if len(edges) > 3:
            b = np.clip(np.digitize(mag, edges[1:-1]), 0, len(edges) - 2)
            cx = np.array([mag[b == i].mean() for i in range(len(edges) - 1) if (b == i).any()])
            cy = np.array([win[b == i].mean() for i in range(len(edges) - 1) if (b == i).any()])
            cw = np.array([(b == i).sum() for i in range(len(edges) - 1) if (b == i).any()])
            iso = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0).fit(cx, cy, sample_weight=cw)
            model.pwin_x, model.pwin_y = cx, iso.predict(cx)
        model.abs_quantiles = np.quantile(mag, np.linspace(0, 1, 101))

    has_edge = bool(slope > 0 and ic_mean > 0 and ic_t - SELECTION_ALLOWANCE >= 2.0)
    wf = np.isfinite(exp_z)
    # the last session counts as "trained through" only if it had finished
    last_complete = (int(panel.k[-1]) + 1 == int(panel.nbar[-1])) or int(days[-1]) != int(panel.day[-1])
    end_day = panel.dates[int(days[-1])] if last_complete or nd < 2 else panel.dates[int(days[-2])]
    model.meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "provider": provider_name,
        "interval_min": cfg.data.interval_min,
        "horizon_bars": m.horizon_bars,
        "target": m.target,
        "train_start": str(panel.dates[int(days[0])]),
        "train_end": str(end_day),
        "sessions": int(nd),
        "symbols": int(len(np.unique(ds.s))),
        "rows": int(len(ds.y)),
        "features": int(ds.X.shape[1]),
        "boosting_rounds": n_iter,
        "ridge_penalty": RIDGE_ALPHAS[int(c["a"])],
        "weight_trees": round(w_gb, 3),
        "weight_ridge": round(w_lin, 3),
        "calibration_slope": round(slope, 4),
        "edge_haircut": m.edge_haircut,
        "evidence_factor": round(evidence, 3),
        "tail_calibration": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
                             for d in c["tail"]],
        "oos_sessions": int(n_ic_days),
        "oos_rows": int(oos_rows.sum()),
        "oos_rank_ic": round(ic_mean, 5),
        "oos_ic_tstat": round(ic_t, 2),
        "oos_corr_trees": round(ic_gb, 5),
        "oos_corr_ridge": round(ic_lin, 5),
        "has_edge": has_edge,
        "edge_table": table,
        "train_seconds": round(_time.time() - t0, 1),
    }
    oos = pd.DataFrame({"t": ds.t[wf], "s": ds.s[wf], "day": ds.day[wf], "exp_z": exp_z[wf]})
    return TrainResult(model=model, oos=oos, report=model.meta, folds=[fd for fd in folds if fd["c"] is not None],
                       ckpts=ckpts, haircut=m.edge_haircut)


def _top_label(group: str) -> str:
    lo, hi = group.rstrip("%").split("-")
    return "%g-%g%%" % (100 - float(hi), 100 - float(lo))


def format_report(meta: Dict[str, Any]) -> str:
    lines = []
    lines.append("  Model trained on %s -> %s (%d sessions, %d symbols, %s rows) in %ss"
                 % (meta["train_start"], meta["train_end"], meta["sessions"], meta["symbols"],
                    format(meta["rows"], ","), meta["train_seconds"]))
    lines.append("  Out-of-sample check over %d sessions:" % meta["oos_sessions"])
    lines.append("    rank IC %+.4f (t-stat %+.1f)   blend: trees %.0f%% / ridge %.0f%%   calibration %.2f x evidence %.2f x haircut %.2f"
                 % (meta["oos_rank_ic"], meta["oos_ic_tstat"], 100 * meta["weight_trees"],
                    100 * meta["weight_ridge"], meta["calibration_slope"], meta.get("evidence_factor", 0.0),
                    meta.get("edge_haircut", 1.0)))
    tail = [d for d in meta.get("tail_calibration") or [] if d["group"].split("-")[0] not in ("0", "50")]
    if tail:
        h = meta.get("edge_haircut", 1.0)
        lines.append("    Strongest-signal calibration (expected move per trade%s):"
                     % (", after the %.0f%% haircut" % (100 * (1 - h)) if h < 1 else ""))
        for d in tail:
            plain = h * d["plain_z"] * d["sigH_bps"]
            cal = h * d["calibrated_z"] * d["sigH_bps"]
            note = "used (t %+.1f)" % d["t"] if d["weight"] > 0 else "not enough evidence (t %+.1f)" % d["t"]
            if d["sessions"] < TAIL_MIN_DAYS:
                note = "too few sessions (%d)" % d["sessions"]
            lines.append("      top %-9s plain %5.1f bp -> calibrated %5.1f bp   %s"
                         % (_top_label(d["group"]), plain, cal, note))
    if meta.get("edge_table"):
        lines.append("    %-12s %9s %11s %10s %8s %9s %7s %13s" % ("signals", "count", "predicted", "realized", "cost",
                                                            "net", "hit", "sessions up"))
        for r in meta["edge_table"]:
            ses = "%d of %d" % (r["sessions_up"], r["sessions"]) if "sessions" in r else "-"
            lines.append("    %-12s %9s %8.1f bp %7.1f bp %5.1f bp %+6.1f bp %6.1f%% %13s"
                         % (r["bucket"], format(r["n"], ","), r["predicted_bps"], r["realized_bps"],
                            r["cost_bps"], r["net_bps"], 100 * r["hit_rate"], ses))
    if meta["has_edge"]:
        lines.append("    Verdict: statistically detectable edge out-of-sample. Whether it survives costs is the 'net' column.")
    else:
        lines.append("    Verdict: NO reliable edge detected out-of-sample (IC t-stat below 2.5). Predicted edges are")
        lines.append("    shrunk accordingly, so expect few or no trades until the evidence improves.")
    return "\n".join(lines)

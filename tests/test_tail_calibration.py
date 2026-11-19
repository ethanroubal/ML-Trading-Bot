"""Strongest-signal calibration: uses real evidence, ignores luck, never flips a signal."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from daytrader.model import Model, apply_calibration, tail_calibration  # noqa: E402

DAYS, PER_DAY = 40, 3000


def _sample(rng, day_effect_mean=0.0, day_effect_sd=0.0, top_q=0.98, days=DAYS):
    raw = rng.normal(size=days * PER_DAY)
    day = np.repeat(np.arange(days), PER_DAY)
    y = rng.normal(size=raw.size)
    top = np.abs(raw) >= np.quantile(np.abs(raw), top_q)
    # the strongest signals share one outcome per session: they win or lose together
    shock = rng.normal(day_effect_mean, day_effect_sd, days)[day]
    y = y + np.sign(raw) * top * shock
    return raw, y, day


def _top_weight(diag):
    return max(d["weight"] for d in diag)


def test_pure_noise_falls_back_to_the_plain_scaling():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(40):
        raw, y, day = _sample(rng)
        kx, kg, diag = tail_calibration(raw, y, day, base_scale=0.0)
        hits += _top_weight(diag) > 0
        assert np.all(kg >= 0)
    assert hits <= 3                                  # rare false positives, and they are shrunk


def test_a_few_lucky_sessions_are_not_mistaken_for_an_edge():
    """Top signals that win big on some days and lose on others: a row-level test would call
    this significant; measured per session it is not."""
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(40):
        raw, y, day = _sample(rng, day_effect_mean=0.0, day_effect_sd=1.0)
        _, _, diag = tail_calibration(raw, y, day, base_scale=0.0)
        hits += _top_weight(diag) > 0
    assert hits <= 3


def test_a_consistent_edge_in_the_strongest_signals_is_used():
    rng = np.random.default_rng(3)
    raw, y, day = _sample(rng, day_effect_mean=0.4, day_effect_sd=0.3)
    kx, kg, diag = tail_calibration(raw, y, day, base_scale=0.0)
    top = diag[-1]
    assert top["weight"] > 0.5 and top["t"] > 5
    assert 0.2 < top["calibrated_z"] <= 0.45          # close to the true 0.4, shrunk a little
    assert diag[0]["calibrated_z"] == pytest.approx(0.0, abs=1e-9)   # the bulk stays at the plain value
    assert np.all(np.diff(kg) >= 0)                   # stronger signal, never a smaller edge
    z = apply_calibration(np.array([-4.0, -0.1, 0.1, 4.0]), 0.0, 0.75, kx, kg)
    assert z[0] < 0 < z[3] and z[1] == z[2] == 0.0    # direction kept, haircut applied
    assert z[3] == pytest.approx(0.75 * kg[-1])


def test_never_reverses_a_signal_and_needs_enough_sessions():
    rng = np.random.default_rng(4)
    raw, y, day = _sample(rng, day_effect_mean=-0.5, day_effect_sd=0.2)     # strongest signals reliably wrong
    kx, kg, diag = tail_calibration(raw, y, day, base_scale=0.05)
    assert np.all(kg >= 0)
    assert 0.0 <= diag[-1]["calibrated_z"] < diag[-1]["plain_z"]     # trusted less, but never flipped
    raw, y, day = _sample(rng, day_effect_mean=0.6, day_effect_sd=0.1, days=10)   # strong but only 10 sessions
    _, _, diag = tail_calibration(raw, y, day, base_scale=0.0)
    assert _top_weight(diag) == 0.0


def test_model_without_knots_behaves_as_before():
    m = Model()
    m.slope, m.evidence, m.haircut = 0.5, 0.8, 0.75
    raw = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(apply_calibration(raw, m.slope * m.evidence, m.haircut, None, None), 0.3 * raw)


def test_haircut_can_change_without_retraining():
    m = Model()
    m.slope, m.evidence, m.haircut, m.haircut_trained = 0.5, 1.0, 0.75, 0.75
    m.pwin_x, m.pwin_y = np.array([0.0, 0.1, 0.2]), np.array([0.5, 0.55, 0.6])
    m.abs_quantiles = np.linspace(0, 0.2, 101)
    m.raw_predict = lambda X: np.asarray(X, dtype=float)
    before_z = m.expected_z(np.array([0.2]))
    before_p, before_s = m.p_win(before_z), m.strength(before_z)
    m.use_haircut(1.0)
    after_z = m.expected_z(np.array([0.2]))
    assert after_z[0] == pytest.approx(before_z[0] / 0.75)          # bigger expected move...
    assert m.p_win(after_z) == pytest.approx(before_p)               # ...same signal, same hit rate
    assert m.strength(after_z) == pytest.approx(before_s)

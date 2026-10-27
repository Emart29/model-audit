import numpy as np
import pytest

from model_audit.calibration import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)


def test_perfect_probabilities_have_zero_error():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(y, p) == pytest.approx(0.0)
    assert brier_score(y, p) == pytest.approx(0.0)
    assert maximum_calibration_error(y, p) == pytest.approx(0.0)


def test_overconfidence_is_measured():
    # Says 0.9 every time and is right half the time: ECE is 0.4.
    y = np.array([1, 0] * 500)
    p = np.full(1000, 0.9)

    assert expected_calibration_error(y, p, n_bins=5) == pytest.approx(0.4, abs=1e-6)
    assert brier_score(y, p) == pytest.approx(0.41, abs=1e-6)


def test_bins_cover_every_row():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1000)
    y = rng.binomial(1, p)

    bins = reliability_bins(y, p, n_bins=10)

    assert sum(b.n for b in bins) == 1000
    assert all(0.0 <= b.accuracy <= 1.0 for b in bins)


def test_quantile_bins_are_non_empty_when_possible():
    p = np.linspace(0.01, 0.99, 200)
    y = (p > 0.5).astype(int)

    bins = reliability_bins(y, p, n_bins=10, strategy="quantile")

    assert len(bins) >= 8
    assert all(b.n > 0 for b in bins)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        expected_calibration_error([0, 1], [0.2, 0.3, 0.4])


def test_probabilities_outside_unit_interval_are_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([0, 1], [0.2, 1.4])

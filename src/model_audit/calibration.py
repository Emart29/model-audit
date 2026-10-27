"""Calibration: are the predicted probabilities telling the truth?

A model can rank well and still be badly calibrated — it says 0.9 and is right
60% of the time. That matters more than accuracy when the score is read by a
person making a decision, and it is the one fairness check that catches a model
which is accurate overall but over-confident for one group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    n: int
    confidence: float
    accuracy: float


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    if y_prob.size and (y_prob.min() < 0 or y_prob.max() > 1):
        raise ValueError("y_prob must lie in [0, 1]")
    if not np.isin(y_true, [0.0, 1.0]).all():
        raise ValueError("y_true must be binary (0/1) for calibration")
    return y_true, y_prob


def _edges(y_prob: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy != "quantile":
        raise ValueError(f"unknown strategy {strategy!r}")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_prob, quantiles))
    if edges.size < 2:
        return np.array([0.0, 1.0])
    edges[0], edges[-1] = 0.0, 1.0
    return edges


def reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> list[ReliabilityBin]:
    """Bin predictions and report confidence vs observed accuracy in each bin."""
    y_true, y_prob = _validate(y_true, y_prob)
    if y_prob.size == 0:
        return []
    edges = _edges(y_prob, n_bins, strategy)
    # Right-closed bins except the first, so 0.0 lands in the first bin.
    index = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, len(edges) - 2)
    bins: list[ReliabilityBin] = []
    for b in range(len(edges) - 1):
        mask = index == b
        if not mask.any():
            continue
        bins.append(
            ReliabilityBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                n=int(mask.sum()),
                confidence=float(y_prob[mask].mean()),
                accuracy=float(y_true[mask].mean()),
            )
        )
    return bins


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """Weighted mean |accuracy - confidence| across bins."""
    y_true, y_prob = _validate(y_true, y_prob)
    if y_prob.size == 0:
        return float("nan")
    bins = reliability_bins(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    total = sum(b.n for b in bins)
    if total == 0:
        return float("nan")
    return float(sum(b.n * abs(b.accuracy - b.confidence) for b in bins) / total)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of the probability. Lower is better."""
    y_true, y_prob = _validate(y_true, y_prob)
    if y_prob.size == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """The worst single bin. Catches a local failure that an average hides."""
    bins = reliability_bins(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    if not bins:
        return float("nan")
    return float(max(abs(b.accuracy - b.confidence) for b in bins))

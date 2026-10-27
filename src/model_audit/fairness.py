"""Fairness metrics, computed from scratch.

Three canonical checks plus a calibration gap, each with the threshold it is
judged against written down next to it. The thresholds are conventions, not
laws: 0.10 for parity-style differences and 0.80 for the four-fifths rule are
the values used in most fairness toolkits and in US employment guidance, and
they are exposed as arguments so a report can say which ones it used.

Every group carries its ``n``. A demographic parity gap computed on twelve
people is noise, and a toolkit that hides that is worse than no toolkit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .calibration import brier_score, expected_calibration_error
from .types import AttributeAudit, FairnessMetric, GroupStats


def _binary(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    unique = np.unique(values)
    if unique.size > 2:
        raise ValueError(f"{name} must be binary, got {unique.size} distinct values")
    if unique.size == 0:
        return values.astype(int)
    if set(unique.tolist()) <= {0, 1, False, True}:
        return values.astype(int)
    if unique.size == 2:
        return (values == unique[1]).astype(int)
    return np.zeros(values.shape, dtype=int)


def _spread(values: Sequence[float | None]) -> float:
    present = [v for v in values if v is not None and not np.isnan(v)]
    if len(present) < 2:
        return 0.0
    return float(max(present) - min(present))


def audit_attribute(
    attribute: str,
    values: Sequence,
    y_true: Sequence,
    y_pred: Sequence,
    y_score: Sequence | None = None,
    *,
    min_group_size: int = 30,
    dp_threshold: float = 0.10,
    eo_threshold: float = 0.10,
    di_threshold: float = 0.80,
    ece_threshold: float = 0.10,
    n_bins: int = 10,
) -> AttributeAudit:
    """Audit one protected attribute.

    ``dp_threshold``  maximum allowed difference in positive rate.
    ``eo_threshold``  maximum allowed difference in TPR or FPR.
    ``di_threshold``  minimum allowed ratio between the lowest and highest
                      positive rate (the four-fifths rule).
    ``ece_threshold`` maximum allowed spread in per-group calibration error.
    """
    values = np.asarray(values)
    y_true = _binary(np.asarray(y_true), "y_true")
    y_pred = _binary(np.asarray(y_pred), "y_pred")
    if not (len(values) == len(y_true) == len(y_pred)):
        raise ValueError("values, y_true and y_pred must be the same length")
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        if len(y_score) != len(y_true):
            raise ValueError("y_score must be the same length as y_true")

    labels = np.array([str(v) for v in values])
    unique, counts = np.unique(labels, return_counts=True)
    order = sorted(range(len(unique)), key=lambda i: (-counts[i], unique[i]))

    groups: list[GroupStats] = []
    for i in order:
        label = str(unique[i])
        mask = labels == unique[i]
        n = int(mask.sum())
        positive_rate = float(y_pred[mask].mean()) if n else 0.0

        positives = int(((y_true == 1) & mask).sum())
        negatives = int(((y_true == 0) & mask).sum())
        tpr = float(((y_pred == 1) & (y_true == 1) & mask).sum() / positives) if positives else None
        fpr = float(((y_pred == 1) & (y_true == 0) & mask).sum() / negatives) if negatives else None

        ece = brier = None
        if y_score is not None and n:
            ece = expected_calibration_error(y_true[mask], y_score[mask], n_bins=n_bins)
            brier = brier_score(y_true[mask], y_score[mask])

        groups.append(
            GroupStats(
                group=label,
                n=n,
                positive_rate=positive_rate,
                tpr=tpr,
                fpr=fpr,
                ece=ece,
                brier=brier,
            )
        )

    rates = [g.positive_rate for g in groups]
    dp = _spread(rates)
    di = float(min(rates) / max(rates)) if max(rates) > 0 else 1.0
    eo = max(_spread([g.tpr for g in groups]), _spread([g.fpr for g in groups]))
    cal = _spread([g.ece for g in groups])

    metrics = [
        FairnessMetric(
            name="demographic_parity_difference",
            value=dp,
            threshold=dp_threshold,
            passed=dp <= dp_threshold,
            rule=f"max-min positive rate <= {dp_threshold}",
        ),
        FairnessMetric(
            name="disparate_impact_ratio",
            value=di,
            threshold=di_threshold,
            passed=di >= di_threshold,
            rule=f"min/max positive rate >= {di_threshold}",
        ),
        FairnessMetric(
            name="equalized_odds_difference",
            value=eo,
            threshold=eo_threshold,
            passed=eo <= eo_threshold,
            rule=f"max TPR/FPR spread <= {eo_threshold}",
        ),
    ]
    if y_score is not None and len(groups) > 1:
        metrics.append(
            FairnessMetric(
                name="calibration_gap",
                value=cal,
                threshold=ece_threshold,
                passed=cal <= ece_threshold,
                rule=f"max-min ECE <= {ece_threshold}",
            )
        )

    return AttributeAudit(
        attribute=attribute,
        groups=tuple(groups),
        metrics=tuple(metrics),
        min_group_size=min_group_size,
    )

"""Anchors: a rule that is sufficient for this prediction, not just correlated.

Greedy rule growth. Start from the whole dataset, repeatedly add the predicate
that most increases precision while keeping coverage above a floor, and stop at
the precision target. It is the greedy version of the Anchors algorithm — no
beam search — and it is deliberately reported with both precision *and*
coverage, because a rule that covers four rows and is always right is not an
explanation of the model.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .types import Anchor


def find_anchor(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    index: int,
    feature_names: Sequence[str],
    *,
    precision_threshold: float = 0.95,
    min_coverage: float = 0.05,
    max_predicates: int = 5,
    n_thresholds: int = 5,
) -> Anchor:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(predict(X)).astype(int).ravel()
    target = int(labels[index])
    n = len(labels)

    candidates: list[tuple[str, str, float, np.ndarray]] = []
    for j, name in enumerate(feature_names):
        column = X[:, j]
        unique = np.unique(column)
        if unique.size <= 2 and set(unique.tolist()) <= {0.0, 1.0}:
            if column[index] == 1.0:
                candidates.append((str(name), "==", 1.0, column == 1.0))
            continue
        thresholds = np.unique(np.quantile(column, np.linspace(0.1, 0.9, n_thresholds)))
        for threshold in thresholds:
            if column[index] <= threshold:
                candidates.append((str(name), "<=", float(threshold), column <= threshold))
            if column[index] > threshold:
                candidates.append((str(name), ">", float(threshold), column > threshold))

    mask = np.ones(n, dtype=bool)
    rules: list[tuple[str, str, float]] = []
    used: set[tuple[str, str, float]] = set()

    def precision_of(m: np.ndarray) -> float:
        if not m.any():
            return 0.0
        return float((labels[m] == target).mean())

    current_precision = precision_of(mask)
    for _ in range(max_predicates):
        if current_precision >= precision_threshold:
            break
        best = None
        for feature, op, value, candidate_mask in candidates:
            key = (feature, op, value)
            if key in used:
                continue
            combined = mask & candidate_mask
            coverage = combined.mean()
            if coverage < min_coverage or not combined.any():
                continue
            precision = precision_of(combined)
            if best is None or (precision, coverage) > (best[0], best[1]):
                best = (precision, coverage, feature, op, value, combined, key)
        if best is None:
            break
        precision, _, feature, op, value, combined, key = best
        if precision <= current_precision and rules:
            break
        mask = combined
        rules.append((feature, op, value))
        used.add(key)
        current_precision = precision

    return Anchor(
        index=index,
        rules=tuple(rules),
        precision=float(current_precision),
        coverage=float(mask.mean()),
        n_matched=int(mask.sum()),
        prediction=target,
        target_precision=precision_threshold,
    )

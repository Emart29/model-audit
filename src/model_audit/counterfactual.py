"""Counterfactuals: the smallest change that flips the decision.

The search is greedy over *raw* features, not encoded columns, so a suggestion
reads "education = Masters" rather than "education_Masters = 1". Two rules keep
the output honest:

* **Immutable features are never touched.** Age, sex, race and anything else the
  caller marks immutable cannot appear in a suggestion, even if changing them
  would flip the prediction fastest. A counterfactual you are not allowed to act
  on is not a counterfactual.
* **No flip, no result.** If the search cannot flip the prediction within the
  allowed features and steps, it returns nothing. A "closest attempt" that does
  not change the outcome is not a counterfactual.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .preprocessing import Encoder
from .types import Counterfactual


def _allowed(encoder: Encoder, immutable: Sequence[str], actionable: Sequence[str] | None) -> set[str]:
    every = set(encoder.raw_columns)
    blocked = set(immutable)
    if actionable is not None:
        allowed = set(actionable) - blocked
    else:
        allowed = every - blocked
    unknown = allowed - every
    if unknown:
        raise KeyError(f"unknown features: {sorted(unknown)}")
    return allowed


def _distance(encoder: Encoder, changes: dict[str, tuple[Any, Any]]) -> float:
    total = 0.0
    for feature, (old, new) in changes.items():
        if feature in encoder.categories:
            total += 1.0
        else:
            low, high = encoder.numeric_bounds.get(feature, (0.0, 1.0))
            span = high - low
            total += abs(float(new) - float(old)) / span if span > 0 else 1.0
    return total


def _numeric_candidates(encoder: Encoder, feature: str, current: float) -> list[float]:
    """Values worth trying for a numeric feature.

    The nearest values above and below the current one, plus the observed
    extremes. Taking only the nearest values would never reach a decision
    boundary that sits further out than the local quantiles.
    """
    low, high = encoder.numeric_bounds.get(feature, (current, current))
    quantiles = encoder.numeric_quantiles.get(feature, (low, high))
    options = [q for q in quantiles if abs(q - current) > 1e-9]
    if not options:
        return [low, high] if high > low else []
    below = [q for q in options if q < current]
    above = [q for q in options if q > current]
    chosen: list[float] = []
    if below:
        chosen.append(max(below))
    if above:
        chosen.append(min(above))
    chosen.extend([min(options), max(options)])
    unique: list[float] = []
    for value in chosen:
        if all(abs(value - seen) > 1e-9 for seen in unique):
            unique.append(value)
    return unique[:4]


def generate_counterfactuals(
    predict_proba: Callable[[np.ndarray], np.ndarray],
    encoder: Encoder,
    row: Mapping[str, Any],
    *,
    target_label: int | None = None,
    k: int = 3,
    max_steps: int = 4,
    immutable: Sequence[str] = (),
    actionable: Sequence[str] | None = None,
) -> list[Counterfactual]:
    """Search for up to ``k`` actionable changes that flip the prediction."""
    allowed = _allowed(encoder, immutable, actionable)
    original = {column: row[column] for column in encoder.raw_columns}
    p_original = float(predict_proba(encoder.transform_row(original))[0])
    original_prediction = int(p_original >= 0.5)
    if target_label is None:
        target_label = 1 - original_prediction

    def score(probability: float) -> float:
        return probability if target_label == 1 else 1.0 - probability

    def predict_row(candidate: Mapping[str, Any]) -> float:
        return float(predict_proba(encoder.transform_row(candidate))[0])

    moves: list[tuple[str, Any]] = []
    for feature in encoder.numeric:
        if feature in allowed:
            for value in _numeric_candidates(encoder, feature, float(original[feature])):
                moves.append((feature, value))
    for feature, levels in encoder.categories.items():
        if feature in allowed:
            current = str(original[feature])
            moves.extend((feature, level) for level in levels if level != current)

    found: list[Counterfactual] = []
    seen: set[tuple[str, str]] = set()

    def record(candidate: Mapping[str, Any], changes: dict[str, tuple[Any, Any]]) -> None:
        probability = predict_row(candidate)
        prediction = int(probability >= 0.5)
        if prediction != target_label:
            return
        key = tuple(sorted((f, str(v[1])) for f, v in changes.items()))
        if key in seen:
            return
        seen.add(key)
        found.append(
            Counterfactual(
                index=-1,
                changes=tuple((f, v[0], v[1]) for f, v in changes.items()),
                original_prediction=original_prediction,
                counterfactual_prediction=prediction,
                original_probability=p_original,
                counterfactual_probability=probability,
                distance=_distance(encoder, changes),
            )
        )

    # One change is usually enough, and a one-line suggestion is worth more than
    # a shorter distance that needs three.
    for feature, value in moves:
        candidate = dict(original)
        candidate[feature] = value
        record(candidate, {feature: (original[feature], value)})

    if not found:
        current = dict(original)
        changes: dict[str, tuple[Any, Any]] = {}
        used: set[str] = set()
        for _ in range(max_steps):
            options: list[tuple[float, float, str, Any]] = []
            base = score(predict_row(current))
            for feature, value in moves:
                if feature in used:
                    continue
                candidate = dict(current)
                candidate[feature] = value
                gain = score(predict_row(candidate)) - base
                step = _distance(encoder, {feature: (current[feature], value)})
                options.append((step, -gain, feature, value))
            if not options:
                break
            # Smallest plausible step first, not the biggest probability jump:
            # a suggestion nobody can act on is not a counterfactual.
            options.sort()
            applied = False
            for _, negative_gain, feature, value in options:
                if -negative_gain <= 0.01:
                    continue
                changes.setdefault(feature, (original[feature], value))
                current[feature] = value
                used.add(feature)
                applied = True
                break
            if not applied:
                break
            if int(predict_row(current) >= 0.5) == target_label:
                record(current, changes)
                break

    found.sort(key=lambda c: (len(c.changes), c.distance))
    return [
        Counterfactual(index=c.index, changes=c.changes, original_prediction=c.original_prediction,
                       counterfactual_prediction=c.counterfactual_prediction,
                       original_probability=c.original_probability,
                       counterfactual_probability=c.counterfactual_probability,
                       distance=c.distance)
        for c in found[:k]
    ]

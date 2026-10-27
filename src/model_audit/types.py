"""Result types shared across the audit.

Everything here is immutable and serialisable. The audit methods return these
rather than loose dicts so that a report, an API response and a test all read
the same object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GroupStats:
    """Per-group outcome statistics. ``None`` means the group had no members of
    the class the statistic needs (e.g. no negatives, so no false-positive rate)."""

    group: str
    n: int
    positive_rate: float
    tpr: float | None
    fpr: float | None
    ece: float | None
    brier: float | None


@dataclass(frozen=True)
class FairnessMetric:
    name: str
    value: float
    threshold: float
    passed: bool
    rule: str


@dataclass(frozen=True)
class AttributeAudit:
    attribute: str
    groups: tuple[GroupStats, ...]
    metrics: tuple[FairnessMetric, ...]
    min_group_size: int

    @property
    def reliable(self) -> bool:
        """False when at least one group is too small for its rate to mean much."""
        return all(g.n >= self.min_group_size for g in self.groups)

    @property
    def smallest_group(self) -> int:
        return min((g.n for g in self.groups), default=0)

    def metric(self, name: str) -> FairnessMetric:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(f"no metric named {name!r} on attribute {self.attribute!r}")

    @property
    def failures(self) -> tuple[FairnessMetric, ...]:
        return tuple(m for m in self.metrics if not m.passed)


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    rank: int


@dataclass(frozen=True)
class LocalExplanation:
    index: int
    method: str
    prediction: Any
    probability: float | None
    contributions: tuple[FeatureContribution, ...]
    intercept: float | None = None
    r2: float | None = None


@dataclass(frozen=True)
class Counterfactual:
    index: int
    changes: tuple[tuple[str, Any, Any], ...]
    original_prediction: Any
    counterfactual_prediction: Any
    original_probability: float
    counterfactual_probability: float
    distance: float

    @property
    def flipped(self) -> bool:
        return self.original_prediction != self.counterfactual_prediction

    def as_dict(self) -> dict[str, Any]:
        return {
            "changes": {f: {"from": a, "to": b} for f, a, b in self.changes},
            "original_prediction": self.original_prediction,
            "counterfactual_prediction": self.counterfactual_prediction,
            "original_probability": round(self.original_probability, 4),
            "counterfactual_probability": round(self.counterfactual_probability, 4),
            "distance": round(self.distance, 4),
        }


@dataclass(frozen=True)
class Anchor:
    index: int
    rules: tuple[tuple[str, str, Any], ...]
    precision: float
    coverage: float
    n_matched: int
    prediction: Any
    target_precision: float = 0.95

    @property
    def meets_target(self) -> bool:
        return self.precision >= self.target_precision

    def text(self) -> str:
        if not self.rules:
            return "always"
        parts: list[str] = []
        for feature, op, value in self.rules:
            # One-hot columns already read as "sex=Female"; "== 1.0" adds nothing.
            if op == "==" and "=" in feature:
                parts.append(feature)
            else:
                parts.append(f"{feature} {op} {value!r}")
        return " AND ".join(parts)


@dataclass(frozen=True)
class FeatureImportance:
    feature: str
    importance: float
    rank: int
    method: str


@dataclass
class ModelCard:
    model_name: str
    version: str
    description: str
    intended_use: str
    limitations: str
    task: str
    target: str
    dataset_name: str
    dataset_source: str
    n_rows: int
    n_features: int
    features: list[str]
    metrics: dict[str, float]
    fairness: list[dict[str, Any]] = field(default_factory=list)
    global_importance: list[dict[str, Any]] = field(default_factory=list)
    protected_attributes: list[str] = field(default_factory=list)
    generated_at: str = ""
    generator: str = "model-audit 0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "name": self.model_name,
                "version": self.version,
                "description": self.description,
                "intended_use": self.intended_use,
                "limitations": self.limitations,
            },
            "task": self.task,
            "target": self.target,
            "dataset": {
                "name": self.dataset_name,
                "source": self.dataset_source,
                "n_rows": self.n_rows,
                "n_features": self.n_features,
                "features": self.features,
                "protected_attributes": self.protected_attributes,
            },
            "metrics": self.metrics,
            "fairness": self.fairness,
            "global_importance": self.global_importance,
            "generated_at": self.generated_at,
            "generator": self.generator,
        }

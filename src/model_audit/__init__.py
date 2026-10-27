"""model-audit: explain any model, audit it for bias, and write the model card."""

from .auditor import ModelAuditor
from .preprocessing import Encoder, fit_encoder
from .types import (
    Anchor,
    AttributeAudit,
    Counterfactual,
    FairnessMetric,
    FeatureContribution,
    FeatureImportance,
    GroupStats,
    LocalExplanation,
    ModelCard,
)

__all__ = [
    "ModelAuditor",
    "Encoder",
    "fit_encoder",
    "Anchor",
    "AttributeAudit",
    "Counterfactual",
    "FairnessMetric",
    "FeatureContribution",
    "FeatureImportance",
    "GroupStats",
    "LocalExplanation",
    "ModelCard",
]

__version__ = "0.1.0"

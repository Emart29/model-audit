"""The auditor: one object that holds a model, its data and every check."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .anchors import find_anchor
from .calibration import brier_score, expected_calibration_error, maximum_calibration_error, reliability_bins
from .counterfactual import generate_counterfactuals
from .explain_global import global_importance
from .explain_local import local_surrogate
from .fairness import audit_attribute
from .preprocessing import Encoder
from .types import Anchor, AttributeAudit, Counterfactual, FeatureImportance, LocalExplanation, ModelCard


def _to_binary(y: Sequence) -> tuple[np.ndarray, tuple[Any, Any]]:
    values = np.asarray(y)
    unique = np.unique(values)
    if unique.size > 2:
        raise ValueError(
            f"expected a binary target, got {unique.size} classes. "
            "The fairness and calibration checks are defined for binary outcomes only."
        )
    if unique.size == 1:
        return np.zeros(len(values), dtype=int), (unique[0], unique[0])
    if set(unique.tolist()) <= {0, 1, False, True}:
        return values.astype(int), (0, 1)
    return (values == unique[1]).astype(int), (unique[0], unique[1])


class ModelAuditor:
    """Explain, audit and document a fitted model.

    ``X`` is the numeric design matrix the model was fitted on. ``frame`` is the
    raw frame it came from, if you want counterfactuals in raw feature names.
    """

    def __init__(
        self,
        model,
        X: np.ndarray,
        y: Sequence,
        *,
        feature_names: Sequence[str] | None = None,
        task: str = "classification",
        name: str = "model",
        target_name: str = "target",
        dataset_name: str = "dataset",
        dataset_source: str = "",
        protected_attributes: Mapping[str, Sequence] | None = None,
        frame=None,
        encoder: Encoder | None = None,
        immutable_features: Sequence[str] = (),
        random_state: int = 0,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.model = model
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim != 2:
            raise ValueError("X must be a 2-D matrix")
        self.task = task
        self.name = name
        self.target_name = target_name
        self.dataset_name = dataset_name
        self.dataset_source = dataset_source
        self.frame = frame
        self.encoder = encoder
        self.immutable_features = tuple(immutable_features)
        self.random_state = random_state
        self.feature_names = list(feature_names) if feature_names is not None else [
            f"x{i}" for i in range(self.X.shape[1])
        ]
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature_names does not match the number of columns in X")

        if task == "classification":
            self.y, self.classes = _to_binary(y)
        else:
            self.y = np.asarray(y, dtype=float)
            self.classes = (None, None)
        if len(self.y) != self.X.shape[0]:
            raise ValueError("X and y have different lengths")

        self.protected_attributes = {k: np.asarray(v) for k, v in (protected_attributes or {}).items()}
        for key, values in self.protected_attributes.items():
            if len(values) != len(self.y):
                raise ValueError(f"protected attribute {key!r} has {len(values)} rows, expected {len(self.y)}")

        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------ prediction

    def _predict_score(self, X: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return np.asarray(self.model.predict(X), dtype=float).ravel()
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(X))
            if probabilities.ndim == 1:
                return probabilities.ravel()
            return probabilities[:, -1].ravel()
        if hasattr(self.model, "decision_function"):
            raw = np.asarray(self.model.decision_function(X), dtype=float).ravel()
            return 1.0 / (1.0 + np.exp(-raw))
        if callable(self.model):
            return np.asarray(self.model(X), dtype=float).ravel()
        raise TypeError("model must expose predict_proba, decision_function or be callable")

    def predict(self, X: np.ndarray | None = None) -> np.ndarray:
        X = self.X if X is None else np.asarray(X, dtype=float)
        if self.task == "regression":
            return self._predict_score(X)
        return (self._predict_score(X) >= 0.5).astype(int)

    @property
    def y_score(self) -> np.ndarray:
        if "y_score" not in self._cache:
            self._cache["y_score"] = self._predict_score(self.X)
        return self._cache["y_score"]

    @property
    def y_pred(self) -> np.ndarray:
        if "y_pred" not in self._cache:
            self._cache["y_pred"] = self.predict()
        return self._cache["y_pred"]

    # --------------------------------------------------------------------- metrics

    def metrics(self) -> dict[str, float]:
        if "metrics" in self._cache:
            return self._cache["metrics"]
        if self.task == "regression":
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            result = {
                "mae": float(mean_absolute_error(self.y, self.y_score)),
                "rmse": float(np.sqrt(mean_squared_error(self.y, self.y_score))),
                "r2": float(r2_score(self.y, self.y_score)),
            }
        else:
            from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score

            result = {
                "accuracy": float(accuracy_score(self.y, self.y_pred)),
                "precision": float(precision_score(self.y, self.y_pred, zero_division=0)),
                "recall": float(recall_score(self.y, self.y_pred, zero_division=0)),
                "f1": float(f1_score(self.y, self.y_pred, zero_division=0)),
                "brier": brier_score(self.y, self.y_score),
            }
            if len(np.unique(self.y)) == 2:
                result["roc_auc"] = float(roc_auc_score(self.y, self.y_score))
                result["log_loss"] = float(log_loss(self.y, self.y_score, labels=[0, 1]))
        self._cache["metrics"] = result
        return result

    # -------------------------------------------------------------------- fairness

    def fairness(
        self,
        *,
        min_group_size: int = 30,
        dp_threshold: float = 0.10,
        eo_threshold: float = 0.10,
        di_threshold: float = 0.80,
        ece_threshold: float = 0.10,
    ) -> tuple[AttributeAudit, ...]:
        if self.task != "classification":
            raise ValueError("fairness auditing is defined for classification models")
        if not self.protected_attributes:
            return ()
        audits = []
        for attribute, values in self.protected_attributes.items():
            audits.append(
                audit_attribute(
                    attribute,
                    values,
                    self.y,
                    self.y_pred,
                    self.y_score,
                    min_group_size=min_group_size,
                    dp_threshold=dp_threshold,
                    eo_threshold=eo_threshold,
                    di_threshold=di_threshold,
                    ece_threshold=ece_threshold,
                )
            )
        return tuple(audits)

    # ----------------------------------------------------------------- calibration

    def calibration(self, *, n_bins: int = 10) -> dict[str, Any]:
        if self.task != "classification":
            raise ValueError("calibration is defined for classification models")
        bins = reliability_bins(self.y, self.y_score, n_bins=n_bins)
        return {
            "ece": expected_calibration_error(self.y, self.y_score, n_bins=n_bins),
            "mce": maximum_calibration_error(self.y, self.y_score, n_bins=n_bins),
            "brier": brier_score(self.y, self.y_score),
            "bins": [
                {
                    "lower": round(b.lower, 3),
                    "upper": round(b.upper, 3),
                    "n": b.n,
                    "confidence": round(b.confidence, 3),
                    "accuracy": round(b.accuracy, 3),
                }
                for b in bins
            ],
        }

    # --------------------------------------------------------------- explanations

    def global_importance(self, *, method: str = "auto") -> list[FeatureImportance]:
        key = f"global:{method}"
        if key not in self._cache:
            self._cache[key] = global_importance(
                self.model,
                self.X,
                self.y,
                self.feature_names,
                method=method,
                seed=self.random_state,
            )
        return self._cache[key]

    def local(self, index: int, *, method: str = "surrogate") -> LocalExplanation:
        if method != "surrogate":
            raise ValueError("supported local method: 'surrogate'")
        return local_surrogate(
            self._predict_score,
            self.X,
            index,
            self.feature_names,
            seed=self.random_state,
        )

    def anchors(self, index: int, **kwargs) -> Anchor:
        return find_anchor(
            self.predict,
            self.X,
            index,
            self.feature_names,
            **kwargs,
        )

    def counterfactuals(
        self,
        index: int,
        *,
        k: int = 3,
        max_steps: int = 4,
        immutable: Sequence[str] | None = None,
        actionable: Sequence[str] | None = None,
    ) -> list[Counterfactual]:
        if self.encoder is None or self.frame is None:
            raise ValueError(
                "counterfactuals need the raw frame and the encoder used to build X; "
                "pass frame= and encoder= to ModelAuditor"
            )
        row = self.frame.iloc[index].to_dict()
        results = generate_counterfactuals(
            self._predict_score,
            self.encoder,
            row,
            k=k,
            max_steps=max_steps,
            immutable=self.immutable_features if immutable is None else immutable,
            actionable=actionable,
        )
        return [
            Counterfactual(
                index=index,
                changes=c.changes,
                original_prediction=c.original_prediction,
                counterfactual_prediction=c.counterfactual_prediction,
                original_probability=c.original_probability,
                counterfactual_probability=c.counterfactual_probability,
                distance=c.distance,
            )
            for c in results
        ]

    # ------------------------------------------------------------------- reporting

    def model_card(self, **kwargs) -> ModelCard:
        from .model_card import build_model_card

        return build_model_card(self, **kwargs)

    def report(self, directory: str = "reports", *, stem: str = "audit") -> dict[str, str]:
        from .report import write_report

        return write_report(self, directory, stem=stem)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ModelAuditor(name={self.name!r}, task={self.task!r}, "
            f"n={self.X.shape[0]}, features={self.X.shape[1]})"
        )

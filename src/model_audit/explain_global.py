"""Global explanations: which features move the model, and by how much.

SHAP is used where it is exact and cheap (trees, linear models) and sampled
where it is not. Permutation importance is always available as the
model-agnostic cross-check, and the report shows both because the two disagree
often enough to be worth seeing.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .types import FeatureImportance


def _positive_class(values) -> np.ndarray:
    """Normalise SHAP output to an (n_samples, n_features) matrix."""
    if isinstance(values, list):
        values = values[-1] if len(values) > 1 else values[0]
    array = np.asarray(values)
    if array.ndim == 3:  # (n, features, classes)
        array = array[:, :, -1]
    if array.ndim != 2:
        raise ValueError(f"unexpected SHAP shape {array.shape}")
    return array


def _is_tree(model) -> bool:
    # ``_predictors`` is HistGradientBoosting's forest; ``estimators_`` covers
    # random forests and gradient boosting; ``get_booster`` covers XGBoost/LightGBM.
    return any(
        hasattr(model, attr) for attr in ("get_booster", "tree_", "estimators_", "_predictors")
    )


def _is_linear(model) -> bool:
    return hasattr(model, "coef_")


def _predict_positive(model) -> Callable[[np.ndarray], np.ndarray]:
    if hasattr(model, "predict_proba"):
        return lambda x: np.asarray(model.predict_proba(x))[:, -1]
    return lambda x: np.asarray(model.predict(x), dtype=float)


def shap_importance(
    model,
    X: np.ndarray,
    *,
    max_samples: int = 500,
    background_size: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, str]:
    """Mean absolute SHAP value per feature, plus which explainer produced it."""
    import shap

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    rows = rng.choice(n, size=min(max_samples, n), replace=False) if n > max_samples else np.arange(n)
    sample = X[rows]

    if _is_tree(model):
        explainer = shap.TreeExplainer(model)
        method = "shap_tree"
    elif _is_linear(model):
        background = X[rng.choice(n, size=min(background_size, n), replace=False)]
        explainer = shap.LinearExplainer(model, background)
        method = "shap_linear"
    else:
        background = X[rng.choice(n, size=min(background_size, n), replace=False)]
        explainer = shap.KernelExplainer(_predict_positive(model), background)
        method = "shap_kernel"

    values = explainer.shap_values(sample)
    return np.abs(_positive_class(values)).mean(axis=0), method


def _score(model, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

    y = np.asarray(y)
    if np.unique(y).size <= 2:
        if hasattr(model, "predict_proba") or callable(model):
            try:
                return float(roc_auc_score(y, _predict_positive(model)(X)))
            except Exception:
                pass
        prediction = model.predict(X) if hasattr(model, "predict") else model(X)
        prediction = np.asarray(prediction).ravel()
        if prediction.dtype.kind == "f":
            prediction = (prediction >= 0.5).astype(int)
        return float(accuracy_score(y, prediction))
    prediction = model.predict(X) if hasattr(model, "predict") else model(X)
    return float(r2_score(y, np.asarray(prediction).ravel()))


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_repeats: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Drop in the score when one feature's values are shuffled.

    Implemented here rather than calling scikit-learn's, so it also works on a
    plain callable or an adapter that has no ``fit`` method.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    baseline = _score(model, X, y)
    importances = np.zeros(X.shape[1], dtype=float)
    for column in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            rng.shuffle(shuffled[:, column])
            drops.append(baseline - _score(model, shuffled, y))
        importances[column] = float(np.mean(drops))
    return importances


def rank_importance(
    importance: Sequence[float], feature_names: Sequence[str], method: str
) -> list[FeatureImportance]:
    order = np.argsort(np.asarray(importance))[::-1]
    return [
        FeatureImportance(
            feature=str(feature_names[i]),
            importance=float(importance[i]),
            rank=rank,
            method=method,
        )
        for rank, i in enumerate(order, start=1)
    ]


def global_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    method: str = "auto",
    seed: int = 0,
) -> list[FeatureImportance]:
    """``auto`` prefers exact SHAP, falls back to permutation importance.

    Kernel SHAP costs O(samples x features x background) model evaluations, so
    above ``kernel_max_features`` features it is not worth the wait and the
    model-agnostic permutation ranking is used instead. The method that ran is
    recorded on every row, so the report never claims SHAP when it was not.
    """
    kernel_max_features = 40
    if method in {"auto", "shap"}:
        cheap = _is_tree(model) or _is_linear(model)
        if cheap or method == "shap" or X.shape[1] <= kernel_max_features:
            try:
                values, used = shap_importance(model, X, seed=seed)
                return rank_importance(values, feature_names, used)
            except Exception:
                if method == "shap":
                    raise
    values = permutation_importance(model, X, y, seed=seed)
    return rank_importance(values, feature_names, "permutation")

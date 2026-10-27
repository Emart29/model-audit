import numpy as np
import pytest

from model_audit.explain_global import permutation_importance, rank_importance, shap_importance
from model_audit.explain_local import local_surrogate


def linear_proba(X: np.ndarray) -> np.ndarray:
    # Deliberately gentle and linear: the surrogate should recover it almost exactly.
    return 0.5 + 0.15 * X[:, 0] - 0.10 * X[:, 1]


def test_local_surrogate_recovers_contribution_signs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 2))
    X[0] = [2.0, 2.0]

    explanation = local_surrogate(linear_proba, X, 0, ["a", "b"], n_samples=2000, seed=0)

    values = {c.feature: c.value for c in explanation.contributions}
    assert values["a"] > 0
    assert values["b"] < 0
    assert explanation.r2 > 0.95


def test_local_surrogate_adds_up_to_the_prediction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3))
    explanation = local_surrogate(linear_proba, X, 5, ["a", "b", "c"], n_samples=1500, seed=1)

    reconstructed = explanation.intercept + sum(c.value for c in explanation.contributions)
    assert reconstructed == pytest.approx(explanation.probability, abs=0.05)


def test_permutation_importance_ranks_the_informative_feature_first():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.1 * rng.normal(size=400) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)

    importance = permutation_importance(model, X, y, n_repeats=3, seed=0)
    ranked = rank_importance(importance, ["signal", "noise1", "noise2"], "permutation")

    assert ranked[0].feature == "signal"
    assert ranked[0].rank == 1


def test_shap_importance_is_non_negative_and_ranked():
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 3))
    y = (X[:, 1] > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

    values, method = shap_importance(model, X, max_samples=200, seed=0)
    ranked = rank_importance(values, ["a", "signal", "c"], method)

    assert (values >= 0).all()
    assert method == "shap_tree"
    assert ranked[0].feature == "signal"

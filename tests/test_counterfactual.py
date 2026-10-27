import numpy as np
import pandas as pd
import pytest

from model_audit.counterfactual import generate_counterfactuals
from model_audit.preprocessing import fit_encoder


def frame() -> pd.DataFrame:
    x = np.array([0.1, 0.2, 0.6, 0.9] * 25)
    flag = np.array(["a", "b"] * 50)
    return pd.DataFrame({"x": x, "flag": flag})


def threshold_model(X: np.ndarray) -> np.ndarray:
    # Column 0 is x; the categorical columns are ignored.
    return 1.0 / (1.0 + np.exp(-20.0 * (X[:, 0] - 0.5)))


def flag_model(X: np.ndarray) -> np.ndarray:
    # Columns: [x, flag=a, flag=b]. Only the category matters.
    return np.where(X[:, 2] == 1.0, 0.9, 0.1)


def test_single_change_flips_a_threshold_model():
    df = frame()
    encoder = fit_encoder(df, ["flag"])
    row = {"x": 0.2, "flag": "a"}

    results = generate_counterfactuals(threshold_model, encoder, row, k=3)

    assert results, "expected at least one counterfactual"
    first = results[0]
    assert first.flipped
    assert first.original_prediction == 0
    assert first.counterfactual_prediction == 1
    changed = dict((f, (a, b)) for f, a, b in first.changes)
    assert changed["x"][1] > 0.5


def test_immutable_features_are_never_changed():
    df = frame()
    encoder = fit_encoder(df, ["flag"])
    row = {"x": 0.2, "flag": "a"}

    results = generate_counterfactuals(threshold_model, encoder, row, immutable=["x"])

    assert results == []


def test_categorical_change_is_suggested_and_actionable():
    df = frame()
    encoder = fit_encoder(df, ["flag"])
    row = {"x": 0.2, "flag": "a"}

    results = generate_counterfactuals(flag_model, encoder, row, k=2)

    assert results
    changes = dict((f, (a, b)) for f, a, b in results[0].changes)
    assert changes["flag"] == ("a", "b")
    assert "x" not in changes


def test_actionable_list_limits_the_search():
    df = frame()
    encoder = fit_encoder(df, ["flag"])
    row = {"x": 0.2, "flag": "a"}

    results = generate_counterfactuals(threshold_model, encoder, row, actionable=["flag"])

    assert results == []


def test_unknown_feature_is_rejected():
    encoder = fit_encoder(frame(), ["flag"])
    with pytest.raises(KeyError, match="unknown"):
        generate_counterfactuals(threshold_model, encoder, {"x": 0.2, "flag": "a"}, actionable=["nope"])


def test_distance_prefers_the_smaller_move():
    df = frame()
    encoder = fit_encoder(df, ["flag"])
    row = {"x": 0.2, "flag": "a"}

    results = generate_counterfactuals(threshold_model, encoder, row, k=3)

    distances = [r.distance for r in results]
    assert distances == sorted(distances)

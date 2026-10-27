import numpy as np

from model_audit.anchors import find_anchor


def rule_model(X: np.ndarray) -> np.ndarray:
    return ((X[:, 0] > 0.0) & (X[:, 1] > 0.0)).astype(int)


def test_anchor_recovers_a_rule_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    index = int(np.argmax((X[:, 0] > 0.5) & (X[:, 1] > 0.5)))

    anchor = find_anchor(rule_model, X, index, ["a", "b", "c"], min_coverage=0.02)

    assert anchor.prediction == 1
    assert anchor.precision >= 0.95
    assert anchor.n_matched > 0
    assert any(feature == "a" for feature, _, _ in anchor.rules)
    assert any(feature == "b" for feature, _, _ in anchor.rules)


def test_anchor_always_covers_the_instance():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(500, 2))
    index = 10

    anchor = find_anchor(rule_model, X, index, ["a", "b"], min_coverage=0.01)

    for feature, op, value in anchor.rules:
        column = X[:, 0] if feature == "a" else X[:, 1]
        if op == "<=":
            assert column[index] <= value
        elif op == ">":
            assert column[index] > value


def test_anchor_reports_coverage_not_just_precision():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, size=(1000, 2))

    anchor = find_anchor(rule_model, X, 0, ["a", "b"], min_coverage=0.05)

    assert 0.0 < anchor.coverage <= 1.0
    assert anchor.n_matched == int(round(anchor.coverage * len(X)))


def test_anchor_flags_when_it_cannot_reach_the_precision_target():
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, size=(1000, 2))

    anchor = find_anchor(rule_model, X, 0, ["a", "b"], precision_threshold=0.999, max_predicates=1)

    assert anchor.target_precision == 0.999
    assert anchor.meets_target == (anchor.precision >= 0.999)

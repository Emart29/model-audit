import numpy as np
import pytest

from model_audit.fairness import audit_attribute


def test_perfect_parity_passes_every_check():
    groups = np.array(["a"] * 100 + ["b"] * 100)
    y_true = np.array([1, 0] * 100)
    y_pred = np.array([1, 0] * 100)
    y_score = y_pred.astype(float)

    audit = audit_attribute("group", groups, y_true, y_pred, y_score)

    assert audit.metric("demographic_parity_difference").value == 0.0
    assert audit.metric("disparate_impact_ratio").value == 1.0
    assert audit.metric("equalized_odds_difference").value == 0.0
    assert all(m.passed for m in audit.metrics)
    assert audit.reliable


def test_known_gap_is_measured_and_fails():
    groups = np.array(["a"] * 100 + ["b"] * 100)
    y_true = np.array([1] * 200)
    y_pred = np.array([1] * 100 + [0] * 100)

    audit = audit_attribute("group", groups, y_true, y_pred)

    assert audit.metric("demographic_parity_difference").value == pytest.approx(1.0)
    assert audit.metric("disparate_impact_ratio").value == pytest.approx(0.0)
    assert not audit.metric("demographic_parity_difference").passed
    assert not audit.metric("disparate_impact_ratio").passed
    assert audit.groups[0].group == "a"
    assert audit.groups[0].n == 100


def test_equalized_odds_catches_false_positive_gap():
    # Same positive rate, different error composition: b flags negatives twice
    # as often. Demographic parity is blind to this; equalized odds is not.
    y_true = np.array([1, 0] * 100)
    y_pred = np.array([1, 0] * 50 + [1, 1] * 50)
    groups = np.array(["a"] * 100 + ["b"] * 100)

    audit = audit_attribute("group", groups, y_true, y_pred)

    assert audit.metric("demographic_parity_difference").value == pytest.approx(0.5)
    assert audit.metric("equalized_odds_difference").value == pytest.approx(1.0)
    assert not audit.metric("equalized_odds_difference").passed


def test_small_group_is_flagged_unreliable():
    groups = np.array(["a"] * 200 + ["b"] * 5)
    y_true = np.array([1, 0] * 100 + [1, 1, 0, 0, 1])
    y_pred = np.array([1, 0] * 100 + [1, 1, 0, 0, 1])

    audit = audit_attribute("group", groups, y_true, y_pred, min_group_size=30)

    assert audit.smallest_group == 5
    assert not audit.reliable


def test_multiclass_target_is_rejected():
    groups = np.array(["a", "b", "a", "b"])
    with pytest.raises(ValueError, match="binary"):
        audit_attribute("group", groups, [0, 1, 2, 0], [0, 1, 2, 0])


def test_group_without_negatives_has_no_false_positive_rate():
    groups = np.array(["a"] * 50 + ["b"] * 50)
    y_true = np.array([1] * 50 + [0, 1] * 25)
    y_pred = np.array([1] * 50 + [0, 1] * 25)

    audit = audit_attribute("group", groups, y_true, y_pred)

    assert audit.groups[0].fpr is None
    assert audit.groups[1].fpr == pytest.approx(0.0)

import numpy as np
import pytest

from model_audit import ModelAuditor
from model_audit.datasets import make_synthetic
from model_audit.preprocessing import fit_encoder


@pytest.fixture(scope="module")
def auditor():
    from sklearn.linear_model import LogisticRegression

    frame = make_synthetic(1500, seed=1)
    features = frame.drop(columns=["hired"])
    encoder = fit_encoder(features, ["region", "group"])
    X = encoder.transform(features)
    model = LogisticRegression(max_iter=1000).fit(X, frame["hired"])
    return ModelAuditor(
        model,
        X,
        frame["hired"].to_numpy(),
        feature_names=encoder.feature_names,
        name="synthetic",
        target_name="hired",
        protected_attributes={"group": frame["group"], "region": frame["region"]},
        frame=features,
        encoder=encoder,
    )


def test_metrics_are_complete(auditor):
    metrics = auditor.metrics()
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_fairness_returns_one_audit_per_attribute(auditor):
    audits = auditor.fairness()
    assert [a.attribute for a in audits] == ["group", "region"]
    assert all(a.groups for a in audits)


def test_global_importance_is_ranked(auditor):
    importance = auditor.global_importance()
    assert importance
    assert [f.rank for f in importance] == list(range(1, len(importance) + 1))
    assert all(f.importance >= 0 for f in importance)


def test_local_explanation_is_consistent(auditor):
    explanation = auditor.local(3)
    assert explanation.index == 3
    assert explanation.contributions
    assert explanation.probability == pytest.approx(auditor.y_score[3])
    assert explanation.prediction == int(auditor.y_score[3] >= 0.5)


def test_anchor_and_counterfactual_respect_constraints(auditor):
    anchor = auditor.anchors(0)
    assert 0.0 <= anchor.precision <= 1.0
    assert 0.0 <= anchor.coverage <= 1.0

    results = auditor.counterfactuals(0, k=3, immutable=["skill"])
    for result in results:
        changed = {feature for feature, _, _ in result.changes}
        assert "skill" not in changed
        assert result.flipped


def test_counterfactuals_need_the_raw_frame():
    frame = make_synthetic(200)
    encoder = fit_encoder(frame.drop(columns=["hired"]), ["region", "group"])
    X = encoder.transform(frame.drop(columns=["hired"]))
    auditor = ModelAuditor(
        lambda matrix: np.clip(matrix[:, 0], 0, 1),
        X,
        frame["hired"].to_numpy(),
        feature_names=encoder.feature_names,
    )
    with pytest.raises(ValueError, match="raw frame"):
        auditor.counterfactuals(0)


def test_report_writes_all_artifacts(auditor, tmp_path):
    paths = auditor.report(tmp_path, stem="test")
    assert paths["report"].endswith("test-report.md")
    assert paths["card_json"].endswith("test-model-card.json")
    assert paths["card_pdf"].endswith("test-model-card.pdf")
    assert (tmp_path / "test-report.md").stat().st_size > 1000


def test_string_labels_are_mapped_to_binary():
    frame = make_synthetic(400, seed=2)
    encoder = fit_encoder(frame.drop(columns=["hired"]), ["region", "group"])
    X = encoder.transform(frame.drop(columns=["hired"]))
    labels = np.where(frame["hired"].to_numpy() == 1, "yes", "no")

    auditor = ModelAuditor(lambda matrix: np.full(len(matrix), 0.5), X, labels)

    assert auditor.classes == ("no", "yes")
    assert set(np.unique(auditor.y)) == {0, 1}
    # A constant 0.5 scores as a positive prediction, so accuracy is the base rate.
    assert auditor.metrics()["accuracy"] == pytest.approx(frame["hired"].mean())


def test_torch_adapter_makes_a_module_auditable():
    import torch
    from torch import nn

    from model_audit.adapters import TorchClassifier

    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    clf = TorchClassifier(module)
    X = np.random.default_rng(0).normal(size=(60, 4))

    proba = clf.predict_proba(X)
    assert proba.shape == (60, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    auditor = ModelAuditor(clf, X, (X[:, 0] > 0).astype(int), feature_names=[f"f{i}" for i in range(4)])
    assert "accuracy" in auditor.metrics()
    assert auditor.global_importance(method="permutation")

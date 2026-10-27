import json

from model_audit.model_card import build_model_card, render_markdown, write_json, write_markdown, write_pdf


def make_auditor():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from model_audit import ModelAuditor
    from model_audit.datasets import make_synthetic
    from model_audit.preprocessing import fit_encoder

    frame = make_synthetic(1200, seed=0)
    features = frame.drop(columns=["hired"])
    encoder = fit_encoder(features, ["region", "group"])
    X = encoder.transform(features)
    model = LogisticRegression(max_iter=1000).fit(X, frame["hired"])
    return ModelAuditor(
        model,
        X,
        frame["hired"].to_numpy(),
        feature_names=encoder.feature_names,
        name="synthetic-hiring",
        target_name="hired",
        dataset_name="synthetic hiring",
        dataset_source="generated",
        protected_attributes={"group": frame["group"]},
        frame=features,
        encoder=encoder,
    )


def test_card_has_every_section():
    card = build_model_card(make_auditor(), version="1.2.3")

    assert card.model_name == "synthetic-hiring"
    assert card.version == "1.2.3"
    assert card.n_rows == 1200
    assert card.metrics["accuracy"] > 0.5
    assert card.fairness and card.fairness[0]["attribute"] == "group"
    assert card.global_importance
    assert card.generated_at


def test_markdown_contains_the_fairness_numbers():
    card = build_model_card(make_auditor())
    text = render_markdown(card)

    assert "# Model Card" in text
    assert "Fairness audit" in text
    assert "demographic_parity_difference" in text
    assert "Limitations" in text


def test_json_and_markdown_and_pdf_are_written(tmp_path):
    card = build_model_card(make_auditor())

    json_path = write_json(card, tmp_path / "card.json")
    md_path = write_markdown(card, tmp_path / "card.md")
    pdf_path = write_pdf(card, tmp_path / "card.pdf")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["model"]["name"] == "synthetic-hiring"
    assert payload["fairness"][0]["groups"]
    assert md_path.stat().st_size > 500
    assert pdf_path.stat().st_size > 1000

"""Streamlit demo: upload nothing, audit the Adult model end to end.

Run from the repository root:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_audit import ModelAuditor  # noqa: E402
from model_audit.datasets import ADULT_CATEGORICAL, load_adult, make_synthetic  # noqa: E402
from model_audit.preprocessing import fit_encoder  # noqa: E402

IMMUTABLE_DEFAULT = [
    "age", "sex", "race", "native-country", "marital-status", "relationship",
    "capital-gain", "capital-loss",
]


@st.cache_resource(show_spinner="Training the model…")
def build(sample: int) -> tuple[ModelAuditor, pd.DataFrame]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    frame = load_adult(sample=sample)
    train, test = train_test_split(
        frame, test_size=0.25, random_state=0, stratify=frame["income"]
    )
    categorical = [c for c in ADULT_CATEGORICAL if c in frame.columns]
    encoder = fit_encoder(train.drop(columns=["income"]), categorical)
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=0
    )
    model.fit(encoder.transform(train.drop(columns=["income"])), train["income"])

    raw_test = test.drop(columns=["income"]).reset_index(drop=True)
    auditor = ModelAuditor(
        model,
        encoder.transform(raw_test),
        test["income"].to_numpy(),
        feature_names=encoder.feature_names,
        name="adult-income-hgb",
        target_name="income > 50K",
        dataset_name="UCI Adult",
        dataset_source="OpenML adult v2, 25% held-out split",
        protected_attributes={"sex": raw_test["sex"], "race": raw_test["race"]},
        frame=raw_test,
        encoder=encoder,
    )
    return auditor, raw_test


@st.cache_resource(show_spinner="Training the synthetic model…")
def build_synthetic() -> tuple[ModelAuditor, pd.DataFrame]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    frame = make_synthetic(6000, seed=0)
    train, test = train_test_split(frame, test_size=0.3, random_state=0, stratify=frame["hired"])
    encoder = fit_encoder(train.drop(columns=["hired"]), ["region", "group"])
    model = LogisticRegression(max_iter=1000)
    model.fit(encoder.transform(train.drop(columns=["hired"])), train["hired"])
    raw_test = test.drop(columns=["hired"]).reset_index(drop=True)
    auditor = ModelAuditor(
        model,
        encoder.transform(raw_test),
        test["hired"].to_numpy(),
        feature_names=encoder.feature_names,
        name="synthetic-hiring",
        target_name="hired",
        dataset_name="synthetic hiring",
        dataset_source="generated with a known group penalty",
        protected_attributes={"group": raw_test["group"]},
        frame=raw_test,
        encoder=encoder,
    )
    return auditor, raw_test


def fairness_frame(audits) -> pd.DataFrame:
    """One row per check, carrying the smallest group's n so no gap is read without it."""
    rows = []
    for audit in audits:
        for metric in audit.metrics:
            rows.append(
                {
                    "attribute": audit.attribute,
                    "check": metric.name,
                    "value": round(metric.value, 3),
                    "threshold": metric.threshold,
                    "result": "pass" if metric.passed else "FAIL",
                    "smallest group n": audit.smallest_group,
                    "reliable": "yes" if audit.reliable else "no, indicative only",
                }
            )
    return pd.DataFrame(rows)


def ranked_bars(frame: pd.DataFrame, label: str, value: str) -> None:
    """Horizontal bars in value order.

    ``st.bar_chart`` orders categories alphabetically, which makes a ranking
    chart read as if the first feature alphabetically were the most important.
    """
    import altair as alt

    order = frame.reindex(frame[value].abs().sort_values(ascending=False).index)[label].tolist()
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title=value),
            y=alt.Y(f"{label}:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
            tooltip=[label, alt.Tooltip(f"{value}:Q", format=".3f")],
        )
        .properties(height=max(120, 30 * len(frame)))
    )
    st.altair_chart(chart, width="stretch")


def main() -> None:
    st.set_page_config(page_title="model-audit", page_icon="🔍", layout="wide")
    st.title("model-audit")
    st.caption("Explain any model, audit it for bias, and generate the model card.")

    with st.sidebar:
        dataset = st.radio("Dataset", ["UCI Adult", "Synthetic hiring"], index=0)
        if dataset == "UCI Adult":
            sample = st.select_slider(
                "Rows", options=[4000, 8000, 12000, 20000, 48842], value=48842,
                help="The full dataset reproduces the README. Smaller samples shrink the "
                "minority race groups below the reliability floor.",
            )
            auditor, raw = build(int(sample))
        else:
            auditor, raw = build_synthetic()

        st.divider()
        attributes = st.multiselect(
            "Protected attributes", list(auditor.protected_attributes), default=list(auditor.protected_attributes)
        )
        immutable = st.multiselect(
            "Immutable features (never changed in counterfactuals)",
            list(auditor.encoder.raw_columns),
            default=[c for c in IMMUTABLE_DEFAULT if c in auditor.encoder.raw_columns],
        )

    metrics = auditor.metrics()
    audits = auditor.fairness()
    calibration = auditor.calibration()
    importance = auditor.global_importance()

    overview, fairness, explain, card = st.tabs(
        ["Overview", "Fairness", "Explain a row", "Model card"]
    )

    with overview:
        columns = st.columns(4)
        for column, key in zip(columns, ["accuracy", "roc_auc", "precision", "recall"]):
            if key in metrics:
                column.metric(key.upper(), f"{metrics[key]:.3f}")
        st.write(
            f"{auditor.X.shape[0]:,} held-out rows · {auditor.X.shape[1]} features · "
            f"ECE {calibration['ece']:.3f} · Brier {calibration['brier']:.3f}"
        )
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Top features")
            top = pd.DataFrame(
                [{"feature": f.feature, "importance": f.importance} for f in importance[:12]]
            )
            ranked_bars(top, "feature", "importance")
            if importance:
                st.caption(f"Mean absolute contribution · method: {importance[0].method}")
        with right:
            st.subheader("Calibration")
            bins = pd.DataFrame(calibration["bins"])
            if not bins.empty:
                st.dataframe(
                    bins.assign(gap=lambda d: (d["accuracy"] - d["confidence"]).round(3)),
                    hide_index=True,
                    width="stretch",
                )
        st.subheader("Checks")
        st.dataframe(fairness_frame(audits), hide_index=True, width="stretch")

    with fairness:
        for audit in audits:
            if audit.attribute not in attributes:
                continue
            st.subheader(f"{audit.attribute}")
            if not audit.reliable:
                st.warning(
                    f"Smallest group is n={audit.smallest_group}, below {audit.min_group_size}. "
                    "Treat these gaps as indicative only."
                )
            groups = pd.DataFrame(
                [
                    {
                        "group": g.group,
                        "n": g.n,
                        "positive rate": round(g.positive_rate, 3),
                        "TPR": None if g.tpr is None else round(g.tpr, 3),
                        "FPR": None if g.fpr is None else round(g.fpr, 3),
                        "ECE": None if g.ece is None else round(g.ece, 3),
                    }
                    for g in audit.groups
                ]
            )
            st.dataframe(groups, hide_index=True, width="stretch")
            ranked_bars(groups[["group", "positive rate"]], "group", "positive rate")
            checks = fairness_frame([audit])
            st.dataframe(checks, hide_index=True, width="stretch")

    with explain:
        index = st.number_input(
            "Row index", min_value=0, max_value=auditor.X.shape[0] - 1, value=0, step=1
        )
        explanation = auditor.local(int(index))
        st.write(
            f"Prediction **{explanation.prediction}** · probability "
            f"**{explanation.probability:.3f}** · surrogate R² {explanation.r2:.3f}"
        )
        contributions = pd.DataFrame(
            [{"feature": c.feature, "contribution": c.value} for c in explanation.contributions[:12]]
        )
        ranked_bars(contributions, "feature", "contribution")

        left, right = st.columns(2)
        with left:
            st.subheader("Anchor")
            anchor = auditor.anchors(int(index))
            st.code(anchor.text())
            st.write(
                f"precision {anchor.precision:.3f} · coverage {anchor.coverage:.3f} · "
                f"n={anchor.n_matched}"
            )
            if not anchor.meets_target:
                st.caption(f"Precision target {anchor.target_precision:.2f} not reached.")
        with right:
            st.subheader("Counterfactual")
            try:
                results = auditor.counterfactuals(int(index), k=3, immutable=immutable)
                if results:
                    for result in results:
                        changes = ", ".join(f"{f}: {a!r} → {b!r}" for f, a, b in result.changes)
                        st.write(
                            f"{changes} — p {result.original_probability:.3f} → "
                            f"{result.counterfactual_probability:.3f}"
                        )
                else:
                    st.info("No allowed change flips this prediction.")
            except ValueError as error:
                st.info(str(error))

        st.subheader("Row")
        st.dataframe(raw.iloc[[int(index)]], hide_index=True, width="stretch")

    with card:
        markdown = auditor.model_card()
        from model_audit.model_card import render_markdown

        text = render_markdown(markdown)
        st.download_button("Download Markdown", text, file_name="model-card.md")
        st.download_button(
            "Download JSON", json.dumps(markdown.to_dict(), indent=2), file_name="model-card.json"
        )
        st.markdown(text)


if __name__ == "__main__":
    main()

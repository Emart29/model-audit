"""Datasets used by the demo and the tests.

The demo uses the UCI Adult census extract, fetched through scikit-learn. It is
real, public, and has two protected attributes that produce genuinely different
error rates — which is the point of auditing it.

The synthetic generator exists so the test suite never needs the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ADULT_CATEGORICAL = (
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
)


def load_adult(*, sample: int | None = None, seed: int = 0) -> pd.DataFrame:
    """UCI Adult (48,842 rows) with a binary ``income`` target.

    Two columns are dropped. ``education`` is the free-text twin of
    ``education-num``, so keeping both gives the model the same information
    twice. ``fnlwgt`` is the census sampling weight — a property of the survey,
    not of the person — and it has no business driving a prediction.
    """
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml("adult", version=2, as_frame=True, n_retries=2, delay=1)
    frame = bunch.data.copy()
    target = bunch.target.astype(str).str.strip()

    for column in frame.columns:
        if frame[column].dtype == object or str(frame[column].dtype) == "category":
            frame[column] = frame[column].astype(str).str.strip().replace({"?": np.nan})

    frame = frame.drop(columns=["education", "fnlwgt"], errors="ignore")
    frame["income"] = (target == ">50K").astype(int)
    frame = frame.dropna(subset=["income"])

    if sample is not None and sample < len(frame):
        frame = frame.sample(n=sample, random_state=seed).reset_index(drop=True)
    return frame


def make_synthetic(
    n: int = 4000,
    *,
    seed: int = 0,
    bias: float = 0.35,
) -> pd.DataFrame:
    """A small frame with a known, deliberate unfairness.

    ``score = 2*skill + experience/10 - bias*(group == 'B')`` and the label is
    drawn from ``score``. A model trained on it will learn the group penalty, so
    the fairness audit has something real to find.
    """
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], size=n, p=[0.7, 0.3])
    skill = rng.normal(0, 1, size=n)
    experience = rng.integers(0, 40, size=n)
    region = rng.choice(["north", "south", "east", "west"], size=n)

    score = 2.0 * skill + experience / 10.0 - bias * (group == "B")
    probability = 1.0 / (1.0 + np.exp(-(score - 1.0)))
    hired = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "skill": skill.round(4),
            "experience": experience,
            "region": region,
            "group": group,
            "hired": hired,
        }
    )

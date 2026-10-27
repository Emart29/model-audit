"""Local explanations by a sparse linear surrogate (the LIME idea).

Sample points around the instance, weight them by distance, fit a ridge
regression to the model's probability, and read the coefficients back as
contributions. Writing it here rather than importing LIME keeps the weighting
and the binarisation visible, and makes it testable: on a linear model the
surrogate should recover the sign of the true coefficients.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.linear_model import Ridge

from .types import FeatureContribution, LocalExplanation


def _is_binary(column: np.ndarray) -> bool:
    unique = np.unique(column[~np.isnan(column)])
    return unique.size <= 2 and set(np.unique(unique).tolist()) <= {0.0, 1.0}


def local_surrogate(
    predict_proba: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    index: int,
    feature_names: Sequence[str],
    *,
    n_samples: int = 1000,
    alpha: float = 1.0,
    seed: int = 0,
) -> LocalExplanation:
    X = np.asarray(X, dtype=float)
    x0 = X[index]
    n_features = X.shape[1]
    rng = np.random.default_rng(seed)

    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    binary = np.array([_is_binary(X[:, j]) for j in range(n_features)])
    prevalence = np.clip(X.mean(axis=0), 0.01, 0.99)

    sample = np.empty((n_samples, n_features), dtype=float)
    for j in range(n_features):
        if binary[j]:
            sample[:, j] = rng.binomial(1, prevalence[j], size=n_samples)
        else:
            sample[:, j] = x0[j] + rng.normal(0.0, 1.0, size=n_samples) * scale[j]

    distance = np.abs(sample - x0).sum(axis=1) / (scale.sum() or 1.0)
    kernel_width = 0.75 * np.sqrt(n_features)
    weights = np.exp(-(distance**2) / (kernel_width**2))

    standardised = (sample - X.mean(axis=0)) / scale
    target = predict_proba(sample)
    model = Ridge(alpha=alpha)
    model.fit(standardised, target, sample_weight=weights)

    centred = (x0 - X.mean(axis=0)) / scale
    contributions = model.coef_ * centred
    p0 = float(predict_proba(x0.reshape(1, -1))[0])

    order = np.argsort(np.abs(contributions))[::-1]
    return LocalExplanation(
        index=index,
        method="local_surrogate",
        prediction=int(p0 >= 0.5),
        probability=p0,
        contributions=tuple(
            FeatureContribution(feature=str(feature_names[j]), value=float(contributions[j]), rank=r)
            for r, j in enumerate(order, start=1)
        ),
        intercept=float(model.intercept_),
        r2=float(model.score(standardised, target, sample_weight=weights)),
    )

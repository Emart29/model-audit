"""Turning a raw frame into the numeric matrix a model expects.

The audit itself works on a numeric design matrix, because that is what SHAP,
permutation importance and the linear surrogate all need. Counterfactuals work
the other way round: they need raw feature names and levels so the suggested
change is something a person could act on ("education = Masters", not
"education_Masters = 1").

``Encoder`` is the bridge. Fit it once on the training frame, then use it for
the audit matrix and for every counterfactual candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

MISSING_LEVEL = "__missing__"

#: Percentiles recorded per numeric column and used as counterfactual candidates.
QUANTILE_GRID = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass
class Encoder:
    numeric: tuple[str, ...]
    categories: dict[str, tuple[str, ...]]
    numeric_fill: dict[str, float] = field(default_factory=dict)
    numeric_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    numeric_quantiles: dict[str, tuple[float, ...]] = field(default_factory=dict)
    drop_first: bool = False

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric)
        for column, levels in self.categories.items():
            names.extend(f"{column}={level}" for level in levels)
        return names

    @property
    def groups(self) -> dict[str, list[int]]:
        """Raw categorical column -> the matrix columns that encode it."""
        out: dict[str, list[int]] = {}
        offset = len(self.numeric)
        for column, levels in self.categories.items():
            out[column] = list(range(offset, offset + len(levels)))
            offset += len(levels)
        return out

    @property
    def raw_columns(self) -> list[str]:
        return [*self.numeric, *self.categories]

    def _clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.raw_columns if c not in frame.columns]
        if missing:
            raise KeyError(f"frame is missing columns: {missing}")
        return frame

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        frame = self._clean(frame)
        blocks: list[np.ndarray] = []
        if self.numeric:
            numeric = frame.loc[:, list(self.numeric)].apply(pd.to_numeric, errors="coerce")
            numeric = numeric.fillna({c: self.numeric_fill.get(c, 0.0) for c in self.numeric})
            blocks.append(numeric.to_numpy(dtype=float))
        for column, levels in self.categories.items():
            values = frame[column].astype(object).where(frame[column].notna(), MISSING_LEVEL)
            values = values.astype(str)
            blocks.append(
                np.column_stack([(values == level).to_numpy(dtype=float) for level in levels])
                if levels
                else np.empty((len(frame), 0))
            )
        if not blocks:
            return np.empty((len(frame), 0))
        return np.hstack(blocks)

    def transform_row(self, row: Mapping[str, Any]) -> np.ndarray:
        return self.transform(pd.DataFrame([dict(row)]))

    def to_frame(self, matrix: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(matrix, columns=self.feature_names)


def fit_encoder(
    frame: pd.DataFrame,
    categorical_features: Sequence[str] = (),
    *,
    drop_first: bool = False,
) -> Encoder:
    """Learn column order, category levels and numeric fill values."""
    categorical = tuple(categorical_features)
    unknown = [c for c in categorical if c not in frame.columns]
    if unknown:
        raise KeyError(f"categorical features not in frame: {unknown}")
    numeric = tuple(c for c in frame.columns if c not in categorical)

    categories: dict[str, tuple[str, ...]] = {}
    for column in categorical:
        values = frame[column].astype(object).where(frame[column].notna(), MISSING_LEVEL)
        levels = sorted(values.astype(str).unique().tolist())
        if drop_first and len(levels) > 1:
            levels = levels[1:]
        categories[column] = tuple(levels)

    numeric_fill: dict[str, float] = {}
    numeric_bounds: dict[str, tuple[float, float]] = {}
    numeric_quantiles: dict[str, tuple[float, ...]] = {}
    for column in numeric:
        series = pd.to_numeric(frame[column], errors="coerce")
        fill = float(series.median()) if series.notna().any() else 0.0
        numeric_fill[column] = fill
        numeric_bounds[column] = (
            float(series.min()) if series.notna().any() else 0.0,
            float(series.max()) if series.notna().any() else 0.0,
        )
        # Counterfactual candidates come from this grid. It stops at the 1st and
        # 99th percentiles because the raw extremes are often caps or sentinel
        # codes (Adult tops capital-gain out at 99,999), and a suggestion to move
        # to one is not advice anyone can act on.
        quantiles = (
            series.quantile(QUANTILE_GRID).to_numpy(dtype=float)
            if series.notna().any()
            else np.zeros(len(QUANTILE_GRID))
        )
        numeric_quantiles[column] = tuple(float(q) for q in quantiles)

    return Encoder(
        numeric=numeric,
        categories=categories,
        numeric_fill=numeric_fill,
        numeric_bounds=numeric_bounds,
        numeric_quantiles=numeric_quantiles,
        drop_first=drop_first,
    )

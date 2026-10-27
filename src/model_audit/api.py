"""HTTP surface: the same audit, callable from a deployment pipeline.

The app is built around an already-loaded auditor, so the model is trained once
and every request reads the cached results.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from .model_card import audit_to_dict, render_markdown


def _local_dict(explanation) -> dict[str, Any]:
    return {
        "index": explanation.index,
        "method": explanation.method,
        "prediction": explanation.prediction,
        "probability": None if explanation.probability is None else round(explanation.probability, 4),
        "intercept": None if explanation.intercept is None else round(explanation.intercept, 4),
        "r2": None if explanation.r2 is None else round(explanation.r2, 4),
        "contributions": [
            {"feature": c.feature, "value": round(c.value, 4), "rank": c.rank}
            for c in explanation.contributions
        ],
    }


def create_app(auditor) -> FastAPI:
    app = FastAPI(
        title="model-audit",
        version="0.1.0",
        description="Explanations, fairness audit and model card for a fitted model.",
    )

    def _check_index(index: int) -> None:
        if index < 0 or index >= auditor.X.shape[0]:
            raise HTTPException(status_code=404, detail=f"row {index} out of range")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": auditor.name,
            "rows": int(auditor.X.shape[0]),
            "features": int(auditor.X.shape[1]),
        }

    @app.get("/metrics")
    def metrics() -> dict[str, float]:
        return auditor.metrics()

    @app.get("/calibration")
    def calibration() -> dict[str, Any]:
        return auditor.calibration()

    @app.get("/fairness")
    def fairness() -> list[dict[str, Any]]:
        return [audit_to_dict(a) for a in auditor.fairness()]

    @app.get("/explain/global")
    def explain_global(top: int = 10) -> list[dict[str, Any]]:
        return [
            {"feature": f.feature, "importance": round(f.importance, 4), "rank": f.rank, "method": f.method}
            for f in auditor.global_importance()[:top]
        ]

    @app.get("/explain/local/{index}")
    def explain_local(index: int) -> dict[str, Any]:
        _check_index(index)
        return _local_dict(auditor.local(index))

    @app.get("/anchors/{index}")
    def anchors(index: int) -> dict[str, Any]:
        _check_index(index)
        anchor = auditor.anchors(index)
        return {
            "index": anchor.index,
            "rule": anchor.text(),
            "precision": round(anchor.precision, 4),
            "coverage": round(anchor.coverage, 4),
            "n_matched": anchor.n_matched,
            "prediction": anchor.prediction,
        }

    @app.get("/counterfactuals/{index}")
    def counterfactuals(index: int, k: int = 3) -> list[dict[str, Any]]:
        _check_index(index)
        try:
            results = auditor.counterfactuals(index, k=k)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return [c.as_dict() for c in results]

    @app.get("/model-card", response_class=PlainTextResponse)
    def model_card() -> str:
        return render_markdown(auditor.model_card())

    return app

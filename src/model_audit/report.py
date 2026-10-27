"""The report: everything the audit found, in reading order.

Markdown is the source of truth. The model card is written alongside it in JSON,
Markdown and PDF so the same numbers go to a repo, a pipeline and a reviewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def _local_examples(auditor) -> list[int]:
    scores = np.asarray(auditor.y_score, dtype=float)
    return sorted(
        {
            int(np.argmax(scores)),
            int(np.argmin(scores)),
            int(np.argmin(np.abs(scores - 0.5))),
        }
    )


def render_report(auditor, *, top_features: int = 15) -> str:
    metrics = auditor.metrics()
    fairness = auditor.fairness()
    calibration = auditor.calibration()
    importance = auditor.global_importance()

    lines: list[str] = [
        f"# Audit report — {auditor.name}",
        "",
        _table(
            ["Field", "Value"],
            [
                ["Dataset", auditor.dataset_name],
                ["Source", auditor.dataset_source or "not recorded"],
                ["Rows", auditor.X.shape[0]],
                ["Features", auditor.X.shape[1]],
                ["Task", auditor.task],
                ["Target", auditor.target_name],
            ],
        ),
        "",
        "## Performance",
        "",
        _table(["Metric", "Value"], [[k, f"{v:.4f}"] for k, v in metrics.items()]),
        "",
        "## Fairness",
        "",
    ]

    if not fairness:
        lines += ["No protected attributes were supplied.", ""]
    else:
        failed = sum(len(a.failures) for a in fairness)
        total = sum(len(a.metrics) for a in fairness)
        lines += [
            f"{failed} of {total} checks failed across {len(fairness)} attributes.",
            "",
        ]
        for audit in fairness:
            lines += [f"### {audit.attribute}", ""]
            lines.append(_table(
                ["Group", "n", "Positive rate", "TPR", "FPR", "ECE", "Brier"],
                [
                    [
                        g.group,
                        g.n,
                        f"{g.positive_rate:.3f}",
                        "" if g.tpr is None else f"{g.tpr:.3f}",
                        "" if g.fpr is None else f"{g.fpr:.3f}",
                        "" if g.ece is None else f"{g.ece:.3f}",
                        "" if g.brier is None else f"{g.brier:.3f}",
                    ]
                    for g in audit.groups
                ],
            ))
            lines.append("")
            lines.append(_table(
                ["Check", "Value", "Threshold", "Result"],
                [
                    [m.name, f"{m.value:.3f}", m.threshold, "pass" if m.passed else "**FAIL**"]
                    for m in audit.metrics
                ],
            ))
            lines.append("")
            if not audit.reliable:
                lines += [
                    f"> Smallest group is n={audit.smallest_group}; below the "
                    f"{audit.min_group_size} needed for a stable rate. Gaps are indicative only.",
                    "",
                ]

    lines += [
        "## Calibration",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Expected calibration error", f"{calibration['ece']:.4f}"],
                ["Maximum calibration error", f"{calibration['mce']:.4f}"],
                ["Brier score", f"{calibration['brier']:.4f}"],
            ],
        ),
        "",
        _table(
            ["Bin", "n", "Confidence", "Accuracy", "Gap"],
            [
                [
                    f"{b['lower']:.2f}–{b['upper']:.2f}",
                    b["n"],
                    f"{b['confidence']:.3f}",
                    f"{b['accuracy']:.3f}",
                    f"{b['accuracy'] - b['confidence']:+.3f}",
                ]
                for b in calibration["bins"]
            ],
        ),
        "",
        "## Global feature importance",
        "",
        _table(
            ["Rank", "Feature", "Mean abs contribution", "Method"],
            [[f.rank, f.feature, f"{f.importance:.4f}", f.method] for f in importance[:top_features]],
        ),
        "",
        "## Local explanations",
        "",
        "Three rows: the most confident positive, the most confident negative, and the "
        "row closest to the decision boundary. Contributions are from a local linear "
        "surrogate; anchors and counterfactuals are computed on the raw features where "
        "available.",
        "",
    ]

    for index in _local_examples(auditor):
        explanation = auditor.local(index)
        lines += [
            f"### Row {index} — predicted {explanation.prediction} "
            f"(p={explanation.probability:.3f})",
            "",
            _table(
                ["Feature", "Contribution"],
                [[c.feature, f"{c.value:+.4f}"] for c in explanation.contributions[:6]],
            ),
            "",
        ]
        try:
            anchor = auditor.anchors(index)
            note = "" if anchor.meets_target else f" — target {anchor.target_precision} not met"
            lines += [
                f"**Anchor:** `{anchor.text()}` — precision {anchor.precision:.3f}, "
                f"coverage {anchor.coverage:.3f} (n={anchor.n_matched}){note}.",
                "",
            ]
        except Exception as error:  # pragma: no cover - defensive
            lines += [f"Anchor unavailable: {error}", ""]
        try:
            counterfactuals = auditor.counterfactuals(index, k=2)
            if counterfactuals:
                for counterfactual in counterfactuals:
                    changes = ", ".join(
                        f"{feature}: {old!r} → {new!r}" for feature, old, new in counterfactual.changes
                    )
                    lines += [
                        f"**Counterfactual:** {changes} — p goes from "
                        f"{counterfactual.original_probability:.3f} to "
                        f"{counterfactual.counterfactual_probability:.3f} "
                        f"(distance {counterfactual.distance:.2f}).",
                        "",
                    ]
            else:
                lines += ["**Counterfactual:** none found within the allowed changes.", ""]
        except ValueError as error:
            lines += [f"Counterfactual unavailable: {error}", ""]

    lines += [
        "## Method and caveats",
        "",
        "- Global importance is mean absolute SHAP where an exact explainer exists, "
        "permutation importance otherwise. The method used is shown per row.",
        "- Local contributions come from a ridge surrogate fitted on perturbed samples "
        "weighted by distance, the LIME construction.",
        "- Anchors are greedy rules: precision and coverage are both reported because "
        "either one alone is easy to game.",
        "- Counterfactuals only change features the caller allows. If no allowed change "
        "flips the prediction, none is reported.",
        "- Fairness metrics describe this dataset. They are not a legal finding, and a "
        "gap does not by itself prove discrimination.",
        "",
    ]
    return "\n".join(lines)


def write_report(auditor, directory: str | Path = "reports", *, stem: str = "audit") -> dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    report_path = directory / f"{stem}-report.md"
    report_path.write_text(render_report(auditor), encoding="utf-8")

    card = auditor.model_card()
    from .model_card import write_json, write_markdown, write_pdf

    paths = {
        "report": str(report_path),
        "card_json": str(write_json(card, directory / f"{stem}-model-card.json")),
        "card_md": str(write_markdown(card, directory / f"{stem}-model-card.md")),
    }
    try:
        paths["card_pdf"] = str(write_pdf(card, directory / f"{stem}-model-card.pdf"))
    except ImportError:  # reportlab is optional
        paths["card_pdf"] = ""
    return paths

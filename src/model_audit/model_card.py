"""Model card generation: the audit, written down in the format people expect.

JSON for machines, Markdown for the repo, PDF for whoever signs off. All three
come from the same object, so they cannot drift apart.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import AttributeAudit, ModelCard


def audit_to_dict(audit: AttributeAudit) -> dict[str, Any]:
    return {
        "attribute": audit.attribute,
        "reliable": audit.reliable,
        "smallest_group": audit.smallest_group,
        "min_group_size": audit.min_group_size,
        "groups": [
            {
                "group": g.group,
                "n": g.n,
                "positive_rate": round(g.positive_rate, 4),
                "tpr": None if g.tpr is None else round(g.tpr, 4),
                "fpr": None if g.fpr is None else round(g.fpr, 4),
                "ece": None if g.ece is None else round(g.ece, 4),
                "brier": None if g.brier is None else round(g.brier, 4),
            }
            for g in audit.groups
        ],
        "metrics": [
            {
                "name": m.name,
                "value": round(m.value, 4),
                "threshold": m.threshold,
                "passed": m.passed,
                "rule": m.rule,
            }
            for m in audit.metrics
        ],
    }


def build_model_card(
    auditor,
    *,
    version: str = "0.1.0",
    description: str | None = None,
    intended_use: str | None = None,
    limitations: str | None = None,
) -> ModelCard:
    importance = auditor.global_importance()[:15]
    return ModelCard(
        model_name=auditor.name,
        version=version,
        description=description
        or f"A {auditor.task} model audited with model-audit. Predicts {auditor.target_name}.",
        intended_use=intended_use
        or "Decision support only. Not validated for autonomous decisions affecting people.",
        limitations=limitations
        or (
            "Metrics are measured on the evaluation split named in the report and do not "
            "transfer to populations that differ from it. Fairness metrics are descriptive: "
            "they quantify gaps in this dataset, they do not establish legally protected "
            "status or prove discrimination."
        ),
        task=auditor.task,
        target=auditor.target_name,
        dataset_name=auditor.dataset_name,
        dataset_source=auditor.dataset_source,
        n_rows=int(auditor.X.shape[0]),
        n_features=int(auditor.X.shape[1]),
        features=list(auditor.feature_names),
        metrics={k: round(float(v), 4) for k, v in auditor.metrics().items() if v is not None},
        fairness=[audit_to_dict(a) for a in auditor.fairness()],
        global_importance=[
            {"feature": f.feature, "importance": round(f.importance, 4), "rank": f.rank}
            for f in importance
        ],
        protected_attributes=list(auditor.protected_attributes or {}),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def render_markdown(card: ModelCard) -> str:
    lines: list[str] = [
        f"# Model Card — {card.model_name} {card.version}",
        "",
        f"_Generated {card.generated_at} by {card.generator}._",
        "",
        "## Model details",
        "",
        _table(
            ["Field", "Value"],
            [
                ["Model", card.model_name],
                ["Version", card.version],
                ["Task", card.task],
                ["Target", card.target],
                ["Description", card.description],
                ["Intended use", card.intended_use],
            ],
        ),
        "",
        "## Training data",
        "",
        _table(
            ["Field", "Value"],
            [
                ["Dataset", card.dataset_name],
                ["Source", card.dataset_source],
                ["Rows", card.n_rows],
                ["Features", card.n_features],
                ["Protected attributes", ", ".join(card.protected_attributes) or "none"],
            ],
        ),
        "",
        "## Performance",
        "",
        _table(["Metric", "Value"], [[k, f"{v:.4f}"] for k, v in card.metrics.items()]),
        "",
        "## Fairness audit",
        "",
    ]

    if not card.fairness:
        lines += ["No protected attributes were supplied, so no fairness audit was run.", ""]
    for audit in card.fairness:
        lines.append(f"### {audit['attribute']}")
        lines.append("")
        rows = [
            [
                g["group"],
                g["n"],
                f"{g['positive_rate']:.3f}",
                "" if g["tpr"] is None else f"{g['tpr']:.3f}",
                "" if g["fpr"] is None else f"{g['fpr']:.3f}",
                "" if g["ece"] is None else f"{g['ece']:.3f}",
            ]
            for g in audit["groups"]
        ]
        lines.append(_table(["Group", "n", "Positive rate", "TPR", "FPR", "ECE"], rows))
        lines.append("")
        lines.append(_table(
            ["Check", "Value", "Threshold", "Result"],
            [
                [
                    m["name"],
                    f"{m['value']:.3f}",
                    m["threshold"],
                    "pass" if m["passed"] else "**FAIL**",
                ]
                for m in audit["metrics"]
            ],
        ))
        lines.append("")
        if not audit["reliable"]:
            lines.append(
                f"> Smallest group is n={audit['smallest_group']}, below the "
                f"{audit['min_group_size']} needed for a stable rate. Treat these gaps as "
                "indicative only."
            )
            lines.append("")

    lines += ["## Global feature importance", ""]
    if card.global_importance:
        lines.append(_table(
            ["Rank", "Feature", "Mean abs contribution"],
            [[f["rank"], f["feature"], f"{f['importance']:.4f}"] for f in card.global_importance],
        ))
    else:
        lines.append("Not computed.")
    lines += [
        "",
        "## Limitations",
        "",
        card.limitations,
        "",
        "---",
        "",
        "Fairness metrics are descriptive, not causal. They quantify gaps in this dataset; "
        "they do not establish that a model is unlawful, nor that removing the gap would "
        "improve outcomes for the group concerned.",
        "",
    ]
    return "\n".join(lines)


def write_json(card: ModelCard, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
    return path


def write_markdown(card: ModelCard, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(card), encoding="utf-8")
    return path


def write_pdf(card: ModelCard, path: str | Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Model Card — {card.model_name}",
    )

    def grid(header: list[str], rows: list[list[Any]], widths: list[float] | None = None) -> Table:
        data = [header] + [["" if v is None else str(v) for v in row] for row in rows]
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return table

    story: list[Any] = [
        Paragraph(escape(f"Model Card — {card.model_name} {card.version}"), styles["Title"]),
        Paragraph(escape(f"Generated {card.generated_at} by {card.generator}"), styles["Italic"]),
        Spacer(1, 6),
        Paragraph("Model details", styles["Heading2"]),
        grid(
            ["Field", "Value"],
            [
                ["Task", card.task],
                ["Target", card.target],
                ["Description", card.description],
                ["Intended use", card.intended_use],
            ],
            [35 * mm, 135 * mm],
        ),
        Spacer(1, 8),
        Paragraph("Training data", styles["Heading2"]),
        grid(
            ["Field", "Value"],
            [
                ["Dataset", card.dataset_name],
                ["Source", card.dataset_source],
                ["Rows", card.n_rows],
                ["Features", card.n_features],
                ["Protected attributes", ", ".join(card.protected_attributes) or "none"],
            ],
            [35 * mm, 135 * mm],
        ),
        Spacer(1, 8),
        Paragraph("Performance", styles["Heading2"]),
        grid(["Metric", "Value"], [[k, f"{v:.4f}"] for k, v in card.metrics.items()], [70 * mm, 40 * mm]),
    ]

    if card.fairness:
        story.append(PageBreak())
        story.append(Paragraph("Fairness audit", styles["Heading2"]))
        for audit in card.fairness:
            story.append(Paragraph(escape(str(audit["attribute"])), styles["Heading3"]))
            rows = [
                [
                    g["group"],
                    g["n"],
                    f"{g['positive_rate']:.3f}",
                    "" if g["tpr"] is None else f"{g['tpr']:.3f}",
                    "" if g["fpr"] is None else f"{g['fpr']:.3f}",
                    "" if g["ece"] is None else f"{g['ece']:.3f}",
                ]
                for g in audit["groups"]
            ]
            story.append(grid(["Group", "n", "Pos rate", "TPR", "FPR", "ECE"], rows))
            story.append(Spacer(1, 4))
            story.append(
                grid(
                    ["Check", "Value", "Threshold", "Result"],
                    [
                        [
                            m["name"],
                            f"{m['value']:.3f}",
                            m["threshold"],
                            "pass" if m["passed"] else "FAIL",
                        ]
                        for m in audit["metrics"]
                    ],
                    [60 * mm, 20 * mm, 25 * mm, 20 * mm],
                )
            )
            story.append(Spacer(1, 8))

    if card.global_importance:
        story.append(PageBreak())
        story.append(Paragraph("Global feature importance", styles["Heading2"]))
        story.append(
            grid(
                ["Rank", "Feature", "Mean abs contribution"],
                [[f["rank"], f["feature"], f"{f['importance']:.4f}"] for f in card.global_importance],
                [15 * mm, 100 * mm, 40 * mm],
            )
        )

    story += [
        Spacer(1, 10),
        Paragraph("Limitations", styles["Heading2"]),
        Paragraph(escape(card.limitations), styles["BodyText"]),
        Spacer(1, 6),
        Paragraph(
            "Fairness metrics are descriptive, not causal. They quantify gaps in this dataset; "
            "they do not establish that a model is unlawful, nor that removing the gap would "
            "improve outcomes for the group concerned.",
            styles["Italic"],
        ),
    ]

    doc.build(story)
    return path

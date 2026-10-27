"""Command line: run the audit end to end, or serve it.

    python -m model_audit demo --sample 12000 --out reports
    python -m model_audit serve --port 8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preprocessing import fit_encoder

IMMUTABLE = ("age", "sex", "race", "native-country", "marital-status", "relationship")


def build_demo(sample: int | None = None, seed: int = 0):
    """Train the Adult model and return an auditor over the held-out split."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    from .auditor import ModelAuditor
    from .datasets import ADULT_CATEGORICAL, load_adult

    frame = load_adult(sample=sample, seed=seed)
    train, test = train_test_split(
        frame, test_size=0.25, random_state=seed, stratify=frame["income"]
    )

    encoder = fit_encoder(train.drop(columns=["income"]), ADULT_CATEGORICAL)
    X_train = encoder.transform(train.drop(columns=["income"]))
    X_test = encoder.transform(test.drop(columns=["income"]))
    y_train = train["income"].to_numpy()
    y_test = test["income"].to_numpy()

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    auditor = ModelAuditor(
        model,
        X_test,
        y_test,
        feature_names=encoder.feature_names,
        task="classification",
        name="adult-income-hgb",
        target_name="income > 50K",
        dataset_name="UCI Adult",
        dataset_source="OpenML adult v2 (48,842 rows), 25% held-out split",
        protected_attributes={
            "sex": test["sex"].to_numpy(),
            "race": test["race"].to_numpy(),
        },
        frame=test.drop(columns=["income"]).reset_index(drop=True),
        encoder=encoder,
        immutable_features=IMMUTABLE,
        random_state=seed,
    )
    return auditor


def _demo(args: argparse.Namespace) -> int:
    auditor = build_demo(sample=args.sample)
    metrics = auditor.metrics()
    fairness = auditor.fairness()
    importance = auditor.global_importance()

    print(f"{auditor.name}: {auditor.X.shape[0]} held-out rows, {auditor.X.shape[1]} features")
    print("  " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    failures = 0
    for audit in fairness:
        print(f"\n{audit.attribute} (smallest group n={audit.smallest_group})")
        for group in audit.groups:
            print(
                f"  {group.group:<22} n={group.n:<5} pos={group.positive_rate:.3f} "
                f"tpr={'-' if group.tpr is None else f'{group.tpr:.3f}'} "
                f"fpr={'-' if group.fpr is None else f'{group.fpr:.3f}'}"
            )
        for metric in audit.metrics:
            status = "pass" if metric.passed else "FAIL"
            print(f"  {status:<4} {metric.name:<32} {metric.value:.3f} (threshold {metric.threshold})")
            failures += 0 if metric.passed else 1

    print("\nTop features")
    for item in importance[:8]:
        print(f"  {item.rank:>2}. {item.feature:<28} {item.importance:.4f}  [{item.method}]")

    out = Path(args.out)
    paths = auditor.report(out, stem="adult")
    numbers = {
        "n_test": int(auditor.X.shape[0]),
        "n_features": int(auditor.X.shape[1]),
        "metrics": metrics,
        "fairness": {
            audit.attribute: {
                "smallest_group": audit.smallest_group,
                "reliable": audit.reliable,
                "metrics": {m.name: round(m.value, 4) for m in audit.metrics},
                "groups": {
                    g.group: {"n": g.n, "positive_rate": round(g.positive_rate, 4)}
                    for g in audit.groups
                },
            }
            for audit in fairness
        },
        "failed_checks": failures,
        "importance_method": importance[0].method if importance else None,
        "top_features": [f.feature for f in importance[:10]],
    }
    (out / "numbers.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print("\nWrote:")
    for key, value in paths.items():
        print(f"  {key:<10} {value}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    app = create_app(build_demo(sample=args.sample))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-audit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="train the Adult model, audit it, write reports")
    demo.add_argument("--sample", type=int, default=None, help="rows to sample (default: all)")
    demo.add_argument("--out", default="reports", help="output directory")
    demo.set_defaults(func=_demo)

    serve = sub.add_parser("serve", help="serve the audit over HTTP")
    serve.add_argument("--sample", type=int, default=12000)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

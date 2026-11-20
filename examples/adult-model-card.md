# Model Card — adult-income-hgb 0.1.0

_Generated 2026-11-20T10:49:25+00:00 by model-audit 0.1.0._

## Model details

| Field | Value |
|---|---|
| Model | adult-income-hgb |
| Version | 0.1.0 |
| Task | classification |
| Target | income > 50K |
| Description | A classification model audited with model-audit. Predicts income > 50K. |
| Intended use | Decision support only. Not validated for autonomous decisions affecting people. |

## Training data

| Field | Value |
|---|---|
| Dataset | UCI Adult |
| Source | OpenML adult v2 (48,842 rows), 25% held-out split |
| Rows | 12211 |
| Features | 91 |
| Protected attributes | sex, race |

## Performance

| Metric | Value |
|---|---|
| accuracy | 0.8730 |
| precision | 0.7824 |
| recall | 0.6499 |
| f1 | 0.7100 |
| brier | 0.0874 |
| roc_auc | 0.9292 |
| log_loss | 0.2748 |

## Fairness audit

### sex

| Group | n | Positive rate | TPR | FPR | ECE |
|---|---|---|---|---|---|
| Male | 8147 | 0.256 | 0.660 | 0.080 | 0.008 |
| Female | 4064 | 0.084 | 0.596 | 0.020 | 0.004 |

| Check | Value | Threshold | Result |
|---|---|---|---|
| demographic_parity_difference | 0.172 | 0.1 | **FAIL** |
| disparate_impact_ratio | 0.327 | 0.8 | **FAIL** |
| equalized_odds_difference | 0.064 | 0.1 | pass |
| calibration_gap | 0.004 | 0.1 | pass |

### race

| Group | n | Positive rate | TPR | FPR | ECE |
|---|---|---|---|---|---|
| White | 10449 | 0.213 | 0.658 | 0.060 | 0.006 |
| Black | 1175 | 0.082 | 0.523 | 0.028 | 0.011 |
| Asian-Pac-Islander | 371 | 0.251 | 0.616 | 0.118 | 0.054 |
| Amer-Indian-Eskimo | 113 | 0.080 | 0.538 | 0.020 | 0.058 |
| Other | 103 | 0.078 | 0.533 | 0.000 | 0.067 |

| Check | Value | Threshold | Result |
|---|---|---|---|
| demographic_parity_difference | 0.173 | 0.1 | **FAIL** |
| disparate_impact_ratio | 0.310 | 0.8 | **FAIL** |
| equalized_odds_difference | 0.135 | 0.1 | **FAIL** |
| calibration_gap | 0.062 | 0.1 | pass |

## Global feature importance

| Rank | Feature | Mean abs contribution |
|---|---|---|
| 1 | marital-status=Married-civ-spouse | 1.1498 |
| 2 | age | 0.8901 |
| 3 | capital-gain | 0.5632 |
| 4 | education-num | 0.4742 |
| 5 | hours-per-week | 0.3150 |
| 6 | sex=Female | 0.1726 |
| 7 | capital-loss | 0.1648 |
| 8 | relationship=Own-child | 0.1265 |
| 9 | marital-status=Never-married | 0.1079 |
| 10 | occupation=Exec-managerial | 0.1037 |
| 11 | occupation=Prof-specialty | 0.0905 |
| 12 | occupation=Other-service | 0.0692 |
| 13 | occupation=Machine-op-inspct | 0.0601 |
| 14 | occupation=Handlers-cleaners | 0.0492 |
| 15 | race=Black | 0.0457 |

## Limitations

Metrics are measured on the evaluation split named in the report and do not transfer to populations that differ from it. Fairness metrics are descriptive: they quantify gaps in this dataset, they do not establish legally protected status or prove discrimination.

---

Fairness metrics are descriptive, not causal. They quantify gaps in this dataset; they do not establish that a model is unlawful, nor that removing the gap would improve outcomes for the group concerned.

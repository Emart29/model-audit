# Audit report — adult-income-hgb

| Field | Value |
|---|---|
| Dataset | UCI Adult |
| Source | OpenML adult v2 (48,842 rows), 25% held-out split |
| Rows | 12211 |
| Features | 91 |
| Task | classification |
| Target | income > 50K |

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

## Fairness

5 of 8 checks failed across 2 attributes.

### sex

| Group | n | Positive rate | TPR | FPR | ECE | Brier |
|---|---|---|---|---|---|---|
| Male | 8147 | 0.256 | 0.660 | 0.080 | 0.008 | 0.108 |
| Female | 4064 | 0.084 | 0.596 | 0.020 | 0.004 | 0.046 |

| Check | Value | Threshold | Result |
|---|---|---|---|
| demographic_parity_difference | 0.173 | 0.1 | **FAIL** |
| disparate_impact_ratio | 0.327 | 0.8 | **FAIL** |
| equalized_odds_difference | 0.064 | 0.1 | pass |
| calibration_gap | 0.004 | 0.1 | pass |

### race

| Group | n | Positive rate | TPR | FPR | ECE | Brier |
|---|---|---|---|---|---|---|
| White | 10449 | 0.213 | 0.658 | 0.060 | 0.006 | 0.091 |
| Black | 1175 | 0.082 | 0.523 | 0.028 | 0.011 | 0.051 |
| Asian-Pac-Islander | 371 | 0.251 | 0.616 | 0.118 | 0.054 | 0.121 |
| Amer-Indian-Eskimo | 113 | 0.080 | 0.538 | 0.020 | 0.058 | 0.061 |
| Other | 103 | 0.078 | 0.533 | 0.000 | 0.067 | 0.051 |

| Check | Value | Threshold | Result |
|---|---|---|---|
| demographic_parity_difference | 0.173 | 0.1 | **FAIL** |
| disparate_impact_ratio | 0.310 | 0.8 | **FAIL** |
| equalized_odds_difference | 0.135 | 0.1 | **FAIL** |
| calibration_gap | 0.062 | 0.1 | pass |

## Calibration

| Metric | Value |
|---|---|
| Expected calibration error | 0.0071 |
| Maximum calibration error | 0.0193 |
| Brier score | 0.0874 |

| Bin | n | Confidence | Accuracy | Gap |
|---|---|---|---|---|
| 0.00–0.00 | 1222 | 0.001 | 0.001 | +0.000 |
| 0.00–0.01 | 1221 | 0.003 | 0.002 | -0.001 |
| 0.01–0.01 | 1221 | 0.009 | 0.009 | +0.000 |
| 0.01–0.03 | 1221 | 0.020 | 0.019 | -0.001 |
| 0.03–0.07 | 1221 | 0.044 | 0.044 | +0.000 |
| 0.07–0.16 | 1221 | 0.105 | 0.124 | +0.019 |
| 0.16–0.31 | 1221 | 0.231 | 0.246 | +0.015 |
| 0.31–0.50 | 1221 | 0.397 | 0.386 | -0.011 |
| 0.50–0.76 | 1221 | 0.632 | 0.621 | -0.011 |
| 0.76–1.00 | 1221 | 0.932 | 0.942 | +0.010 |

## Global feature importance

| Rank | Feature | Mean abs contribution | Method |
|---|---|---|---|
| 1 | marital-status=Married-civ-spouse | 1.1498 | shap_tree |
| 2 | age | 0.8901 | shap_tree |
| 3 | capital-gain | 0.5632 | shap_tree |
| 4 | education-num | 0.4742 | shap_tree |
| 5 | hours-per-week | 0.3150 | shap_tree |
| 6 | sex=Female | 0.1726 | shap_tree |
| 7 | capital-loss | 0.1648 | shap_tree |
| 8 | relationship=Own-child | 0.1265 | shap_tree |
| 9 | marital-status=Never-married | 0.1079 | shap_tree |
| 10 | occupation=Exec-managerial | 0.1037 | shap_tree |
| 11 | occupation=Prof-specialty | 0.0905 | shap_tree |
| 12 | occupation=Other-service | 0.0692 | shap_tree |
| 13 | occupation=Machine-op-inspct | 0.0601 | shap_tree |
| 14 | occupation=Handlers-cleaners | 0.0492 | shap_tree |
| 15 | race=Black | 0.0457 | shap_tree |

## Local explanations

Three rows: the most confident positive, the most confident negative, and the row closest to the decision boundary. Contributions are from a local linear surrogate; anchors and counterfactuals are computed on the raw features where available.

### Row 2173 — predicted 1 (p=0.500)

| Feature | Contribution |
|---|---|
| marital-status=Married-civ-spouse | +0.0921 |
| capital-gain | -0.0305 |
| hours-per-week | +0.0241 |
| occupation=Sales | +0.0154 |
| occupation=Exec-managerial | -0.0136 |
| occupation=Other-service | +0.0108 |

**Anchor:** `marital-status=Married-civ-spouse AND education-num > 9.0 AND hours-per-week > 40.0 AND age > 37.0 AND race=White` — precision 0.788, coverage 0.071 (n=866) — target 0.95 not met.

**Counterfactual:** education-num: 10 → 9.0 — p goes from 0.500 to 0.403 (distance 0.07).

**Counterfactual:** hours-per-week: 65 → 99.0 — p goes from 0.500 to 0.351 (distance 0.35).

### Row 5024 — predicted 0 (p=0.000)

| Feature | Contribution |
|---|---|
| age | -0.0866 |
| hours-per-week | -0.0284 |
| relationship=Own-child | -0.0191 |
| marital-status=Married-civ-spouse | -0.0183 |
| marital-status=Never-married | +0.0165 |
| capital-gain | -0.0098 |

**Anchor:** `age <= 22.0` — precision 0.999, coverage 0.118 (n=1446).

**Counterfactual:** none found within the allowed changes.

### Row 11055 — predicted 1 (p=1.000)

| Feature | Contribution |
|---|---|
| capital-gain | +0.3097 |
| occupation=Prof-specialty | +0.0291 |
| marital-status=Married-civ-spouse | +0.0262 |
| education-num | +0.0218 |
| workclass=Federal-gov | +0.0174 |
| relationship=Husband | -0.0102 |

**Anchor:** `education-num > 13.0 AND marital-status=Married-civ-spouse AND hours-per-week > 25.0` — precision 0.956, coverage 0.051 (n=619).

**Counterfactual:** capital-gain: 15024 → 0.0, education-num: 15 → 16.0, hours-per-week: 50 → 1.0, occupation: 'Prof-specialty' → 'Handlers-cleaners' — p goes from 1.000 to 0.434 (distance 1.72).

## Method and caveats

- Global importance is mean absolute SHAP where an exact explainer exists, permutation importance otherwise. The method used is shown per row.
- Local contributions come from a ridge surrogate fitted on perturbed samples weighted by distance, the LIME construction.
- Anchors are greedy rules: precision and coverage are both reported because either one alone is easy to game.
- Counterfactuals only change features the caller allows. If no allowed change flips the prediction, none is reported.
- Fairness metrics describe this dataset. They are not a legal finding, and a gap does not by itself prove discrimination.

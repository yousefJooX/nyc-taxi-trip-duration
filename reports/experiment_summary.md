# NYC Taxi Trip Duration — Experiment Summary

## 1. Project Goal

Predict NYC taxi trip duration from trip information. This is a **regression** problem: the output is a numerical duration.

The course requires **`Ridge(alpha=1)` as the Official Course Model**. We also compared Polynomial Degree 2 + Ridge, Random Forest, Gradient Boosting, and XGBoost to understand how model capacity affects prediction quality. Their results do not replace the course-required model.

## 2. Data Workflow

```text
Train → Validation → Test
Fit     Compare      Report once after decisions are locked
```

We use the provided, separate train and validation partitions. Train fits model parameters, preprocessing, and KMeans. Validation is used for feature experiments and model comparison, using transformations learned from train. Test remains untouched until all decisions are locked.

**We did NOT use test results for feature engineering decisions, model selection, or hyperparameter tuning. Test evaluation is still pending.**

The main experiment documented here is the saved Colab run: **993,415 usable training rows and 229,319 validation rows**. These are historical measurements, not results from rerunning the simplified source code.

## 3. Data Quality Issue We Found

The initial Colab training attempt failed with:

```text
ValueError: trip_duration must be finite and nonnegative
```

We investigated the target instead of blindly dropping rows:

| Target check | Train | Validation |
|---|---:|---:|
| Original rows | 993,416 | 229,319 |
| Missing trip_duration (NaN) | 1 | 0 |
| Infinite | 0 | 0 |
| Negative | 0 | 0 |
| Zero | 0 | 0 |

The bad training row had no target, `y`. A supervised regression model cannot learn a duration from a row without its correct duration. We therefore removed **only that one training row**, leaving **993,415** usable training rows. Validation was not modified based on its target; no test rows were removed or evaluated.

The Colab repair rewrote its training CSV after removing the missing label. The simplified training code instead removes missing training labels in memory and leaves input CSVs unchanged.

## 4. Target Transformation

Trip duration has a strong right skew: most trips are relatively short, with a long tail of very long trips. We train on:

```python
y = np.log1p(trip_duration)
```

This reduces the scale and influence of extremely long trips and gives regression a more manageable target distribution. **All experiment RMSE and R² values below are in log-target space**, not seconds. Lower RMSE and higher R² indicate better validation predictions.

Predictions can return to seconds with `np.expm1(predicted_log_duration)`. Exported duration predictions are clamped to zero if this inverse transformation produces a negative value.

## 5. Initial Feature Engineering

### Time

- `hour`: pickup hour.
- `dayofweek`: pickup day of the week.
- `month`: pickup month.
- `dayofyear`: pickup day within the year.

These describe when a trip happens, allowing the model to learn time-related patterns.

### Geographic distance

- `distance_km`: Haversine distance between pickup and dropoff, using Earth radius 6371.0088 km.
- `log_distance`: `log1p(distance_km)`, which represents the strongly skewed distance feature on a smaller scale.
- `lat_diff` and `lon_diff`: absolute endpoint coordinate differences in the measured implementation; early EDA also explored signed differences.
- `same_location`: an indicator for `distance_km <= 0.1`.

Haversine approximates direct geographic separation, not actual road distance. Identical endpoints do not prove that a taxi never moved.

The initial 5,000-row EDA sample had no missing values or duplicates. Nine trips longer than 20 hours were examined rather than automatically discarded. Historical sample distance experiments gave R²≈0.403 with raw distance, ≈0.526 with log distance, and ≈0.543 with both. These are **early EDA observations**, separate from the main Colab comparison.

## 6. Location Clustering with KMeans

We experimented with separate pickup and dropoff KMeans models. Early clustering exposed coordinate outliers, including an unreasonable singleton-like cluster. Elbow curves explored K=1–20 and suggested K≈5 as a candidate, not a guaranteed optimum. The final settings use **K=5**, `random_state=42`, and `n_init="auto"` for each model.

We cleaned the coordinates used to **fit** KMeans. A training trip contributes fitting coordinates only when both endpoints fall within the project's NYC bounds:

| Coordinate | Allowed fitting range |
|---|---|
| Latitude | 40.4 to 41.0 |
| Longitude | -74.3 to -73.6 |

This mask changes the rows used to fit clusters; it does **not** remove those trips from regression training or validation. All usable labeled training rows and all validation rows receive cluster predictions.

**KMeans is fitted on training coordinates only.** Validation uses `kmeans.predict(...)`; test will also use `predict(...)`, never fitting clusters on those partitions.

`pickup_cluster` and `dropoff_cluster` are categorical labels, represented with one-hot encoding. Pickup cluster 2 and dropoff cluster 2 need not mean the same geographic region because they come from separate fitted models. Later, `route_cluster = pickup_cluster -> dropoff_cluster` represents an approximate origin–destination area pair.

## 7. Additional Feature Ideas We Tested

### Manhattan-style distance

`manhattan_km` approximates north/south plus east/west movement using geographic distance legs instead of only direct Haversine separation. It slightly worsened validation RMSE and was **dropped**. It is not in the final retained feature set.

### Bearing

`bearing_sin` and `bearing_cos` describe direction: distance tells us how far apart the endpoints are; bearing tells us which way the trip points. We use sine/cosine because direction is circular: 1° and 359° are close directions despite their very different raw numbers. Both components are zero for identical endpoints. **Keep.**

### Airport proximity

Six flags indicate whether pickup or dropoff is within a fixed **2 km** approximation of JFK, LGA, or EWR. The fixed reference coordinates are JFK (40.6413, -73.7781), LGA (40.7769, -73.8740), and EWR (40.6895, -74.1745). These circles are approximations, not exact airport boundaries or target-tuned regions.

The flags describe location type: airport trips may involve different roads and traffic patterns, beyond what endpoint distance captures. The group improved validation performance. **Keep**, without attributing the group improvement to any one airport flag.

### Rush Hour

Two trips with similar distance and direction can take different times depending on traffic. We added `morning_rush` and `evening_rush`, treating supplied datetimes as NYC local wall time:

- Weekday morning: **07:00–09:59**.
- Weekday evening: **16:00–18:59**.

**Keep.**

### Weekend

`is_weekend` indicates Saturday or Sunday, giving the model a simple weekday/weekend distinction. **Keep.**

### Route Cluster

`route_cluster` combines pickup and dropoff cluster IDs into a categorical origin → destination region. This allows the model to represent approximate route-area combinations. **Keep.**

## 8. Ridge Feature Ablation

The experiment was **sequential/greedy**: each candidate group was tested on top of the currently retained features. We kept a candidate when validation RMSE decreased by more than 0.000001. Rejected features were absent from later candidates; specifically, Manhattan distance was not carried into the bearing or later experiments.

These are **conditional, order-dependent improvements**, not independent feature importance values.

| Experiment | Validation RMSE | Validation R² | Decision |
|---|---:|---:|---|
| A_baseline | 0.511068 | 0.591914 | Baseline |
| A_clean_clusters | 0.507635 | 0.597378 | Keep |
| B_manhattan | 0.507707 | 0.597264 | Drop |
| C_bearing | 0.506711 | 0.598841 | Keep |
| D_airports | 0.505704 | 0.600434 | Keep |
| E_rush | 0.501983 | 0.606293 | Keep |
| F_weekend | 0.500666 | 0.608356 | Keep |
| G_route | 0.498973 | 0.611001 | Keep |

Feature Engineering improved the official Ridge from **RMSE 0.511068 → 0.498973** and **R² 0.591914 → 0.611001**. Rush-hour flags produced one of the clearest incremental improvements in this sequence: RMSE 0.505704 → 0.501983.

These are observed validation gains, not statistical significance claims. Repeatedly using validation for selection can make selected validation scores optimistic.

## 9. Final Ridge Feature Set

### Numerical

- `log_distance`
- `distance_km`
- `lat_diff`
- `lon_diff`
- `hour`
- `dayofweek`
- `month`
- `dayofyear`
- `bearing_sin`
- `bearing_cos`

### Categorical

- `vendor_id`
- `store_and_fwd_flag`
- `passenger_count`
- `same_location`
- `pickup_cluster`
- `dropoff_cluster`
- Airport proximity flags: `pickup_near_jfk`, `pickup_near_lga`, `pickup_near_ewr`, `dropoff_near_jfk`, `dropoff_near_lga`, `dropoff_near_ewr`
- `morning_rush`
- `evening_rush`
- `is_weekend`
- `route_cluster`

There are 10 numerical and 16 categorical input columns before encoding. **`manhattan_km` was tested but rejected.**

## 10. Polynomial Regression Experiment

Normal Ridge is linear in its input features. We tested `PolynomialFeatures(degree=2)` to give Ridge explicit squared terms and interactions such as `x1²`, `x2²`, and `x1*x2`.

Only numerical features are expanded. The numeric path is StandardScaler → PolynomialFeatures → StandardScaler, with `include_bias=False`: the 10 numeric inputs become 65 degree-1/2 terms. Categorical features are one-hot encoded separately and **are not polynomial-expanded**.

| Model | Validation RMSE | Validation R² |
|---|---:|---:|
| Plain selected Ridge | 0.498973 | 0.611001 |
| Polynomial Degree 2 + Ridge | 0.466298 | 0.660279 |

This improvement is strong evidence that useful nonlinear relationships/interactions help predict taxi duration on this split. It does not establish causality. Polynomial Ridge remains a benchmark; **plain `Ridge(alpha=1)` remains the Official Course Model**.

## 11. Model Comparison

All main scores below come from the saved Colab experiment on the same validation partition and log target.

| Model | Validation RMSE | Validation R² |
|---|---:|---:|
| Ridge — Official Course Model | 0.498973 | 0.611001 |
| Polynomial Degree 2 + Ridge | 0.466298 | 0.660279 |
| Random Forest | 0.426733 | 0.715483 |
| Gradient Boosting | 0.441517 | 0.695428 |
| XGBoost | 0.420183 | 0.724151 |

The approximate printed training/evaluation runtimes from the completed Colab output were:

| Benchmark | Approximate runtime |
|---|---:|
| Polynomial Ridge | 87 s |
| Random Forest | 720 s |
| Gradient Boosting | 207 s |
| XGBoost | 32 s |

These are run timings, not a controlled hardware comparison or total project runtime. Environment and hardware affect them.

Ridge is our linear baseline and official model. Polynomial Ridge adds explicit nonlinear interactions. Tree ensembles can naturally represent nonlinear relationships and interactions. **XGBoost had the lowest RMSE and highest R² in this validation comparison**, but is not the new official model.

For reproducibility, the measured settings were preserved during the educational refactor:

| Model | Settings |
|---|---|
| Ridge and Polynomial Ridge | `alpha=1`, `solver="lsqr"`, `tol=1e-8` |
| Random Forest | 80 trees, `max_depth=16`, `min_samples_leaf=5`, `max_features=0.8`, `random_state=42` |
| Gradient Boosting | 100 trees, `max_depth=3`, `subsample=0.3`, `random_state=42` |
| XGBoost | 300 trees, `learning_rate=0.05`, `max_depth=6`, `tree_method="hist"`, `random_state=42` |

Random Forest and XGBoost default to `n_jobs=4`. Random Forest/Gradient Boosting use dense preprocessed inputs; Ridge/XGBoost retain sparse preprocessing. An earlier slow forced-sparse forest attempt did not produce a reported score. No model or feature hyperparameters were changed in the educational refactor.

## 12. What We Learned About Model Capacity

```text
Ridge baseline                         RMSE 0.511068
    ↓ Feature Engineering
Selected Ridge                         RMSE 0.498973
    ↓ Explicit nonlinear polynomial interactions
Polynomial Ridge                       RMSE 0.466298
    ↓ Compare alternative nonlinear tree ensembles
    ├─ Random Forest                   RMSE 0.426733
    ├─ Gradient Boosting               RMSE 0.441517
    └─ XGBoost                         RMSE 0.420183
```

Feature Engineering helped the fixed Ridge model. Polynomial expansion and tree models improved predictions further, suggesting useful nonlinear relationships in this problem. The three tree models are alternatives, not sequential improvements over one another. This is evidence from one validation experiment, not a causal explanation or a universal ranking of algorithms.

## 13. Leakage Prevention Checklist

- [x] Train and validation are separate.
- [x] IDs checked for overlap where applicable.
- [x] Target transformation does not use validation statistics.
- [x] KMeans fitted on train only.
- [x] Validation receives `KMeans.predict()`.
- [ ] Test will receive `KMeans.predict()` when evaluated.
- [x] StandardScaler fitted through the training pipeline only.
- [x] OneHotEncoder fitted through the training pipeline only.
- [x] Test was not used for feature selection.
- [x] Test was not used for model selection.
- [x] Test was not used for hyperparameter tuning.

## 14. Official Model vs Benchmarks

> **Official Course Model: `Ridge(alpha=1)`**  
> Final selected Colab validation: **RMSE = 0.498973; R² = 0.611001**.

**Benchmarks:** Polynomial Ridge, Random Forest, Gradient Boosting, and XGBoost. Their performance helps us learn about model capacity and compare approaches. Better benchmark performance does not replace the course-required official Ridge model.

## 15. Historical / Provenance Note

An earlier local run used **1,000,000 train / 229,319 validation rows**, with selected Ridge **RMSE = 0.495187; R² = 0.616881**. Its `feature_ablation.csv`, `benchmark_results.csv`, and `run_metadata.json` are preserved for provenance. The exact reason for its training-row difference from Colab was not established; the single missing label does not explain it. **Do not combine the two runs.**

The main experiment is **Colab: 993,415 train / 229,319 validation**. `colab_feature_ablation.csv` and `colab_benchmark_results.csv` transcribe the six-decimal saved outputs of `notebooks/Untitled13.ipynb`, without inventing extra precision. They are not new executions of the simplified code. A future training run will write fresh results; the Colab-prefixed files preserve this history. Input-file and environment-version differences may affect reproducibility.

## 16. Current Project State

**COMPLETED:**

- EDA.
- Target transformation.
- Baseline Ridge.
- Distance features.
- Leakage-safe KMeans.
- Coordinate cleaning for KMeans fitting.
- Feature Engineering.
- Sequential feature ablation.
- Polynomial benchmark.
- Random Forest benchmark.
- Gradient Boosting benchmark.
- XGBoost benchmark.
- Educational source-code refactor.

The refactor corrected an older notebook workflow that reused precomputed EDA clusters before splitting, shared a preprocessor across models, and required MLflow. The simplified workflow starts from the provided raw partitions, fits clusters on train only, and creates fresh preprocessors. Old processed EDA cluster columns are not model inputs. The pre-refactor comparison notebook was backed up in the task workspace; the Colab notebook remains preserved.

The source exposes `add_features`, `fit_kmeans`, and `add_clusters`, followed by a visible training sequence. The comparison notebook shows the actual feature, clustering, preprocessing, and model steps without hiding them behind source imports; some duplication is deliberate for learning. Existing EDA plots were preserved, with a raw-target histogram and route-category example added. Automatic ablation, mandatory MLflow, and old cluster-count options were removed from the simplified workflow; legacy metadata remains historical evidence.

Model-local `validation_results.csv` pairs scores with the exact saved pipelines, preventing accidental pairing with a different historical run. The test workflow checks that all five model artifacts are present before opening test data. The existing local model directory contains only the older Ridge and two KMeans artifacts; all five final models still need to be trained and saved.

Previous refactor verification passed six unit tests, syntax/import/help checks, and a synthetic CLI exercise with 79 usable training rows (one missing label removed from 80) and 30 validation rows. All five reloaded synthetic models reproduced their recorded scores, and notebook/source features matched. Those checks covered train-only preprocessing, predict-only clustering, target independence, separate preprocessors, and numeric-only polynomial expansion. Synthetic scores are not project results. This documentation update runs no training or test evaluation.

**NEXT:**

1. Review the simplified source code.
2. Train/save all five final model artifacts using locked settings.
3. Upload/use `test.csv`.
4. Evaluate all five models **once** on test.
5. Save `reports/final_test_results.csv`.
6. Generate prediction CSVs.
7. Do **not** tune anything after seeing test results.

**FINAL TEST STATUS: NOT RUN YET.**

## 17. Final Test Plan

Evaluate Ridge, Polynomial Degree 2 + Ridge, Random Forest, Gradient Boosting, and XGBoost together using saved preprocessing and train-fitted KMeans. Prediction must not fit or refit anything on test.

For each model, report **Validation RMSE, Test RMSE, Validation R², and Test R²** in log-target space. Validation scores must come from the matching saved model's validation records, not an unrelated local or Colab report.

Save the comparison to `reports/final_test_results.csv`. It currently contains only the header, with no test result rows. Save per-trip predictions to:

- `predictions/ridge_predictions.csv`
- `predictions/polynomial_ridge_predictions.csv`
- `predictions/random_forest_predictions.csv`
- `predictions/gradient_boosting_predictions.csv`
- `predictions/xgboost_predictions.csv`

Prediction exports include IDs when present, predicted log duration, and predicted seconds using `maximum(0, expm1(predicted_log_duration))`.

**The final test is reporting only. After seeing its results: no feature changes, no hyperparameter tuning, and no model-selection changes.** Ridge remains the Official Course Model.

## Quick Recall

- Predicting taxi duration is regression; `Ridge(alpha=1)` is the official model.
- Main Colab data: 993,415 usable train rows and 229,319 validation rows.
- Only one training row was removed because its target was missing.
- Train and score on `log1p(trip_duration)`; reported metrics are not in seconds.
- Fit KMeans, scalers, and encoders on train only; validation/test use learned transforms.
- Separate pickup/dropoff cluster numbers are categorical labels, not matching regions.
- Greedy ablation measures conditional gains; Manhattan distance was rejected.
- Feature Engineering improved Ridge RMSE from 0.511068 to 0.498973.
- Polynomial Ridge reached 0.466298 RMSE, supporting useful nonlinear interactions.
- XGBoost led the benchmarks at 0.420183 RMSE; it does not replace official Ridge.
- Keep the earlier million-row local results separate from Colab results.
- Test is not run yet: evaluate once after locking decisions, then report without tuning.

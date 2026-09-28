# NYC Taxi Trip Duration — Final Experiment Summary

## Objective and data

Predict trip duration from trip information. This is regression, with **Ridge(alpha=1)** as the Official Course Model. Polynomial Ridge, Random Forest, Gradient Boosting and XGBoost provide benchmark comparisons.

The completed Colab run used the provided partitions:

- Training: **1,000,000 trips**.
- Validation: **229,319 trips**.
- Test: **229,322 trips**.

No missing training targets were found in this run. The training script removes missing training labels in memory if present; invalid validation/test labels cause an error rather than row removal. The processed EDA sample is not the training input.

## Target and metrics

Models learn `log1p(trip_duration)`. RMSE and R² are measured in this log-target space, not in seconds. Lower RMSE and higher R² indicate better predictions on the evaluated partition. R² is not a classification accuracy percentage.

Prediction exports contain trip IDs, predicted log duration and `maximum(0, expm1(predicted_log_duration))` seconds.

## Feature pipeline

### Time

Extract hour, day of week, month and day of year from the pickup timestamp. Add weekday morning rush (07:00–09:59), weekday evening rush (16:00–18:59), and weekend indicators. These flags give Ridge explicit representations of selected time patterns.

### Distance and direction

Use Haversine distance in kilometers, its `log1p` transformation, absolute latitude/longitude differences, a same-location flag for distance at most 0.1 km, and bearing sine/cosine. Haversine measures direct endpoint separation, not road distance. Sine/cosine represent direction without a discontinuity between 0° and 360°.

### Airports and location clusters

Mark pickup/dropoff within 2 km of approximate JFK, LGA and EWR coordinates. Fit separate pickup and dropoff KMeans models with K=5 and random seed 42. Fit clusters only using training trips whose endpoints are within latitude 40.4–41.0 and longitude −74.3 to −73.6.

This geographic mask selects KMeans fitting coordinates; it does not remove regression-training or evaluation rows. Cluster IDs are categorical labels, not ordered values. Their pickup/dropoff pair forms a route category.

The final input also includes vendor ID, passenger count and the store-and-forward flag. Manhattan distance is not part of the final feature set. This final run evaluates a fixed feature set; it does not measure the independent contribution of each feature.

## Preprocessing and model settings

Each model owns a separate preprocessing pipeline. Numerical features are standardized, categorical features are one-hot encoded with unknown categories ignored, and all learned transformations fit on training data only.

Polynomial degree 2 expands only the 10 numerical features into 65 terms, with scaling before and after expansion. Categorical features are not polynomially expanded.

| Model | Fixed settings |
|---|---|
| Ridge | alpha=1, lsqr solver, tolerance 1e-8 |
| Polynomial Ridge | numerical degree 2, no bias term; same Ridge settings |
| Random Forest | 80 trees, depth 16, minimum leaf size 5, max features 0.8, seed 42 |
| Gradient Boosting | 100 trees, depth 3, subsample 0.3, seed 42 |
| XGBoost | 300 trees, learning rate 0.05, depth 6, histogram tree method, seed 42 |

Random Forest and XGBoost use four worker threads. Random Forest and Gradient Boosting receive dense preprocessed inputs; Ridge and XGBoost retain sparse preprocessing.

## Training and final evaluation

Six synthetic pipeline checks passed before full training. Training and validation completed for all five pipelines in approximately **26.3 minutes**. Each pipeline was saved with its matching validation record.

Final evaluation loaded the saved pipelines and train-fitted KMeans models. Test processing used learned transformations and prediction only. No feature, hyperparameter or official-model changes followed the test.

| Model | Validation RMSE | Test RMSE | Validation R² | Test R² |
|---|---:|---:|---:|---:|
| Ridge (Official Course Model) | 0.495187 | 0.490155 | 0.616881 | 0.620731 |
| Polynomial Degree 2 + Ridge | 0.464108 | 0.458792 | 0.663463 | 0.667715 |
| Random Forest | 0.424157 | 0.419676 | 0.718908 | 0.721960 |
| Gradient Boosting | 0.440879 | 0.436288 | 0.696308 | 0.699512 |
| XGBoost | 0.417929 | 0.412810 | 0.727102 | 0.730983 |

XGBoost achieved the lowest test RMSE among the fixed models. Ridge remains the Official Course Model. Test RMSE is slightly lower than validation RMSE for each model in these partitions; this alone does not establish performance on future or differently distributed trips.

## Saved results

- [Validation results](benchmark_results.csv): full-run validation metrics and partition sizes.
- [Final test results](final_test_results.csv): matching validation/test RMSE and R².
- [Run manifest](final_colab_run/run_manifest.json): Python/package versions, source hashes and training duration.
- [Training log](final_colab_run/training.log) and [test log](final_colab_run/final_test.log): recorded execution output.

Five saved model pipelines and two KMeans artifacts are available locally in `models/`. Five prediction CSVs are available locally in `predictions/`, each containing 229,322 rows. Binary artifacts, predictions and raw input partitions are excluded from Git.

## Key takeaways

- Feature engineering converts time and coordinates into useful model inputs.
- Learned preprocessing and clustering belong to the training partition only.
- Nonlinear models outperformed Ridge on these evaluated partitions.
- The official model is determined by course requirements, not changed after viewing test scores.
- The final test is complete; use its saved results for reporting, not further tuning.

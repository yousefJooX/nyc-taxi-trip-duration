# NYC Taxi Trip Duration Prediction

Classical machine learning regression project predicting `log1p(trip_duration)`. RMSE and R² below are measured on the log target, not in seconds.

**Official Course Model: Ridge(alpha=1).** Polynomial Ridge, Random Forest, Gradient Boosting and XGBoost are benchmarks. Better benchmark scores do not change the official course model.

## Model evaluation

All five pipelines were trained with the locked settings, saved, and evaluated on the held-out test partition. No feature or hyperparameter tuning followed the test.

- Training: **1,000,000 rows**.
- Validation: **229,319 rows**.
- Test: **229,322 rows**, with a prediction export for each model.
- Full training time: approximately **26.3 minutes** on Colab CPU.
- Six synthetic pipeline checks passed before training.

| Model | Validation RMSE | Test RMSE | Validation R² | Test R² |
|---|---:|---:|---:|---:|
| Ridge (Official Course Model) | 0.495187 | 0.490155 | 0.616881 | 0.620731 |
| Polynomial Degree 2 + Ridge | 0.464108 | 0.458792 | 0.663463 | 0.667715 |
| Random Forest | 0.424157 | 0.419676 | 0.718908 | 0.721960 |
| Gradient Boosting | 0.440879 | 0.436288 | 0.696308 | 0.699512 |
| XGBoost | 0.417929 | 0.412810 | 0.727102 | 0.730983 |

XGBoost has the lowest test RMSE in this run. Ridge remains the official model.

[Executed Colab notebook](https://colab.research.google.com/drive/1EETmiZuSLPmrZ0TDzVm07DDsKNowc1KW) (Google account access may be required).

## Workflow

1. Explore trip durations, timestamps and pickup/dropoff coordinates.
2. Engineer time, distance, direction, airport and location-cluster features.
3. Fit preprocessing and five regression models on the training partition.
4. Compare models on validation data with fixed evaluation metrics.
5. Evaluate the final pipelines on held-out test data and export predictions.

## Project structure

```text
notebooks/
  01_data_understanding.ipynb    # data exploration and visualizations
  02_model_comparison.ipynb     # step-by-step model implementation
src/
  __init__.py
  features.py                  # feature engineering and location clustering
  train.py                     # train, validate and save five pipelines
  test.py                      # evaluate saved pipelines and export predictions
tests/
  test_pipeline.py             # synthetic checks for preprocessing and inference
reports/
  benchmark_results.csv        # validation metrics
  final_test_results.csv       # matching validation and test metrics
  final_colab_run/             # execution logs and environment details
  historical_local/           # supporting local experiment records
  colab_*.csv                  # feature-selection experiment results
  experiment_summary.md        # experiment methodology and findings
models/                        # trained pipelines and KMeans artifacts
predictions/                   # per-model trip-duration predictions
split/                         # full-data input partitions
split_sample/                  # small train/validation/test learning samples
data/
  processed_taxi.csv           # processed sample for exploratory analysis
README.md
requirements.txt
.gitignore
```

Datasets, trained model binaries and prediction exports are local files excluded from Git. The repository includes source code, notebooks, a processed EDA sample and compact experiment reports.

## Features and leakage prevention

Time features include hour, weekday, month, day of year, weekday rush flags and weekend. Geographic features include Haversine/log distance, absolute coordinate differences, same-location flag, bearing sine/cosine and airport proximity within 2 km. Separate pickup/dropoff KMeans use K=5, fit on training coordinates inside the NYC bounds. The fitting mask does not drop regression or evaluation rows. Cluster IDs and route pairs are categorical. Manhattan distance is excluded.

Each model has its own preprocessing pipeline. Scalers and encoders fit on train only. Polynomial degree 2 expands only the numerical features. Test uses saved transformations and `predict()` only.

## Install and check the code

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The measured Colab package versions and source hashes are recorded in `reports/final_colab_run/run_manifest.json`. Loading binary models requires a compatible environment.

## Optional sample training

From the project root, use separate output folders so the final models stay intact:

```bash
python src/train.py --train split_sample/train.csv --val split_sample/val.csv --model-dir models/sample --report-dir reports/sample --jobs 4
```

These learning samples are local and are not distributed with the repository. Sample scores are not the reported full-data scores.

## Full-data training and evaluation

Provide the course partitions as `split/train.csv`, `split/val.csv` and `split/test.csv`. Use these provided partitions directly; the processed EDA sample is not the training input.

Train and validate all five models:

```bash
python src/train.py --train split/train.csv --val split/val.csv --jobs 4
```

After all model and feature choices are fixed, evaluate the saved pipelines:

```bash
python src/test.py --test split/test.csv
```

Training saves the pipelines, two KMeans models and matching validation records. Evaluation exports the validation/test comparison and five prediction CSVs. Predictions include trip IDs, predicted log duration and predicted duration in seconds.

The results reported above are from the completed final evaluation. Test results are for reporting only, not further feature selection or hyperparameter tuning.

## Experiment documentation

See [Experiment summary](reports/experiment_summary.md) for feature engineering, ablation findings and model comparisons. Feature-selection reports use a separate 993,415-row training experiment; the final evaluation above uses 1,000,000 training rows. Each run's results are documented separately for reproducibility.

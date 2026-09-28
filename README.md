# NYC Taxi Trip Duration Prediction

Classical machine learning regression project predicting `log1p(trip_duration)`. RMSE and R² below are measured on the log target, not in seconds.

**Official Course Model: Ridge(alpha=1).** Polynomial Ridge, Random Forest, Gradient Boosting and XGBoost are benchmarks. Better benchmark scores do not change the official course model.

## Completed Colab training and final test

All five pipelines were trained with the locked settings, saved, and evaluated on the held-out test partition. No feature or hyperparameter tuning followed the test.

- Training: **1,000,000 rows**, no missing-target rows removed.
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

## Project structure

```text
notebooks/
  01_data_understanding.ipynb    # exploratory analysis
  02_model_comparison.ipynb      # educational model workflow
src/
  features.py                   # feature engineering and train-fitted KMeans
  train.py                      # fixed five-model training and validation
  test.py                       # prediction-only final evaluation
  __init__.py
tests/test_pipeline.py           # small synthetic software tests
reports/
  benchmark_results.csv         # NEW Colab validation results
  final_test_results.csv        # NEW matched validation/test comparison
  final_colab_run/               # environment, source hashes, logs, provenance
  historical_local/             # previous local results, kept separate
  colab_*.csv                   # historical 993,415-row experiment
  experiment_summary.md         # learning narrative and final results
models/                         # current trained pipelines; local, ignored
predictions/                    # five final prediction CSVs; local, ignored
split/                          # retained test.csv and val.csv; local, ignored
split_sample/                   # retained train/val/test samples; local, ignored
data/processed_taxi.csv          # historical EDA sample, not training input
```

The older full `split/train.csv` and previous model artifacts were moved to a backup outside this repository during cleanup. Sample data and the existing test/validation files were preserved. The current trained artifacts remain available locally; GitHub contains code, notebooks and compact reports.

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

## Reproduce the full run

Restore the original full training data into `split/train.csv` and supply the matching `val.csv` and `test.csv`. Do not create a new random split from the processed EDA sample. For Colab, use the executed notebook linked above with the matching input files. The temporary upload/train/test notebook is not included in this repository. Download trained artifacts before the runtime ends.

The final test for the recorded run is **already complete**. Use the saved CSVs for reporting; do not rerun it for tuning or model selection.

## Historical experiments

The earlier feature-selection experiment used 993,415 usable training rows. Its six-decimal metrics are preserved in `reports/colab_*.csv`. The prior local ablation and metadata are in `reports/historical_local/`. Keep these separate from the new million-row Colab run. See `reports/experiment_summary.md` for the full study narrative.

## GitHub contents

Raw splits, learning splits, model binaries, prediction exports, ZIP transfers, caches and local credentials are ignored. Compact validation/test reports and execution provenance are included. The historical tracked EDA sample is retained. No datasets or binary models need to be uploaded to reproduce the source code.

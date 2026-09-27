# NYC Taxi Trip Duration Prediction

A Classical ML learning project: predict trip duration using `log1p(trip_duration)`. RMSE and R² are measured on the log target, not in seconds.

## Official Course Model
**Ridge(alpha=1)** remains the final course model. Polynomial Ridge, Random Forest, Gradient Boosting, and XGBoost are comparisons, even when their validation scores are better.

## Data and split
Put the instructor's `train.csv`, `val.csv`, and held-out `test.csv` in `split/`. Keep the provided split; do not split the EDA export again. Training removes missing `trip_duration` rows only, in memory. Validation/test labels must be valid; their rows are never removed to improve scores.

The preserved Colab run (`notebooks/Untitled13.ipynb`) used **993,415 training rows** after removing one missing label, and **229,319 validation rows**. The existing local training file has 1,000,000 rows; it is a different input. Removing one missing label does not explain that entire size difference. Scores cannot be reproduced exactly without the same input files.

## Features and leakage prevention
- Time: hour, weekday, month, day of year, weekday rush flags (07:00–09:59 / 16:00–18:59), weekend.
- Geography: Haversine km, log1p distance, absolute latitude/longitude differences, same location ≤0.1 km, bearing sine/cosine.
- Airports: pickup/dropoff within 2 km of fixed approximate JFK, LGA, EWR points.
- Separate pickup/dropoff KMeans, **K=5**, fitted only on training coordinates inside the existing NYC bounds. This fitting mask does not remove model-training or evaluation rows.
- Cluster IDs and their route pair are categorical and one-hot encoded. Manhattan distance was rejected and is excluded.

Every model has its own saved preprocessing pipeline. Scalers/encoders fit on train only. Polynomial degree 2 expands **only numerical features**, with scaling before and after expansion. All hyperparameters match the previous Colab implementation; they are explicit in `src/train.py`.

## Validation results — historical Colab run
| Model | Validation RMSE | Validation R² |
|---|---:|---:|
| Ridge (Official Course Model) | 0.498973 | 0.611001 |
| Polynomial Degree 2 + Ridge | 0.466298 | 0.660279 |
| Random Forest | 0.426733 | 0.715483 |
| Gradient Boosting | 0.441517 | 0.695428 |
| XGBoost | 0.420183 | 0.724151 |

The ablation was **sequential/greedy**, not independent single-feature tests. See `reports/experiment_summary.md`. Existing local-run CSVs are preserved separately from `reports/colab_*.csv`; the rewritten code was not fully retrained during this refactor.

## Structure
```text
notebooks/
  01_data_understanding.ipynb  # EDA only; existing plots preserved
  02_model_comparison.ipynb    # visible implementation, step by step
  Untitled13.ipynb             # unchanged Colab evidence
src/
  __init__.py
  features.py                 # add_features, fit_kmeans, add_clusters
  train.py                    # train, validate, save all five pipelines
  test.py                     # manual final reporting for all five
tests/test_pipeline.py        # small synthetic checks
reports/                      # historical validation CSVs and summary
models/                       # generated pipelines, KMeans, validation scores
predictions/                  # five generated prediction CSVs
split/                        # instructor's data
split_sample/                 # optional learning samples
data/processed_taxi.csv        # existing historical EDA export, not model input
README.md
requirements.txt
.gitignore
```

## Install and train
Run from the project root:
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/train.py --train split/train.csv --val split/val.csv
```
This now trains the **fixed retained features and all five models** by default. It saves five `.joblib` pipelines, two KMeans models, and their matching `models/validation_results.csv`. The report CSV is updated only after the full training loop finishes. Full tree benchmarks can take several minutes. Use `--jobs 4` (default) to bound CPU threads. For a quick learning run, pass sample train/validation paths and separate `--model-dir` / `--report-dir` folders.

The old `--experiments`, `--polynomial`, `--benchmarks`, `--n-clusters`, and `--mlflow-uri` options were removed to keep the final workflow simple. Historical ablation reports remain available. No new feature selection, metadata JSON generation, or MLflow dependency is in the core workflow.

## FINAL test — manually, after review
**Do not run test.py during model/feature selection.** Train/save all five models first; the old artifacts only include Ridge.
```bash
python src/test.py --test split/test.csv
```
This command checks that all models and matching validation scores exist **before reading test data**. It loads saved train-fitted KMeans, calls `predict()` only, and creates:
- `reports/final_test_results.csv`: validation and test RMSE/R² for all five models.
- `predictions/{ridge,polynomial_ridge,random_forest,gradient_boosting,xgboost}_predictions.csv`: IDs, log predictions, and `maximum(0, expm1(prediction))` seconds.

The final-test CSV currently has a header only: **no final test has run**. Test results are final reporting only; do not tune features, hyperparameters, or model choice after seeing them. Ridge remains official.

Raw splits, model binaries, predictions, caches and local credentials are ignored. The previously tracked EDA sample is preserved unchanged; no new dataset or model binary is added to Git.

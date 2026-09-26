# NYC Taxi Trip Duration Prediction

## Goal
Predict taxi trip duration from pickup time, trip endpoints, and available trip metadata. Train on `log1p(trip_duration)` and evaluate **RMSE and R² on that log target**; RMSE is not in seconds.

## Official Model
**`Ridge(alpha=1)` is the official course model.** Polynomial Ridge, Random Forest, Gradient Boosting, and XGBoost are diagnostic benchmarks only. Only the official plain Ridge and its train-fitted KMeans models are saved.

## Dataset
Use the instructor-provided files, not a replacement Kaggle split. Obtain them from the course project folder and place `train.csv`, `val.csv`, and the held-out `test.csv` under `split/`. The measured run uses 1,000,000 training rows and 229,319 validation rows. Put optional course samples under `split_sample/`.

The test set is reserved for one final evaluation after development. It was not opened or evaluated during this refactor. All validation rows remain in every comparison, including coordinate outliers. `data/processed_taxi.csv` is a previously committed, small EDA export retained intentionally; training uses the provided raw split instead.

## Project Structure
```text
.
├── data/processed_taxi.csv              # existing tracked EDA sample
├── notebooks/
│   ├── 01_data_understanding.ipynb      # preserved EDA, portable paths
│   └── 02_model_comparison.ipynb        # experiments through shared source
├── src/
│   ├── __init__.py
│   ├── features.py                     # shared target-independent features
│   ├── train.py                        # Ridge, ablations, optional benchmarks
│   └── test.py                         # final inference/evaluation only
├── tests/test_pipeline.py               # synthetic leakage/roundtrip checks
├── reports/
│   ├── experiment_summary.md
│   ├── feature_ablation.csv
│   ├── benchmark_results.csv
│   └── run_metadata.json
├── models/                             # generated, ignored
├── split/                              # course data, ignored
├── split_sample/                       # course samples, ignored
├── README.md
├── requirements.txt
└── .gitignore
```
The course PDF, warmstart script, and editor settings remain locally available and ignored. Empty existing report subdirectories are preserved.

## Feature Engineering
Baseline numerical features: `log_distance`, `distance_km`, `lat_diff`, `lon_diff`, `hour`, `dayofweek`, `month`, `dayofyear`. Haversine uses Earth radius 6371.0088 km; `log_distance=log1p(distance_km)`. Coordinate differences are absolute, matching the original large-data script (historical EDA used signed differences).

Baseline categoricals: `vendor_id`, `store_and_fwd_flag`, `passenger_count`, `same_location` (distance ≤ 0.1 km), `pickup_cluster`, `dropoff_cluster`. Cluster IDs have no numerical ordering and are one-hot encoded.

Retained additions:
- Initial bearing as `bearing_sin` and `bearing_cos`, avoiding the 0°/360° discontinuity. Both are zero at identical endpoints.
- Pickup/dropoff flags within 2 km of approximate reference points for JFK (40.6413, -73.7781), LaGuardia (40.7769, -73.8740), and Newark (40.6895, -74.1745). These fixed geographic heuristics are not terminal boundaries or target-tuned radii. Newark falls within the existing study-area bounds.
- Weekday morning rush 07:00–09:59 and evening rush 16:00–18:59; Saturday/Sunday `is_weekend`. Pickup datetimes are treated as supplied local NYC wall time.
- Categorical `route_cluster`, the pickup/dropoff cluster pair.

Manhattan-style distance sums north/south and east/west Haversine legs in km. It worsened validation RMSE and was rejected. This geographic approximation is not a road-network distance.

## Leakage Prevention
Each model owns a fresh ColumnTransformer: StandardScaler for numerical features and OneHotEncoder(handle_unknown="ignore") for categoricals. Every learned transformation is fitted on training rows only. Numerical polynomial expansion uses degree 2 only, with scaling before and after expansion; categorical columns are never polynomial-expanded.

Separate pickup and dropoff KMeans models use K=5, random_state=42, n_init="auto". Baseline A fits all training coordinates to reproduce the previous script. A separate controlled experiment restricts KMeans fitting to training trips whose **both** endpoints are within latitude [40.4, 41.0], longitude [-74.3, -73.6], the bounds already used in EDA. This improved validation and was retained. All training and validation rows still receive cluster predictions; no rows are removed from Ridge fitting or scoring. Validation and later test data use only `predict()`.

## Experiments
The previous sample results (different sample preparation/split) included Ridge RMSE ≈ 0.476596, R² ≈ 0.647851. They are historical observations, not full-data results or rerun measurements.

On the supplied large split, the baseline reproduced RMSE **0.510391**, R² **0.592994**. The selected plain Ridge reached RMSE **0.495187**, R² **0.616881**. Polynomial degree 2 reached RMSE **0.464111**, R² **0.663458**, while remaining experimental.

See [the full experiment report](reports/experiment_summary.md) for each candidate, rejected features, historical evidence, and measured benchmark scores. CSVs contain unrounded metrics. Additions are tried against the retained incumbent and accepted only for an RMSE decrease greater than 0.000001. This is a transparent selection rule, not a statistical significance claim. Delta columns named `previous` refer to the retained incumbent, not necessarily the preceding rejected row.

## Running
From the project root, in a Python environment with the requirements installed:
```bash
pip install -r requirements.txt
# Baseline only; writes baseline artifacts/results to separate directories:
python src/train.py --train split/train.csv --val split/val.csv --model-dir models/baseline --report-dir reports/baseline
# Reproduce retained feature selection and all benchmark results:
python src/train.py --train split/train.csv --val split/val.csv --experiments --benchmarks --jobs 4
# Feature selection and polynomial comparison, without tree benchmarks:
python src/train.py --train split/train.csv --val split/val.csv --experiments --polynomial
python -m unittest discover -s tests -v
```
`python src/train.py --train split/train.csv --val split/val.csv` is also supported; without `--experiments` it trains baseline A and overwrites default saved models/reports. Use separate output directories to preserve the selected model. `--n-clusters` defaults to 5; no cluster-count search was performed.

Optional MLflow logging:
```bash
python src/train.py --experiments --mlflow-uri http://127.0.0.1:5000
```
The experiment is `NYC Taxi - Model Comparison`. Missing MLflow or an unavailable server produces a warning and training continues. Logging is off by default. Saved artifacts contain only scikit-learn pipelines plus separate KMeans; the selected feature columns are embedded in the pipeline and recorded in metadata.

## Models
Random Forest uses 80 trees, depth 16, min_samples_leaf=5, max_features=0.8. Gradient Boosting uses 100 depth-3 trees and subsample=0.3. XGBoost uses 300 depth-6 trees, learning_rate=0.05, CPU histogram training. These bounded settings avoid costly searches; all models receive the same full training split and selected feature columns. Gradient Boosting draws a training subsample per boosting iteration. All random seeds are 42. XGBoost is imported only for requested benchmarks and is skipped with a warning if unavailable.

## Final Test — Later Only
After locking the official model, the following command loads the saved Ridge and train-fitted KMeans, writes duration predictions via `expm1` (clipped at zero), and explicitly evaluates log-target metrics:
```bash
python src/test.py --test split/test.csv --model-dir models --output predictions.csv --evaluate
```
**This command was not executed during development.** Omit `--evaluate` for prediction only; unlabeled CSVs are supported. Predictions preserve input row order and IDs when present. Only load trusted joblib artifacts.

## Reproducibility
Measured environment versions and retained columns are in `reports/run_metadata.json`; model hyperparameters are fixed in `src/train.py`. Numerical libraries may produce small differences across environments. Core dependencies include NumPy, pandas, scikit-learn, SciPy, joblib and threadpoolctl; notebooks additionally use Jupyter, matplotlib, seaborn and haversine. XGBoost and MLflow support optional benchmark/tracking workflows. No GPU is required.

Models, raw/split datasets, predictions, caches, credentials and local MLflow artifacts are ignored. The existing processed EDA sample remains tracked deliberately. There are no machine-specific paths in project source or notebook code.

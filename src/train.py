"""Train the five fixed models. Ridge(alpha=1) is the Official Course Model."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

try:
    from .features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, add_features, fit_kmeans, add_clusters
except ImportError:
    from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, add_features, fit_kmeans, add_clusters

MODEL_NAMES = {
    "ridge": "Ridge (Official Course Model)",
    "polynomial_ridge": "Polynomial Degree 2 + Ridge",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}


def clean_training_target(train):
    """Remove missing labels from training only; do not rewrite the CSV."""
    return train.dropna(subset=["trip_duration"]).copy()


def get_log_target(df):
    duration = df["trip_duration"].to_numpy(dtype=float)
    if not np.isfinite(duration).all() or (duration < 0).any():
        raise ValueError("trip_duration must be finite and nonnegative; validation/test rows are never dropped")
    return np.log1p(duration)


def make_preprocessor(polynomial=False, dense=False):
    numeric = StandardScaler()
    if polynomial:
        numeric = Pipeline([
            ("scale_input", StandardScaler()),
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale_expanded", StandardScaler()),
        ])
    # Dense output keeps sklearn tree fitting practical; feature values are unchanged.
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], sparse_threshold=0.0 if dense else 1.0)


def build_models(jobs=4):
    from xgboost import XGBRegressor

    # Each model gets its OWN preprocessing object. Hyperparameters are unchanged.
    models = {
        "ridge": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", Ridge(alpha=1, solver="lsqr", tol=1e-8)),
        ]),
        "polynomial_ridge": Pipeline([
            ("preprocessor", make_preprocessor(polynomial=True)),
            ("model", Ridge(alpha=1, solver="lsqr", tol=1e-8)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", make_preprocessor(dense=True)),
            ("model", RandomForestRegressor(
                n_estimators=80, max_depth=16, min_samples_leaf=5,
                max_features=0.8, random_state=42, n_jobs=jobs, verbose=1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocessor", make_preprocessor(dense=True)),
            ("model", GradientBoostingRegressor(
                n_estimators=100, max_depth=3, subsample=0.3, random_state=42, verbose=1,
            )),
        ]),
        "xgboost": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                tree_method="hist", random_state=42, n_jobs=jobs,
            )),
        ]),
    }
    return models


def evaluate_model(model, X, y):
    prediction = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, prediction))
    r2 = r2_score(y, prediction)
    return float(rmse), float(r2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("split/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("split/val.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    # 1. Load the provided split and clean ONLY missing training targets.
    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)
    before = len(train)
    train = clean_training_target(train)
    print(f"Removed {before - len(train)} training rows with missing targets.")
    if "id" in train and "id" in val and train["id"].isin(val["id"]).any():
        raise ValueError("Training and validation IDs overlap")
    y_train = get_log_target(train)
    y_val = get_log_target(val)

    # 2. Add features, learn clusters on train, then predict both splits.
    train = add_features(train)
    val = add_features(val)
    with threadpool_limits(limits=args.jobs):
        pickup_kmeans, dropoff_kmeans = fit_kmeans(train)
    train = add_clusters(train, pickup_kmeans, dropoff_kmeans)
    val = add_clusters(val, pickup_kmeans, dropoff_kmeans)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train = train[feature_columns]
    X_val = val[feature_columns]

    # 3. Prepare the five fixed models and output folders.
    models = build_models(args.jobs)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    columns = ["model", "rmse", "r2", "train_rows", "validation_rows"]
    # An incomplete run must not look like five newly trained models.
    pd.DataFrame(columns=columns).to_csv(args.model_dir / "validation_results.csv", index=False)
    joblib.dump(pickup_kmeans, args.model_dir / "pickup_kmeans.joblib")
    joblib.dump(dropoff_kmeans, args.model_dir / "dropoff_kmeans.joblib")

    # 4. Fit on train, evaluate on validation, and save each complete pipeline.
    for key, model in models.items():
        print(f"Training {MODEL_NAMES[key]}...", flush=True)
        with threadpool_limits(limits=args.jobs):
            model.fit(X_train, y_train)
            rmse, r2 = evaluate_model(model, X_val, y_val)
        joblib.dump(model, args.model_dir / f"{key}.joblib")
        results.append({"model": MODEL_NAMES[key], "rmse": rmse, "r2": r2,
                        "train_rows": len(train), "validation_rows": len(val)})
        # Keep validation scores beside the exact models that produced them.
        pd.DataFrame(results).to_csv(args.model_dir / "validation_results.csv", index=False)
        print(f"Validation RMSE={rmse:.6f}, R2={r2:.6f}", flush=True)

    comparison = pd.DataFrame(results)
    comparison.to_csv(args.report_dir / "benchmark_results.csv", index=False)
    print("\nAll metrics use log1p(trip_duration).")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

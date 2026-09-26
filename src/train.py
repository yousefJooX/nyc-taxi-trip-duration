"""Train official Ridge(alpha=1); optionally run controlled experiments."""
import argparse
import json
import os
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from threadpoolctl import threadpool_limits

try:
    from .features import BASE_NUM, BASE_CAT, ADDITIONS, prepare, fit_clusters, apply_clusters
except ImportError:
    from features import BASE_NUM, BASE_CAT, ADDITIONS, prepare, fit_clusters, apply_clusters


def make_pipeline(num, cat, estimator=None, degree=1):
    numeric = StandardScaler() if degree == 1 else Pipeline([
        ("scale_input", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale_expanded", StandardScaler()),
    ])
    # A fresh preprocessor for EVERY model prevents fitted state being overwritten.
    # Dense matrices avoid very slow sparse splits in sklearn's tree benchmarks.
    # Ridge and XGBoost retain sparse output (XGBoost distinguishes missing zeros).
    preprocessor = ColumnTransformer([
        ("num", numeric, list(num)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), list(cat)),
    ], sparse_threshold=0.0 if isinstance(estimator, (
        RandomForestRegressor, GradientBoostingRegressor)) else 1.0)
    return Pipeline([("preprocessor", preprocessor),
                     ("model", Ridge(alpha=1, solver="lsqr", tol=1e-8)
                      if estimator is None else estimator)])


def target(frame):
    y = frame["trip_duration"].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("trip_duration must be finite and nonnegative")
    return np.log1p(y)


def metrics(y, prediction):
    return {"rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction))}


def optional_logger(uri):
    if not uri:
        return None
    # Logging must never block training on an unavailable local server.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("NYC Taxi - Model Comparison")
        return mlflow
    except Exception as exc:
        warnings.warn(f"MLflow disabled: {exc}")
        return None


def log_run(logger, name, scores, params):
    if logger is not None:
        try:
            with logger.start_run(run_name=name):
                logger.log_params(params)
                logger.log_metrics(scores)
        except Exception as exc:
            warnings.warn(f"MLflow logging skipped: {exc}")


def run(args):
    if args.train.resolve() == args.val.resolve():
        raise ValueError("Train and validation must be different files")
    train_raw, val_raw = pd.read_csv(args.train), pd.read_csv(args.val)
    if "id" in train_raw and "id" in val_raw:
        if set(train_raw.id) & set(val_raw.id):
            raise ValueError("Training and validation IDs overlap")
    y_train, y_val = target(train_raw), target(val_raw)
    train_raw, val_raw = prepare(train_raw), prepare(val_raw)
    print(f"Train={len(train_raw):,}; validation={len(val_raw):,}; test not loaded", flush=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    logger = optional_logger(args.mlflow_uri)
    num, cat = BASE_NUM.copy(), BASE_CAT.copy()
    clusters = fit_clusters(train_raw, k=args.n_clusters)
    train, val = (apply_clusters(frame, *clusters) for frame in (train_raw, val_raw))
    rows, benchmark_rows = [], []
    clean = False

    def evaluate(name, pipe, tr=train, va=val, numerical=None, categorical=None, degree=1):
        start = time.perf_counter()
        print(f"Training {name}...", flush=True)
        pipe.fit(tr, y_train)
        scores = metrics(y_val, pipe.predict(va))
        params = {"model_name": type(pipe.named_steps["model"]).__name__,
                  "feature_set": name, "n_clusters": args.n_clusters,
                  "cluster_fit_scope": "train_only", "polynomial_degree": degree,
                  "numeric_features": ",".join(numerical or num),
                  "categorical_features": ",".join(categorical or cat)}
        if isinstance(pipe.named_steps["model"], Ridge):
            params["alpha"] = 1
        log_run(logger, name, scores, params)
        print(f"{name}: RMSE={scores['rmse']:.6f} R2={scores['r2']:.6f} "
              f"({time.perf_counter()-start:.1f}s)", flush=True)
        return pipe, scores

    best, score = evaluate("A_baseline", make_pipeline(num, cat))
    baseline = score.copy()

    def record(name, added, result, reference, keep, numerical, categorical):
        rows.append({"experiment": name, "added_feature": added, **result,
                     "delta_rmse_previous": result["rmse"]-reference["rmse"],
                     "delta_r2_previous": result["r2"]-reference["r2"],
                     "delta_rmse_baseline": result["rmse"]-baseline["rmse"],
                     "delta_r2_baseline": result["r2"]-baseline["r2"],
                     "keep": keep, "numeric_features": ",".join(numerical),
                     "categorical_features": ",".join(categorical)})
        pd.DataFrame(rows).to_csv(args.report_dir/"feature_ablation.csv", index=False)

    record("A_baseline", "existing large-data features", score, score, True, num, cat)
    if args.experiments:
        clean_clusters = fit_clusters(train_raw, clean=True, k=args.n_clusters)
        clean_train, clean_val = (apply_clusters(f, *clean_clusters) for f in (train_raw, val_raw))
        candidate, result = evaluate("A_clean_clusters", make_pipeline(num, cat), clean_train, clean_val)
        keep = result["rmse"] < score["rmse"] - 1e-6
        record("A_clean_clusters", "NYC-only training coordinates for KMeans", result, score, keep, num, cat)
        if keep:
            best, score, clusters, train, val, clean = candidate, result, clean_clusters, clean_train, clean_val, True
        for name, (new_num, new_cat) in ADDITIONS.items():
            trial_num, trial_cat = num + new_num, cat + new_cat
            candidate, result = evaluate(name, make_pipeline(trial_num, trial_cat), train, val,
                                         trial_num, trial_cat)
            keep = result["rmse"] < score["rmse"] - 1e-6
            record(name, ",".join(new_num+new_cat), result, score, keep, trial_num, trial_cat)
            if keep:
                best, score, num, cat = candidate, result, trial_num, trial_cat
    # Save only official plain Ridge and its corresponding train-fitted clusters.
    joblib.dump(best, args.model_dir/"ridge.joblib")
    for name, cluster in zip(("pickup", "dropoff"), clusters):
        joblib.dump(cluster, args.model_dir/f"{name}_kmeans.joblib")
    metadata = {"official_model": "Ridge(alpha=1)", "numeric_features": num,
                "categorical_features": cat, "clean_cluster_fit": clean,
                "n_clusters": args.n_clusters, "random_state": 42,
                "train_rows": len(train), "validation_rows": len(val),
                "validation_metrics": score, "target": "log1p(trip_duration)",
                "python": platform.python_version(), "sklearn": sklearn.__version__,
                "numpy": np.__version__, "pandas": pd.__version__,
                "selection_rule": "accept only RMSE decrease > 0.000001 vs retained incumbent",
                "test_evaluated": False}
    (args.model_dir/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (args.report_dir/"run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    benchmark_rows.append({"model": "Ridge (official)", **score})
    if args.polynomial or args.benchmarks:
        _, result = evaluate("Polynomial degree 2 + Ridge", make_pipeline(num, cat, degree=2),
                             train, val, degree=2)
        benchmark_rows.append({"model": "Polynomial degree 2 + Ridge", **result})
    if args.benchmarks:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=80, max_depth=16,
                min_samples_leaf=5, max_features=0.8, random_state=42, n_jobs=args.jobs, verbose=1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100,
                max_depth=3, subsample=0.3, random_state=42, verbose=1),
        }
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05,
                max_depth=6, tree_method="hist", random_state=42, n_jobs=args.jobs)
        except ImportError:
            warnings.warn("XGBoost is not installed; benchmark skipped")
        for name, estimator in models.items():
            _, result = evaluate(name, make_pipeline(num, cat, estimator), train, val)
            benchmark_rows.append({"model": name, **result})
            pd.DataFrame(benchmark_rows).to_csv(args.report_dir/"benchmark_results.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(args.report_dir/"benchmark_results.csv", index=False)
    print("\nFEATURE EXPERIMENTS\n"+pd.DataFrame(rows).drop(
        columns=["numeric_features", "categorical_features"]).to_string(index=False))
    print("\nMODEL COMPARISON\n"+pd.DataFrame(benchmark_rows).to_string(index=False))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("split/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("split/val.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--experiments", action="store_true", help="Run greedy Ridge ablations A-G")
    parser.add_argument("--polynomial", action="store_true")
    parser.add_argument("--benchmarks", action="store_true", help="Run bounded tree and polynomial benchmarks")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--mlflow-uri", default=None, help="Optional, e.g. http://127.0.0.1:5000")
    args = parser.parse_args()
    if args.jobs < 1 or args.n_clusters < 1:
        parser.error("jobs and n-clusters must be positive")
    with threadpool_limits(limits=args.jobs):
        run(args)


if __name__ == "__main__":
    main()

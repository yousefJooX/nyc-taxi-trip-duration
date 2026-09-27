"""FINAL reporting only. Run manually after all feature/model choices are locked."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

try:
    from .features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, add_features, add_clusters
    from .train import MODEL_NAMES, get_log_target
except ImportError:
    from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, add_features, add_clusters
    from train import MODEL_NAMES, get_log_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=Path("split/test.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions"))
    args = parser.parse_args()

    # Check training is complete BEFORE opening the held-out data.
    required = [f"{key}.joblib" for key in MODEL_NAMES]
    required += ["pickup_kmeans.joblib", "dropoff_kmeans.joblib", "validation_results.csv"]
    for filename in required:
        if not (args.model_dir / filename).is_file():
            raise FileNotFoundError(f"Missing {filename}. Run train.py to save all five models first.")
    validation = pd.read_csv(args.model_dir / "validation_results.csv")
    if len(validation) != 5 or set(validation["model"]) != set(MODEL_NAMES.values()):
        raise ValueError("Training did not finish all five models; final evaluation cannot start")
    validation = validation.set_index("model")

    test = pd.read_csv(args.test)
    y_test = get_log_target(test)  # Invalid labels cause an error, never row removal.
    pickup_kmeans = joblib.load(args.model_dir / "pickup_kmeans.joblib")
    dropoff_kmeans = joblib.load(args.model_dir / "dropoff_kmeans.joblib")
    test = add_features(test)
    test = add_clusters(test, pickup_kmeans, dropoff_kmeans)  # predict only
    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for key, name in MODEL_NAMES.items():
        model = joblib.load(args.model_dir / f"{key}.joblib")
        predicted_log = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predicted_log))
        r2 = r2_score(y_test, predicted_log)
        results.append({
            "model": name,
            "validation_rmse": validation.loc[name, "rmse"], "test_rmse": rmse,
            "validation_r2": validation.loc[name, "r2"], "test_r2": r2,
        })
        predictions = pd.DataFrame({
            "predicted_log_duration": predicted_log,
            "predicted_trip_duration": np.maximum(0, np.expm1(predicted_log)),
        })
        if "id" in test:
            predictions.insert(0, "id", test["id"].to_numpy())
        predictions.to_csv(args.predictions_dir / f"{key}_predictions.csv", index=False)

    comparison = pd.DataFrame(results)
    comparison.to_csv(args.report_dir / "final_test_results.csv", index=False)
    print(comparison.to_string(index=False))
    print("Final reporting only. Ridge(alpha=1) remains the Official Course Model.")


if __name__ == "__main__":
    main()

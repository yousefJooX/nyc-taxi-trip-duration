"""Final inference/evaluation: run only after feature/model selection is locked."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
try:
    from .features import prepare, apply_clusters
except ImportError:
    from features import prepare, apply_clusters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=Path("split/test.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--evaluate", action="store_true", help="Explicitly evaluate labels if present")
    args = parser.parse_args()
    if args.output.resolve() == args.test.resolve():
        parser.error("Prediction output must not overwrite input data")
    raw = pd.read_csv(args.test)
    model = joblib.load(args.model_dir/"ridge.joblib")
    pickup = joblib.load(args.model_dir/"pickup_kmeans.joblib")
    dropoff = joblib.load(args.model_dir/"dropoff_kmeans.joblib")
    frame = apply_clusters(prepare(raw), pickup, dropoff)
    prediction = model.predict(frame)
    if args.evaluate:
        if "trip_duration" not in raw:
            parser.error("Evaluation requires trip_duration labels")
        target = raw.trip_duration.to_numpy(dtype=float)
        if not np.isfinite(target).all() or (target < 0).any():
            parser.error("trip_duration must be finite and nonnegative")
        y = np.log1p(target)
        print(f"TEST RMSE (log): {np.sqrt(mean_squared_error(y, prediction)):.6f}")
        print(f"TEST R2 (log): {r2_score(y, prediction):.6f}")
    output = pd.DataFrame({"predicted_trip_duration": np.maximum(0, np.expm1(prediction))})
    if "id" in raw:
        output.insert(0, "id", raw.id.to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output):,} predictions to {args.output}")


if __name__ == "__main__":
    main()

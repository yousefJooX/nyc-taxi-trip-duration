"""Small synthetic checks. No course CSVs are opened."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.features import (NUMERIC_FEATURES, CATEGORICAL_FEATURES, add_features,
                          fit_kmeans, add_clusters, haversine_distance)
from src import test as final_test
from src.train import clean_training_target, get_log_target, make_preprocessor, build_models


class PipelineChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.raw = pd.DataFrame({
            "pickup_latitude": 40.7 + rng.random(60) * 0.1,
            "pickup_longitude": -74 + rng.random(60) * 0.1,
            "dropoff_latitude": 40.7 + rng.random(60) * 0.1,
            "dropoff_longitude": -74 + rng.random(60) * 0.1,
            "pickup_datetime": pd.date_range("2016-01-01", periods=60, freq="h"),
            "vendor_id": 1, "store_and_fwd_flag": "N", "passenger_count": 1,
            "trip_duration": np.arange(60) + 100,
        })

    def test_expected_features_and_target_independence(self):
        features = add_features(self.raw)
        different_labels = add_features(self.raw.assign(trip_duration=9999))
        pd.testing.assert_frame_equal(features.drop(columns="trip_duration"),
                                      different_labels.drop(columns="trip_duration"))
        features = add_clusters(features, *fit_kmeans(features))
        self.assertTrue(set(NUMERIC_FEATURES + CATEGORICAL_FEATURES).issubset(features.columns))
        self.assertNotIn("manhattan_km", features.columns)
        self.assertAlmostEqual(float(haversine_distance(0, 0, 1, 0)), 111.19508, places=4)

    def test_missing_training_target_only(self):
        frame = self.raw.copy()
        frame.loc[0, "trip_duration"] = np.nan
        cleaned = clean_training_target(frame)
        self.assertEqual(len(cleaned), 59)
        self.assertEqual(len(frame), 60)  # Original input is unchanged.
        self.assertEqual(len(get_log_target(cleaned)), 59)
        with self.assertRaises(ValueError):
            get_log_target(frame)  # Validation/test labels cannot be silently removed.

    def test_kmeans_train_only_and_predict_only(self):
        train = add_features(self.raw.iloc[:40])
        val = add_features(self.raw.iloc[40:])
        train.loc[0, "pickup_latitude"] = 45.0
        pickup, dropoff = fit_kmeans(train)
        self.assertEqual(len(pickup.labels_), 39)
        self.assertEqual(len(dropoff.labels_), 39)
        centers = pickup.cluster_centers_.copy()
        with patch.object(KMeans, "fit", side_effect=AssertionError("Inference must not fit")):
            result = add_clusters(val, pickup, dropoff)
        self.assertEqual(len(result), len(val))
        np.testing.assert_array_equal(pickup.cluster_centers_, centers)

    def test_pipeline_roundtrip_and_train_scaler(self):
        train = add_features(self.raw.iloc[:40])
        val = add_features(self.raw.iloc[40:].assign(vendor_id=99))
        clusters = fit_kmeans(train)
        train = add_clusters(train, *clusters)
        val = add_clusters(val, *clusters)
        model = Pipeline([("preprocessor", make_preprocessor()),
                          ("model", Ridge(alpha=1, solver="lsqr", tol=1e-8))])
        model.fit(train, get_log_target(train))
        scaler = model.named_steps["preprocessor"].named_transformers_["num"]
        np.testing.assert_allclose(scaler.mean_, train[NUMERIC_FEATURES].mean())
        expected = model.predict(val)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ridge.joblib"
            joblib.dump(model, path)
            np.testing.assert_allclose(joblib.load(path).predict(val), expected)
        self.assertTrue(np.isfinite(expected).all())

    def test_missing_models_fail_before_reading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ["test.py", "--model-dir", directory, "--test", "must_not_be_read.csv"]
            with patch("sys.argv", argv):
                with patch("src.test.pd.read_csv", side_effect=AssertionError("Data was opened")):
                    with self.assertRaises(FileNotFoundError):
                        final_test.main()

    def test_polynomial_numeric_only_and_separate_preprocessors(self):
        frame = add_features(self.raw)
        frame = add_clusters(frame, *fit_kmeans(frame))
        models = build_models(jobs=1)
        self.assertEqual(len({id(m.named_steps["preprocessor"]) for m in models.values()}), 5)
        poly = models["polynomial_ridge"]
        poly.fit(frame, get_log_target(frame))
        pre = poly.named_steps["preprocessor"]
        numeric = pre.named_transformers_["num"]
        self.assertEqual(numeric.named_steps["polynomial"].n_features_in_, 10)
        self.assertEqual(numeric.named_steps["polynomial"].n_output_features_, 65)
        self.assertEqual(len(pre.named_transformers_["cat"].categories_), 16)


if __name__ == "__main__":
    unittest.main()

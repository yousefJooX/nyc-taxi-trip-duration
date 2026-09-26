"""Synthetic checks only; never opens the held-out test CSV or runs src/test.py."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from src.features import prepare, fit_clusters, apply_clusters, BASE_NUM, BASE_CAT, haversine
from src.train import make_pipeline, optional_logger
from sklearn.ensemble import RandomForestRegressor


class PipelineChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.raw = pd.DataFrame({
            "pickup_latitude": 40.7+rng.random(60)*0.1,
            "pickup_longitude": -74+rng.random(60)*0.1,
            "dropoff_latitude": 40.7+rng.random(60)*0.1,
            "dropoff_longitude": -74+rng.random(60)*0.1,
            "pickup_datetime": pd.date_range("2016-01-01", periods=60, freq="h"),
            "vendor_id": 1, "store_and_fwd_flag": "N", "passenger_count": 1,
        })

    def test_geography_and_target_independence(self):
        self.assertAlmostEqual(float(haversine(0, 0, 1, 0)), 111.19508, places=4)
        first = prepare(self.raw)
        second = prepare(self.raw.assign(trip_duration=99999)).drop(columns="trip_duration")
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((first.manhattan_km >= first.distance_km-1e-9).all())
        same = self.raw.copy()
        same["dropoff_latitude"] = same.pickup_latitude
        same["dropoff_longitude"] = same.pickup_longitude
        same = prepare(same)
        np.testing.assert_allclose(same[["bearing_sin", "bearing_cos", "distance_km"]], 0)

    def test_no_validation_fit_and_roundtrip(self):
        train = prepare(self.raw.iloc[:40])
        val = prepare(self.raw.iloc[40:].assign(vendor_id=99))
        clusters = fit_clusters(train, clean=True)
        self.assertEqual(len(clusters[0].labels_), len(train))
        centers = [c.cluster_centers_.copy() for c in clusters]
        with patch.object(type(clusters[0]), "fit", side_effect=AssertionError("Inference refitted KMeans")):
            train = apply_clusters(train, *clusters)
            val = apply_clusters(val, *clusters)
        for model, expected in zip(clusters, centers):
            np.testing.assert_array_equal(model.cluster_centers_, expected)
        model = make_pipeline(BASE_NUM, BASE_CAT)
        model.fit(train, np.linspace(4, 7, len(train)))
        scaler = model.named_steps["preprocessor"].named_transformers_["num"]
        np.testing.assert_allclose(scaler.mean_, train[BASE_NUM].mean())
        prediction = model.predict(val)
        self.assertTrue(np.isfinite(prediction).all())
        other = make_pipeline(BASE_NUM, BASE_CAT)
        other.fit(val, np.ones(len(val)))
        np.testing.assert_array_equal(prediction, model.predict(val))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.joblib"
            joblib.dump(model, path)
            np.testing.assert_array_equal(prediction, joblib.load(path).predict(val))

    def test_clean_cluster_scope_and_polynomial(self):
        train = prepare(self.raw.iloc[:40])
        train.loc[train.index[0], "pickup_latitude"] = 45.0
        clusters = fit_clusters(train, clean=True)
        self.assertEqual(len(clusters[0].labels_), 39)
        self.assertEqual(len(clusters[1].labels_), 39)
        train = apply_clusters(train, *clusters)
        model = make_pipeline(BASE_NUM, BASE_CAT, degree=2)
        model.fit(train, np.linspace(4, 7, len(train)))
        preprocessing = model.named_steps["preprocessor"]
        numeric = preprocessing.named_transformers_["num"]
        self.assertEqual(numeric.named_steps["polynomial"].n_features_in_, len(BASE_NUM))
        self.assertEqual(len(preprocessing.named_transformers_["cat"].categories_), len(BASE_CAT))

    def test_tree_dense_values_match_ridge_sparse_values(self):
        train = prepare(self.raw)
        train = apply_clusters(train, *fit_clusters(train))
        y = np.linspace(4, 7, len(train))
        ridge = make_pipeline(BASE_NUM, BASE_CAT).fit(train, y)
        forest = make_pipeline(BASE_NUM, BASE_CAT,
            RandomForestRegressor(n_estimators=1, max_depth=2, random_state=42)).fit(train, y)
        sparse = ridge.named_steps["preprocessor"].transform(train)
        dense = forest.named_steps["preprocessor"].transform(train)
        self.assertIsInstance(dense, np.ndarray)
        np.testing.assert_allclose(dense, sparse.toarray())

    def test_mlflow_missing_is_optional(self):
        self.assertIsNone(optional_logger(None))
        with patch.dict("sys.modules", {"mlflow": None}):
            with self.assertWarns(UserWarning):
                self.assertIsNone(optional_logger("http://127.0.0.1:5000"))


if __name__ == "__main__":
    unittest.main()

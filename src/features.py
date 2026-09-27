"""Features used by both training and final prediction."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

NUMERIC_FEATURES = [
    "log_distance", "distance_km", "lat_diff", "lon_diff",
    "hour", "dayofweek", "month", "dayofyear", "bearing_sin", "bearing_cos",
]
CATEGORICAL_FEATURES = [
    "vendor_id", "store_and_fwd_flag", "passenger_count", "same_location",
    "pickup_cluster", "dropoff_cluster",
    "pickup_near_jfk", "pickup_near_lga", "pickup_near_ewr",
    "dropoff_near_jfk", "dropoff_near_lga", "dropoff_near_ewr",
    "morning_rush", "evening_rush", "is_weekend", "route_cluster",
]
AIRPORTS = {
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
    "ewr": (40.6895, -74.1745),
}
PICKUP_COORDS = ["pickup_latitude", "pickup_longitude"]
DROPOFF_COORDS = ["dropoff_latitude", "dropoff_longitude"]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Straight-line distance on Earth, in kilometers."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def add_features(df):
    df = df.copy()
    pickup_time = pd.to_datetime(df["pickup_datetime"], errors="raise")
    if pickup_time.isna().any():
        raise ValueError("pickup_datetime contains missing values")
    df["hour"] = pickup_time.dt.hour
    df["dayofweek"] = pickup_time.dt.dayofweek
    df["month"] = pickup_time.dt.month
    df["dayofyear"] = pickup_time.dt.dayofyear

    for end in ["pickup", "dropoff"]:
        valid = (df[f"{end}_latitude"].between(-90, 90)
                 & df[f"{end}_longitude"].between(-180, 180))
        if not valid.all():
            raise ValueError(f"Missing or invalid {end} coordinates")

    df["distance_km"] = haversine_distance(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    df["log_distance"] = np.log1p(df["distance_km"])
    df["lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    df["lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["same_location"] = (df["distance_km"] <= 0.1).astype(int)

    # Sine/cosine keep directions near 0 and 360 degrees close together.
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    delta_lon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    bearing = np.arctan2(
        np.sin(delta_lon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon),
    )
    df["bearing_sin"] = np.where(df["distance_km"] > 0, np.sin(bearing), 0)
    df["bearing_cos"] = np.where(df["distance_km"] > 0, np.cos(bearing), 0)

    for end in ["pickup", "dropoff"]:
        for airport, (airport_lat, airport_lon) in AIRPORTS.items():
            distance = haversine_distance(
                df[f"{end}_latitude"], df[f"{end}_longitude"], airport_lat, airport_lon,
            )
            df[f"{end}_near_{airport}"] = (distance <= 2).astype(int)

    weekday = df["dayofweek"] < 5
    df["morning_rush"] = (weekday & df["hour"].between(7, 9)).astype(int)
    df["evening_rush"] = (weekday & df["hour"].between(16, 18)).astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def fit_kmeans(train):
    # This mask chooses rows for KMeans fitting, not for model evaluation.
    in_nyc = (
        train["pickup_latitude"].between(40.4, 41.0)
        & train["dropoff_latitude"].between(40.4, 41.0)
        & train["pickup_longitude"].between(-74.3, -73.6)
        & train["dropoff_longitude"].between(-74.3, -73.6)
    )
    pickup_kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
    dropoff_kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
    pickup_kmeans.fit(train.loc[in_nyc, PICKUP_COORDS])
    dropoff_kmeans.fit(train.loc[in_nyc, DROPOFF_COORDS])
    return pickup_kmeans, dropoff_kmeans


def add_clusters(df, pickup_kmeans, dropoff_kmeans):
    df = df.copy()
    df["pickup_cluster"] = pickup_kmeans.predict(df[PICKUP_COORDS])
    df["dropoff_cluster"] = dropoff_kmeans.predict(df[DROPOFF_COORDS])
    df["route_cluster"] = (df["pickup_cluster"].astype(str)
                           + "->" + df["dropoff_cluster"].astype(str))
    return df

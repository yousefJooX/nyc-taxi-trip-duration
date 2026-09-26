"""Target-independent features shared by training and inference."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

BASE_NUM = ["log_distance", "distance_km", "lat_diff", "lon_diff",
            "hour", "dayofweek", "month", "dayofyear"]
BASE_CAT = ["vendor_id", "store_and_fwd_flag", "passenger_count", "same_location",
            "pickup_cluster", "dropoff_cluster"]
ADDITIONS = {
    "B_manhattan": (["manhattan_km"], []),
    "C_bearing": (["bearing_sin", "bearing_cos"], []),
    "D_airports": ([], [f"{end}_near_{airport}" for end in ("pickup", "dropoff")
                       for airport in ("jfk", "lga", "ewr")]),
    "E_rush": ([], ["morning_rush", "evening_rush"]),
    "F_weekend": ([], ["is_weekend"]),
    "G_route": ([], ["route_cluster"]),
}
# Approximate airport reference points; fixed 2 km radius, not tuned to targets.
AIRPORTS = {"jfk": (40.6413, -73.7781), "lga": (40.7769, -73.8740),
            "ewr": (40.6895, -74.1745)}


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def prepare(frame):
    df = frame.copy()
    for end in ("pickup", "dropoff"):
        lat, lon = df[f"{end}_latitude"], df[f"{end}_longitude"]
        if not (lat.between(-90, 90) & lon.between(-180, 180)).all():
            raise ValueError("Coordinates must be finite valid latitude/longitude values")
    dt = pd.to_datetime(df["pickup_datetime"], errors="raise")
    if dt.isna().any():
        raise ValueError("pickup_datetime contains missing values")
    for name in ("hour", "dayofweek", "month", "dayofyear"):
        df[name] = getattr(dt.dt, name)
    lat1, lon1 = df.pickup_latitude, df.pickup_longitude
    lat2, lon2 = df.dropoff_latitude, df.dropoff_longitude
    df["distance_km"] = haversine(lat1, lon1, lat2, lon2)
    df["log_distance"] = np.log1p(df.distance_km)
    # Preserve the existing large-data script's absolute differences.
    df["lat_diff"], df["lon_diff"] = (lat2-lat1).abs(), (lon2-lon1).abs()
    df["same_location"] = (df.distance_km <= 0.1).astype(int)
    # Sum north/south and east/west great-circle legs in km, not degrees.
    df["manhattan_km"] = haversine(lat1, lon1, lat2, lon1) + haversine(lat2, lon1, lat2, lon2)
    phi1, phi2, delta = np.radians(lat1), np.radians(lat2), np.radians(lon2-lon1)
    angle = np.arctan2(np.sin(delta)*np.cos(phi2),
                      np.cos(phi1)*np.sin(phi2)-np.sin(phi1)*np.cos(phi2)*np.cos(delta))
    # Direction is undefined at identical endpoints: neutral components.
    df["bearing_sin"] = np.where(df.distance_km > 0, np.sin(angle), 0)
    df["bearing_cos"] = np.where(df.distance_km > 0, np.cos(angle), 0)
    for end in ("pickup", "dropoff"):
        for airport, (lat, lon) in AIRPORTS.items():
            df[f"{end}_near_{airport}"] = (haversine(
                df[f"{end}_latitude"], df[f"{end}_longitude"], lat, lon) <= 2).astype(int)
    weekday = df.dayofweek < 5
    df["morning_rush"] = (weekday & df.hour.between(7, 9)).astype(int)
    df["evening_rush"] = (weekday & df.hour.between(16, 18)).astype(int)
    df["is_weekend"] = (df.dayofweek >= 5).astype(int)
    return df


def coordinates(end):
    return [f"{end}_latitude", f"{end}_longitude"]


def fit_clusters(train, clean=False, k=5):
    """Only training rows enter this function; never pass validation here."""
    mask = np.ones(len(train), dtype=bool)
    if clean:
        for end in ("pickup", "dropoff"):
            mask &= train[f"{end}_latitude"].between(40.4, 41.0).to_numpy()
            mask &= train[f"{end}_longitude"].between(-74.3, -73.6).to_numpy()
    if mask.sum() < k:
        raise ValueError("Too few in-area training coordinates for KMeans")
    return tuple(KMeans(n_clusters=k, random_state=42, n_init="auto").fit(
        train.loc[mask, coordinates(end)]) for end in ("pickup", "dropoff"))


def apply_clusters(frame, pickup, dropoff):
    df = frame.copy()
    for end, model in (("pickup", pickup), ("dropoff", dropoff)):
        df[f"{end}_cluster"] = model.predict(df[coordinates(end)])
    df["route_cluster"] = df.pickup_cluster.astype(str) + "->" + df.dropoff_cluster.astype(str)
    return df

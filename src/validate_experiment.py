from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

PROJECT = Path.home() / "Documents" / "MapMatchingProject"

TRIPS_DIR = PROJECT / "data" / "processed" / "trips"
ROADS_FILE = PROJECT / "data" / "processed" / "osm_roads.geojson"

SEED = 42
NOISE_LEVELS = [5, 10, 20]


def main():
    print("Loading roads and one trajectory...")

    roads = gpd.read_file(ROADS_FILE)

    roads = roads[
        roads.geometry.notna()
        & roads.geometry.geom_type.isin(
            ["LineString", "MultiLineString"]
        )
    ].copy()

    roads = roads.to_crs("EPSG:32650").reset_index(drop=True)

    trip_files = sorted(TRIPS_DIR.glob("trip_[0-9][0-9][0-9].csv"))

    if not trip_files:
        raise FileNotFoundError("No extracted trajectories found.")

    trip = pd.read_csv(trip_files[0])

    trip["latitude"] = pd.to_numeric(
        trip["latitude"], errors="coerce"
    )
    trip["longitude"] = pd.to_numeric(
        trip["longitude"], errors="coerce"
    )

    trip = trip.dropna(subset=["latitude", "longitude"])

    if len(trip) > 100:
        trip = trip.iloc[
            np.linspace(0, len(trip) - 1, 100, dtype=int)
        ]

    points = gpd.GeoDataFrame(
        trip,
        geometry=gpd.points_from_xy(
            trip["longitude"], trip["latitude"]
        ),
        crs="EPSG:4326",
    ).to_crs("EPSG:32650")

    xy = np.column_stack([
        points.geometry.x.to_numpy(),
        points.geometry.y.to_numpy(),
    ])

    # Distance to the nearest OSM road before adding noise.
    original_distances = []

    for x, y in xy:
        point = Point(float(x), float(y))
        nearest = roads.geometry.distance(point).min()
        original_distances.append(float(nearest))

    print(
        "\nOriginal GPS-to-road distances (m):",
        pd.Series(original_distances).describe().round(2).to_dict(),
    )

    rng = np.random.default_rng(SEED)

    for noise in NOISE_LEVELS:
        noisy = xy + rng.normal(0, noise, xy.shape)

        displacements = np.linalg.norm(noisy - xy, axis=1)

        print(f"\nGaussian noise setting: {noise} m")
        print(
            "Actual displacement (m):",
            pd.Series(displacements).describe().round(2).to_dict(),
        )

        noisy_distances = []

        for x, y in noisy:
            point = Point(float(x), float(y))
            nearest = roads.geometry.distance(point).min()
            noisy_distances.append(float(nearest))

        print(
            "Noisy GPS-to-road distances (m):",
            pd.Series(noisy_distances).describe().round(2).to_dict(),
        )

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
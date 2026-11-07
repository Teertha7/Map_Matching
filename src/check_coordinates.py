from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT = Path.home() / "Documents" / "MapMatchingProject"

TRIPS_DIR = PROJECT / "data" / "processed" / "trips"
ROADS_FILE = PROJECT / "data" / "processed" / "osm_roads.geojson"

# Load the first trajectory
trip_files = sorted(TRIPS_DIR.glob("trip_[0-9][0-9][0-9].csv"))
trip = pd.read_csv(trip_files[0])

print("\n--- GPS COORDINATES ---")
print(trip[["latitude", "longitude"]].describe())

print("\nGPS sample:")
print(trip[["latitude", "longitude"]].head())

# Load the roads
roads = gpd.read_file(ROADS_FILE)

print("\n--- ROAD CRS ---")
print(roads.crs)

print("\n--- ROAD BOUNDS (ORIGINAL CRS) ---")
print(roads.total_bounds)

print("\n--- ROAD GEOMETRY SAMPLE ---")
print(roads.geometry.head(3))

# Convert GPS points to a GeoDataFrame
points = gpd.GeoDataFrame(
    trip,
    geometry=gpd.points_from_xy(
        trip["longitude"],
        trip["latitude"],
    ),
    crs="EPSG:4326",
)

print("\n--- GPS BOUNDS (WGS84) ---")
print(points.total_bounds)

# Project both datasets into the same CRS
if roads.crs is None:
    raise ValueError("OSM roads have no CRS metadata.")

roads_projected = roads.to_crs("EPSG:4326")
points_projected = points.to_crs("EPSG:4326")

print("\n--- GPS BOUNDS (WGS84) ---")
print(points_projected.total_bounds)

print("\n--- ROADS BOUNDS (WGS84) ---")
print(roads_projected.total_bounds)

print("\n--- OVERLAP CHECK ---")
print(
    "GPS and road bounding boxes overlap:",
    (
        points_projected.total_bounds[0] <= roads_projected.total_bounds[2]
        and points_projected.total_bounds[2] >= roads_projected.total_bounds[0]
        and points_projected.total_bounds[1] <= roads_projected.total_bounds[3]
        and points_projected.total_bounds[3] >= roads_projected.total_bounds[1]
    ),
)
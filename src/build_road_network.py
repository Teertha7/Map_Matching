from pathlib import Path
import math

import geopandas as gpd
import osmnx as ox
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path.home() / "Documents" / "MapMatchingProject"

TRIPS_DIR = PROJECT_DIR / "data" / "processed" / "trips"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "osm_roads.geojson"


# ============================================================
# SETTINGS
# ============================================================

# Extra distance around all GPS trajectories.
MARGIN_METERS = 2000

# Coordinate reference system for GPS and downloaded OSM data.
GPS_CRS = "EPSG:4326"

# Download driving-accessible roads.
NETWORK_TYPE = "drive"


# ============================================================
# LOAD ALL TRAJECTORIES
# ============================================================

def load_all_trajectories():
    """
    Load all trip CSV files and combine their GPS coordinates.
    """

    if not TRIPS_DIR.exists():
        raise FileNotFoundError(
            f"Trajectories directory not found: {TRIPS_DIR}"
        )

    trip_files = sorted(
        TRIPS_DIR.glob("trip_*.csv")
    )

    if not trip_files:
        raise FileNotFoundError(
            f"No trip CSV files found in {TRIPS_DIR}"
        )

    all_trips = []

    print(f"Found {len(trip_files)} trip files.")

    for trip_file in trip_files:

        try:
            trip = pd.read_csv(trip_file)

            required_columns = {
                "latitude",
                "longitude"
            }

            if not required_columns.issubset(trip.columns):
                print(
                    f"Skipping {trip_file.name}: "
                    "missing latitude/longitude columns."
                )
                continue

            trip = trip[
                ["latitude", "longitude"]
            ].copy()

            trip["latitude"] = pd.to_numeric(
                trip["latitude"],
                errors="coerce"
            )

            trip["longitude"] = pd.to_numeric(
                trip["longitude"],
                errors="coerce"
            )

            trip = trip.dropna(
                subset=["latitude", "longitude"]
            )

            # Remove invalid geographic coordinates.
            trip = trip[
                trip["latitude"].between(-90, 90)
                & trip["longitude"].between(-180, 180)
            ]

            if not trip.empty:
                all_trips.append(trip)

                print(
                    f"Loaded {trip_file.name}: "
                    f"{len(trip)} GPS points"
                )

        except Exception as error:
            print(
                f"Could not read {trip_file.name}: {error}"
            )

    if not all_trips:
        raise RuntimeError(
            "No valid GPS trajectories were found."
        )

    gps = pd.concat(
        all_trips,
        ignore_index=True
    )

    print("\nTotal GPS points:", len(gps))

    print("\nGPS coordinate summary:")
    print(
        gps[["latitude", "longitude"]].describe()
    )

    return gps


# ============================================================
# CALCULATE BOUNDING BOX
# ============================================================

def calculate_bounding_box(gps):
    """
    Calculate a bounding box around all trajectories,
    including a configurable margin in meters.

    Returns:
        north, south, east, west
    """

    lat_min = gps["latitude"].min()
    lat_max = gps["latitude"].max()

    lon_min = gps["longitude"].min()
    lon_max = gps["longitude"].max()

    lat_mid = (lat_min + lat_max) / 2

    # Approximate conversion from meters to degrees.
    lat_margin = MARGIN_METERS / 111320

    cos_lat = abs(
        math.cos(math.radians(lat_mid))
    )

    # Prevent division by zero near the poles.
    cos_lat = max(cos_lat, 0.01)

    lon_margin = (
        MARGIN_METERS / (111320 * cos_lat)
    )

    north = min(
        90,
        lat_max + lat_margin
    )

    south = max(
        -90,
        lat_min - lat_margin
    )

    east = min(
        180,
        lon_max + lon_margin
    )

    west = max(
        -180,
        lon_min - lon_margin
    )

    print("\nCalculated bounding box:")
    print(f"North: {north:.6f}")
    print(f"South: {south:.6f}")
    print(f"East:  {east:.6f}")
    print(f"West:  {west:.6f}")

    return north, south, east, west


# ============================================================
# DOWNLOAD ROAD NETWORK
# ============================================================

def download_road_network(north, south, east, west):
    """
    Download driving-accessible OpenStreetMap roads
    for the calculated bounding box.
    """

    print("\nConfiguring OpenStreetMap download...")

    ox.settings.use_cache = True
    ox.settings.log_console = True

    print("Downloading roads. This may take some time...")

    # OSMnx 2.x expects bbox=(left, bottom, right, top).
    # These values correspond to:
    # (west, south, east, north).
    graph = ox.graph_from_bbox(
        bbox=(west, south, east, north),
        network_type=NETWORK_TYPE,
        simplify=True
    )

    edges = ox.graph_to_gdfs(
        graph,
        nodes=False,
        edges=True
    )

    edges = edges.reset_index()

    if edges.empty:
        raise RuntimeError(
            "The OpenStreetMap download returned no roads."
        )

    # Ensure that the road network has a known CRS.
    if edges.crs is None:
        raise RuntimeError(
            "Downloaded road network has no CRS."
        )

    # GeoJSON should use longitude/latitude coordinates.
    edges = edges.to_crs(GPS_CRS)

    print("\nRoad network downloaded successfully.")
    print("Number of road segments:", len(edges))
    print("Road CRS:", edges.crs)

    if "highway" in edges.columns:
        print("\nMost common road types:")

        print(
            edges["highway"]
            .astype(str)
            .value_counts()
            .head(15)
        )

    return edges


# ============================================================
# VERIFY GEOGRAPHIC OVERLAP
# ============================================================

def verify_road_coverage(gps, roads):
    """
    Check whether the GPS bounding box overlaps the road
    network bounding box.
    """

    points = gpd.GeoDataFrame(
        gps.copy(),
        geometry=gpd.points_from_xy(
            gps["longitude"],
            gps["latitude"]
        ),
        crs=GPS_CRS
    )

    if roads.crs is None:
        raise RuntimeError(
            "Road network CRS is missing."
        )

    roads_wgs84 = roads.to_crs(GPS_CRS)

    gps_bounds = points.total_bounds
    road_bounds = roads_wgs84.total_bounds

    print("\nGPS bounds:")
    print(gps_bounds)

    print("\nRoad bounds:")
    print(road_bounds)

    overlap = (
        gps_bounds[0] <= road_bounds[2]
        and gps_bounds[2] >= road_bounds[0]
        and gps_bounds[1] <= road_bounds[3]
        and gps_bounds[3] >= road_bounds[1]
    )

    print("\nGPS and road bounding boxes overlap:", overlap)

    if not overlap:
        raise RuntimeError(
            "GPS trajectories and road network do not overlap. "
            "The road network will not be saved."
        )

    # Calculate nearest-road distances for a sample of points.
    # Project to a local metric CRS for distance calculations.
    sample_size = min(100, len(points))

    sample = points.sample(
        n=sample_size,
        random_state=42
    )

    local_crs = sample.estimate_utm_crs()

    if local_crs is None:
        raise RuntimeError(
            "Could not determine a suitable projected CRS."
        )

    sample_projected = sample.to_crs(local_crs)
    roads_projected = roads_wgs84.to_crs(local_crs)

    nearest = gpd.sjoin_nearest(
        sample_projected,
        roads_projected[["geometry"]],
        how="left",
        distance_col="distance_m"
    )

    distances = nearest["distance_m"].dropna()

    if distances.empty:
        raise RuntimeError(
            "Could not calculate GPS-to-road distances."
        )

    print("\nGPS-to-nearest-road distance summary (meters):")
    print(distances.describe())

    # A bounding-box overlap alone is not sufficient:
    # reject a network if sampled points are still extremely far
    # from every road.
    if distances.median() > 5000:
        raise RuntimeError(
            "Median GPS-to-road distance exceeds 5 km. "
            "Check whether the downloaded roads cover the "
            "actual trajectory locations."
        )

    print("\nRoad coverage verification passed.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BUILDING OPENSTREETMAP ROAD NETWORK")
    print("=" * 60)

    # 1. Load all trajectories.
    gps = load_all_trajectories()

    # 2. Calculate the bounding box around every trajectory.
    north, south, east, west = calculate_bounding_box(gps)

    # 3. Download the corresponding road network.
    roads = download_road_network(
        north,
        south,
        east,
        west
    )

    # 4. Verify geographic coverage before saving.
    verify_road_coverage(
        gps,
        roads
    )

    # 5. Save the verified road network.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    roads.to_file(
        OUTPUT_FILE,
        driver="GeoJSON"
    )

    print("\n" + "=" * 60)
    print("ROAD NETWORK SAVED SUCCESSFULLY")
    print("=" * 60)

    print(f"Output file: {OUTPUT_FILE}")
    print(f"Road segments: {len(roads)}")


if __name__ == "__main__":
    main()
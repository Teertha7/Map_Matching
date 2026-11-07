from pathlib import Path
import ast
import warnings

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path.home() / "Documents" / "MapMatchingProject"

GPS_FILE = PROJECT / "data" / "processed" / "selected_trip.csv"
ROADS_FILE = PROJECT / "data" / "processed" / "osm_roads.geojson"

RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

NOISE_LEVELS = [5, 10, 20]
DEGRADATION_TYPES = [
    "gaussian",
    "persistent_bias",
    "correlated_drift",
]

MAX_GPS_POINTS = 600
MAX_CANDIDATES = 8
CANDIDATE_RADIUS_M = 100
RANDOM_SEED = 42
DRIFT_CORRELATION = 0.95

ALLOWED_CLASSES = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential", "unclassified", "service",
}


# ============================================================
# GPS DEGRADATION
# ============================================================

def degrade_gps_positions(
    gps_xy,
    degradation_type,
    severity_m,
    seed=42,
    correlation=DRIFT_CORRELATION,
):
    """
    Add simulated GNSS errors to projected x/y coordinates.

    gaussian:
        Independent zero-mean Gaussian error per point and axis.
    persistent_bias:
        A constant offset vector for the entire trajectory. The
        vector's magnitude is severity_m.
    correlated_drift:
        An AR(1) error process with temporally correlated errors.
        Each coordinate's stationary standard deviation is
        approximately severity_m.

    These are simulated error models, not real GNSS measurements.
    """
    rng = np.random.default_rng(seed)
    n_points = len(gps_xy)

    if degradation_type == "gaussian":
        errors = rng.normal(
            loc=0.0,
            scale=severity_m,
            size=(n_points, 2),
        )

    elif degradation_type == "persistent_bias":
        angle = rng.uniform(0.0, 2.0 * np.pi)
        bias = severity_m * np.array(
            [np.cos(angle), np.sin(angle)]
        )
        errors = np.tile(bias, (n_points, 1))

    elif degradation_type == "correlated_drift":
        if not 0 <= correlation < 1:
            raise ValueError("correlation must be in [0, 1).")

        errors = np.zeros((n_points, 2), dtype=float)
        if n_points:
            # Initialize from the stationary distribution.
            errors[0] = rng.normal(
                loc=0.0,
                scale=severity_m,
                size=2,
            )

        innovation_scale = severity_m * np.sqrt(
            1.0 - correlation ** 2
        )

        for i in range(1, n_points):
            innovation = rng.normal(
                loc=0.0,
                scale=innovation_scale,
                size=2,
            )
            errors[i] = (
                correlation * errors[i - 1] + innovation
            )

    else:
        raise ValueError(
            f"Unknown degradation type: {degradation_type}"
        )

    return gps_xy + errors


# ============================================================
# ROAD CLASS NORMALIZATION
# ============================================================

def normalize_road_class(value):
    """Normalize OSM highway values, including list-like values."""
    if isinstance(value, (list, tuple, set)):
        values = [str(v).strip() for v in value]
        return next(
            (v for v in values if v in ALLOWED_CLASSES),
            values[0] if values else "unclassified",
        )

    value = str(value).strip()

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                values = [str(v).strip() for v in parsed]
                return next(
                    (v for v in values if v in ALLOWED_CLASSES),
                    values[0] if values else "unclassified",
                )
        except (ValueError, SyntaxError):
            pass

    return value


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not GPS_FILE.exists():
        raise FileNotFoundError(f"GPS file not found: {GPS_FILE}")
    if not ROADS_FILE.exists():
        raise FileNotFoundError(f"Road file not found: {ROADS_FILE}")

    print("Loading GPS trajectory...")
    gps = pd.read_csv(GPS_FILE)
    gps = gps.dropna(subset=["latitude", "longitude"]).copy()

    if gps.empty:
        raise ValueError("GPS file contains no valid latitude/longitude rows.")

    if len(gps) > MAX_GPS_POINTS:
        step = int(np.ceil(len(gps) / MAX_GPS_POINTS))
        gps = gps.iloc[::step].copy()

    gps = gps.reset_index(drop=True)
    gps["point_id"] = np.arange(len(gps))
    # Nearby points are kept in the same split group. All degradation
    # variants of a point also share this group, preventing variant leakage.
    gps["group"] = gps["point_id"] // 10

    gps_geo = gpd.GeoDataFrame(
        gps,
        geometry=gpd.points_from_xy(
            gps["longitude"], gps["latitude"]
        ),
        crs="EPSG:4326",
    )

    print("Loading OpenStreetMap roads...")
    roads = gpd.read_file(ROADS_FILE)
    roads = roads[
        roads.geometry.notna()
        & roads.geometry.geom_type.isin(
            ["LineString", "MultiLineString"]
        )
    ].copy()

    if "highway" not in roads.columns:
        raise ValueError("No 'highway' column found in the road data.")

    roads["road_class"] = roads["highway"].apply(normalize_road_class)
    roads = roads[roads["road_class"].isin(ALLOWED_CLASSES)].copy()

    if roads.empty:
        raise ValueError(
            "No usable roads remain. Check the OSM highway values."
        )

    print("\nRoad classes retained:")
    print(roads["road_class"].value_counts().head(15).to_string())

    # GeoLife trajectories are in the Beijing area; use UTM zone 50N.
    gps_metric = gps_geo.to_crs("EPSG:32650")
    roads_metric = roads.to_crs("EPSG:32650")

    gps_xy = np.column_stack([
        gps_metric.geometry.x.to_numpy(),
        gps_metric.geometry.y.to_numpy(),
    ])

    road_geometries = roads_metric.geometry.to_list()
    road_classes = roads_metric["road_class"].to_numpy()
    road_index = roads_metric.sindex

    print(f"\nGPS points used: {len(gps_xy)}")
    print(f"Road segments available: {len(road_geometries)}")

    return gps, gps_xy, road_geometries, road_classes, road_index


# ============================================================
# SPATIAL CANDIDATE SEARCH
# ============================================================

def nearby_road_indices(point, road_index, radius_m):
    """Return integer positions of road geometries near a point."""
    result = road_index.query(point.buffer(radius_m))
    indices = [int(idx) for idx in np.asarray(result).reshape(-1)]

    if not indices:
        nearest_result = road_index.nearest(point)
        nearest_array = np.asarray(nearest_result).reshape(-1)
        if len(nearest_array) == 0:
            return []
        # Shapely/GeoPandas spatial-index nearest may return query and
        # tree indices. For one query, the final index is the tree index.
        indices = [int(nearest_array[-1])]

    return indices


def find_nearest_road(point, road_index, road_geometries):
    indices = nearby_road_indices(
        point, road_index, CANDIDATE_RADIUS_M
    )
    if not indices:
        raise RuntimeError("Spatial index did not return any road candidates.")

    return min(
        indices,
        key=lambda i: road_geometries[i].distance(point),
    )


def road_heading(geometry):
    """Approximate road orientation in radians."""
    if geometry.geom_type == "MultiLineString":
        geometry = max(geometry.geoms, key=lambda g: g.length)

    coords = list(geometry.coords)
    if len(coords) < 2:
        return 0.0

    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    return float(np.arctan2(dy, dx))


# ============================================================
# GENERATE CANDIDATES AND PROXY LABELS
# ============================================================

def build_candidates(
    gps, gps_xy, road_geometries, road_classes, road_index
):
    print("\nFinding proxy target roads...")
    target_roads = []

    # Proxy target = road nearest to the original, unperturbed GPS point.
    # This is not independently verified ground truth.
    for i, (x, y) in enumerate(gps_xy):
        point = Point(float(x), float(y))
        target = find_nearest_road(
            point, road_index, road_geometries
        )
        target_roads.append(int(target))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(gps_xy)} GPS points")

    records = []

    for degradation_type in DEGRADATION_TYPES:
        for noise_m in NOISE_LEVELS:
            print(
                f"\nGenerating candidates for {degradation_type}, "
                f"severity={noise_m} m..."
            )

            # The seed differs for each condition but remains repeatable.
            seed = RANDOM_SEED + 100 * DEGRADATION_TYPES.index(
                degradation_type
            ) + noise_m

            noisy_xy = degrade_gps_positions(
                gps_xy=gps_xy,
                degradation_type=degradation_type,
                severity_m=noise_m,
                seed=seed,
            )

            for i, (x, y) in enumerate(noisy_xy):
                observed = Point(float(x), float(y))

                indices = nearby_road_indices(
                    observed, road_index, CANDIDATE_RADIUS_M
                )
                if not indices:
                    continue

                distances = [
                    (
                        int(idx),
                        float(
                            road_geometries[int(idx)].distance(observed)
                        ),
                    )
                    for idx in indices
                ]
                distances.sort(key=lambda item: item[1])
                candidates = distances[:MAX_CANDIDATES]

                target_idx = int(target_roads[i])
                candidate_ids = [idx for idx, _ in candidates]

                # Evaluate only points whose proxy target is in the
                # candidate set; report coverage in the final table.
                if target_idx not in candidate_ids:
                    continue

                for rank, (idx, distance) in enumerate(candidates):
                    records.append({
                        "degradation": degradation_type,
                        "noise_m": int(noise_m),
                        "point_id": int(i),
                        "group": int(gps.iloc[i]["group"]),
                        "candidate_id": int(idx),
                        "distance_m": distance,
                        "road_class": str(road_classes[idx]),
                        "road_heading": road_heading(
                            road_geometries[idx]
                        ),
                        "is_target": int(idx == target_idx),
                        "rank": int(rank),
                    })

    data = pd.DataFrame(records)

    if data.empty:
        raise RuntimeError(
            "No candidates generated. Check GPS/road overlap or "
            "increase CANDIDATE_RADIUS_M."
        )

    print(f"\nCandidate rows generated: {len(data):,}")
    print(f"Positive proxy labels: {int(data['is_target'].sum()):,}")
    print(
        "Unique evaluated point-condition pairs: "
        f"{data[['degradation', 'noise_m', 'point_id']].drop_duplicates().shape[0]:,}"
    )
    return data


# ============================================================
# FEATURES AND MODEL
# ============================================================

def make_features(data):
    return pd.get_dummies(
        data[["distance_m", "road_heading", "rank", "road_class"]],
        columns=["road_class"],
        dtype=float,
    )


def train_model(data):
    features = make_features(data)
    labels = data["is_target"].astype(int)
    groups = data["group"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_SEED,
    )
    train_idx, test_idx = next(
        splitter.split(features, labels, groups=groups)
    )

    X_train = features.iloc[train_idx]
    y_train = labels.iloc[train_idx]

    if y_train.nunique() < 2:
        raise RuntimeError(
            "Training data contains only one label class. "
            "Try a different split seed or more GPS points."
        )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    print(f"Training candidate rows: {len(train_idx):,}")
    print(f"Test candidate rows: {len(test_idx):,}")

    return model, features.columns.tolist(), test_idx


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(data, model, feature_columns, test_idx):
    features = make_features(data)
    features = features.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    scored = data.copy()
    scored["ml_score"] = model.predict_proba(features)[:, 1]
    scored["is_test_row"] = False
    scored.loc[test_idx, "is_test_row"] = True

    results = []

    for degradation_type in DEGRADATION_TYPES:
        for noise_m in NOISE_LEVELS:
            subset = scored[
                (scored["degradation"] == degradation_type)
                & (scored["noise_m"] == noise_m)
                & scored["is_test_row"]
            ]

            predictions = []

            for point_id, candidates in subset.groupby("point_id"):
                targets = candidates[candidates["is_target"] == 1]
                if targets.empty:
                    continue

                target_id = int(targets.iloc[0]["candidate_id"])
                nearest = candidates.loc[
                    candidates["distance_m"].idxmin()
                ]
                ml_choice = candidates.loc[
                    candidates["ml_score"].idxmax()
                ]

                predictions.append({
                    "target": target_id,
                    "nearest": int(nearest["candidate_id"]),
                    "random_forest": int(ml_choice["candidate_id"]),
                })

            pred_df = pd.DataFrame(predictions)
            if pred_df.empty:
                print(
                    f"No test points for {degradation_type}, {noise_m} m."
                )
                continue

            results.append({
                "degradation": degradation_type,
                "noise_m": int(noise_m),
                "evaluated_points": int(len(pred_df)),
                "nearest_accuracy": accuracy_score(
                    pred_df["target"], pred_df["nearest"]
                ),
                "rf_accuracy": accuracy_score(
                    pred_df["target"], pred_df["random_forest"]
                ),
                "nearest_macro_f1": f1_score(
                    pred_df["target"], pred_df["nearest"],
                    average="macro", zero_division=0,
                ),
                "rf_macro_f1": f1_score(
                    pred_df["target"], pred_df["random_forest"],
                    average="macro", zero_division=0,
                ),
            })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError(
            "No evaluation results produced. Check test-set coverage."
        )

    return results_df


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(results, model, feature_columns):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = TABLES_DIR / "experiment_metrics.csv"
    results.to_csv(metrics_path, index=False)

    model_path = RESULTS_DIR / "map_matching_rf.joblib"
    joblib.dump({
        "model": model,
        "feature_columns": feature_columns,
        "degradation_types": DEGRADATION_TYPES,
        "noise_levels_m": NOISE_LEVELS,
    }, model_path)

    # One plot per degradation type avoids mixing multiple conditions
    # on the same x-axis.
    for degradation_type in DEGRADATION_TYPES:
        subset = results[
            results["degradation"] == degradation_type
        ].sort_values("noise_m")

        if subset.empty:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(
            subset["noise_m"],
            subset["nearest_accuracy"],
            marker="o",
            label="Nearest-road baseline",
        )
        plt.plot(
            subset["noise_m"],
            subset["rf_accuracy"],
            marker="o",
            label="Random Forest",
        )
        plt.xlabel("GPS degradation severity (m)")
        plt.ylabel("Agreement with proxy labels")
        plt.title(
            "Road-selection agreement under "
            + degradation_type.replace("_", " ").title()
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = FIGURES_DIR / f"robustness_{degradation_type}.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved plot: {plot_path}")

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved model: {model_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(RANDOM_SEED)

    for folder in [FIGURES_DIR, TABLES_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    gps, gps_xy, road_geometries, road_classes, road_index = load_data()

    data = build_candidates(
        gps, gps_xy, road_geometries, road_classes, road_index
    )

    model, feature_columns, test_idx = train_model(data)
    results = evaluate_model(
        data, model, feature_columns, test_idx
    )

    save_outputs(results, model, feature_columns)

    print("\n========== EXPERIMENT RESULTS ==========")
    print(results.to_string(index=False))

    print(
        "\nIMPORTANT LIMITATIONS:\n"
        "1. Target roads are the nearest roads to original GPS coordinates. "
        "These are proxy labels, not verified ground truth.\n"
        "2. The split holds out chunks of one trajectory, not entire unseen trips.\n"
        "3. Noise conditions are simulated. They do not prove performance "
        "under real GNSS failures.\n"
        "4. Accuracy means agreement with proxy road-segment IDs; it is not "
        "highway/service-road classification accuracy."
    )


if __name__ == "__main__":
    main()

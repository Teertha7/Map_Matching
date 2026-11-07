#!/usr/bin/env python3
"""Multi-trajectory map-matching experiment with honest candidate evaluation.

Run from the project root:
    source .venv/bin/activate
    python src/run_experiment_multitrip.py

Labels are proxy labels: the nearest OSM road to the clean GPS point. They are
not independently verified ground truth.

Important evaluation design:
- Candidate generation uses ONLY the noisy point and the road network.
- The clean-GPS proxy target is NOT inserted into the candidate list.
- Candidate recall is measured on every test point.
- Ranking accuracy is measured only on points where the proxy target was
  naturally present in the candidate list. This is conditional ranking
  performance, and must be reported alongside candidate recall.
"""
from pathlib import Path
import ast
import warnings

import geopandas as gpd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

PROJECT = Path.home() / "Documents" / "MapMatchingProject"
TRIPS_DIR = PROJECT / "data" / "processed" / "trips"
ROADS_FILE = PROJECT / "data" / "processed" / "osm_roads.geojson"
RESULTS = PROJECT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

NOISE_LEVELS = [5, 10, 20]
DEGRADATIONS = ["gaussian", "persistent_bias", "correlated_drift"]
MAX_POINTS = 300
RADIUS_M = 100
MAX_CANDIDATES = 8
SEED = 42
RHO = 0.95

ROAD_CLASSES = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "residential", "unclassified", "service",
}


def perturb(xy, kind, scale, seed):
    """Apply one of three synthetic coordinate-error models."""
    rng = np.random.default_rng(seed)
    n = len(xy)

    if kind == "gaussian":
        # scale is the standard deviation of EACH coordinate axis.
        error = rng.normal(0, scale, (n, 2))
    elif kind == "persistent_bias":
        # A fixed offset vector with magnitude scale for the whole trajectory.
        angle = rng.uniform(0, 2 * np.pi)
        offset = scale * np.array([np.cos(angle), np.sin(angle)])
        error = np.tile(offset, (n, 1))
    elif kind == "correlated_drift":
        # Stationary AR(1) correlated error; not an accumulating random walk.
        error = np.zeros((n, 2), dtype=float)
        if n:
            error[0] = rng.normal(0, scale, 2)
        innovation_sd = scale * np.sqrt(1 - RHO**2)
        for i in range(1, n):
            error[i] = RHO * error[i - 1] + rng.normal(
                0, innovation_sd, 2
            )
    else:
        raise ValueError(f"Unknown degradation type: {kind}")

    return xy + error


def normalize_class(value):
    if isinstance(value, (list, tuple, set)):
        values = [str(v).strip() for v in value]
        return next(
            (v for v in values if v in ROAD_CLASSES),
            values[0] if values else "unclassified",
        )

    value = str(value).strip()
    if value.startswith("[") and value.endswith("]"):
        try:
            values = ast.literal_eval(value)
            if isinstance(values, (list, tuple, set)):
                values = [str(v).strip() for v in values]
                return next(
                    (v for v in values if v in ROAD_CLASSES),
                    values[0] if values else "unclassified",
                )
        except (ValueError, SyntaxError):
            pass
    return value


def load_trips():
    trips = []
    for path in sorted(TRIPS_DIR.glob("trip_[0-9][0-9][0-9].csv")):
        df = pd.read_csv(path)
        if not {"latitude", "longitude"}.issubset(df.columns):
            print(f"Skipping {path.name}: missing latitude/longitude columns")
            continue

        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])
        df = df[
            df.latitude.between(-90, 90)
            & df.longitude.between(-180, 180)
        ]

        if len(df) > MAX_POINTS:
            keep = np.linspace(0, len(df) - 1, MAX_POINTS, dtype=int)
            df = df.iloc[keep]

        if len(df) >= 20:
            df = df.reset_index(drop=True)
            df["trip_id"] = path.stem
            trips.append(df)

    if len(trips) < 4:
        raise RuntimeError(
            f"Only {len(trips)} usable trips found under {TRIPS_DIR}; "
            "expected at least 4."
        )

    print(f"Loaded {len(trips)} trajectories.")
    return trips


def load_roads():
    if not ROADS_FILE.exists():
        raise FileNotFoundError(f"Road file not found: {ROADS_FILE}")

    roads = gpd.read_file(ROADS_FILE)
    roads = roads[
        roads.geometry.notna()
        & roads.geometry.geom_type.isin(["LineString", "MultiLineString"])
    ].copy()

    if "highway" not in roads.columns:
        raise RuntimeError("OSM road file has no 'highway' field.")

    roads["road_class"] = roads["highway"].apply(normalize_class)
    roads = roads[roads.road_class.isin(ROAD_CLASSES)].copy()

    if roads.empty:
        raise RuntimeError("No usable OSM roads after filtering.")

    roads = roads.to_crs("EPSG:32650").reset_index(drop=True)
    print(f"Loaded {len(roads):,} OSM road geometries.")
    return roads


def line_heading(geom):
    if geom.geom_type == "MultiLineString":
        geom = max(geom.geoms, key=lambda g: g.length)
    coords = list(geom.coords)
    if len(coords) < 2:
        return 0.0
    return float(
        np.arctan2(
            coords[-1][1] - coords[0][1],
            coords[-1][0] - coords[0][0],
        )
    )


def spatial_candidates(point, roads, index):
    """Return naturally generated nearby road candidates; never insert labels."""
    ids = []
    for radius in (RADIUS_M, 250, 500, 1000):
        found = np.asarray(index.query(point.buffer(radius))).reshape(-1)
        ids = sorted(set(found.astype(int).tolist()))
        if len(ids) >= 2:
            break

    if len(ids) < 2:
        try:
            near = np.asarray(index.nearest(point)).reshape(-1).astype(int).tolist()
            ids = sorted(set(ids + near))
        except Exception:
            pass

    # Rank by distance and cap the candidate list. Target labels are not used.
    ranked = sorted(
        ((cid, float(roads.geometry.iloc[cid].distance(point))) for cid in ids),
        key=lambda item: item[1],
    )
    return ranked[:MAX_CANDIDATES]


def nearest_index(point, roads, index):
    """Find nearest road globally using the spatial index, with safe fallback."""
    try:
        nearest_ids = np.asarray(index.nearest(point)).reshape(-1).astype(int)
        if len(nearest_ids):
            return min(
                set(nearest_ids.tolist()),
                key=lambda i: roads.geometry.iloc[i].distance(point),
            )
    except Exception:
        pass

    # Fallback if the installed spatial-index backend does not support nearest.
    ids = spatial_candidates(point, roads, index)
    if not ids:
        return None
    return min(ids, key=lambda item: item[1])[0]


def feature_frame(df):
    return pd.get_dummies(
        df[["distance_m", "road_heading", "rank", "road_class"]],
        columns=["road_class"],
        dtype=float,
    )


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    trips = load_trips()
    roads = load_roads()
    geoms = roads.geometry.to_list()
    classes = roads.road_class.astype(str).to_numpy()
    index = roads.sindex

    trip_ids = np.array([str(t.trip_id.iloc[0]) for t in trips])
    train_idx, test_idx = next(
        GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=SEED
        ).split(np.zeros(len(trips)), groups=trip_ids)
    )
    train_ids = set(trip_ids[train_idx])
    test_ids = set(trip_ids[test_idx])
    print(f"Train trajectories ({len(train_ids)}): {sorted(train_ids)}")
    print(f"Test trajectories ({len(test_ids)}): {sorted(test_ids)}")

    candidate_rows = []
    point_rows = []

    for ti, trip in enumerate(trips, 1):
        tid = str(trip.trip_id.iloc[0])
        projected = gpd.GeoDataFrame(
            trip,
            geometry=gpd.points_from_xy(trip.longitude, trip.latitude),
            crs="EPSG:4326",
        ).to_crs("EPSG:32650")

        xy = np.column_stack([
            projected.geometry.x.to_numpy(),
            projected.geometry.y.to_numpy(),
        ])
        print(f"[{ti}/{len(trips)}] Processing {tid}: {len(xy)} points")

        # Proxy labels are computed once from the original, clean GPS points.
        targets = [
            nearest_index(Point(float(x), float(y)), roads, index)
            for x, y in xy
        ]

        for di, kind in enumerate(DEGRADATIONS):
            for severity in NOISE_LEVELS:
                noisy = perturb(
                    xy, kind, severity,
                    SEED + 10000 * ti + 100 * di + severity,
                )

                for pi, (x, y) in enumerate(noisy):
                    target = targets[pi]
                    if target is None:
                        continue

                    point = Point(float(x), float(y))
                    ranked = spatial_candidates(point, roads, index)
                    candidate_ids = [cid for cid, _ in ranked]
                    target_present = target in candidate_ids

                    point_rows.append({
                        "trip_id": tid,
                        "split": "train" if tid in train_ids else "test",
                        "degradation": kind,
                        "noise_m": severity,
                        "point_id": pi,
                        "target_id": int(target),
                        "target_class": str(classes[target]),
                        "candidate_count": len(ranked),
                        "target_in_candidates": bool(target_present),
                    })

                    # Ranking is learnable only when the proxy target is
                    # naturally in the candidate set. Do not force it in.
                    if not target_present:
                        continue

                    for rank, (cid, dist) in enumerate(ranked):
                        candidate_rows.append({
                            "trip_id": tid,
                            "split": "train" if tid in train_ids else "test",
                            "degradation": kind,
                            "noise_m": severity,
                            "point_id": pi,
                            "candidate_id": int(cid),
                            "distance_m": float(dist),
                            "road_heading": line_heading(geoms[cid]),
                            "road_class": str(classes[cid]),
                            "is_target": int(cid == target),
                            "rank": rank,
                        })

    data = pd.DataFrame(candidate_rows)
    point_data = pd.DataFrame(point_rows)
    if point_data.empty:
        raise RuntimeError("No valid points were processed; check GPS/OSM overlap.")

    # Diagnostics: each retained candidate set must contain exactly one proxy
    # target. Candidate recall itself is computed from point_data separately.
    if data.empty:
        raise RuntimeError(
            "No target-containing candidate sets were generated. "
            "Check coordinate systems, road coverage, and candidate radius."
        )

    set_counts = data.groupby(
        ["trip_id", "degradation", "noise_m", "point_id"]
    ).is_target.agg(["sum", "count"])

    if not (set_counts["sum"] == 1).all():
        bad = set_counts[set_counts["sum"] != 1].head()
        raise RuntimeError(
            "Expected exactly one target in every retained candidate set. "
            f"Examples:\n{bad}"
        )

    print(
        f"\nAll point-condition records: {len(point_data):,}; "
        f"naturally target-containing sets: {len(set_counts):,}"
    )
    print(
        "Candidate-set sizes (retained sets):",
        set_counts["count"].value_counts().sort_index().to_dict(),
    )
    print(
        "Target-in-candidate rate by split:",
        point_data.groupby("split").target_in_candidates.mean().round(4).to_dict(),
    )

    train = data[data.split == "train"].copy()
    if train.empty:
        raise RuntimeError("No training candidate sets contain the proxy target.")

    X_train = feature_frame(train)
    feature_cols = X_train.columns.tolist()
    y_train = train.is_target.astype(int)
    print("Training labels:", y_train.value_counts().to_dict())

    if y_train.nunique() < 2:
        raise RuntimeError(
            "Training data has only one label class. Inspect candidate generation."
        )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    print(f"Training Random Forest on {len(train):,} candidate rows...")
    model.fit(X_train, y_train)

    test = data[data.split == "test"].copy()
    X_test = feature_frame(test).reindex(columns=feature_cols, fill_value=0)
    test["score"] = model.predict_proba(X_test)[:, 1]

    result_rows = []
    for kind in DEGRADATIONS:
        for severity in NOISE_LEVELS:
            all_points = point_data[
                (point_data.split == "test")
                & (point_data.degradation == kind)
                & (point_data.noise_m == severity)
            ]
            retained = test[
                (test.degradation == kind)
                & (test.noise_m == severity)
            ]

            # Candidate recall denominator includes every test point with a
            # valid proxy target, including points whose target was not found.
            candidate_recall = (
                float(all_points.target_in_candidates.mean())
                if len(all_points) else np.nan
            )

            predictions = []
            for (_, _), candidates in retained.groupby(["trip_id", "point_id"]):
                target_rows = candidates[candidates.is_target == 1]
                if len(target_rows) != 1:
                    continue

                target_row = target_rows.iloc[0]
                nearest_row = candidates.loc[candidates.distance_m.idxmin()]
                rf_row = candidates.loc[candidates.score.idxmax()]

                predictions.append((
                    int(target_row.candidate_id),
                    int(nearest_row.candidate_id),
                    int(rf_row.candidate_id),
                    str(target_row.road_class),
                    str(nearest_row.road_class),
                    str(rf_row.road_class),
                ))

            p = pd.DataFrame(
                predictions,
                columns=[
                    "target", "nearest", "rf", "target_class",
                    "nearest_class", "rf_class",
                ],
            )

            if p.empty:
                result_rows.append({
                    "degradation": kind,
                    "noise_m": severity,
                    "test_points": len(all_points),
                    "candidate_recall": candidate_recall,
                    "ranked_points": 0,
                    "nearest_segment_accuracy_conditional": np.nan,
                    "rf_segment_accuracy_conditional": np.nan,
                    "nearest_road_class_accuracy_conditional": np.nan,
                    "rf_road_class_accuracy_conditional": np.nan,
                    "nearest_road_class_macro_f1_conditional": np.nan,
                    "rf_road_class_macro_f1_conditional": np.nan,
                })
                continue

            result_rows.append({
                "degradation": kind,
                "noise_m": severity,
                "test_points": len(all_points),
                "candidate_recall": candidate_recall,
                "ranked_points": len(p),
                "nearest_segment_accuracy_conditional": accuracy_score(
                    p.target, p.nearest
                ),
                "rf_segment_accuracy_conditional": accuracy_score(
                    p.target, p.rf
                ),
                "nearest_road_class_accuracy_conditional": accuracy_score(
                    p.target_class, p.nearest_class
                ),
                "rf_road_class_accuracy_conditional": accuracy_score(
                    p.target_class, p.rf_class
                ),
                # Road-class macro-F1 is more interpretable than macro-F1
                # over thousands of individual OSM segment IDs.
                "nearest_road_class_macro_f1_conditional": f1_score(
                    p.target_class, p.nearest_class,
                    average="macro", zero_division=0,
                ),
                "rf_road_class_macro_f1_conditional": f1_score(
                    p.target_class, p.rf_class,
                    average="macro", zero_division=0,
                ),
            })

    result = pd.DataFrame(result_rows)
    if result.empty:
        raise RuntimeError("No evaluation results were produced.")

    metrics_path = TABLES / "experiment_metrics_multitrip.csv"
    result.to_csv(metrics_path, index=False)

    point_path = TABLES / "candidate_recall_multitrip.csv"
    point_data.to_csv(point_path, index=False)

    model_path = RESULTS / "map_matching_rf_multitrip.joblib"
    joblib.dump({
        "model": model,
        "feature_columns": feature_cols,
        "train_trip_ids": sorted(train_ids),
        "test_trip_ids": sorted(test_ids),
        "label_definition": (
            "nearest OSM road to clean GPS coordinate; proxy label only"
        ),
        "candidate_policy": (
            "natural noisy-coordinate candidates; target is never forcibly inserted"
        ),
        "ranking_metrics_are_conditional": True,
        "noise_models": {
            "gaussian": "per-axis standard deviation in metres",
            "persistent_bias": "fixed offset vector magnitude in metres",
            "correlated_drift": f"stationary AR(1), rho={RHO}",
        },
    }, model_path)

    for kind in DEGRADATIONS:
        part = result[result.degradation == kind].sort_values("noise_m")
        if part.empty:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(
            part.noise_m, part.candidate_recall,
            marker="o", label="Candidate recall",
        )
        plt.plot(
            part.noise_m, part.nearest_segment_accuracy_conditional,
            marker="o", label="Nearest-road accuracy (conditional)",
        )
        plt.plot(
            part.noise_m, part.rf_segment_accuracy_conditional,
            marker="o", label="Random Forest accuracy (conditional)",
        )
        plt.xlabel("Simulated GNSS error severity (m)")
        plt.ylabel("Score")
        plt.ylim(0, 1.02)
        plt.title(f"Unseen-trajectory test: {kind.replace('_', ' ').title()}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES / f"multitrip_{kind}.png", dpi=180)
        plt.close()

    print("\n========== MULTI-TRAJECTORY TEST RESULTS ==========")
    print(result.to_string(index=False))
    print(
        f"\nMetrics: {metrics_path}\n"
        f"Candidate diagnostics: {point_path}\n"
        f"Model: {model_path}"
    )
    print(
        "\nINTERPRETATION: candidate_recall is the fraction of all test points "
        "whose clean-GPS proxy road appears naturally in the noisy-point "
        "candidate set. Ranking metrics are conditional on that event and "
        "must not be reported alone. The clean-GPS nearest-road label is only "
        "a proxy, noise is simulated, and GeoLife includes mixed transport "
        "modes; these results do not establish verified highway/service-road "
        "classification."
    )


if __name__ == "__main__":
    main()

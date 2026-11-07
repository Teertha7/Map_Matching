
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path.home() / "Documents" / "MapMatchingProject"
DATA_DIR = PROJECT_DIR / "data" / "raw" / "Geolife Trajectories 1.3" / "Data"
OUTPUT_DIR = PROJECT_DIR / "results" / "tables"

TARGET_USERS = {"128", "163"}

def load_trajectory(path):
    try:
        df = pd.read_csv(path, skiprows=6, header=None)

        if df.shape[1] < 7:
            return None

        df = df.iloc[:, :7].copy()
        df.columns = [
            "latitude", "longitude", "unused",
            "altitude", "days", "date", "time"
        ]

        df["latitude"] = pd.to_numeric(
            df["latitude"], errors="coerce"
        )
        df["longitude"] = pd.to_numeric(
            df["longitude"], errors="coerce"
        )
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce"
        )

        df = df.dropna(
            subset=["latitude", "longitude", "timestamp"]
        )
        df = df[
            df["latitude"].between(-90, 90)
            & df["longitude"].between(-180, 180)
        ]

        return df.sort_values("timestamp")

    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return None


def main():
    results = []

    for user in sorted(TARGET_USERS):
        user_dir = DATA_DIR / user
        label_path = user_dir / "labels.txt"

        if not label_path.exists():
            print(f"No labels.txt for user {user}")
            continue

        labels = pd.read_csv(label_path, sep="\t")
        labels.columns = [str(c).strip() for c in labels.columns]

        labels["Transportation Mode"] = (
            labels["Transportation Mode"]
            .astype(str).str.strip().str.lower()
        )

        labels = labels[
            labels["Transportation Mode"].isin(["car", "taxi"])
        ].copy()

        labels["start"] = pd.to_datetime(
            labels["Start Time"], errors="coerce"
        )
        labels["end"] = pd.to_datetime(
            labels["End Time"], errors="coerce"
        )
        labels = labels.dropna(subset=["start", "end"])

        trajectory_files = sorted(
            (user_dir / "Trajectory").glob("*.plt")
        )

        print(f"\nUser {user}: {len(labels)} vehicle labels, "
              f"{len(trajectory_files)} trajectory files")

        # Load each trajectory once, then match label intervals.
        trajectories = []
        for path in trajectory_files:
            gps = load_trajectory(path)
            if gps is not None and not gps.empty:
                trajectories.append((path.name, gps))

        for idx, label in labels.iterrows():
            matched_parts = []

            for filename, gps in trajectories:
                points = gps[
                    (gps["timestamp"] >= label["start"])
                    & (gps["timestamp"] <= label["end"])
                ]

                if not points.empty:
                    part = points.copy()
                    part["trajectory_file"] = filename
                    matched_parts.append(part)

            if matched_parts:
                matched = pd.concat(matched_parts, ignore_index=True)
                matched = matched.drop_duplicates(
                    subset=["timestamp", "latitude", "longitude"]
                )

                results.append({
                    "user": user,
                    "label_index": idx,
                    "mode": label["Transportation Mode"],
                    "start": label["start"],
                    "end": label["end"],
                    "label_duration_minutes": (
                        label["end"] - label["start"]
                    ).total_seconds() / 60,
                    "matched_gps_points": len(matched),
                    "trajectory_files": (
                        matched["trajectory_file"].nunique()
                    ),
                    "first_gps_time": matched["timestamp"].min(),
                    "last_gps_time": matched["timestamp"].max(),
                    "latitude_min": matched["latitude"].min(),
                    "latitude_max": matched["latitude"].max(),
                    "longitude_min": matched["longitude"].min(),
                    "longitude_max": matched["longitude"].max()
                })
            else:
                results.append({
                    "user": user,
                    "label_index": idx,
                    "mode": label["Transportation Mode"],
                    "start": label["start"],
                    "end": label["end"],
                    "label_duration_minutes": (
                        label["end"] - label["start"]
                    ).total_seconds() / 60,
                    "matched_gps_points": 0,
                    "trajectory_files": 0,
                    "first_gps_time": None,
                    "last_gps_time": None,
                    "latitude_min": None,
                    "latitude_max": None,
                    "longitude_min": None,
                    "longitude_max": None
                })

    if not results:
        print("No label intervals processed.")
        return

    result_df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "matched_vehicle_gps.csv"
    result_df.to_csv(output_path, index=False)

    print("\n=== GPS points matched to vehicle labels ===")
    print(
        result_df.groupby(["user", "mode"]).agg(
            segments=("label_index", "count"),
            segments_with_gps=(
                "matched_gps_points",
                lambda x: (x > 0).sum()
            ),
            total_matched_points=("matched_gps_points", "sum"),
            median_points_per_segment=("matched_gps_points", "median")
        ).to_string()
    )

    print("\nSample matched intervals:")
    print(
        result_df[result_df["matched_gps_points"] > 0]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
